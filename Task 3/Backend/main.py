import json
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from Backend.Services.UtilitiesService import (
    TransformIntoGrayScale,
    NormalizeImage
)
from Backend.Services.LatestImageService import (
    save_latest_image,
    load_latest_image,
    load_latest_image_bytes,
)
from Backend.Services.FeatureMatchingService import (
    match_feature_sets,
    draw_matches,
    to_rgb_uint8,
    to_gray,
)
from Backend.Services.SIFTService import (
    extract_sift_features,
    draw_sift_keypoints,
)



app = FastAPI()

# Allow the frontend (opened from a file or another port) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Lets the browser read match stats and SIFT stats sent with visualization PNGs.
    expose_headers=[
        "X-Match-Count",
        "X-Matching-Time-Ms",
        "X-Descriptor-Time-Ms",
        "X-Keypoint-Count",
        "X-Execution-Time-Ms",
    ],
)


@app.get("/")
def read_root():
    return {"message": "Edge Detection API is running"}

def process_image(image: np.ndarray) -> StreamingResponse:
    """Encode the processed image as PNG and return it."""

    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    # Ensure valid PNG pixel values and data type if normalized.
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    success, encoded_image = cv2.imencode(".png", image)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode the processed image."
        )

    return StreamingResponse(
        BytesIO(encoded_image.tobytes()),
        media_type="image/png"
    )

# Keep upload sizes manageable. Adjust these for your project's expected images.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


async def read_uploaded_image(
    file: Optional[UploadFile] = None,
    convert_to_grayscale: bool = True,
    save_as_latest: bool = True,
) -> np.ndarray:
    """Read an uploaded image, or reuse the most recently saved image.

    A new upload is validated and saved as a lossless PNG before processing.
    Requests without a file load the saved PNG (which survives API restarts).
    """
    if file is None:
        try:
            image = await run_in_threadpool(load_latest_image)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No image has been uploaded yet. Upload one using "
                    "POST /images/upload or include 'file' in an image request."
                ),
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load the stored image: {exc}",
            ) from exc
    else:
        # Read one byte beyond the limit to reject oversized uploads.
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Uploaded image exceeds the 25 MiB limit.",
            )
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        try:
            image = cv2.imdecode(
                np.frombuffer(contents, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
        except cv2.error:
            image = None

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid or unsupported image file.",
            )

    if image.ndim not in (2, 3) or (
        image.ndim == 3 and image.shape[2] not in (3, 4)
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported image channel count.",
        )

    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 16-megapixel limit.",
        )

    # Only a newly supplied image changes the saved file. The original
    # OpenCV BGR/BGRA pixel data is stored before any color conversion.
    if file is not None and save_as_latest:
        try:
            await run_in_threadpool(save_latest_image, image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (OSError, cv2.error) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not save the uploaded image: {exc}",
            ) from exc

    # Keep the previous API's RGB convention for downstream services.
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    if convert_to_grayscale and image.ndim == 3:
        image = TransformIntoGrayScale(image, mode="weighted")

    return image


# ----------------------------------------------------------------------------
# Latest uploaded image
# ----------------------------------------------------------------------------

@app.post("/images/upload")
async def upload_image(file: UploadFile = File(...)):
    """Replace the single saved image; return its dimensions."""
    image = await read_uploaded_image(file, convert_to_grayscale=False)
    return {
        "message": "Latest image saved successfully.",
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": int(image.shape[2]) if image.ndim == 3 else 1,
    }


@app.get("/images/latest")
def get_latest_image():
    """Return the saved image as PNG (no browser cache)."""
    try:
        image_data = load_latest_image_bytes()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail="No image has been uploaded yet.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load the stored image: {exc}",
        ) from exc
    return Response(
        content=image_data,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# ----------------------------------------------------------------------------
# Feature matching (SSD / NCC) for Harris and SIFT features
# ----------------------------------------------------------------------------
#
# Multipart form fields shared by both endpoints:
#
#   image1     file, optional  first image (falls back to the latest uploaded image)
#   image2     file, required  second image
#   features1  JSON string     {"keypoints": [[x, y], ...], "descriptors": [[...], ...]}
#   features2  JSON string     same layout, for image2
#   method     "sift" | "harris"            (default "sift")
#   metric     "ssd" | "ncc"                (default "ssd")
#   ratio      float in (0, 1]              ratio test; 1 disables it (default 0.8)
#   threshold  float, optional              max SSD (ssd) / min NCC (ncc)
#   cross_check bool                        keep only mutual best matches (default true)
#   patch_size odd int                      Harris-only patch descriptor size (default 11)
#   max_matches int, optional               keep only the N best matches
#
# Input shapes:
#   keypoints   (N, 2)   [x, y] in the pixel frame of the uploaded image
#   descriptors (N, 128) for SIFT (required)
#               (N, D)   for Harris (optional; omitted -> patch_size**2 patch pixels)

def get_match_params(
    method: str = Form("sift"),
    metric: str = Form("ssd"),
    ratio: float = Form(0.8, gt=0, le=1),
    threshold: Optional[float] = Form(None),
    cross_check: bool = Form(True),
    patch_size: int = Form(11, ge=3, le=65),
    max_matches: Optional[int] = Form(None, ge=1),
) -> dict:
    return {
        "method": method,
        "metric": metric,
        "ratio": ratio,
        "threshold": threshold,
        "cross_check": cross_check,
        "patch_size": patch_size,
        "max_matches": max_matches,
    }


async def compute_feature_matches(
    image1: Optional[UploadFile],
    image2: UploadFile,
    features1: str,
    features2: str,
    params: dict,
):
    """Load both images, parse the feature JSON and run the matcher."""
    # Matching must not overwrite the saved "latest" image.
    rgb1 = to_rgb_uint8(
        await read_uploaded_image(image1, convert_to_grayscale=False, save_as_latest=False)
    )
    rgb2 = to_rgb_uint8(
        await read_uploaded_image(image2, convert_to_grayscale=False, save_as_latest=False)
    )

    try:
        feats1 = json.loads(features1)
        feats2 = json.loads(features2)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"features1/features2 must be valid JSON: {exc}",
        ) from exc

    try:
        result = await run_in_threadpool(
            match_feature_sets,
            to_gray(rgb1), feats1,
            to_gray(rgb2), feats2,
            **params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result, rgb1, rgb2


@app.post("/features/match")
async def match_features(
    image1: Optional[UploadFile] = File(None),
    image2: UploadFile = File(...),
    features1: str = Form(...),
    features2: str = Form(...),
    params: dict = Depends(get_match_params),
):
    """Match Harris or SIFT features between two images with SSD or NCC.

    Returns the matches (indices + coordinates + score) and the computation times.
    """
    result, _, _ = await compute_feature_matches(image1, image2, features1, features2, params)
    return result


@app.post("/features/match/visualize")
async def visualize_feature_matches(
    image1: Optional[UploadFile] = File(None),
    image2: UploadFile = File(...),
    features1: str = Form(...),
    features2: str = Form(...),
    circle_radius: int = Form(8, ge=1, le=100),
    draw_lines: bool = Form(True),
    params: dict = Depends(get_match_params),
):
    """Same matching as /features/match, returned as a side-by-side PNG with a
    circle on every matched point (same colour in both images) and, optionally,
    a line joining each pair. Stats are in the X-Match-Count, X-Matching-Time-Ms
    and X-Descriptor-Time-Ms response headers."""
    result, rgb1, rgb2 = await compute_feature_matches(
        image1, image2, features1, features2, params
    )
    canvas = await run_in_threadpool(
        draw_matches, rgb1, rgb2, result["matches"], circle_radius, draw_lines
    )

    response = process_image(canvas)
    response.headers["X-Match-Count"] = str(result["num_matches"])
    response.headers["X-Matching-Time-Ms"] = str(result["matching_time_ms"])
    response.headers["X-Descriptor-Time-Ms"] = str(result["descriptor_time_ms"])
    return response


# ----------------------------------------------------------------------------
# SIFT Feature Extraction and Visualization
# ----------------------------------------------------------------------------
#
# Multipart form fields for SIFT endpoints:
#   file                 file, optional    image to process (defaults to latest uploaded image)
#   max_keypoints        int, optional     maximum keypoints to retain by magnitude of response (default 500)
#   contrast_threshold   float             subpixel contrast threshold (default 0.03)
#   edge_threshold       float             ratio of principal curvatures threshold (default 10.0)
#   num_octaves          int               number of octaves in scale space (default 3)
#   num_intervals        int               number of scale intervals per octave (default 3)
#   sigma                float             base scale standard deviation (default 1.6)
#   assumed_blur         float             initial image blur assumption (default 0.5)
#   num_cells            int               descriptor grid cells per dimension (default 4 -> 4x4 grid)
#   num_bins             int               descriptor orientation bins per cell (default 8 -> 128-D)
#   peak_ratio           float             dominant orientation detection ratio (default 0.8)
#   descriptor_threshold float             descriptor clipping threshold (default 0.2)
#   save_as_latest       bool              whether to persist file as latest image (default false)
#   circle_radius        int, optional     visualization circle radius; None scales with keypoint sigma
#   draw_orientation     bool              whether to draw canonical orientation vector (default true)

def get_sift_params(
    max_keypoints: Optional[int] = Form(500, ge=1),
    contrast_threshold: float = Form(0.03, gt=0),
    edge_threshold: float = Form(10.0, gt=0),
    num_octaves: int = Form(3, ge=1, le=8),
    num_intervals: int = Form(3, ge=1, le=10),
    sigma: float = Form(1.6, gt=0),
    assumed_blur: float = Form(0.5, ge=0),
    num_cells: int = Form(4, ge=2, le=8),
    num_bins: int = Form(8, ge=4, le=36),
    peak_ratio: float = Form(0.8, gt=0, le=1.0),
    descriptor_threshold: float = Form(0.2, gt=0, le=1.0),
) -> dict:
    return {
        "max_keypoints": max_keypoints,
        "contrast_threshold": contrast_threshold,
        "edge_threshold": edge_threshold,
        "num_octaves": num_octaves,
        "num_intervals": num_intervals,
        "sigma": sigma,
        "assumed_blur": assumed_blur,
        "num_cells": num_cells,
        "num_bins": num_bins,
        "peak_ratio": peak_ratio,
        "descriptor_threshold": descriptor_threshold,
    }


@app.post("/features/sift")
@app.post("/sift")
@app.post("/sift/detect")
async def extract_sift_endpoint(
    file: Optional[UploadFile] = File(None),
    save_as_latest: bool = Form(False),
    params: dict = Depends(get_sift_params),
):
    """Extract SIFT keypoints and 128-D descriptors from an image.

    If file is omitted, processes the most recently saved image.
    The response format is directly compatible with POST /features/match:
      {
        "keypoints": [[x, y], ...],
        "descriptors": [[d0, ..., d127], ...],
        "keypoint_details": [{x, y, sigma, angle, response, octave, scale}, ...],
        "num_keypoints": N,
        "execution_time_ms": float
      }
    """
    image = await read_uploaded_image(
        file, convert_to_grayscale=False, save_as_latest=save_as_latest
    )
    result = await run_in_threadpool(extract_sift_features, image, **params)
    return result


@app.post("/features/sift/visualize")
@app.post("/sift/visualize")
async def visualize_sift_endpoint(
    file: Optional[UploadFile] = File(None),
    circle_radius: Optional[int] = Form(None, ge=1, le=100),
    draw_orientation: bool = Form(True),
    save_as_latest: bool = Form(False),
    params: dict = Depends(get_sift_params),
):
    """Extract SIFT keypoints and return a PNG with scale circles and orientation rays.

    Stats are included in the X-Keypoint-Count and X-Execution-Time-Ms headers.
    """
    image = await read_uploaded_image(
        file, convert_to_grayscale=False, save_as_latest=save_as_latest
    )
    result = await run_in_threadpool(extract_sift_features, image, **params)
    canvas = await run_in_threadpool(
        draw_sift_keypoints,
        image,
        result["keypoint_details"],
        circle_radius,
        draw_orientation,
    )
    response = process_image(canvas)
    response.headers["X-Keypoint-Count"] = str(result["num_keypoints"])
    response.headers["X-Execution-Time-Ms"] = str(result["execution_time_ms"])
    return response
