import json
from io import BytesIO

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
from Backend.Services.ActiveContourService import (
    compute_edge_map,
    segment_active_contour,
    draw_contours,
)

from Backend.Services.ChainCodeService import contour_to_chain_code



app = FastAPI()

# Allow the frontend (opened from a file or another port) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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

async def read_uploaded_image(file: UploadFile, convert_to_grayscale = True) -> np.ndarray:
    """Read an uploaded image and convert it to a NumPy array."""
    contents = await file.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image file."
        )

    # Convert color images from OpenCV's BGR format to RGB.
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported image channel count."
            )

    if convert_to_grayscale:
        image = TransformIntoGrayScale(image, mode="weighted")

    return image


# ----------------------------------------------------------------------------
# Active contour (greedy snake)
# ----------------------------------------------------------------------------

def snake_params(
    contour: str = Form(..., description='Initial contour drawn by the user, JSON: [[x, y], [x, y], ...] in image pixels'),
    num_points: int = Form(200, ge=3, le=2000),
    alpha: float = Form(1.0, ge=0),
    beta: float = Form(1.0, ge=0),
    gamma: float = Form(2.0, ge=0),
    scales: str = Form("12,6,3", description="Comma-separated attraction sigmas, coarse to fine"),
    max_iterations: int = Form(200, ge=1, le=2000),
    radius: int = Form(2, ge=1, le=5),
    convergence_threshold: float = Form(0.02, ge=0, le=1),
    use_corners: bool = Form(False),
) -> dict:
    """Parse and validate the form fields shared by the snake endpoints."""
    try:
        points = json.loads(contour)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="'contour' must be valid JSON: [[x, y], ...]")

    try:
        parsed_scales = tuple(float(s) for s in scales.split(",") if s.strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="'scales' must be comma-separated numbers, e.g. '12,6,3'")
    if not parsed_scales:
        raise HTTPException(status_code=400, detail="'scales' must contain at least one value.")

    return dict(
        points=points,
        num_points=num_points,
        scales=parsed_scales,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_iterations=max_iterations,
        radius=radius,
        convergence_threshold=convergence_threshold,
        use_corners=use_corners,
    )


async def run_active_contour(file: UploadFile, params: dict):
    """Read the image, run the snake in a worker thread. Returns (image, initial, final)."""
    image = await read_uploaded_image(file, convert_to_grayscale=False)

    try:
        # The snake is CPU-bound pure Python, keep it off the event loop.
        initial, final = await run_in_threadpool(segment_active_contour, image, **params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return image, initial, final


@app.post("/active-contour/edge-map")
async def active_contour_edge_map(file: UploadFile = File(...)):
    """Step 1: the edge map the snake is attracted to (PNG)."""
    image = await read_uploaded_image(file, convert_to_grayscale=False)
    edge_map = await run_in_threadpool(compute_edge_map, image)
    return process_image((edge_map * 255).astype(np.uint8))


@app.post("/active-contour")
async def active_contour(file: UploadFile = File(...), params: dict = Depends(snake_params)):
    """Run the snake and return the initial and final contour points as JSON."""
    _, initial, final = await run_active_contour(file, params)
    return {
        "initial_contour": initial.tolist(),
        "final_contour": final.tolist(),
    }


@app.post("/active-contour/overlay")
async def active_contour_overlay(file: UploadFile = File(...), params: dict = Depends(snake_params)):
    """Run the snake and return the image with initial (blue) and final (red) contours drawn on it (PNG)."""
    image, initial, final = await run_active_contour(file, params)
    return process_image(draw_contours(image, initial, final))

# ----------------------------------------------------------------------------
# Freeman 8-direction Chain Code
# ----------------------------------------------------------------------------

@app.post("/chain-code")
async def generate_chain_code(
    contour: str = Form(
        ...,
        description="Contour points in JSON format: [[x, y], [x, y], ...]"
    )
):
    """
    Convert contour points into an 8-direction Freeman chain code.

    Returns:
        start_pixel: Starting pixel of the contour.
        chain_code: List of Freeman direction codes.
    """

    try:
        points = json.loads(contour)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="'contour' must be valid JSON: [[x, y], ...]"
        )

    # Validate contour format
    if not isinstance(points, list) or len(points) < 2:
        raise HTTPException(
            status_code=400,
            detail="Contour must contain at least two points."
        )

    try:
        start_pixel, chain_code = contour_to_chain_code(points)

    except (ValueError, IndexError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    return {
        "start_pixel": list(start_pixel),
        "chain_code": chain_code,
        "chain_code_length": len(chain_code)
    }