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




