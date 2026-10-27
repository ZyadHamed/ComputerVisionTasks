from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
import numpy as np
import cv2
from io import BytesIO

from Backend.Services.UtilitiesService import (
    TransformIntoGrayScale,
    NormalizeImage
)

from Backend.Services.NoiseService import add_noise

from Backend.Services.FiltersService import low_pass_filter

from Backend.Services.HistogramService import Histogram

app = FastAPI()


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

