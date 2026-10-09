from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from io import BytesIO

from Backend.Services.EdgeDetectionService import (
    SobelEdgeDetection,
    PrewittEdgeDetection,
    RobertsEdgeDetection,
    CannyEdgeDetection
)

from Backend.Services.UtilitiesService import (
    TransformIntoGrayScale,
    NormalizeImage
)

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


@app.post("/edge-detection/sobel")
async def sobel(
    file: UploadFile = File(...),
    apply_blur: bool = Form(False),
    sigma: float = Form(1.0)
):
    image = await read_uploaded_image(file)

    if sigma <= 0:
        raise HTTPException(
            status_code=422,
            detail="sigma must be greater than zero."
        )

    result = SobelEdgeDetection(image, apply_blur, sigma)
    return process_image(result)


@app.post("/edge-detection/prewitt")
async def prewitt(
    file: UploadFile = File(...),
    apply_blur: bool = Form(False),
    sigma: float = Form(1.0)
):
    image = await read_uploaded_image(file)

    if sigma <= 0:
        raise HTTPException(
            status_code=422,
            detail="sigma must be greater than zero."
        )

    result = PrewittEdgeDetection(image, apply_blur, sigma)
    return process_image(result)


@app.post("/edge-detection/roberts")
async def roberts(
    file: UploadFile = File(...),
    apply_blur: bool = Form(False),
    sigma: float = Form(1.0)
):
    image = await read_uploaded_image(file)

    if sigma <= 0:
        raise HTTPException(
            status_code=422,
            detail="sigma must be greater than zero."
        )

    result = RobertsEdgeDetection(image, apply_blur, sigma)
    return process_image(result)



@app.post("/edge-detection/canny")
async def canny(
    file: UploadFile = File(...),
    threshold_mode: str = Form("manual"),
    threshold1: int = Form(100),
    threshold2: int = Form(200),
    apply_blur: bool = Form(False),
    sigma: float = Form(1.0)
):
    image = await read_uploaded_image(file)

    if threshold_mode not in {
        "image_percentile",
        "gradient_percentile",
        "manual"
    }:
        raise HTTPException(
            status_code=422,
            detail=(
                "threshold_mode must be 'image_percentile', "
                "'gradient_percentile', or 'manual'."
            )
        )

    if sigma <= 0:
        raise HTTPException(
            status_code=422,
            detail="sigma must be greater than zero."
        )

    if threshold_mode == "manual":
        if not 0 <= threshold1 <= 255 or not 0 <= threshold2 <= 255:
            raise HTTPException(
                status_code=422,
                detail="Canny thresholds must be between 0 and 255."
            )

        if threshold1 > threshold2:
            raise HTTPException(
                status_code=422,
                detail="threshold1 must be less than or equal to threshold2."
            )

    result = CannyEdgeDetection(
        image,
        threshold1,
        threshold2,
        apply_blur,
        sigma,
        threshold_mode=threshold_mode
    )

    return process_image(result)

@app.post("/image/normalize")
async def normalize_image(file: UploadFile = File(...)):
    image = await read_uploaded_image(file, convert_to_grayscale=False)

    result = NormalizeImage(image)

    return process_image(result)


@app.post("/image/grayscale")
async def grayscale_image(
    file: UploadFile = File(...),
    mode: str = Form("weighted")
):
    if mode not in ("weighted", "average"):
        raise HTTPException(
            status_code=422,
            detail="Mode must be either 'weighted' or 'average'."
        )

    contents = await file.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image file."
        )

    # OpenCV loads color images as BGR; convert to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = TransformIntoGrayScale(image, mode)

    return process_image(result)