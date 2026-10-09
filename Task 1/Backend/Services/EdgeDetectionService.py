import numpy as np
from .UtilitiesService import ConvolveImage
import cv2

sobel_kernel_x = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]],
                  dtype=np.float64)
sobel_kernel_y = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]],
                  dtype=np.float64)

prewitt_kernel_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float64)

prewitt_kernel_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float64)

roberts_kernel_x = np.array([
    [1,  0],
    [0, -1]
], dtype=np.float64)

roberts_kernel_y = np.array([
    [0,  1],
    [-1, 0]
], dtype=np.float64)



def _prepare_image(image: np.ndarray, apply_blur: bool, sigma: float):
    if apply_blur:
        image = cv2.GaussianBlur(
            image,
            ksize=(0, 0),  # Kernel size determined by sigma
            sigmaX=sigma
        )
    return image


def SobelEdgeDetection(
    image: np.ndarray,
    apply_blur: bool = False,
    sigma: float = 1.0
):
    image = _prepare_image(image, apply_blur, sigma)

    G_x = ConvolveImage(image, sobel_kernel_x)
    G_y = ConvolveImage(image, sobel_kernel_y)

    gradient = np.sqrt(G_x**2 + G_y**2)

    return np.clip(gradient, 0, 255).astype(np.uint8)


def PrewittEdgeDetection(
    image: np.ndarray,
    apply_blur: bool = False,
    sigma: float = 1.0
):
    image = _prepare_image(image, apply_blur, sigma)

    G_x = ConvolveImage(image, prewitt_kernel_x)
    G_y = ConvolveImage(image, prewitt_kernel_y)

    gradient = np.sqrt(G_x**2 + G_y**2)

    return np.clip(gradient, 0, 255).astype(np.uint8)


def RobertsEdgeDetection(
    image: np.ndarray,
    apply_blur: bool = False,
    sigma: float = 1.0
):
    image = _prepare_image(image, apply_blur, sigma)

    G_x = ConvolveImage(image, roberts_kernel_x)
    G_y = ConvolveImage(image, roberts_kernel_y)

    gradient = np.sqrt(G_x**2 + G_y**2)

    return np.clip(gradient, 0, 255).astype(np.uint8)


def CannyEdgeDetection(
    image: np.ndarray,
    threshold1: int = 100,
    threshold2: int = 200,
    apply_blur: bool = False,
    sigma: float = 1.0,
    threshold_mode: str = "manual"
):
    image = _prepare_image(image, apply_blur, sigma)

    image = image.astype(np.uint8)

    if threshold_mode == "image_percentile":
        threshold1, threshold2 = np.percentile(
            image, [25, 75]
        )

    elif threshold_mode == "gradient_percentile":
        
        gradient_magnitude = SobelEdgeDetection(image)

        threshold1, threshold2 = np.percentile(
            gradient_magnitude, [25, 75]
        )

    elif threshold_mode != "manual":
        raise ValueError(
            "threshold_mode must be 'manual', "
            "'image_percentile', or 'gradient_percentile'."
        )

    return cv2.Canny(
        image,
        float(threshold1),
        float(threshold2)
    )