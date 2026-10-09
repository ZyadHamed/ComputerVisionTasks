from .FrequencyFiltersService import LowPassFilter, HighPassFilter
import cv2
import numpy as np


def create_hybrid_image(image1: np.ndarray, image2: np.ndarray, radius: int = 30) -> np.ndarray:
    """Create a hybrid image by combining the low-frequency content of image1 with the high-frequency content of image2."""
    if image1 is None or image2 is None:
        raise ValueError("Both images must not be None.")

    if radius <= 0:
        raise ValueError("radius must be greater than zero.")

    if image1.ndim == 2 and image2.ndim == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    elif image1.ndim == 3 and image2.ndim == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

    if image1.shape[:2] != image2.shape[:2]:
        image2 = cv2.resize(
            image2,
            (image1.shape[1], image1.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    low_freq_image = LowPassFilter(image1, radius=radius)
    high_freq_image = HighPassFilter(image2, radius=radius)

    hybrid_image = low_freq_image + high_freq_image
    return np.clip(hybrid_image, 0, 255).astype(np.uint8)


create_hybird_image = create_hybrid_image

