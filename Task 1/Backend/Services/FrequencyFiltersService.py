import cv2
import numpy as np


def create_gaussian_mask(shape, radius: int = 30, center=None):
    """Create a Gaussian low-pass mask for frequency-domain filtering."""
    rows, cols = shape[:2]
    if center is None:
        center = (rows / 2, cols / 2)

    sigma = max(radius / 3.0, 1.0)
    y, x = np.ogrid[:rows, :cols]
    distance = ((x - center[1]) ** 2 + (y - center[0]) ** 2) / (2 * sigma ** 2)
    return np.exp(-distance)


def prepare_image_for_frequency_filter(image: np.ndarray) -> np.ndarray:
    """Ensure the input is a grayscale 2D array for frequency filtering."""
    if image is None:
        raise ValueError("Image cannot be None.")

    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError("Unsupported image channel count for frequency filtering.")

    return np.asarray(image, dtype=np.float64)


def ApplyFrequencyFilter(
    image: np.ndarray,
    filter_type: str = "low",
    radius: int = 30
) -> np.ndarray:
    """Apply a Gaussian low-pass or high-pass Fourier-domain filter."""

    if filter_type not in {"low", "high"}:
        raise ValueError("filter_type must be either 'low' or 'high'.")

    if radius <= 0:
        raise ValueError("radius must be greater than zero.")

    if image is None:
        raise ValueError("Image cannot be None.")

    image = np.asarray(image)

    if image.ndim not in (2, 3):
        raise ValueError("Image must be grayscale or a color image.")

    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Image must have 3 (RGB) or 4 (RGBA) channels.")

    rows, cols = image.shape[:2]
    gaussian_mask = create_gaussian_mask((rows, cols), radius=radius)

    if filter_type == "high":
        gaussian_mask = 1.0 - gaussian_mask

    frequency_image = np.fft.fft2(image, axes=(0, 1))
    shifted_frequency = np.fft.fftshift(frequency_image, axes=(0, 1))

    if image.ndim == 3:
        gaussian_mask = gaussian_mask[:, :, np.newaxis]

    filtered_frequency = shifted_frequency * gaussian_mask

    filtered_image = np.fft.ifft2(
        np.fft.ifftshift(filtered_frequency, axes=(0, 1)),
        axes=(0, 1)
    ).real

    return np.clip(filtered_image, 0, 255).astype(np.uint8)


def LowPassFilter(image: np.ndarray, radius: int = 30) -> np.ndarray:
    """Apply a Gaussian low-pass filter to an image in the frequency domain."""
    return ApplyFrequencyFilter(image, filter_type="low", radius=radius)


def HighPassFilter(image: np.ndarray, radius: int = 30) -> np.ndarray:
    """Apply a Gaussian high-pass filter to an image in the frequency domain."""
    return ApplyFrequencyFilter(image, filter_type="high", radius=radius)


