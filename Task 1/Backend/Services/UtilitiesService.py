import numpy as np
from scipy.signal import correlate2d


def _convolve_channel_manual(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel_height, kernel_width = kernel.shape

    pad_top = (kernel_height - 1) // 2
    pad_bottom = kernel_height - 1 - pad_top
    pad_left = (kernel_width - 1) // 2
    pad_right = kernel_width - 1 - pad_left

    padded = np.pad(
        channel,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
    )

    img_height, img_width = channel.shape
    output = np.zeros((img_height, img_width), dtype=np.float64)

    for i in range(img_height):
        for j in range(img_width):
            region = padded[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output


def _convolve_channel_scipy(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return correlate2d(
        channel.astype(np.float64),
        kernel.astype(np.float64),
        mode="same",
        boundary="fill",
        fillvalue=0,
    )


def ConvolveImage(image: np.ndarray, kernel: np.ndarray, manual: bool = True):
    """
    Apply a kernel to an image (cross-correlation, no kernel flip).

    manual=True  -> loop-based implementation
    manual=False -> scipy.signal.correlate2d
    """
    convolve_channel = _convolve_channel_manual if manual else _convolve_channel_scipy

    # RGB / multi-channel
    if image.ndim == 3:
        output_image = np.zeros(image.shape, dtype=np.float64)
        for c in range(image.shape[2]):
            output_image[:, :, c] = convolve_channel(image[:, :, c], kernel)
        return output_image

    # Grayscale
    return convolve_channel(image, kernel)


def NormalizeImage(img: np.array):
  max_val = img.max()
  min_val = img.min()
  if max_val == min_val:
    return img #Return same image as no normalization needs to take place. It's unicolor. 
  normalized_image = 255 * ((img - min_val) / (max_val - min_val))
  return normalized_image.astype("uint8")

def TransformIntoGrayScale(rgbImg: np.ndarray, mode: str):

    img = rgbImg.astype(np.float32)

    if mode == "weighted":
        gray_image = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]

    elif mode == "average":
        gray_image = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3

    else:
        raise ValueError("Mode must be either 'weighted' or 'average'")

    return gray_image.astype("uint8")