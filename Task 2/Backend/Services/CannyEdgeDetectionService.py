import numpy as np

from Backend.Services.UtilitiesService import (
    ConvolveImage,
    TransformIntoGrayScale,
)


SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def make_gaussian_kernel(sigma, kernel_size=None):
    if kernel_size is None:
        kernel_size = 2 * int(np.ceil(3 * sigma)) + 1

    radius = kernel_size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    g1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))

    kernel = np.outer(g1d, g1d)
    return kernel / kernel.sum()


def gaussian_blur(gray, sigma, kernel_size=None, manual=False):
    kernel = make_gaussian_kernel(sigma, kernel_size)
    return ConvolveImage(gray, kernel, manual=manual), kernel.shape[0] // 2


def sobel_gradient(image, manual=False):
    gx = ConvolveImage(image, SOBEL_X, manual=manual)
    gy = ConvolveImage(image, SOBEL_Y, manual=manual)

    magnitude = np.hypot(gx, gy)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0

    angle = np.degrees(np.arctan2(gy, gx)) % 180
    return magnitude, angle


def clear_border(magnitude, margin):
    magnitude = magnitude.copy()
    magnitude[:margin, :] = 0
    magnitude[-margin:, :] = 0
    magnitude[:, :margin] = 0
    magnitude[:, -margin:] = 0
    return magnitude


def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    padded = np.pad(magnitude, 1, mode="constant")

    def neighbor(di, dj):
        return padded[1 + di:1 + di + height, 1 + dj:1 + dj + width]

    mask_0 = (angle < 22.5) | (angle >= 157.5)
    mask_45 = (angle >= 22.5) & (angle < 67.5)
    mask_90 = (angle >= 67.5) & (angle < 112.5)
    mask_135 = (angle >= 112.5) & (angle < 157.5)

    is_max = np.zeros((height, width), dtype=bool)
    is_max |= mask_0 & (magnitude >= neighbor(0, -1)) & (magnitude >= neighbor(0, 1))
    is_max |= mask_45 & (magnitude >= neighbor(-1, -1)) & (magnitude >= neighbor(1, 1))
    is_max |= mask_90 & (magnitude >= neighbor(-1, 0)) & (magnitude >= neighbor(1, 0))
    is_max |= mask_135 & (magnitude >= neighbor(-1, 1)) & (magnitude >= neighbor(1, -1))

    return np.where(is_max, magnitude, 0.0)


def has_neighbor_in(mask):
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros((height, width), dtype=bool)

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            result |= padded[1 + di:1 + di + height, 1 + dj:1 + dj + width]

    return result


def hysteresis_threshold(nms, low_threshold, high_threshold, propagate=True):
    strong = nms > high_threshold
    weak = (nms >= low_threshold) & (nms > 0) & ~strong

    edges = strong.copy()

    if not propagate:
        edges |= weak & has_neighbor_in(strong)
    else:
        while True:
            grown = edges | (weak & has_neighbor_in(edges))
            if np.array_equal(grown, edges):
                break
            edges = grown

    return (edges * 255).astype(np.uint8)


def canny_edge_detection(img, sigma=1.4, low_threshold=20, high_threshold=50,
                         kernel_size=None, propagate=True, manual=False,
                         return_steps=False):
    if low_threshold > high_threshold:
        raise ValueError("low_threshold must not be greater than high_threshold.")
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero.")

    if img.ndim == 3:
        gray = TransformIntoGrayScale(img, "weighted")
    else:
        gray = img.copy()
    gray = gray.astype(np.float64)

    blurred, blur_radius = gaussian_blur(gray, sigma, kernel_size, manual=manual)
    magnitude, angle = sobel_gradient(blurred, manual=manual)
    magnitude = clear_border(magnitude, blur_radius + 1)
    nms = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_threshold(nms, low_threshold, high_threshold, propagate)

    if return_steps:
        return edges, blurred, magnitude, nms
    return edges