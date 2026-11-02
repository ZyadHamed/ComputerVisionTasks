"""Hough line detection and rendering for a two-dimensional edge image."""

import cv2
import numpy as np


def BuildLinesAccumulator(image: np.ndarray):
    """Return (votes, angles, distances) for the image dimensions."""
    if image.ndim != 2:
        raise ValueError("Hough line detection expects a 2D grayscale/edge image.")

    # Preserve the 100 theta bins and integer rho bins in the original algorithm.
    thetas = np.linspace(0, np.pi, 100)
    height, width = image.shape
    diagonal = int(np.hypot(height, width))
    rhos = np.arange(-diagonal, diagonal + 1)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    return accumulator, thetas, rhos


# Alias for compatibility with the name used in the original notebook.
BuildLinesAccumilator = BuildLinesAccumulator


def NonMaximumSuppressionLines(accumulator: np.ndarray, thetas: np.ndarray, rhos: np.ndarray,
                              threshold: int = 100, rho_distance: int = 10, theta_distance: int = 5):
    scores = accumulator.copy()
    lines = []

    while True:
        rho_index, theta_index = np.unravel_index(
            np.argmax(scores), scores.shape
        )

        if scores[rho_index, theta_index] <= threshold:
            break

        lines.append((rhos[rho_index], thetas[theta_index]))

        rho_min = max(0, rho_index - rho_distance)
        rho_max = min(len(rhos), rho_index + rho_distance + 1)

        theta_min = max(0, theta_index - theta_distance)
        theta_max = min(len(thetas), theta_index + theta_distance + 1)

        scores[rho_min:rho_max, theta_min:theta_max] = 0

    return np.array(lines).reshape(-1, 2)


def _extract_lines(accumulator, thetas, rhos, threshold):
    lines = NonMaximumSuppressionLines(
        accumulator,
        thetas,
        rhos,
        threshold=threshold,
        rho_distance=10,
        theta_distance=5
    )

    return lines, accumulator


def GetLines(image: np.ndarray, threshold: int = 100):
    """Non-vectorized Hough voting, matching the original implementation."""
    accumulator, thetas, rhos = BuildLinesAccumulator(image)
    cosines = np.cos(thetas)
    sines = np.sin(thetas)

    for y, x in np.argwhere(image > 0):
        for i in range(len(thetas)):
            rho = x * cosines[i] + y * sines[i]
            rho_index = round(rho) - rhos[0]
            accumulator[rho_index, i] += 1

    return _extract_lines(accumulator, thetas, rhos, threshold)


def GetLinesVectorized(image: np.ndarray, threshold: int = 100):
    """Vectorized Hough voting using NumPy histograms for each theta."""
    accumulator, thetas, rhos = BuildLinesAccumulator(image)
    cosines = np.cos(thetas)
    sines = np.sin(thetas)
    ys, xs = np.nonzero(image > 0)

    for i in range(len(thetas)):
        current_rhos = xs * cosines[i] + ys * sines[i]
        rho_indices = np.rint(current_rhos).astype(np.int64) - rhos[0]
        accumulator[:, i] = np.bincount(
            rho_indices, minlength=len(rhos)
        )

    return _extract_lines(accumulator, thetas, rhos, threshold)


def DetectLines(image: np.ndarray, threshold: int = 100, use_vectorized: bool = True,
                use_nms: bool = True, rho_distance: int = 10, theta_distance: int = 5):
    """Return (array of [rho, theta_radians], votes accumulator)."""
    if threshold < 0:
        raise ValueError("threshold must be greater than or equal to zero.")

    if use_vectorized:
        _, accum = GetLinesVectorized(image, threshold=threshold)
    else:
        _, accum = GetLines(image, threshold=threshold)

    _, thetas, rhos = BuildLinesAccumulator(image)

    if use_nms:
        lines = NonMaximumSuppressionLines(
            accum,
            thetas,
            rhos,
            threshold=threshold,
            rho_distance=rho_distance,
            theta_distance=theta_distance
        )
    else:
        rho_indices, theta_indices = np.where(accum > threshold)
        lines = np.column_stack((rhos[rho_indices], thetas[theta_indices]))

    return lines, accum


def DrawLines(image: np.ndarray, polar_coordinates: np.ndarray) -> np.ndarray:
    """Draw detected lines in red on an RGB copy of the uploaded image."""
    if image.ndim == 2:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        overlay = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        overlay = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError("Expected a grayscale, RGB, or RGBA image.")

    height, width = overlay.shape[:2]
    diagonal = np.hypot(height, width)

    for rho, theta in polar_coordinates:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (round(x0 - diagonal * b), round(y0 + diagonal * a))
        pt2 = (round(x0 + diagonal * b), round(y0 - diagonal * a))
        cv2.line(overlay, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)

    return overlay
