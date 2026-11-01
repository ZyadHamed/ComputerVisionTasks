"""Hough circle voting, peak suppression, and RGB overlay rendering.

Circles use the original notebook ordering: (radius, center_y, center_x).
An input image must be a two-dimensional grayscale/edge map; all nonzero
pixels are treated as edges.
"""

import cv2
import numpy as np

_MAX_ACCUMULATOR_BYTES = 128 * 1024 * 1024


def _validate_inputs(image: np.ndarray, radii: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Circle detection expects a 2D grayscale/edge image.")
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("radii must be a non-empty 1D array.")
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("All radii must be finite and greater than zero.")
    if len(np.unique(radii)) != len(radii):
        raise ValueError("radii must not contain duplicates.")
    # A circle with its edge pixel and center both inside the image cannot
    # have a radius larger than the image diagonal.
    max_radius = np.hypot(image.shape[0] - 1, image.shape[1] - 1)
    if np.any(radii > max_radius):
        raise ValueError("radii cannot exceed the image diagonal.")
    required_bytes = len(radii) * image.shape[0] * image.shape[1] * np.dtype(np.int32).itemsize
    if required_bytes > _MAX_ACCUMULATOR_BYTES:
        raise ValueError("Circle accumulator exceeds 128 MiB; reduce the image size or number of radii.")
    return radii


def BuildCirclesAccumulator(image: np.ndarray, radii: np.ndarray):
    """Return (votes[radius, y, x], possible_y_centers, possible_x_centers)."""
    radii = _validate_inputs(image, radii)
    height, width = image.shape
    x_centers = np.arange(width)
    y_centers = np.arange(height)
    accumulator = np.zeros((len(radii), height, width), dtype=np.int32)
    return accumulator, y_centers, x_centers


# Maintain compatibility with the original notebook's spelling.
BuildCirclesAccumilator = BuildCirclesAccumulator


def GetCircles(image: np.ndarray, radii: np.ndarray, threshold: int = 100):
    """Original nested-loop circle voter; returns (all bins > threshold, votes)."""
    radii = _validate_inputs(image, radii)
    accum, y_centers, x_centers = BuildCirclesAccumulator(image, radii)
    height, width = image.shape

    for y, x in np.argwhere(image > 0):
        for r_index, r in enumerate(radii):
            min_y = max(0, int(np.ceil(y - r)))
            max_y = min(height - 1, int(np.floor(y + r)))
            for y_center in range(min_y, max_y + 1):
                dx = np.sqrt(max(0.0, r**2 - (y - y_center)**2))
                x_center_1 = round(x - dx)
                x_center_2 = round(x + dx)
                for x_center in {x_center_1, x_center_2}:
                    if 0 <= x_center < width:
                        accum[r_index, y_center, x_center] += 1

    r_indices, y_indices, x_indices = np.where(accum > threshold)
    circles = np.column_stack((
        radii[r_indices], y_centers[y_indices], x_centers[x_indices]
    ))
    return circles, accum


def GetCirclesVectorized(image: np.ndarray, radii: np.ndarray, batch_size: int = 1024):
    """Batch-vectorized circle voter; returns the complete 3D accumulator."""
    radii = _validate_inputs(image, radii)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    accum, _, _ = BuildCirclesAccumulator(image, radii)
    height, width = image.shape
    ys, xs = np.nonzero(image > 0)

    for r_index, r in enumerate(radii):
        dy = np.arange(-int(np.floor(r)), int(np.floor(r)) + 1)
        dx = np.sqrt(np.maximum(0.0, float(r)**2 - dy**2))

        for start in range(0, len(xs), batch_size):
            x = xs[start:start + batch_size, None]
            y = ys[start:start + batch_size, None]

            yc = y + dy[None, :]
            xc1 = np.rint(x - dx[None, :]).astype(np.int64)
            xc2 = np.rint(x + dx[None, :]).astype(np.int64)

            valid_y = (yc >= 0) & (yc < height)
            valid1 = valid_y & (xc1 >= 0) & (xc1 < width)
            valid2 = valid_y & (xc2 >= 0) & (xc2 < width) & (xc1 != xc2)

            np.add.at(accum[r_index], (yc[valid1], xc1[valid1]), 1)
            np.add.at(accum[r_index], (yc[valid2], xc2[valid2]), 1)

    return accum


def NonMaximumSuppressionCircles(
    accum: np.ndarray,
    radii: np.ndarray,
    threshold: int = 100,
    max_circles: int = 2,
    min_center_distance: int = 80,
    radius_distance: int = 5,
):
    """Keep the strongest circle hypotheses, suppressing nearby accumulator bins.

    radius_distance is the number of radius *indices* on each side (as in the
    original code), not a difference measured in pixels.
    """
    if max_circles < 0 or min_center_distance < 0 or radius_distance < 0:
        raise ValueError("Suppression parameters must be nonnegative.")
    if accum.ndim != 3 or accum.shape[0] != len(radii):
        raise ValueError("accum must have shape (len(radii), height, width).")

    scores = accum.copy()
    circles = []
    _, height, width = scores.shape

    for _ in range(max_circles):
        r_index, y, x = np.unravel_index(np.argmax(scores), scores.shape)
        if scores[r_index, y, x] <= threshold:
            break

        circles.append((radii[r_index], y, x))

        y_min = max(0, y - min_center_distance)
        y_max = min(height, y + min_center_distance + 1)
        x_min = max(0, x - min_center_distance)
        x_max = min(width, x + min_center_distance + 1)
        r_min = max(0, r_index - radius_distance)
        r_max = min(len(radii), r_index + radius_distance + 1)
        scores[r_min:r_max, y_min:y_max, x_min:x_max] = 0

    return np.asarray(circles, dtype=float).reshape(-1, 3)


def DetectCircles(
    image: np.ndarray,
    radii: np.ndarray,
    threshold: int = 100,
    max_circles: int = 2,
    min_center_distance: int = 80,
    radius_distance: int = 5,
    use_vectorized: bool = True,
    batch_size: int = 1024,
):
    """Return (NMS-filtered circles of [r, y, x], votes[r, y, x])."""
    if threshold < 0:
        raise ValueError("threshold must be nonnegative.")
    radii = _validate_inputs(image, radii)

    if use_vectorized:
        accum = GetCirclesVectorized(image, radii, batch_size=batch_size)
    else:
        _, accum = GetCircles(image, radii, threshold=threshold)

    circles = NonMaximumSuppressionCircles(
        accum,
        radii,
        threshold=threshold,
        max_circles=max_circles,
        min_center_distance=min_center_distance,
        radius_distance=radius_distance,
    )
    return circles, accum


def DrawCircles(image: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """Return an RGB image with detected circles drawn in red."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        overlay = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        overlay = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError("Expected grayscale, RGB, or RGBA image.")

    for radius, y_center, x_center in circles:
        cv2.circle(
            overlay,
            (int(round(x_center)), int(round(y_center))),
            int(round(radius)),
            (255, 0, 0),  # RGB red; process_image converts to BGR before PNG encoding.
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return overlay
