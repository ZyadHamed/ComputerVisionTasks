"""Three-point Hough ellipse detection and RGB overlay rendering.

GetEllipses retains the nested-loop implementation from the notebook.
GetEllipsesVectorized retains the optimized Numba/JIT approach (it is not
literally NumPy-vectorized). Numba is optional; without it the optimized
implementation runs as Python, which can be very slow on dense edge maps.
"""

import math

import cv2
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Allow other FastAPI routes to work even when the optional accelerator
    # has not been installed. Install numba to speed up this implementation.
    def njit(*args, **kwargs):
        return lambda func: func


def _validate_image(image: np.ndarray):
    if image.ndim != 2:
        raise ValueError("Ellipse detection expects a 2D grayscale/edge image.")


def _remove_detected_ellipse(remaining_edges: np.ndarray, ellipse: dict) -> bool:
    """Remove edge pixels close to the found ellipse; return whether any changed."""
    mask = np.zeros(remaining_edges.shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (round(ellipse["center_x"]), round(ellipse["center_y"])),
        (round(ellipse["a"]), round(ellipse["b"])),
        float(np.degrees(ellipse["alpha"])),
        0, 360, 255,
        thickness=5,
    )
    pixels_to_remove = remaining_edges & (mask > 0)
    if not np.any(pixels_to_remove):
        return False
    remaining_edges[pixels_to_remove] = False
    return True


def GetEllipses(
    image: np.ndarray,
    pair_tolerance: float = 2,
    center_tolerance: float = 4,
    threshold: int = 100,
    max_ellipses: int | None = None,
):
    """Original pure-Python three-point Hough voter.

    Returns a list of dictionaries with count, center_x, center_y, a, alpha, b.
    """
    _validate_image(image)
    remaining_edges = image > 0
    final_ellipses = []

    while max_ellipses is None or len(final_ellipses) < max_ellipses:
        ys, xs = np.nonzero(remaining_edges)
        if len(ys) < 3:
            break
        global_best_ellipse = None

        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                accm = {}
                x1, y1 = xs[i], ys[i]
                x2, y2 = xs[j], ys[j]
                major_distance = np.hypot(x1 - x2, y1 - y2)

                if major_distance <= pair_tolerance:
                    continue

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                a = major_distance / 2
                alpha = np.arctan2(y2 - y1, x2 - x1)

                for k in range(len(ys)):
                    if k == i or k == j:
                        continue

                    x, y = xs[k], ys[k]
                    minor_distance = np.hypot(center_x - x, center_y - y)
                    if minor_distance <= center_tolerance or minor_distance >= a:
                        continue

                    f = np.hypot(x2 - x, y2 - y)
                    cos_taw = (minor_distance**2 + a**2 - f**2) / (2 * a * minor_distance)
                    if not -1 <= cos_taw <= 1:
                        continue

                    denominator = a**2 - minor_distance**2 * cos_taw**2
                    if denominator <= 0:
                        continue

                    b = int(np.sqrt(
                        (minor_distance**2 * a**2 * np.sin(np.arccos(cos_taw))**2)
                        / denominator
                    ))
                    if b <= 0 or b > a:
                        continue

                    if b in accm:
                        accm[b]["count"] += 1
                    else:
                        accm[b] = {
                            "count": 1,
                            "center_x": center_x,
                            "center_y": center_y,
                            "a": a,
                            "alpha": alpha,
                            "b": b,
                        }

                if not accm:
                    continue
                best_ellipse = max(accm.values(), key=lambda e: e["count"])
                if (
                    global_best_ellipse is None
                    or best_ellipse["count"] > global_best_ellipse["count"]
                ):
                    global_best_ellipse = best_ellipse

        if global_best_ellipse is None or global_best_ellipse["count"] <= threshold:
            break
        if not _remove_detected_ellipse(remaining_edges, global_best_ellipse):
            break
        final_ellipses.append(global_best_ellipse)

    return final_ellipses


@njit(cache=True)
def _FindBestEllipse(xs, ys, pair_tolerance, center_tolerance):
    """JIT three-point Hough search. Mirrors the optimized notebook function."""
    n = len(xs)
    best_count = 0
    best_center_x = 0.0
    best_center_y = 0.0
    best_a = 0.0
    best_b = 0
    best_alpha = 0.0
    min_distance_squared = center_tolerance**2

    for i in range(n):
        x1, y1 = float(xs[i]), float(ys[i])
        for j in range(i + 1, n):
            x2, y2 = float(xs[j]), float(ys[j])
            major_distance = math.hypot(x1 - x2, y1 - y2)
            if major_distance <= pair_tolerance:
                continue

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            a = major_distance / 2
            alpha = math.atan2(y2 - y1, x2 - x1)

            ux = (x2 - x1) / major_distance
            uy = (y2 - y1) / major_distance
            a_squared = a * a
            accm = np.zeros(int(a) + 1, dtype=np.int32)

            for k in range(n):
                if k == i or k == j:
                    continue

                dx = float(xs[k]) - center_x
                dy = float(ys[k]) - center_y
                minor_distance_squared = dx * dx + dy * dy
                if (
                    minor_distance_squared <= min_distance_squared
                    or minor_distance_squared >= a_squared
                ):
                    continue

                along = dx * ux + dy * uy
                perpendicular = dy * ux - dx * uy
                denominator = a_squared - along * along
                if denominator <= 0:
                    continue

                b = int(math.sqrt(
                    a_squared * perpendicular * perpendicular / denominator
                ))
                if b <= 0 or b > a:
                    continue

                accm[b] += 1
                if accm[b] > best_count:
                    best_count = accm[b]
                    best_center_x = center_x
                    best_center_y = center_y
                    best_a = a
                    best_alpha = alpha
                    best_b = b

    return best_count, best_center_x, best_center_y, best_a, best_alpha, best_b


def GetEllipsesVectorized(
    image: np.ndarray,
    pair_tolerance: float = 2,
    center_tolerance: float = 4,
    threshold: int = 100,
    max_ellipses: int | None = None,
):
    """Optimized Hough ellipse search using Numba when available."""
    _validate_image(image)
    remaining_edges = image > 0
    final_ellipses = []

    while max_ellipses is None or len(final_ellipses) < max_ellipses:
        ys, xs = np.nonzero(remaining_edges)
        if len(ys) < 3:
            break

        count, center_x, center_y, a, alpha, b = _FindBestEllipse(
            xs, ys, pair_tolerance, center_tolerance
        )
        if count <= threshold:
            break

        best_ellipse = {
            "count": int(count),
            "center_x": float(center_x),
            "center_y": float(center_y),
            "a": float(a),
            "alpha": float(alpha),
            "b": int(b),
        }
        if not _remove_detected_ellipse(remaining_edges, best_ellipse):
            break
        final_ellipses.append(best_ellipse)

    return final_ellipses


def DetectEllipses(
    image: np.ndarray,
    pair_tolerance: float = 2,
    center_tolerance: float = 4,
    threshold: int = 100,
    use_vectorized: bool = True,
    max_ellipses: int | None = None,
):
    """Find ellipses; preserves notebook output dictionary keys and units."""
    _validate_image(image)
    if pair_tolerance < 0 or center_tolerance < 0 or threshold < 0:
        raise ValueError("pair_tolerance, center_tolerance and threshold must be nonnegative.")
    if max_ellipses is not None and max_ellipses < 1:
        raise ValueError("max_ellipses must be greater than zero.")

    if use_vectorized:
        return GetEllipsesVectorized(
            image, pair_tolerance, center_tolerance, threshold, max_ellipses
        )
    return GetEllipses(
        image, pair_tolerance, center_tolerance, threshold, max_ellipses
    )


def DrawEllipses(image: np.ndarray, ellipses: list[dict]) -> np.ndarray:
    """Return an RGB image with detected ellipses drawn in red."""
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

    for ellipse in ellipses:
        cv2.ellipse(
            overlay,
            (round(ellipse["center_x"]), round(ellipse["center_y"])),
            (round(ellipse["a"]), round(ellipse["b"])),
            float(np.degrees(ellipse["alpha"])),
            0, 360,
            (255, 0, 0),  # RGB red.
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return overlay
