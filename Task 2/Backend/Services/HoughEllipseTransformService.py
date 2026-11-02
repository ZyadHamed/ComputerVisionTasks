"""Three-point Hough ellipse detection and RGB overlay rendering.

GetEllipses retains the original exhaustive implementation.
GetEllipsesRandomized uses sampled major-axis pairs and Numba-compiled voting.
DetectEllipses selects either method; DrawEllipses renders results.
"""

import math

import cv2
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # The original detector works without Numba; randomized detection requires it.
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
def _FindBestEllipseRandomized(
    xs, ys, pairs, pair_tolerance, center_tolerance, threshold,
    fit_tolerance, min_b, min_coverage, min_score, sample_scale
):
    """Randomly sampled major-axis pairs; vote and validate using edge support."""
    n = len(xs)
    best_count = 0
    best_score = 0.0
    best_center_x = 0.0
    best_center_y = 0.0
    best_a = 0.0
    best_alpha = 0.0
    best_b = 0
    min_distance_squared = center_tolerance ** 2

    for q in range(len(pairs)):
        i, j = pairs[q, 0], pairs[q, 1]
        if i == j:
            continue

        x1, y1 = float(xs[i]), float(ys[i])
        x2, y2 = float(xs[j]), float(ys[j])
        major_distance = math.hypot(x2 - x1, y2 - y1)
        if major_distance <= pair_tolerance:
            continue

        a = major_distance / 2
        a_squared = a * a
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        ux = (x2 - x1) / major_distance
        uy = (y2 - y1) / major_distance

        accm = np.zeros(int(a) + 1, dtype=np.int32)

        # Vote for minor-axis length using all sampled third points.
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

            b = int(math.sqrt(a_squared * perpendicular * perpendicular / denominator))
            if b < min_b or b > a:
                continue
            accm[b] += 1

        b = int(np.argmax(accm))
        count = int(accm[b])
        if count <= threshold:
            continue

        # Verify that edge pixels follow the proposed ellipse perimeter.
        b_squared = float(b * b)
        support = 0
        covered = np.zeros(24, dtype=np.uint8)

        for k in range(n):
            dx = float(xs[k]) - center_x
            dy = float(ys[k]) - center_y
            u = dx * ux + dy * uy
            v = -dx * uy + dy * ux

            error = u * u / a_squared + v * v / b_squared - 1.0
            gradient = 2 * math.sqrt(
                (u / a_squared) ** 2 + (v / b_squared) ** 2
            )
            if gradient > 1e-10 and abs(error) / gradient <= fit_tolerance:
                support += 1
                angle = math.atan2(v / b, u / a) + math.pi
                angle_bin = min(23, int(angle * 24 / (2 * math.pi)))
                covered[angle_bin] = 1

        coverage = np.sum(covered) / 24.0
        perimeter = math.pi * (
            3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b))
        )
        score = coverage * (support * sample_scale / perimeter)

        if (
            coverage >= min_coverage
            and score >= min_score
            and score > best_score
        ):
            best_count = count
            best_score = score
            best_center_x = center_x
            best_center_y = center_y
            best_a = a
            best_alpha = math.atan2(y2 - y1, x2 - x1)
            best_b = b

    return best_count, best_center_x, best_center_y, best_a, best_alpha, best_b


def GetEllipsesRandomized(
    image: np.ndarray,
    pair_tolerance: float = 2,
    center_tolerance: float = 4,
    threshold: int = 20,
    num_pairs: int = 5000,
    max_edge_points: int = 2000,
    max_ellipses: int = 5,
    removal_tolerance: float = 2.0,
    min_coverage: float = 0.5,
    min_score: float = 0.5,
    seed: int | None = 42,
    min_b: int = 5,
):
    """Sample major axes, vote for minor axes, verify, and remove detected ellipses.

    Output dict keys match GetEllipses: count, center_x, center_y, a, alpha, b.
    """
    _validate_image(image)
    if not NUMBA_AVAILABLE:
        raise ImportError('Randomized ellipse detection requires numba: pip install numba')
    if pair_tolerance < 0 or center_tolerance < 0 or threshold < 0:
        raise ValueError('pair_tolerance, center_tolerance and threshold must be nonnegative.')
    if num_pairs < 1 or max_edge_points < 3 or max_ellipses < 1 or min_b < 1:
        raise ValueError('num_pairs and max_ellipses must be positive; max_edge_points >= 3 and min_b >= 1.')
    if removal_tolerance <= 0 or not 0 <= min_coverage <= 1 or min_score < 0:
        raise ValueError('Invalid removal_tolerance, min_coverage, or min_score.')

    remaining_edges = image > 0
    final_ellipses = []
    rng = np.random.default_rng(seed)

    for _ in range(max_ellipses):
        ys_all, xs_all = np.nonzero(remaining_edges)
        total = len(xs_all)
        if total < 3:
            break

        # Bound the third-point voting cost per candidate pair.
        if total > max_edge_points:
            chosen = rng.choice(total, max_edge_points, replace=False)
            xs = xs_all[chosen]
            ys = ys_all[chosen]
        else:
            xs, ys = xs_all, ys_all

        # Generate random major-axis endpoint pairs.
        pairs = rng.integers(len(xs), size=(num_pairs, 2), dtype=np.int32)
        count, center_x, center_y, a, alpha, b = _FindBestEllipseRandomized(
            xs, ys, pairs, pair_tolerance, center_tolerance, threshold,
            removal_tolerance, min_b, min_coverage, min_score, total / len(xs)
        )
        if count <= threshold:
            break

        # Remove only remaining edge pixels close to this ellipse's perimeter.
        dx = xs_all - center_x
        dy = ys_all - center_y
        cosine = math.cos(alpha)
        sine = math.sin(alpha)
        u = dx * cosine + dy * sine
        v = -dx * sine + dy * cosine
        error = (u / a) ** 2 + (v / b) ** 2 - 1
        gradient = 2 * np.sqrt((u / (a * a)) ** 2 + (v / (b * b)) ** 2)
        pixels_to_remove = np.abs(error) / np.maximum(gradient, 1e-10) <= removal_tolerance

        if not np.any(pixels_to_remove):
            break

        remaining_edges[ys_all[pixels_to_remove], xs_all[pixels_to_remove]] = False
        final_ellipses.append({
            'count': int(count),
            'center_x': float(center_x),
            'center_y': float(center_y),
            'a': float(a),
            'alpha': float(alpha),
            'b': int(b),
        })

    return final_ellipses


def GetEllipsesVectorized(
    image: np.ndarray,
    pair_tolerance: float = 2,
    center_tolerance: float = 4,
    threshold: int = 20,
    max_ellipses: int | None = None,
    **kwargs,
):
    """Backward-compatible name; now uses the randomized Hough implementation."""
    return GetEllipsesRandomized(
        image,
        pair_tolerance=pair_tolerance,
        center_tolerance=center_tolerance,
        threshold=threshold,
        max_ellipses=5 if max_ellipses is None else max_ellipses,
        **kwargs,
    )


def DetectEllipses(
    image: np.ndarray,
    pair_tolerance: float = 2,
    center_tolerance: float = 4,
    threshold: int | None = None,
    use_randomized: bool = True,
    max_ellipses: int | None = None,
    *,
    num_pairs: int = 5000,
    max_edge_points: int = 2000,
    removal_tolerance: float = 2.0,
    min_coverage: float = 0.5,
    min_score: float = 0.5,
    seed: int | None = 42,
    min_b: int = 5,
    use_vectorized: bool | None = None,
):
    _validate_image(image)
    if use_vectorized is not None:
        use_randomized = use_vectorized
    if pair_tolerance < 0 or center_tolerance < 0 or (threshold is not None and threshold < 0):
        raise ValueError('pair_tolerance, center_tolerance and threshold must be nonnegative.')
    if max_ellipses is not None and max_ellipses < 1:
        raise ValueError('max_ellipses must be greater than zero.')

    if use_randomized:
        return GetEllipsesRandomized(
            image=image,
            pair_tolerance=pair_tolerance,
            center_tolerance=center_tolerance,
            threshold=20 if threshold is None else threshold,
            num_pairs=num_pairs,
            max_edge_points=max_edge_points,
            max_ellipses=5 if max_ellipses is None else max_ellipses,
            removal_tolerance=removal_tolerance,
            min_coverage=min_coverage,
            min_score=min_score,
            seed=seed,
            min_b=min_b,
        )
    return GetEllipses(
        image=image,
        pair_tolerance=pair_tolerance,
        center_tolerance=center_tolerance,
        threshold=100 if threshold is None else threshold,
        max_ellipses=max_ellipses,
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
