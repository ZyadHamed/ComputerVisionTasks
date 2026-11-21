"""Feature matching (SSD and normalized cross-correlation) for Harris and SIFT.

Input contract (one "feature set" per image, JSON-like dict)
------------------------------------------------------------
    {
        "keypoints":   [[x, y], ...],          # shape (N, 2), pixel coordinates
        "descriptors": [[d0, ..., dD-1], ...]  # shape (N, D), row i <-> keypoint i
    }

    SIFT  : "descriptors" is REQUIRED with shape (N, 128).
    Harris: "descriptors" is OPTIONAL. If omitted, a flattened square patch of
            `patch_size` x `patch_size` grayscale pixels around each corner is
            used as the descriptor, i.e. shape (N, patch_size**2). Corners
            closer than patch_size // 2 pixels to the border are dropped.

    Keypoint coordinates must be in the pixel frame of the images that are
    uploaded alongside them (x = column, y = row, origin at the top-left).

Both descriptor sets must have the same length D.
"""
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from Backend.Services.UtilitiesService import TransformIntoGrayScale
except ModuleNotFoundError:
    from Services.UtilitiesService import TransformIntoGrayScale

SUPPORTED_METHODS = ("sift", "harris")
SUPPORTED_METRICS = ("ssd", "ncc")
SIFT_DESCRIPTOR_SIZE = 128

# The score matrix is N1 x N2 float64 (4000 x 4000 = ~128 MB), so cap the input.
MAX_FEATURES = 4000


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------

def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Return an HxWx3 uint8 RGB copy of a gray/RGB/RGBA image (alpha dropped)."""
    if image.dtype != np.uint8:
        if image.dtype == np.uint16:
            image = image // 257                      # 16-bit -> 8-bit range
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Weighted grayscale (same conversion as the rest of the API), as float64."""
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    return np.asarray(TransformIntoGrayScale(rgb, mode="weighted"), dtype=np.float64)


# ----------------------------------------------------------------------------
# Scores: all descriptor pairs at once
# ----------------------------------------------------------------------------

def ssd_matrix(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """SSD between every row of d1 (N1, D) and every row of d2 (N2, D) -> (N1, N2).

    Lower = better. Uses sum (a-b)^2 = sum a^2 - 2 sum a*b + sum b^2, so the
    whole matrix is one matrix multiplication.
    """
    d1, d2 = d1.astype(np.float64), d2.astype(np.float64)   # float64: no overflow / less cancellation
    out = (d1 ** 2).sum(axis=1)[:, None] - 2.0 * (d1 @ d2.T) + (d2 ** 2).sum(axis=1)[None, :]
    return np.maximum(out, 0.0)                              # clamp round-off negatives


def ncc_matrix(d1: np.ndarray, d2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean normalized cross-correlation between all row pairs -> (N1, N2).

    Higher = better, range [-1, 1]. Flat (constant) descriptors score 0 instead of NaN.
    """
    a = d1.astype(np.float64)
    b = d2.astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)                    # zero-mean each descriptor
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    score = np.where(denom > eps, (a @ b.T) / np.maximum(denom, eps), 0.0)
    return np.clip(score, -1.0, 1.0)


# ----------------------------------------------------------------------------
# Descriptors for Harris corners (patch around each point)
# ----------------------------------------------------------------------------

def extract_patch_descriptors(
    gray: np.ndarray, keypoints: np.ndarray, patch_size: int = 11
) -> Tuple[np.ndarray, np.ndarray]:
    """Flattened patch descriptors, shape (M, patch_size**2).

    Returns (descriptors, valid_mask). valid_mask has length N; keypoints too
    close to the border for a full patch are False and get no descriptor.
    """
    half = patch_size // 2
    h, w = gray.shape
    xs = np.rint(keypoints[:, 0]).astype(int)
    ys = np.rint(keypoints[:, 1]).astype(int)
    valid = (xs >= half) & (xs < w - half) & (ys >= half) & (ys < h - half)

    windows = np.lib.stride_tricks.sliding_window_view(gray, (patch_size, patch_size))
    patches = windows[ys[valid] - half, xs[valid] - half]    # (M, p, p), top-left indexing
    return patches.reshape(patches.shape[0], -1).astype(np.float64), valid


# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------

def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    metric: str = "ssd",
    ratio: Optional[float] = 0.8,
    threshold: Optional[float] = None,
    cross_check: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match rows of desc1 to rows of desc2.

    For every descriptor in image 1 the best partner in image 2 is kept if it passes:
      * ratio test (Lowe): best distance < ratio * second-best distance, where
        distance = sqrt(SSD) for "ssd" and 1 - NCC for "ncc" (ratio >= 1 disables it)
      * cross-check: the partner's own best match is the same descriptor
      * threshold (optional): SSD <= threshold, or NCC >= threshold

    Returns (idx1, idx2, score) sorted best-first; score is the raw SSD / NCC value.
    """
    n1, n2 = len(desc1), len(desc2)
    if n1 == 0 or n2 == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0)

    if metric == "ssd":
        score = ssd_matrix(desc1, desc2)
        dist = np.sqrt(score)
    else:
        score = ncc_matrix(desc1, desc2)
        dist = 1.0 - score

    rows = np.arange(n1)
    nn12 = dist.argmin(axis=1)                               # best partner in image 2
    best = dist[rows, nn12]
    keep = np.ones(n1, dtype=bool)

    if ratio is not None and ratio < 1.0 and n2 >= 2:
        second = np.partition(dist, 1, axis=1)[:, 1]
        keep &= best < ratio * second

    if cross_check:
        keep &= dist.argmin(axis=0)[nn12] == rows

    if threshold is not None:
        matched_score = score[rows, nn12]
        keep &= (matched_score <= threshold) if metric == "ssd" else (matched_score >= threshold)

    selected = np.flatnonzero(keep)
    selected = selected[np.argsort(best[selected], kind="stable")]
    return selected, nn12[selected], score[selected, nn12[selected]]


# ----------------------------------------------------------------------------
# Input validation / orchestration
# ----------------------------------------------------------------------------

def parse_feature_set(data: dict, name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate one feature set dict -> (keypoints (N, 2), descriptors (N, D) or None)."""
    if not isinstance(data, dict) or "keypoints" not in data:
        raise ValueError(f"{name} must be a JSON object with a 'keypoints' list.")
    try:
        keypoints = np.asarray(data["keypoints"], dtype=np.float64)
        raw = data.get("descriptors")
        descriptors = None if raw is None else np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains non-numeric or ragged data.") from exc

    if keypoints.size == 0:
        return np.empty((0, 2)), None
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"{name}.keypoints must have shape (N, 2) as [x, y]; got {keypoints.shape}.")
    if len(keypoints) > MAX_FEATURES:
        raise ValueError(f"{name} has {len(keypoints)} keypoints; the limit is {MAX_FEATURES}.")
    if not np.isfinite(keypoints).all():
        raise ValueError(f"{name}.keypoints contains NaN or infinite values.")
    return keypoints, descriptors


def _prepare(name, gray, keypoints, descriptors, method, patch_size):
    """Return (original_indices, keypoints, descriptors, dropped_count)."""
    n = len(keypoints)
    if n == 0:
        return np.empty(0, int), keypoints, np.empty((0, 0)), 0

    if descriptors is None:
        if method == "sift":
            raise ValueError(f"{name}: SIFT needs 'descriptors' with shape (N, {SIFT_DESCRIPTOR_SIZE}).")
        descriptors, valid = extract_patch_descriptors(gray, keypoints, patch_size)
        idx = np.flatnonzero(valid)
        return idx, keypoints[idx], descriptors, n - len(idx)

    if descriptors.ndim != 2 or descriptors.shape[0] != n:
        raise ValueError(
            f"{name}.descriptors must have shape (N, D) with N = {n} (one row per keypoint); "
            f"got {descriptors.shape}."
        )
    if method == "sift" and descriptors.shape[1] != SIFT_DESCRIPTOR_SIZE:
        raise ValueError(
            f"{name}.descriptors must have shape (N, {SIFT_DESCRIPTOR_SIZE}) for SIFT; "
            f"got {descriptors.shape}."
        )
    if not np.isfinite(descriptors).all():
        raise ValueError(f"{name}.descriptors contains NaN or infinite values.")
    return np.arange(n), keypoints, descriptors, 0


def match_feature_sets(
    gray1: np.ndarray,
    features1: dict,
    gray2: np.ndarray,
    features2: dict,
    method: str = "sift",
    metric: str = "ssd",
    patch_size: int = 11,
    ratio: Optional[float] = 0.8,
    threshold: Optional[float] = None,
    cross_check: bool = True,
    max_matches: Optional[int] = None,
) -> Dict:
    """Match two feature sets and report timings. Raises ValueError on bad input."""
    method, metric = method.lower(), metric.lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}.")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {SUPPORTED_METRICS}.")
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd.")

    kp1, desc1 = parse_feature_set(features1, "features1")
    kp2, desc2 = parse_feature_set(features2, "features2")

    # Descriptor generation (only does real work for Harris without descriptors)
    t0 = time.perf_counter()
    idx1, pts1, d1, dropped1 = _prepare("features1", gray1, kp1, desc1, method, patch_size)
    idx2, pts2, d2, dropped2 = _prepare("features2", gray2, kp2, desc2, method, patch_size)
    descriptor_ms = (time.perf_counter() - t0) * 1000

    if len(d1) and len(d2) and d1.shape[1] != d2.shape[1]:
        raise ValueError(
            f"Descriptor lengths differ: features1 has {d1.shape[1]}, features2 has {d2.shape[1]}."
        )

    # Matching
    t0 = time.perf_counter()
    i, j, scores = match_descriptors(d1, d2, metric, ratio, threshold, cross_check)
    matching_ms = (time.perf_counter() - t0) * 1000

    if max_matches is not None:
        i, j, scores = i[:max_matches], j[:max_matches], scores[:max_matches]

    matches: List[Dict] = [
        {
            "index1": int(idx1[a]),                           # index into the ORIGINAL keypoints list
            "index2": int(idx2[b]),
            "point1": [float(pts1[a, 0]), float(pts1[a, 1])],
            "point2": [float(pts2[b, 0]), float(pts2[b, 1])],
            "score": float(s),                                # raw SSD (lower=better) or NCC (higher=better)
        }
        for a, b, s in zip(i, j, scores)
    ]

    return {
        "method": method,
        "metric": metric,
        "num_keypoints1": int(len(kp1)),
        "num_keypoints2": int(len(kp2)),
        "dropped_near_border1": int(dropped1),
        "dropped_near_border2": int(dropped2),
        "num_matches": len(matches),
        "descriptor_time_ms": round(descriptor_ms, 3),
        "matching_time_ms": round(matching_ms, 3),
        "matches": matches,
    }


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------

def draw_matches(
    rgb1: np.ndarray,
    rgb2: np.ndarray,
    matches: List[Dict],
    circle_radius: int = 8,
    draw_lines: bool = True,
) -> np.ndarray:
    """Place the images side by side; circle each matched point (same colour in
    both images) and optionally connect the pairs with a line. Returns RGB uint8."""
    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = rgb1
    canvas[:h2, w1:] = rgb2

    n = len(matches)
    if n == 0:
        return canvas

    # Well-separated colours (golden-ratio hue steps), converted HSV -> RGB once.
    hues = ((np.arange(n) * 0.61803398875) % 1.0 * 179).astype(np.uint8)
    hsv = np.ascontiguousarray(
        np.stack([hues, np.full(n, 255, np.uint8), np.full(n, 255, np.uint8)], axis=1)[None]
    )
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]

    points = []
    for m, c in zip(matches, colors):
        color = (int(c[0]), int(c[1]), int(c[2]))
        p1 = (int(round(m["point1"][0])), int(round(m["point1"][1])))
        p2 = (int(round(m["point2"][0])) + w1, int(round(m["point2"][1])))
        points.append((p1, p2, color))

    if draw_lines:                                            # lines first so circles sit on top
        for p1, p2, color in points:
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
    for p1, p2, color in points:
        cv2.circle(canvas, p1, circle_radius, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, p2, circle_radius, color, 2, cv2.LINE_AA)

    return canvas