import logging

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from Backend.Services.UtilitiesService import (
    ConvolveImage,
    TransformIntoGrayScale,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Layer 1: Edge map
# ----------------------------------------------------------------------------

def gaussian_blur(img: np.ndarray, kernel_size=3, sigma=1.0, manual: bool = False):
    half = kernel_size // 2
    offsets = np.arange(-half, half + 1)  # position of each column relative to the center
    x, y = np.meshgrid(offsets, offsets)

    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # normalize so the weights sum to 1, otherwise the image gets brighter
    gaussian_kernel = kernel / kernel.sum()

    return np.round(ConvolveImage(img, gaussian_kernel, manual=manual)).astype(np.uint8)


SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def compute_edge_map(img: np.ndarray, kernel_size=5, sigma=2.0, manual=False):
    # 1) grayscale (skip if already 2D)
    if img.ndim == 3:
        gray = TransformIntoGrayScale(img, "weighted")
    else:
        gray = img.copy()

    blurred = gaussian_blur(gray, kernel_size, sigma, manual).astype(np.float64)

    # 2) Sobel: horizontal and vertical brightness change
    gx = ConvolveImage(blurred, SOBEL_X, manual=manual)
    gy = ConvolveImage(blurred, SOBEL_Y, manual=manual)

    # 3) edge strength (gradient magnitude)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # 4) remove fake edges at the border (caused by zero padding)
    m = kernel_size // 2 + 1
    magnitude[:m, :] = 0
    magnitude[-m:, :] = 0
    magnitude[:, :m] = 0
    magnitude[:, -m:] = 0

    # 5) scale to [0, 1] using the 99th percentile
    nonzero_values = magnitude[magnitude > 0]
    if len(nonzero_values) > 0:
        scale = np.percentile(nonzero_values, 99)
        magnitude = np.clip(magnitude / (scale + 1e-12), 0, 1)

    return magnitude


# ----------------------------------------------------------------------------
# Layer 2: Contour initialization 
# ----------------------------------------------------------------------------

def make_circle(center, radius, num_points=100):
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.round(np.column_stack((x, y))).astype(int)


def make_rectangle(center, width, height):
    cx, cy = center
    half_width = width // 2
    half_height = height // 2
    contour = np.array([
        [cx - half_width, cy - half_height],  # Top-left
        [cx + half_width, cy - half_height],  # Top-right
        [cx + half_width, cy + half_height],  # Bottom-right
        [cx - half_width, cy + half_height],  # Bottom-left
    ])
    return np.round(contour).astype(int)


def make_square(center, side_length):
    return make_rectangle(center, side_length, side_length)


def resample_contour(contour, num_points=100):
    contour = np.asarray(contour, dtype=float)

    # Close the polygon by connecting the last point to the first.
    closed_contour = np.vstack([contour, contour[0]])

    segments = np.diff(closed_contour, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)

    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    perimeter = cumulative_lengths[-1]

    if perimeter == 0:
        raise ValueError("Contour perimeter must be greater than zero.")

    # Target positions, excluding the duplicate endpoint.
    target_distances = np.linspace(0, perimeter, num_points, endpoint=False)

    # Find which segment contains each target position.
    segment_indices = np.searchsorted(
        cumulative_lengths, target_distances, side="right"
    ) - 1
    segment_indices = np.clip(segment_indices, 0, len(segment_lengths) - 1)

    start_distances = cumulative_lengths[segment_indices]
    lengths = segment_lengths[segment_indices]

    t = np.zeros(num_points)
    valid = lengths > 0
    t[valid] = (target_distances[valid] - start_distances[valid]) / lengths[valid]

    resampled = closed_contour[segment_indices] + t[:, None] * segments[segment_indices]

    return np.rint(resampled).astype(int)


def prepare_contour(points, image_shape, num_points=None):
    """
    Validate the contour sent by the client (list of [x, y]), optionally
    resample it to `num_points` evenly spaced points, and clip it to the image.
    """
    contour = np.asarray(points, dtype=float)

    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("Contour must be a list of [x, y] points.")
    if len(contour) < 3:
        raise ValueError("Contour needs at least 3 points.")

    if num_points:
        contour = resample_contour(contour, num_points)
    else:
        contour = np.rint(contour).astype(int)

    contour[:, 0] = np.clip(contour[:, 0], 0, image_shape[1] - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, image_shape[0] - 1)
    return contour


# ----------------------------------------------------------------------------
# Layer 3: Greedy snake
# ----------------------------------------------------------------------------

def get_neighborhood(point, image_shape, radius=2):
    """
    Valid candidate positions around a point (radius=1 -> 3x3, radius=2 -> 5x5).
    point: (x, y), x = column, y = row
    """
    x, y = point
    height, width = image_shape[:2]

    candidates = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < width and 0 <= new_y < height:
                candidates.append((new_x, new_y))

    return candidates


def compute_average_distance(contour):
    """d-bar: average distance between consecutive points (closed contour)."""
    contour = np.asarray(contour, dtype=float)
    diffs = contour - np.roll(contour, 1, axis=0)
    return np.linalg.norm(diffs, axis=1).mean()


def continuity_energy(candidates, prev_point, avg_spacing):
    """
    (|candidate - prev| - d-bar)^2 for every candidate. Lower = better.
    Vectorised: accepts a single (x, y) or an array of K candidates.
    """
    candidates = np.asarray(candidates, dtype=float)
    prev_point = np.asarray(prev_point, dtype=float)
    distance = np.linalg.norm(candidates - prev_point, axis=-1)
    return (distance - avg_spacing) ** 2


def curvature_energy(candidates, prev_point, next_point):
    """|x[i+1] - 2*x[i] + x[i-1]|^2 for each candidate."""
    candidates = np.asarray(candidates, dtype=float)
    prev_point = np.asarray(prev_point, dtype=float)
    next_point = np.asarray(next_point, dtype=float)
    return np.linalg.norm(next_point - 2 * candidates + prev_point, axis=1) ** 2


def image_energy(image=None, sigma=2.0, attraction_sigma=3.0, edge_map=None):
    """
    Edge-based image energy (negative, because the snake minimises energy).

    attraction_sigma controls the capture range:
        large (10-12) -> pulled from far away, edges localised badly
        small (2-4)   -> sits exactly on the edge, but only feels it nearby
    Pass a precomputed `edge_map` to avoid recomputing it for every scale.
    """
    if edge_map is None:
        edge_map = compute_edge_map(image, sigma=sigma)

    if attraction_sigma and attraction_sigma > 0:
        widened_edges = gaussian_filter(edge_map, sigma=attraction_sigma)
    else:
        widened_edges = edge_map

    widened_edges = widened_edges / (widened_edges.max() + 1e-12)
    return -widened_edges


def normalize_energy(energy, min_range=1e-12):
    """Min-max normalise to [0, 1] (min_range avoids amplifying flat-region noise)."""
    energy = np.asarray(energy, dtype=float)
    min_val = np.min(energy)
    value_range = max(np.max(energy) - min_val, min_range)
    return (energy - min_val) / value_range


def move_point(contour, index, image_energy_map,
               avg_spacing, alpha=1.0, beta=1.0, gamma=1.0, radius=2):
    """Greedy step for one point; every energy term is normalised over the window."""
    current_point = contour[index]
    prev_point = contour[(index - 1) % len(contour)]
    next_point = contour[(index + 1) % len(contour)]

    candidates = np.asarray(
        get_neighborhood(current_point, image_energy_map.shape, radius)
    )

    e_cont = continuity_energy(candidates, prev_point, avg_spacing)
    e_curv = curvature_energy(candidates, prev_point, next_point)
    e_img = image_energy_map[candidates[:, 1], candidates[:, 0]]  # (x, y) -> [row, col]

    total_energy = (
        alpha * normalize_energy(e_cont)
        + beta * normalize_energy(e_curv)
        + gamma * normalize_energy(e_img, min_range=0.05)
    )

    return candidates[np.argmin(total_energy)]


def detect_corners(contour, edge_map, angle_threshold=45, edge_threshold=0.5):
    corners = []
    n = len(contour)

    for i in range(n):
        prev_point = np.asarray(contour[(i - 1) % n], dtype=float)
        curr_point = np.asarray(contour[i], dtype=float)
        next_point = np.asarray(contour[(i + 1) % n], dtype=float)

        v1 = prev_point - curr_point
        v2 = next_point - curr_point

        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denominator == 0:
            corners.append(False)
            continue

        cos_angle = np.clip(np.dot(v1, v2) / denominator, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        bend_angle = 180 - angle

        x, y = contour[i]
        strong_edge = edge_map[y, x] >= edge_threshold

        corners.append(bend_angle >= angle_threshold and strong_edge)

    return np.array(corners, dtype=bool)


def run_snake(contour, image, alpha=1.0, beta=1.0, gamma=1.0,
              attraction_sigma=3.0, max_iterations=300,
              convergence_threshold=0.02, radius=2,
              use_corners=False, edge_map=None, verbose=False):
    """
    convergence_threshold: fraction of points allowed to move in an iteration
                           before the snake is considered converged.
    use_corners:           switch curvature off at sharp bends (off by default,
                           it fires on pixel jitter with integer coordinates).
    edge_map:              optional precomputed edge map.
    """
    contour = np.asarray(contour, dtype=int).copy()

    contour[:, 0] = np.clip(contour[:, 0], 0, image.shape[1] - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, image.shape[0] - 1)

    if edge_map is None:
        edge_map = compute_edge_map(image)
    energy_map = image_energy(edge_map=edge_map, attraction_sigma=attraction_sigma)

    moved_points = 0
    for iteration in range(max_iterations):
        avg_spacing = compute_average_distance(contour)

        if use_corners:
            corners = detect_corners(contour, edge_map)
        else:
            corners = np.zeros(len(contour), dtype=bool)

        moved_points = 0
        for i in range(len(contour)):
            beta_i = 0 if corners[i] else beta

            new_point = move_point(contour, i, energy_map, avg_spacing,
                                   alpha=alpha, beta=beta_i, gamma=gamma,
                                   radius=radius)

            if not np.array_equal(new_point, contour[i]):
                contour[i] = new_point
                moved_points += 1

        if moved_points <= convergence_threshold * len(contour):
            if verbose:
                logger.info("converged at iteration %d", iteration + 1)
            break
    else:
        if verbose:
            logger.info("stopped after %d iterations (%d points still moving)",
                        max_iterations, moved_points)

    return contour


def run_snake_multiscale(contour, image, scales=(12, 6, 3), **kwargs):
    """Coarse-to-fine: wide capture range first, then precise localisation."""
    contour = np.asarray(contour, dtype=int).copy()
    edge_map = compute_edge_map(image)  # computed once, reused for every scale

    for s in scales:
        contour = run_snake(contour, image, attraction_sigma=s,
                            edge_map=edge_map, **kwargs)
    return contour


# ----------------------------------------------------------------------------
# High-level entry points used by the API
# ----------------------------------------------------------------------------

def segment_active_contour(image, points, num_points=200, scales=(12, 6, 3),
                           alpha=1.0, beta=1.0, gamma=2.0,
                           max_iterations=200, radius=2,
                           convergence_threshold=0.02, use_corners=False):
    """Returns (initial_contour, final_contour) as (N, 2) int arrays of (x, y)."""
    initial = prepare_contour(points, image.shape, num_points=num_points)

    final = run_snake_multiscale(
        initial, image, scales=scales,
        alpha=alpha, beta=beta, gamma=gamma,
        max_iterations=max_iterations, radius=radius,
        convergence_threshold=convergence_threshold,
        use_corners=use_corners,
    )
    return initial, final


def draw_contours(image, initial_contour=None, final_contour=None):
    """
    Draw the contours on an RGB copy of the image (initial = blue, final = red,
    final points = yellow). Returns an RGB uint8 image.
    """
    if image.ndim == 2:
        canvas = np.stack([image] * 3, axis=-1)
    else:
        canvas = image[:, :, :3].copy()

    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    canvas = np.ascontiguousarray(canvas)

    if initial_contour is not None:
        pts = np.asarray(initial_contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

    if final_contour is not None:
        pts = np.asarray(final_contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        for x, y in pts.reshape(-1, 2):
            cv2.circle(canvas, (int(x), int(y)), 2, (255, 255, 0), -1)

    return canvas