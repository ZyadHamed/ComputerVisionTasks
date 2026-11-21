"""Scale-Invariant Feature Transform (SIFT) Service.

Implements the end-to-end SIFT algorithm following the structure in SIFT_(1).ipynb:
  1. Detect keypoints:
     - Generate_Scale_Space
     - Calculate_DoG
     - Detect_Local_Extrema
     - Localize_And_Filter_Keypoints
     - Detect_Keypoints
  2. Orientation Assignment:
     - Calculate_Gradients
     - Apply_Gaussian_Weighting
     - Build_Orientation_Histogram
     - Assign_Dominant_Orientation
     - Assign_Orientation_To_Keypoint
  3. Descriptor Neighborhood Preparation:
     - Extract_Local_Neighborhood
     - Rotate_Neighborhood
     - Normalize_Orientation
     - Apply_Descriptor_Gaussian_Weighting
  4. Descriptor Histogram:
     - Divide_Into_Cells
     - Calculate_Local_Gradients
     - Build_Local_Orientation_Histograms
     - Apply_Trilinear_Interpolation
     - Concatenate_Histograms
  5. Descriptor Normalization:
     - Normalize_Descriptor
     - Threshold_And_Renormalize
  6. End-to-End Pipeline & Visualization:
     - compute_descriptor
     - extract_sift_features
     - draw_sift_keypoints
     - circ_diff
"""
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from Backend.Services.UtilitiesService import TransformIntoGrayScale


# Helper / Image preprocessing


def to_sift_gray(image: np.ndarray) -> np.ndarray:
    """Convert an image (RGB uint8, grayscale uint8, or float) to float32 [0.0, 1.0] grayscale."""
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        gray = TransformIntoGrayScale(image, mode="weighted").astype(np.float32)
    else:
        gray = image.astype(np.float32)

    if gray.max() > 1.0:
        gray = gray / 255.0

    return np.clip(gray, 0.0, 1.0)


def circ_diff(a: Any, b: Any) -> np.ndarray:
    """Smallest absolute difference between two angles in degrees (handles 0/360 wrap)."""
    return np.abs(
        (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + 180) % 360 - 180
    )



# 1. Detect keypoints


def Generate_Scale_Space(
    image: np.ndarray,
    num_octaves: int = 3,
    num_intervals: int = 3,
    sigma: float = 1.6,
    assumed_blur: float = 0.5,
) -> List[List[np.ndarray]]:
    """Generate Gaussian scale space pyramid for an image."""
    k = 2 ** (1 / num_intervals)
    scale_space = []

    # Octave base is first brought to exactly `sigma`, and every level is produced
    # with the incremental blur needed to reach its absolute sigma.
    init = np.sqrt(max(sigma ** 2 - assumed_blur ** 2, 0.01))
    octave_image = cv2.GaussianBlur(image, (0, 0), sigmaX=init, sigmaY=init)

    for octave in range(num_octaves):
        gaussian_images = [octave_image]

        for i in range(1, num_intervals + 3):
            target_sigma = sigma * (k ** i)
            incremental = np.sqrt(target_sigma ** 2 - sigma ** 2)

            blurred = cv2.GaussianBlur(
                octave_image,
                (0, 0),
                sigmaX=incremental,
                sigmaY=incremental,
            )
            gaussian_images.append(blurred)

        scale_space.append(gaussian_images)

        # Downsample the image with 2*sigma blur to start the next octave
        base = gaussian_images[num_intervals]
        new_w, new_h = base.shape[1] // 2, base.shape[0] // 2
        if new_w < 1 or new_h < 1:
            break
        octave_image = cv2.resize(
            base,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return scale_space


def Calculate_DoG(scale_space: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    """Compute Difference of Gaussians (DoG) pyramid across adjacent scale levels."""
    dog_space = []

    for octave_images in scale_space:
        dog_images = []
        for i in range(len(octave_images) - 1):
            dog = (
                octave_images[i + 1].astype(np.float32)
                - octave_images[i].astype(np.float32)
            )
            dog_images.append(dog)
        dog_space.append(dog_images)

    return dog_space


def Detect_Local_Extrema(dog_space: List[List[np.ndarray]]) -> List[Dict[str, Any]]:
    """Detect local 3D maxima and minima in the 26-neighborhood of the DoG space."""
    candidates = []

    for octave, dog_images in enumerate(dog_space):
        # We need three consecutive DoG images
        for s in range(1, len(dog_images) - 1):
            prev_img = dog_images[s - 1]
            curr_img = dog_images[s]
            next_img = dog_images[s + 1]

            # Exclude image borders
            center = curr_img[1:-1, 1:-1]
            if center.size == 0:
                continue

            is_max = np.ones(center.shape, dtype=bool)
            is_min = np.ones(center.shape, dtype=bool)

            # Compare with all 26 neighbors
            for ds in [-1, 0, 1]:
                neighbor_img = dog_images[s + ds]
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if ds == 0 and dy == 0 and dx == 0:
                            continue

                        neighbor = neighbor_img[
                            1 + dy : neighbor_img.shape[0] - 1 + dy,
                            1 + dx : neighbor_img.shape[1] - 1 + dx,
                        ]

                        is_max &= center > neighbor
                        is_min &= center < neighbor

            extrema = is_max | is_min
            ys, xs = np.where(extrema)

            for y, x in zip(ys, xs):
                candidates.append({
                    "x": int(x + 1),
                    "y": int(y + 1),
                    "octave": octave,
                    "scale": s,
                    "response": float(curr_img[y + 1, x + 1]),
                })

    return candidates


def Localize_And_Filter_Keypoints(
    dog_space: List[List[np.ndarray]],
    candidates: List[Dict[str, Any]],
    contrast_threshold: float = 0.03,
    edge_threshold: float = 10.0,
) -> List[Dict[str, Any]]:
    """Refine keypoint location via subpixel quadratic fit and filter low-contrast/edges."""
    keypoints = []

    for kp in candidates:
        octave = kp["octave"]
        s = kp["scale"]
        dog = dog_space[octave]
        x = kp["x"]
        y = kp["y"]

        # Ensure the point has enough neighbors
        if x < 1 or y < 1:
            continue
        if x >= dog[0].shape[1] - 1 or y >= dog[0].shape[0] - 1:
            continue
        if s < 1 or s >= len(dog) - 1:
            continue

        img = dog[s]

        # First derivatives
        dx = (img[y, x + 1] - img[y, x - 1]) / 2.0
        dy = (img[y + 1, x] - img[y - 1, x]) / 2.0
        ds = (dog[s + 1][y, x] - dog[s - 1][y, x]) / 2.0

        gradient = np.array([dx, dy, ds])

        # Second derivatives
        dxx = img[y, x + 1] + img[y, x - 1] - 2.0 * img[y, x]
        dyy = img[y + 1, x] + img[y - 1, x] - 2.0 * img[y, x]
        dss = dog[s + 1][y, x] + dog[s - 1][y, x] - 2.0 * img[y, x]

        dxy = (
            img[y + 1, x + 1] - img[y + 1, x - 1]
            - img[y - 1, x + 1] + img[y - 1, x - 1]
        ) / 4.0

        dxs = (
            dog[s + 1][y, x + 1] - dog[s + 1][y, x - 1]
            - dog[s - 1][y, x + 1] + dog[s - 1][y, x - 1]
        ) / 4.0

        dys = (
            dog[s + 1][y + 1, x] - dog[s + 1][y - 1, x]
            - dog[s - 1][y + 1, x] + dog[s - 1][y - 1, x]
        ) / 4.0

        # Hessian matrix in x, y, scale dimensions
        H = np.array([
            [dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss],
        ])

        try:
            offset = -np.linalg.solve(H, gradient)
        except np.linalg.LinAlgError:
            continue

        # An offset > 0.5 in any dimension means the true extremum lies
        # closer to a neighbouring sample, so the quadratic fit is unreliable
        if np.any(np.abs(offset) > 0.5):
            continue

        # Subpixel / subscale localization
        contrast = img[y, x] + 0.5 * np.dot(gradient, offset)

        if abs(contrast) < contrast_threshold:
            continue

        # Edge response elimination
        det = dxx * dyy - dxy ** 2
        trace = dxx + dyy

        if det <= 0:
            continue

        curvature_ratio = (trace ** 2) / det
        threshold = ((edge_threshold + 1) ** 2) / edge_threshold

        if curvature_ratio >= threshold:
            continue

        # Convert coordinates to original image scale
        scale_factor = 2 ** octave

        refined_x = float((x + offset[0]) * scale_factor)
        refined_y = float((y + offset[1]) * scale_factor)
        refined_sigma = 1.6 * (2 ** ((s + offset[2]) / 3.0)) * scale_factor

        keypoints.append({
            "x": refined_x,
            "y": refined_y,
            "sigma": float(refined_sigma),
            "octave": octave,
            "scale": s,
            "response": float(contrast),
        })

    return keypoints


def Detect_Keypoints(
    image: np.ndarray,
    return_scale_space: bool = False,
    num_octaves: int = 3,
    num_intervals: int = 3,
    sigma: float = 1.6,
    assumed_blur: float = 0.5,
    contrast_threshold: float = 0.03,
    edge_threshold: float = 10.0,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[List[np.ndarray]]]]:
    """Complete SIFT keypoint detection pipeline."""
    # Step 1: Generate Gaussian scale space
    scale_space = Generate_Scale_Space(
        image,
        num_octaves=num_octaves,
        num_intervals=num_intervals,
        sigma=sigma,
        assumed_blur=assumed_blur,
    )

    # Step 2: Calculate Difference of Gaussians
    dog_space = Calculate_DoG(scale_space)

    # Step 3: Detect local maxima and minima
    candidates = Detect_Local_Extrema(dog_space)

    # Step 4: Localize and filter candidate keypoints
    keypoints = Localize_And_Filter_Keypoints(
        dog_space,
        candidates,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
    )

    if return_scale_space:
        return keypoints, scale_space

    return keypoints



# 2. Orientation Assignment


def Calculate_Gradients(gaussian_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate gradient magnitude and orientation ([0, 360) deg) using central differences."""
    image = gaussian_image.astype(np.float32)

    Gy, Gx = np.gradient(image)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    orientation = np.arctan2(Gy, Gx)

    orientation = np.degrees(orientation)
    orientation = (orientation + 360) % 360

    return magnitude, orientation


def Apply_Gaussian_Weighting(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    x: int,
    y: int,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Gaussian window weighting to gradient magnitudes around a keypoint."""
    sigma_w = 1.5 * sigma
    radius = int(round(3 * sigma_w))

    x_min = max(0, x - radius)
    x_max = min(magnitude.shape[1], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(magnitude.shape[0], y + radius + 1)

    local_magnitude = magnitude[y_min:y_max, x_min:x_max]
    local_orientation = orientation[y_min:y_max, x_min:x_max]

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    dx = xx - x
    dy = yy - y

    gaussian_weights = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_w ** 2))
    weighted_magnitude = local_magnitude * gaussian_weights

    return weighted_magnitude, local_orientation


def Build_Orientation_Histogram(
    weighted_magnitude: np.ndarray,
    orientation: np.ndarray,
    num_bins: int = 36,
) -> np.ndarray:
    """Build weighted orientation histogram with num_bins bins covering [0, 360)."""
    bin_width = 360 / num_bins
    bin_indices = np.floor(orientation / bin_width).astype(int)
    bin_indices = bin_indices % num_bins

    histogram = np.bincount(
        bin_indices.ravel(),
        weights=weighted_magnitude.ravel(),
        minlength=num_bins,
    ).astype(np.float32)

    return histogram


def Assign_Dominant_Orientation(
    histogram: np.ndarray,
    num_bins: int = 36,
    peak_ratio: float = 0.8,
) -> List[float]:
    """Smooth orientation histogram, find dominant peaks and refine via parabolic interpolation."""
    # 5-tap binomial kernel smoothing (same as OpenCV's SIFT)
    smoothed = (
        6 * histogram
        + 4 * (np.roll(histogram, 1) + np.roll(histogram, -1))
        + (np.roll(histogram, 2) + np.roll(histogram, -2))
    ) / 16.0

    max_value = np.max(smoothed)
    if max_value <= 0:
        return []

    orientations = []
    for i in range(num_bins):
        previous_bin = smoothed[(i - 1) % num_bins]
        current_bin = smoothed[i]
        next_bin = smoothed[(i + 1) % num_bins]

        if (
            current_bin > previous_bin
            and current_bin > next_bin
            and current_bin >= peak_ratio * max_value
        ):
            denominator = previous_bin - 2 * current_bin + next_bin
            if abs(denominator) > 1e-10:
                offset = 0.5 * (previous_bin - next_bin) / denominator
            else:
                offset = 0.0

            bin_width = 360 / num_bins
            angle = (i + 0.5 + offset) * bin_width
            angle = angle % 360

            orientations.append(float(angle))

    return orientations


def Assign_Orientation_To_Keypoint(
    kp: Dict[str, Any],
    scale_space: List[List[np.ndarray]],
    num_intervals: int = 3,
    peak_ratio: float = 0.8,
) -> List[float]:
    """Assign one or more dominant orientations (in degrees) to a keypoint."""
    octave = kp["octave"]
    scale_factor = 2 ** octave

    x = int(round(kp["x"] / scale_factor))
    y = int(round(kp["y"] / scale_factor))
    sigma = kp["sigma"] / scale_factor

    gaussian_scales = [
        1.6 * (2 ** (i / num_intervals))
        for i in range(num_intervals + 3)
    ]
    scale_index = int(np.argmin(np.abs(np.array(gaussian_scales) - sigma)))
    gaussian_image = scale_space[octave][scale_index]

    # 1. Calculate gradients
    magnitude, orientation = Calculate_Gradients(gaussian_image)

    # 2. Apply Gaussian weighting
    weighted_magnitude, local_orientation = Apply_Gaussian_Weighting(
        magnitude, orientation, x, y, sigma
    )

    # 3. Build orientation histogram
    histogram = Build_Orientation_Histogram(weighted_magnitude, local_orientation)

    # 4. Assign dominant orientations
    assigned_orientations = Assign_Dominant_Orientation(
        histogram, peak_ratio=peak_ratio
    )

    return assigned_orientations



# 3. Descriptor Neighborhood Preparation


def Extract_Local_Neighborhood(
    gaussian_image: np.ndarray,
    keypoint: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Extract square image patch around keypoint in its octave scale."""
    octave = keypoint["octave"]
    x = keypoint["x"]
    y = keypoint["y"]
    sigma = keypoint["sigma"]

    scale_factor = 2 ** octave
    x = x / scale_factor
    y = y / scale_factor
    sigma = sigma / scale_factor

    window_size = int(round(8 * sigma))
    if window_size % 2 == 0:
        window_size += 1

    radius = window_size // 2
    center_x = int(round(x))
    center_y = int(round(y))

    x_min = center_x - radius
    x_max = center_x + radius + 1
    y_min = center_y - radius
    y_max = center_y + radius + 1

    if (
        x_min < 0
        or y_min < 0
        or x_max > gaussian_image.shape[1]
        or y_max > gaussian_image.shape[0]
    ):
        return None

    return gaussian_image[y_min:y_max, x_min:x_max]


def Rotate_Neighborhood(
    neighborhood: np.ndarray,
    keypoint_orientation: float,
) -> np.ndarray:
    """Rotate local neighborhood patch to canonical keypoint orientation."""
    height, width = neighborhood.shape
    angle = keypoint_orientation
    center = ((width - 1) / 2.0, (height - 1) / 2.0)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_neighborhood = cv2.warpAffine(
        neighborhood,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return rotated_neighborhood


def Normalize_Orientation(
    local_orientations: np.ndarray,
    keypoint_orientation: float,
) -> np.ndarray:
    """Normalize local gradient angles relative to keypoint orientation."""
    return (local_orientations - keypoint_orientation) % 360


def Apply_Descriptor_Gaussian_Weighting(
    gradient_magnitude: np.ndarray,
    neighborhood_sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weight descriptor gradient magnitudes with Gaussian window centered on patch."""
    height, width = gradient_magnitude.shape
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    sigma_d = neighborhood_sigma if neighborhood_sigma is not None else (min(height, width) / 2.0)

    y, x = np.mgrid[0:height, 0:width]
    dx = x - center_x
    dy = y - center_y

    gaussian_weights = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_d ** 2))
    weighted_magnitude = gradient_magnitude * gaussian_weights

    return weighted_magnitude, gaussian_weights



# 4. Descriptor Histogram


def Divide_Into_Cells(
    neighborhood: np.ndarray,
    num_cells: int = 4,
) -> List[np.ndarray]:
    """Divide neighborhood patch into a grid of num_cells x num_cells sub-patches."""
    height, width = neighborhood.shape
    cell_height = height // num_cells
    cell_width = width // num_cells

    cells = []
    for row in range(num_cells):
        for col in range(num_cells):
            y_start = row * cell_height
            y_end = y_start + cell_height
            x_start = col * cell_width
            x_end = x_start + cell_width

            cell = neighborhood[y_start:y_end, x_start:x_end]
            cells.append(cell)

    return cells


def Calculate_Local_Gradients(neighborhood: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate gradient magnitude and orientation for a local neighborhood."""
    return Calculate_Gradients(neighborhood)


def Build_Local_Orientation_Histograms(
    cells: List[np.ndarray],
    gradient_magnitude: np.ndarray,
    gradient_orientation: np.ndarray,
    num_bins: int = 8,
) -> List[np.ndarray]:
    """Build orientation histograms for each cell without interpolation."""
    histograms = []
    bin_width = 360 / num_bins

    for cell_index, cell in enumerate(cells):
        height, width = cell.shape
        histogram = np.zeros(num_bins, dtype=np.float32)

        num_cells = int(np.sqrt(len(cells)))
        row = cell_index // num_cells
        col = cell_index % num_cells

        cell_height = gradient_magnitude.shape[0] // num_cells
        cell_width = gradient_magnitude.shape[1] // num_cells

        y_start = row * cell_height
        x_start = col * cell_width

        local_magnitude = gradient_magnitude[
            y_start : y_start + height,
            x_start : x_start + width,
        ]
        local_orientation = gradient_orientation[
            y_start : y_start + height,
            x_start : x_start + width,
        ]

        for y in range(height):
            for x in range(width):
                magnitude = local_magnitude[y, x]
                orientation = local_orientation[y, x]
                bin_index = int(orientation // bin_width) % num_bins
                histogram[bin_index] += magnitude

        histograms.append(histogram)

    return histograms


def Apply_Trilinear_Interpolation(
    gradient_magnitude: np.ndarray,
    gradient_orientation: np.ndarray,
    num_cells: int = 4,
    num_bins: int = 8,
) -> np.ndarray:
    """Distribute gradient samples into 3D spatial-orientation histogram grid via trilinear interpolation."""
    height, width = gradient_magnitude.shape
    histograms = np.zeros((num_cells, num_cells, num_bins), dtype=np.float32)

    cell_width = width / num_cells
    cell_height = height / num_cells

    for y in range(height):
        for x in range(width):
            magnitude = gradient_magnitude[y, x]
            orientation = gradient_orientation[y, x]

            # Continuous spatial coordinates
            col = (x + 0.5) / cell_width - 0.5
            row = (y + 0.5) / cell_height - 0.5

            # Continuous orientation-bin coordinate
            orientation_bin = orientation / (360 / num_bins)

            # Find neighboring bins
            col_floor = int(np.floor(col))
            row_floor = int(np.floor(row))
            bin_floor = int(np.floor(orientation_bin))

            # Fractional distances
            col_fraction = col - col_floor
            row_fraction = row - row_floor
            bin_fraction = orientation_bin - bin_floor

            # Distribute contribution across 8 neighboring bin corners
            for dy in [0, 1]:
                for dx in [0, 1]:
                    for db in [0, 1]:
                        cell_row = row_floor + dy
                        cell_col = col_floor + dx
                        bin_index = (bin_floor + db) % num_bins

                        # Ignore spatial cells outside descriptor
                        if not (0 <= cell_row < num_cells):
                            continue
                        if not (0 <= cell_col < num_cells):
                            continue

                        # Calculate interpolation weights
                        wy = 1 - row_fraction if dy == 0 else row_fraction
                        wx = 1 - col_fraction if dx == 0 else col_fraction
                        wb = 1 - bin_fraction if db == 0 else bin_fraction

                        weight = wx * wy * wb
                        histograms[cell_row, cell_col, bin_index] += magnitude * weight

    return histograms


def Concatenate_Histograms(histograms: np.ndarray) -> np.ndarray:
    """Flatten 3D histogram tensor into a 1D feature vector."""
    return histograms.flatten()


# 5. Descriptor Normalization


def Normalize_Descriptor(descriptor: np.ndarray) -> np.ndarray:
    """L2 normalize descriptor vector."""
    norm = np.linalg.norm(descriptor)
    if norm < 1e-10:
        return descriptor
    return descriptor / norm


def Threshold_And_Renormalize(
    descriptor: np.ndarray,
    threshold: float = 0.2,
) -> np.ndarray:
    """Clamp large gradient values to threshold to reduce non-linear illumination effects, then renormalize."""
    clipped_descriptor = np.clip(descriptor, 0, threshold)
    norm = np.linalg.norm(clipped_descriptor)
    if norm < 1e-10:
        return clipped_descriptor
    return clipped_descriptor / norm



# 6. End-to-End Pipeline & Visualization


def compute_descriptor(
    kp: Dict[str, Any],
    scale_space: List[List[np.ndarray]],
    theta: float,
    num_cells: int = 4,
    num_bins: int = 8,
    threshold: float = 0.2,
) -> Optional[np.ndarray]:
    """Compute 128-D SIFT descriptor for keypoint kp with canonical orientation theta."""
    patch_ = Extract_Local_Neighborhood(scale_space[kp["octave"]][kp["scale"]], kp)
    if patch_ is None:
        return None

    patch_ = Rotate_Neighborhood(patch_, theta)
    magnitude, orientation = Calculate_Local_Gradients(patch_)
    magnitude, _ = Apply_Descriptor_Gaussian_Weighting(magnitude)

    interpolated_histograms = Apply_Trilinear_Interpolation(
        magnitude, orientation, num_cells=num_cells, num_bins=num_bins
    )

    descriptor = Concatenate_Histograms(interpolated_histograms)
    normalized_descriptor = Normalize_Descriptor(descriptor)
    final_descriptor = Threshold_And_Renormalize(normalized_descriptor, threshold=threshold)
    return final_descriptor


def extract_sift_features(
    image: np.ndarray,
    max_keypoints: Optional[int] = 500,
    contrast_threshold: float = 0.03,
    edge_threshold: float = 10.0,
    num_octaves: int = 3,
    num_intervals: int = 3,
    sigma: float = 1.6,
    assumed_blur: float = 0.5,
    num_cells: int = 4,
    num_bins: int = 8,
    peak_ratio: float = 0.8,
    descriptor_threshold: float = 0.2,
) -> Dict[str, Any]:
    """Extract SIFT keypoints and 128-D descriptors from an image.
    
    Output format is directly compatible with FeatureMatchingService:
      - keypoints: (N, 2) list of [x, y] coordinates in original image pixel frame.
      - descriptors: (N, 128) list of descriptor floats.
      - keypoint_details: list of dicts with {x, y, sigma, angle, response, octave, scale}.
      - num_keypoints: count of extracted features.
      - execution_time_ms: total elapsed runtime in milliseconds.
    """
    gray = to_sift_gray(image)

    t0 = time.perf_counter()

    kps, scale_space = Detect_Keypoints(
        gray,
        return_scale_space=True,
        num_octaves=num_octaves,
        num_intervals=num_intervals,
        sigma=sigma,
        assumed_blur=assumed_blur,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
    )

    if max_keypoints is not None and len(kps) > max_keypoints:
        kps = sorted(kps, key=lambda k: -abs(k["response"]))[:max_keypoints]

    descriptors: List[np.ndarray] = []
    owners: List[Tuple[Dict[str, Any], float]] = []

    for kp in kps:
        thetas = Assign_Orientation_To_Keypoint(
            kp,
            scale_space,
            num_intervals=num_intervals,
            peak_ratio=peak_ratio,
        )
        for theta in thetas:
            dsc = compute_descriptor(
                kp,
                scale_space,
                theta,
                num_cells=num_cells,
                num_bins=num_bins,
                threshold=descriptor_threshold,
            )
            if dsc is not None:
                descriptors.append(dsc)
                owners.append((kp, theta))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    keypoints_coords = [[float(kp["x"]), float(kp["y"])] for kp, _ in owners]
    descriptors_list = [dsc.astype(float).tolist() for dsc in descriptors]
    keypoint_details = [
        {
            "x": float(kp["x"]),
            "y": float(kp["y"]),
            "sigma": float(kp["sigma"]),
            "angle": float(theta),
            "response": float(kp["response"]),
            "octave": int(kp["octave"]),
            "scale": int(kp["scale"]),
        }
        for kp, theta in owners
    ]

    return {
        "keypoints": keypoints_coords,
        "descriptors": descriptors_list,
        "keypoint_details": keypoint_details,
        "num_keypoints": len(descriptors),
        "execution_time_ms": round(elapsed_ms, 3),
    }


def draw_sift_keypoints(
    image: np.ndarray,
    keypoints: List[Any],
    circle_radius: Optional[int] = None,
    draw_orientation: bool = True,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Draw SIFT keypoints with scale circles and orientation rays.
    
    Accepts RGB/grayscale uint8 image and keypoint dictionaries or coordinates.
    Returns RGB uint8 canvas.
    """
    if image.dtype != np.uint8:
        if image.dtype == np.uint16:
            image = image // 257
        elif np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        canvas = image[:, :, :3].copy()
    else:
        canvas = image.copy()

    n = len(keypoints)
    if n == 0:
        return canvas

    # Determine colors
    if color is not None:
        colors = [color] * n
    else:
        # Golden-ratio hues for vivid, distinct colors
        hues = ((np.arange(n) * 0.61803398875) % 1.0 * 179).astype(np.uint8)
        hsv = np.ascontiguousarray(
            np.stack([hues, np.full(n, 255, np.uint8), np.full(n, 255, np.uint8)], axis=1)[None]
        )
        colors_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]
        colors = [(int(c[0]), int(c[1]), int(c[2])) for c in colors_rgb]

    for item, clr in zip(keypoints, colors):
        if isinstance(item, dict):
            x = float(item["x"])
            y = float(item["y"])
            sigma = float(item.get("sigma", 1.6))
            angle = float(item.get("angle", 0.0)) if "angle" in item else None
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = float(item[0]), float(item[1])
            sigma = 1.6
            angle = None
        else:
            continue

        pt = (int(round(x)), int(round(y)))
        r = int(round(circle_radius)) if circle_radius is not None else max(int(round(2 * sigma)), 3)

        # Draw circle and center point
        cv2.circle(canvas, pt, r, clr, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 1, clr, -1, cv2.LINE_AA)

        # Draw orientation line
        if draw_orientation and angle is not None:
            rad = np.radians(angle)
            end_x = int(round(x + r * np.cos(rad)))
            end_y = int(round(y + r * np.sin(rad)))
            cv2.line(canvas, pt, (end_x, end_y), clr, 1, cv2.LINE_AA)

    return canvas



generate_scale_space = Generate_Scale_Space
calculate_dog = Calculate_DoG
detect_local_extrema = Detect_Local_Extrema
localize_and_filter_keypoints = Localize_And_Filter_Keypoints
detect_keypoints = Detect_Keypoints
calculate_gradients = Calculate_Gradients
apply_gaussian_weighting = Apply_Gaussian_Weighting
build_orientation_histogram = Build_Orientation_Histogram
assign_dominant_orientation = Assign_Dominant_Orientation
assign_orientation_to_keypoint = Assign_Orientation_To_Keypoint
extract_local_neighborhood = Extract_Local_Neighborhood
rotate_neighborhood = Rotate_Neighborhood
normalize_orientation = Normalize_Orientation
apply_descriptor_gaussian_weighting = Apply_Descriptor_Gaussian_Weighting
divide_into_cells = Divide_Into_Cells
calculate_local_gradients = Calculate_Local_Gradients
build_local_orientation_histograms = Build_Local_Orientation_Histograms
apply_trilinear_interpolation = Apply_Trilinear_Interpolation
concatenate_histograms = Concatenate_Histograms
normalize_descriptor = Normalize_Descriptor
threshold_and_renormalize = Threshold_And_Renormalize
