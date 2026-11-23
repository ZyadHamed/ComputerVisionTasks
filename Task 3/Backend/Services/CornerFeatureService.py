"""Harris / Lambda Minus corner features with SIFT or MOPS descriptors.

Uses SIFTService.Generate_Scale_Space for multi-octave scale handling.
Images are grayscale, RGB or RGBA (not BGR).
The returned keypoint coordinates are in the ORIGINAL image pixel frame.
"""

import time
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from Backend.Services.UtilitiesService import ConvolveImage
from Backend.Services.SIFTService import (
    to_sift_gray,
    Generate_Scale_Space,
    Calculate_Gradients,
    Apply_Gaussian_Weighting,
    Build_Orientation_Histogram,
    Assign_Dominant_Orientation,
    compute_descriptor,
)

CORNER_METHODS = ("harris", "lamda_minus")
DESCRIPTOR_METHODS = ("sift", "mops")

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def _gray(image: np.ndarray) -> np.ndarray:
    """Make float32 grayscale [0, 1] using the existing SIFT preprocessing."""
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Expected a nonempty grayscale, RGB or RGBA image.")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Expected 3 or 4 channels for a color image.")
    if image.dtype == np.uint16:
        image = (image // 257).astype(np.uint8)
    return to_sift_gray(image)


def ApplySobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use UtilitiesService's SciPy cross-correlation convolution for Sobel."""
    gx = ConvolveImage(gray, SOBEL_X, manual=False)
    gy = ConvolveImage(gray, SOBEL_Y, manual=False)
    return gx, gy


def ApplyHarrisOperator(gray: np.ndarray, method: str = "harris") -> np.ndarray:
    """Corner response matrix: Harris or the smallest structure-tensor eigenvalue."""
    if method not in CORNER_METHODS:
        raise ValueError(f"method must be one of {CORNER_METHODS}")

    gx, gy = ApplySobel(gray.astype(np.float64))
    sxx = cv2.GaussianBlur(gx * gx, (5, 5), 1.4)
    syy = cv2.GaussianBlur(gy * gy, (5, 5), 1.4)
    sxy = cv2.GaussianBlur(gx * gy, (5, 5), 1.4)

    if method == "harris":
        return sxx * syy - sxy ** 2 - 0.04 * (sxx + syy) ** 2

    return 0.5 * (sxx + syy - np.sqrt(np.maximum((sxx - syy) ** 2 + 4 * sxy ** 2, 0)))


def GetHarrisKeypoints(
    R: np.ndarray, threshold: float = 0.01, window_size: int = 5,
    margin: int = 10, max_keypoints: Optional[int] = None,
) -> np.ndarray:
    """Relative threshold + nonmaximum suppression; return (N,2) [x,y]."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError("window_size must be a positive odd number")
    if margin < 0:
        raise ValueError("margin cannot be negative")
    if max_keypoints is not None and max_keypoints < 1:
        raise ValueError("max_keypoints must be positive when provided")

    if R.size == 0 or not np.isfinite(R).all() or R.max() <= 0:
        return np.empty((0, 2), dtype=int)

    maxima = maximum_filter(R, size=window_size)
    mask = (R > max(threshold * R.max(), 0)) & (R == maxima)
    if margin:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False

    y, x = np.where(mask)
    if len(x):
        strongest_first = np.argsort(R[y, x])[::-1]
        if max_keypoints is not None:
            strongest_first = strongest_first[:max_keypoints]
        x, y = x[strongest_first], y[strongest_first]
    return np.column_stack((x, y))


def DetectCorners(
    image: np.ndarray, method: str = "harris", threshold: float = 0.01,
    window_size: int = 5, margin: int = 10,
    max_keypoints: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-scale detector: return (response matrix, detected keypoints)."""
    gray = _gray(image)
    R = ApplyHarrisOperator(gray, method)
    points = GetHarrisKeypoints(R, threshold, window_size, margin, max_keypoints)
    return R, points


def ExtractMOPSDescriptorsFromHarris(
    gray: np.ndarray, keypoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Single-scale MOPS-style aligned, zero-mean 8x8 patches (64 values)."""
    gx, gy = ApplySobel(gray)
    gx = cv2.GaussianBlur(gx, (5, 5), 1.5)
    gy = cv2.GaussianBlur(gy, (5, 5), 1.5)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)

    h, w = gray.shape
    points, descriptors, angles = [], [], []
    for x, y in keypoints:
        x, y = float(x), float(y)
        if x < 30 or y < 30 or x >= w - 30 or y >= h - 30:
            continue

        angle = float(np.degrees(np.arctan2(gy[round(y), round(x)], gx[round(y), round(x)])))
        matrix = cv2.getRotationMatrix2D((x, y), angle, 8 / 40)
        matrix[0, 2] += 3.5 - x
        matrix[1, 2] += 3.5 - y

        patch = cv2.warpAffine(blurred, matrix, (8, 8), flags=cv2.INTER_LINEAR)
        descriptor = patch.reshape(-1)
        std = float(descriptor.std())
        if std < 1e-6:
            continue

        descriptor = (descriptor - descriptor.mean()) / std
        points.append([x, y])
        descriptors.append(descriptor)
        angles.append(angle % 360)

    return (np.asarray(points, dtype=float).reshape(-1, 2),
            np.asarray(descriptors, dtype=float).reshape(-1, 64), angles)


def ExtractSIFTDescriptorsFromHarris(
    gray: np.ndarray, keypoints: np.ndarray,
    scale_space: Optional[list[list[np.ndarray]]] = None,
    octave: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Describe corners at a pyramid octave; keypoints and returned points are octave-local.

    If scale_space is supplied, reuse it without generating a second pyramid.
    compute_descriptor() expects x/y and sigma in ORIGINAL-image coordinates.
    """
    if not len(keypoints):
        return np.empty((0, 2)), np.empty((0, 128)), []

    if scale_space is None:
        scale_space = Generate_Scale_Space(gray, num_octaves=1)
        octave = 0

    magnitude, orientation = Calculate_Gradients(scale_space[octave][0])
    factor = 2 ** octave
    points, descriptors, angles = [], [], []
    for x, y in keypoints:
        x, y = float(x), float(y)
        kp = {
            "x": x * factor, "y": y * factor,
            "sigma": 1.6 * factor, "octave": octave, "scale": 0,
        }
        weighted, local_ori = Apply_Gaussian_Weighting(
            magnitude, orientation, round(x), round(y), 1.6
        )
        histogram = Build_Orientation_Histogram(weighted, local_ori)
        thetas = Assign_Dominant_Orientation(histogram)
        for theta in thetas:
            descriptor = compute_descriptor(kp, scale_space, theta)
            if descriptor is not None:
                points.append([x, y])
                descriptors.append(descriptor)
                angles.append(float(theta))

    return (np.asarray(points, dtype=float).reshape(-1, 2),
            np.asarray(descriptors, dtype=float).reshape(-1, 128), angles)


def ExtractCornerFeatures(
    image: np.ndarray, method: str = "harris", descriptor_method: str = "mops",
    threshold: float = 0.01, window_size: int = 5,
    margin: Optional[int] = None, max_keypoints: Optional[int] = None,
    num_octaves: int = 3,
) -> dict:
    """Detect multi-octave Harris/Lambda Minus features and return match-ready descriptors.

    Uses the EXISTING SIFT Gaussian pyramid, taking its first Gaussian scale
    at each octave. Coordinates in the response are original-image pixels.
    max_keypoints, if supplied, caps candidates across all octaves together.
    """
    if descriptor_method not in DESCRIPTOR_METHODS:
        raise ValueError(f"descriptor_method must be one of {DESCRIPTOR_METHODS}")
    if not isinstance(num_octaves, int) or num_octaves < 1:
        raise ValueError("num_octaves must be a positive integer")
    if max_keypoints is not None and max_keypoints < 1:
        raise ValueError("max_keypoints must be positive when provided")
    if margin is None:
        margin = 30 if descriptor_method == "mops" else 10

    start = time.perf_counter()
    gray = _gray(image)
    pyramid = Generate_Scale_Space(gray, num_octaves=num_octaves)

    # Detect corners on one Gaussian image at each octave.
    responses = []
    candidates = []
    for octave, levels in enumerate(pyramid):
        R = ApplyHarrisOperator(levels[0], method)
        responses.append(R)
        pts = GetHarrisKeypoints(R, threshold, window_size, margin)
        candidates.extend(
            (octave, int(x), int(y), float(R[y, x])) for x, y in pts
        )

    if max_keypoints is not None:
        candidates.sort(key=lambda item: item[3], reverse=True)
        candidates = candidates[:max_keypoints]

    keypoints, descriptors, details = [], [], []
    for octave, levels in enumerate(pyramid):
        local_points = np.array(
            [[x, y] for o, x, y, _ in candidates if o == octave], dtype=float
        ).reshape(-1, 2)
        if not len(local_points):
            continue

        if descriptor_method == "sift":
            pts, descs, angles = ExtractSIFTDescriptorsFromHarris(
                levels[0], local_points, scale_space=pyramid, octave=octave
            )
        else:
            pts, descs, angles = ExtractMOPSDescriptorsFromHarris(levels[0], local_points)

        factor = 2 ** octave
        for (x, y), desc, angle in zip(pts, descs, angles):
            x_original, y_original = float(x * factor), float(y * factor)
            keypoints.append([x_original, y_original])
            descriptors.append(desc.tolist())
            details.append({
                "x": x_original, "y": y_original,
                "sigma": 1.6 * factor,
                "octave": octave, "scale": 0, "angle": float(angle),
                "response": float(responses[octave][int(round(y)), int(round(x))]),
            })

    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "keypoint_details": details,
        "num_keypoints": len(keypoints),
        "execution_time_ms": round((time.perf_counter() - start) * 1000, 3),
        "detector_method": method,
        "descriptor_method": descriptor_method,
    }


def DrawCorners(image: np.ndarray, keypoints: np.ndarray, circle_radius: int = 5) -> np.ndarray:
    """Draw red circles on an RGB copy suitable for main.py's process_image()."""
    if not 1 <= circle_radius <= 100:
        raise ValueError("circle_radius must be between 1 and 100")
    canvas = np.array(image, copy=True)
    if canvas.dtype == np.uint16:
        canvas = (canvas // 257).astype(np.uint8)
    elif canvas.dtype != np.uint8:
        if np.issubdtype(canvas.dtype, np.floating) and canvas.max() <= 1.0:
            canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
        else:
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    elif canvas.shape[2] == 4:
        canvas = canvas[:, :, :3].copy()
    for x, y in keypoints:
        cv2.circle(canvas, (int(round(x)), int(round(y))), circle_radius,
                   (255, 0, 0), 2, cv2.LINE_AA)
    return canvas
