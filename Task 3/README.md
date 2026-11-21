# Image Processing & Shape Detection

An interactive computer vision application for experimenting with **edge detection, geometric shape detection, and contour-based image segmentation**. The project combines an **Angular frontend** for uploading images, adjusting algorithm parameters, and inspecting results with a **Python FastAPI backend** that performs the image-processing operations.

The application covers Canny edge detection, Hough transforms for lines, circles, and ellipses, greedy active contours (snakes), and Freeman 8-direction chain codes.

## Features

| Feature | What it does |
| --- | --- |
| **Canny Edge Detection** | Produces an edge map with configurable Gaussian smoothing, low/high thresholds, and edge propagation. |
| **Hough Line Transform** | Finds straight lines and reports their polar coordinates (`rho`, `theta`); can draw detected lines over the source image. |
| **Hough Circle Transform** | Detects circles across a configurable radius range and reports center coordinates, radius, and vote count. |
| **Hough Ellipse Transform** | Detects ellipses and reports their center, semi-axes, rotation, and supporting votes. |
| **Active Contour (Snake)** | Refines a user-defined initial contour toward object boundaries with a multi-scale greedy snake algorithm. |
| **Freeman Chain Code** | Encodes contour boundaries as 8-direction steps; active-contour iteration output also includes final area and perimeter. |
| **Image Reuse** | Saves the most recently uploaded image as a lossless PNG so different algorithms can run without re-uploading it. |

## Screenshots

The screenshots show the interface and example image-processing results for the different algorithms.

| Active Contour - Example 1 | Active Contour - Example 2 |
| :---: | :---: |
| ![Active contour example 1](Screenshots/ActiveCountour1.png) | ![Active contour example 2](Screenshots/ActiveCountour2.png) |

| Canny Edge Detection | Hough Line Detection |
| :---: | :---: |
| ![Canny edge detection](Screenshots/CannyEdgeDetection.png) | ![Hough lines](Screenshots/Hough%20Lines.png) |

| Hough Circle Detection | Hough Ellipse Detection |
| :---: | :---: |
| ![Hough circles](Screenshots/HoughCircles.png) | ![Hough ellipses](Screenshots/Hough%20Ellipses.png) |

> **Screenshot setup:** Create a `screenshots/` directory at the repository root and copy the six original PNG screenshots into it. The paths above preserve the filenames shown in the supplied folder screenshot, including `ActiveCountour` and the spaces in the Hough filenames. If you store the screenshots elsewhere, update these relative paths.

## Technology Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Frontend | Angular, TypeScript | Image upload, algorithm configuration, and result visualization |
| Backend | Python, FastAPI | HTTP endpoints, validation, image processing, and streamed responses |
| Image processing | OpenCV, NumPy | Image decoding, grayscale conversion, numerical operations, and PNG encoding |
| Optional acceleration | Numba | Faster Hough ellipse detection on suitable inputs |

## How It Works

1. **Upload an image** using the interface or `POST /images/upload`. The backend validates the file and retains the latest image as a PNG.
2. **Select an algorithm** and configure its parameters. Canny produces an edge map; Hough algorithms can optionally run Canny before shape detection.
3. **Run the operation.** The API returns a PNG image, a JSON representation of detected shapes/contours, or a stream of contour iterations, depending on the endpoint.
4. **Inspect the result** in the frontend, or consume the returned data programmatically.

For Hough transforms, the backend can overlay detected shapes on the original image. For active contours, the overlay endpoint draws the **initial contour in blue** and the **final contour in red**.

### Important Parameters

**Canny (`/canny`)**

| Form field | Default | Description |
| --- | --- | --- |
| `sigma` | `1.4` | Gaussian smoothing strength |
| `low_threshold` | `20` | Lower hysteresis threshold |
| `high_threshold` | `50` | Upper hysteresis threshold |
| `kernel_size` | Derived from `sigma` | Optional odd Gaussian kernel size, from 3 to 51 |
| `propagate` | `true` | Extend weak edges through connected chains |
| `manual` | `false` | Use the manual convolution implementation |

**Hough transforms (shared preprocessing)**

| Form field | Default | Description |
| --- | --- | --- |
| `apply_canny` | `true` | Run Canny before Hough detection; set to `false` for an existing edge map |
| `sigma` | `1.4` | Canny Gaussian smoothing strength |
| `low_threshold` | `20` | Canny lower threshold |
| `high_threshold` | `50` | Canny upper threshold |

Additional detection options:

| Algorithm | Key options | Default values |
| --- | --- | --- |
| Lines | `threshold`, `use_vectorized`, `use_nms`, `rho_distance`, `theta_distance` | `100`, `true`, `true`, `10`, `5` |
| Circles | `threshold`, `min_radius`, `max_radius`, `radius_step`, `max_circles` | `100`, `10`, `70`, `1`, `2` |
| Circles | `radii`, `min_center_distance`, `radius_distance`, `batch_size` | optional radius list, `80`, `5`, `1024` |
| Ellipses | `threshold`, `pair_tolerance`, `center_tolerance`, `max_ellipses` | `100`, `2`, `4`, `2` |
| Ellipses | `use_vectorized`, `max_edge_points` | `true`, `400` |

The circle detector accepts `radii` as comma-separated values (for example, `20,25,30`) instead of the min/max/step range. It limits a request to 256 different radius values. Hough vote thresholds must be **strictly exceeded** for a detection to be accepted.

**Active contour**

The contour endpoints (except `/active-contour/edge-map`) require a `contour` form field containing JSON coordinates such as `[[40,40],[200,40],[200,200],[40,200]]`, using image pixel coordinates.

| Form field | Default | Description |
| --- | --- | --- |
| `contour` | Required | Initial contour as `[x, y]` points |
| `num_points` | `200` | Number of sampled points on the snake |
| `alpha` | `1.0` | Continuity weight |
| `beta` | `1.0` | Smoothness weight |
| `gamma` | `2.0` | Image attraction weight |
| `scales` | `12,6,3` | Coarse-to-fine attraction scales |
| `max_iterations` | `200` | Maximum iterations per stage |
| `radius` | `2` | Local search radius in pixels |
| `convergence_threshold` | `0.02` | Convergence threshold |
| `use_corners` | `false` | Enable corner-related behavior |

## Implementation Notes and Limits

- **Upload limits:** The backend rejects files larger than **25 MiB** or images containing more than **16 million pixels**.
- **Latest image:** A new file is stored as a PNG before processing. Requests without a file reuse that saved image, which survives API restarts. The supplied API has one shared latest-image slot, not separate images per user.
- **Save behavior:** Most image-processing requests support `save_as_latest` (default `true`) when a new file is attached.
- **Overlay output:** All rendered output images are encoded and returned as PNG.
- **Ellipse performance:** The three-point ellipse search is computationally expensive. The default `max_edge_points` is `400`. Without Numba acceleration, inputs with more than `120` edge pixels are rejected; consider a smaller or sparser image.
- **CORS:** The provided backend allows all origins, methods, and headers. Restrict these settings before deploying to an environment where cross-origin access needs to be controlled.
- **Validation:** Invalid images and algorithm parameters can produce `400` responses, oversized images produce `413`, and FastAPI also validates form-field ranges.


## SIFT Feature Extraction & Matching (Task 3)

### SIFT Service Architecture (`Backend/Services/SIFTService.py`)

The SIFT service follows the exact pipeline and algorithmic structure implemented in `SIFT_(1).ipynb`:

1. **Keypoint Detection:**
   - `Generate_Scale_Space(image, num_octaves=3, num_intervals=3, sigma=1.6, assumed_blur=0.5)`: Generates octave pyramid with incremental Gaussian blurring.
   - `Calculate_DoG(scale_space)`: Differences of adjacent Gaussian blurred levels across each octave.
   - `Detect_Local_Extrema(dog_space)`: Vectorized 3D local extrema detection across 26 spatial/scale neighbors.
   - `Localize_And_Filter_Keypoints(dog_space, candidates, contrast_threshold=0.03, edge_threshold=10)`: Subpixel quadratic Taylor expansion, low-contrast removal, and principal curvature ratio edge elimination.
   - `Detect_Keypoints(image, return_scale_space=False, ...)`: Complete detection pipeline returning localized keypoints and optional scale space.

2. **Orientation Assignment:**
   - `Calculate_Gradients(gaussian_image)`: Computes gradient magnitude and orientation in `[0, 360)` degrees.
   - `Apply_Gaussian_Weighting(magnitude, orientation, x, y, sigma)`: Applies `1.5 * sigma` Gaussian window weighting to local gradients.
   - `Build_Orientation_Histogram(weighted_magnitude, orientation, num_bins=36)`: Accumulates 36-bin histogram of weighted orientations.
   - `Assign_Dominant_Orientation(histogram, num_bins=36, peak_ratio=0.8)`: 5-tap binomial smoothing, peak detection, and sub-bin parabolic interpolation.
   - `Assign_Orientation_To_Keypoint(kp, scale_space, num_intervals=3)`: Identifies dominant canonical orientations for each keypoint.

3. **Descriptor Neighborhood Preparation:**
   - `Extract_Local_Neighborhood(gaussian_image, keypoint)`: Extracts local window in octave scale (`8 * sigma`).
   - `Rotate_Neighborhood(neighborhood, keypoint_orientation)`: Rotates patch around center to keypoint orientation via bilinear affine warp.
   - `Normalize_Orientation(local_orientations, keypoint_orientation)`: Normalizes local gradient angles relative to canonical keypoint orientation.
   - `Apply_Descriptor_Gaussian_Weighting(gradient_magnitude, neighborhood_sigma=None)`: Center-weighted Gaussian decay on descriptor patch.

4. **Descriptor Histogram & Trilinear Interpolation:**
   - `Divide_Into_Cells(neighborhood, num_cells=4)`: Splits neighborhood patch into 4x4 spatial cells.
   - `Calculate_Local_Gradients(neighborhood)`: Computes local patch gradients.
   - `Build_Local_Orientation_Histograms(cells, ...)`: Cell-wise histograms.
   - `Apply_Trilinear_Interpolation(gradient_magnitude, gradient_orientation, num_cells=4, num_bins=8)`: Distributes samples across spatial (row, col) and angular bins.
   - `Concatenate_Histograms(histograms)`: Flattens 4x4x8 3D tensor into a 128-D descriptor vector.

5. **Descriptor Normalization:**
   - `Normalize_Descriptor(descriptor)`: L2 unit normalization.
   - `Threshold_And_Renormalize(descriptor, threshold=0.2)`: Clamps large gradient spikes to 0.2 and re-normalizes to unit length.

6. **End-to-End Pipeline & Visualization:**
   - `compute_descriptor(kp, scale_space, theta)`: Produces normalized 128-D descriptor for an oriented keypoint.
   - `extract_sift_features(image, ...)`: High-level entry point returning `{ keypoints, descriptors, keypoint_details, num_keypoints, execution_time_ms }`.
   - `draw_sift_keypoints(image, keypoints, circle_radius, draw_orientation)`: Visualizes detected keypoints with scale circles and canonical orientation rays.
   - `circ_diff(a, b)`: Angular difference accounting for 0/360 wrap-around.

### SIFT Endpoints

- **`POST /features/sift`** (aliases: `POST /sift`, `POST /sift/detect`)
  - Form parameters: `file` (optional, falls back to latest image), `max_keypoints` (default 500), `contrast_threshold` (0.03), `edge_threshold` (10.0), `num_octaves` (3), `num_intervals` (3), `sigma` (1.6), `assumed_blur` (0.5), `save_as_latest` (false).
  - Returns JSON with `keypoints` (`(N, 2)` format) and `descriptors` (`(N, 128)` format), directly compatible with `POST /features/match`.

- **`POST /features/sift/visualize`** (alias: `POST /sift/visualize`)
  - Form parameters: same detection parameters + `circle_radius` (optional), `draw_orientation` (boolean, default true).
  - Returns rendered PNG image with keypoint circles and orientation rays, accompanied by `X-Keypoint-Count` and `X-Execution-Time-Ms` response headers.

### Feature Matching Interface

The Feature Matching tab displays Input 1, Input 2, and the match visualization output side by side. Input 1 can be uploaded, loaded from the latest stored image, or supplied from descriptors extracted with **Detect Keypoints** in the SIFT Keypoints tab. Input 2 is uploaded separately; SIFT descriptors are extracted for it before matching. Descriptor JSON is uploaded as file parts so larger SIFT results are not rejected by multipart text-field size limits. The existing `FeatureMatchingService` performs descriptor distance comparison, nearest/second-nearest selection, Lowe's ratio test (default threshold **0.75**), optional mutual nearest-neighbor filtering, and match visualization. Euclidean distance (SSD) is selected by default; normalized correlation remains available as an alternate metric.
