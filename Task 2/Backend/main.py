import json
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from Backend.Services.UtilitiesService import (
    TransformIntoGrayScale,
    NormalizeImage
)
from Backend.Services.ActiveContourService import (
    compute_edge_map,
    iter_snake,
    prepare_contour,
    segment_active_contour,
    draw_contours,
)
from Backend.Services.CannyEdgeDetectionService import (
    canny_edge_detection,
)
from Backend.Services.LatestImageService import (
    save_latest_image,
    load_latest_image,
    load_latest_image_bytes,
)
from Backend.Services.HoughLineTransformService import (
    DetectLines,
    DrawLines,
)
from Backend.Services.HoughCircleTransformService import (
    DetectCircles,
    DrawCircles,
)
from Backend.Services.HoughEllipseTransformService import (
    DetectEllipses,
    DrawEllipses,
    NUMBA_AVAILABLE,
)

from Backend.Services.ChainCodeService import (
    compute_area,
    compute_perimeter,
    contour_to_chain_code,
)



app = FastAPI()

# Allow the frontend (opened from a file or another port) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Edge Detection API is running"}

def process_image(image: np.ndarray) -> StreamingResponse:
    """Encode the processed image as PNG and return it."""

    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    # Ensure valid PNG pixel values and data type if normalized.
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    success, encoded_image = cv2.imencode(".png", image)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode the processed image."
        )

    return StreamingResponse(
        BytesIO(encoded_image.tobytes()),
        media_type="image/png"
    )

# Keep upload sizes manageable. Adjust these for your project's expected images.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


async def read_uploaded_image(
    file: Optional[UploadFile] = None,
    convert_to_grayscale: bool = True,
    save_as_latest: bool = True,
) -> np.ndarray:
    """Read an uploaded image, or reuse the most recently saved image.

    A new upload is validated and saved as a lossless PNG before processing.
    Requests without a file load the saved PNG (which survives API restarts).
    """
    if file is None:
        try:
            image = await run_in_threadpool(load_latest_image)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No image has been uploaded yet. Upload one using "
                    "POST /images/upload or include 'file' in an image request."
                ),
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load the stored image: {exc}",
            ) from exc
    else:
        # Read one byte beyond the limit to reject oversized uploads.
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Uploaded image exceeds the 25 MiB limit.",
            )
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        try:
            image = cv2.imdecode(
                np.frombuffer(contents, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
        except cv2.error:
            image = None

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid or unsupported image file.",
            )

    if image.ndim not in (2, 3) or (
        image.ndim == 3 and image.shape[2] not in (3, 4)
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported image channel count.",
        )

    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 16-megapixel limit.",
        )

    # Only a newly supplied image changes the saved file. The original
    # OpenCV BGR/BGRA pixel data is stored before any color conversion.
    if file is not None and save_as_latest:
        try:
            await run_in_threadpool(save_latest_image, image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (OSError, cv2.error) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not save the uploaded image: {exc}",
            ) from exc

    # Keep the previous API's RGB convention for downstream services.
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    if convert_to_grayscale and image.ndim == 3:
        image = TransformIntoGrayScale(image, mode="weighted")

    return image


# ----------------------------------------------------------------------------
# Latest uploaded image
# ----------------------------------------------------------------------------

@app.post("/images/upload")
async def upload_image(file: UploadFile = File(...)):
    """Replace the single saved image; return its dimensions."""
    image = await read_uploaded_image(file, convert_to_grayscale=False)
    return {
        "message": "Latest image saved successfully.",
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": int(image.shape[2]) if image.ndim == 3 else 1,
    }


@app.get("/images/latest")
def get_latest_image():
    """Return the saved image as PNG (no browser cache)."""
    try:
        image_data = load_latest_image_bytes()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail="No image has been uploaded yet.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load the stored image: {exc}",
        ) from exc

    return Response(
        content=image_data,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# ----------------------------------------------------------------------------
# Canny edge detection
# ----------------------------------------------------------------------------

@app.post("/canny")
async def canny(
    file: Optional[UploadFile] = File(None),
    sigma: float = Form(1.4, gt=0, description="Gaussian blur strength"),
    low_threshold: float = Form(20, ge=0, le=255, description="Low hysteresis threshold (0-255 gradient scale)"),
    high_threshold: float = Form(50, ge=0, le=255, description="High hysteresis threshold (0-255 gradient scale)"),
    kernel_size: Optional[int] = Form(None, ge=3, le=51, description="Gaussian kernel size (odd). Derived from sigma if omitted"),
    propagate: bool = Form(True, description="Grow weak edges along connected chains (classic Canny)"),
    manual: bool = Form(False, description="Use the manual convolution implementation"),
    save_as_latest: bool = Form(True, description="Replace stored image when a file is supplied"),
):
    """Run Canny edge detection and return the edge map (PNG)."""
    image = await read_uploaded_image(file, convert_to_grayscale=True, save_as_latest=save_as_latest)

    try:
        edges = await run_in_threadpool(
            canny_edge_detection,
            image,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            kernel_size=kernel_size,
            propagate=propagate,
            manual=manual,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return process_image(edges)


# ----------------------------------------------------------------------------
# Hough transforms: shared preprocessing
# ----------------------------------------------------------------------------

def hough_edge_params(
    apply_canny: bool = Form(True, description="Apply Canny first (false for an existing edge map)"),
    sigma: float = Form(1.4, gt=0, description="Canny Gaussian blur strength"),
    low_threshold: float = Form(20, ge=0, le=255),
    high_threshold: float = Form(50, ge=0, le=255),
    save_as_latest: bool = Form(True, description="Replace stored image when a file is supplied"),
) -> dict:
    if low_threshold > high_threshold:
        raise HTTPException(status_code=400, detail="low_threshold must not exceed high_threshold.")
    return dict(
        apply_canny=apply_canny,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        save_as_latest=save_as_latest,
    )


async def prepare_hough_image(file: UploadFile, params: dict):
    """Return the original RGB/grayscale image and 2D edge image."""
    image = await read_uploaded_image(
        file, convert_to_grayscale=False, save_as_latest=params["save_as_latest"]
    )
    gray = TransformIntoGrayScale(image, mode="weighted") if image.ndim == 3 else image

    if not params["apply_canny"]:
        return image, gray  # Consistent with the notebook: all nonzero pixels vote.

    try:
        edges = await run_in_threadpool(
            canny_edge_detection,
            gray,
            sigma=params["sigma"],
            low_threshold=params["low_threshold"],
            high_threshold=params["high_threshold"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return image, edges


# ----------------------------------------------------------------------------
# Hough line transform
# ----------------------------------------------------------------------------

def hough_line_params(
    threshold: int = Form(100, ge=0, description="Minimum votes (strictly greater)"),
    use_vectorized: bool = Form(True, description="Use fast NumPy implementation"),
    use_nms: bool = Form(True),
    rho_distance: int = Form(10, ge=0),
    theta_distance: int = Form(5, ge=0),
    edge_params: dict = Depends(hough_edge_params),
) -> dict:
    return {
        **edge_params,
        "threshold": threshold,
        "use_vectorized": use_vectorized,
        "use_nms": use_nms,
        "rho_distance": rho_distance,
        "theta_distance": theta_distance,
    }


async def run_hough_lines(file: UploadFile, params: dict):
    """Return (original image, [rho, theta] array, accumulator)."""
    image, edges = await prepare_hough_image(file, params)
    try:
        lines, accumulator = await run_in_threadpool(
            DetectLines,
            edges,
            threshold=params["threshold"],
            use_vectorized=params["use_vectorized"],
            use_nms=params["use_nms"],
            rho_distance=params["rho_distance"],
            theta_distance=params["theta_distance"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return image, lines, accumulator    


@app.post("/hough-lines")
async def hough_lines(file: Optional[UploadFile] = File(None), params: dict = Depends(hough_line_params)):
    """Return detected lines in JSON (rho, theta radians and theta degrees)."""
    _, lines, accumulator = await run_hough_lines(file, params)
    return {
        "count": len(lines),
        "lines": [
            {"rho": float(rho), "theta": float(theta), "theta_degrees": float(np.degrees(theta))}
            for rho, theta in lines
        ],
        "accumulator_shape": list(accumulator.shape),
    }


@app.post("/hough-lines/overlay")
async def hough_lines_overlay(file: Optional[UploadFile] = File(None), params: dict = Depends(hough_line_params)):
    """Return the uploaded image with detected lines drawn in red (PNG)."""
    image, lines, _ = await run_hough_lines(file, params)
    overlay = await run_in_threadpool(DrawLines, image, lines)
    return process_image(overlay)


# ----------------------------------------------------------------------------
# Hough circle transform
# ----------------------------------------------------------------------------

def hough_circle_params(
    threshold: int = Form(100, ge=0, description="Minimum circle votes (strictly greater)"),
    min_radius: int = Form(10, ge=1, description="Minimum circle radius in pixels"),
    max_radius: int = Form(70, ge=1, description="Maximum circle radius in pixels (inclusive)"),
    radius_step: int = Form(1, ge=1, description="Step size between searched radii"),
    radii: Optional[str] = Form(None, description="Optional comma-separated radius list; overrides min/max/step"),
    max_circles: int = Form(2, ge=1, le=100),
    min_center_distance: int = Form(80, ge=0, description="Suppression distance in center pixels"),
    radius_distance: int = Form(5, ge=0, description="Suppression distance in radius bin indices"),
    use_vectorized: bool = Form(True),
    batch_size: int = Form(1024, ge=1, le=8192),
    edge_params: dict = Depends(hough_edge_params),
) -> dict:
    if radii is not None:
        try:
            tokens = [piece.strip() for piece in radii.split(",")]
            if not tokens or any(not piece for piece in tokens):
                raise ValueError()
            selected_radii = np.asarray([float(piece) for piece in tokens], dtype=float)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="radii must be comma-separated numbers, e.g. '20,25,30'.") from exc
        if (
            np.any(~np.isfinite(selected_radii))
            or np.any(selected_radii <= 0)
            or len(np.unique(selected_radii)) != len(selected_radii)
        ):
            raise HTTPException(status_code=400, detail="radii must contain distinct positive finite values.")
        selected_radii.sort()
    else:
        if max_radius < min_radius:
            raise HTTPException(status_code=400, detail="max_radius must be >= min_radius.")
        requested_count = 1 + (max_radius - min_radius) // radius_step
        if requested_count > 256:
            raise HTTPException(status_code=400, detail="Limit searches to 256 different radius values.")
        selected_radii = np.arange(min_radius, max_radius + 1, radius_step, dtype=float)

    if len(selected_radii) > 256:
        raise HTTPException(status_code=400, detail="Limit searches to 256 different radius values.")
    return {
        **edge_params,
        "threshold": threshold,
        "radii": selected_radii,
        "max_circles": max_circles,
        "min_center_distance": min_center_distance,
        "radius_distance": radius_distance,
        "use_vectorized": use_vectorized,
        "batch_size": batch_size,
    }


async def run_hough_circles(file: UploadFile, params: dict):
    """Return (original image, [radius, y, x] array, accumulator)."""
    image, edges = await prepare_hough_image(file, params)
    try:
        circles, accumulator = await run_in_threadpool(
            DetectCircles,
            edges,
            params["radii"],
            threshold=params["threshold"],
            max_circles=params["max_circles"],
            min_center_distance=params["min_center_distance"],
            radius_distance=params["radius_distance"],
            use_vectorized=params["use_vectorized"],
            batch_size=params["batch_size"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return image, circles, accumulator


@app.post("/hough-circles")
async def hough_circles(file: Optional[UploadFile] = File(None), params: dict = Depends(hough_circle_params)):
    """Return detected circle centers, radii and vote counts as JSON."""
    _, circles, accumulator = await run_hough_circles(file, params)
    items = []
    for radius, center_y, center_x in circles:
        r_index = int(np.flatnonzero(params["radii"] == radius)[0])
        items.append({
            "radius": float(radius),
            "center_x": int(center_x),
            "center_y": int(center_y),
            "votes": int(accumulator[r_index, int(center_y), int(center_x)]),
        })
    return {
        "count": len(items),
        "circles": items,
        "accumulator_shape": list(accumulator.shape),
    }


@app.post("/hough-circles/overlay")
async def hough_circles_overlay(file: Optional[UploadFile] = File(None), params: dict = Depends(hough_circle_params)):
    """Return the uploaded image with detected circles drawn in red (PNG)."""
    image, circles, _ = await run_hough_circles(file, params)
    overlay = await run_in_threadpool(DrawCircles, image, circles)
    return process_image(overlay)


# ----------------------------------------------------------------------------
# Hough ellipse transform
# ----------------------------------------------------------------------------

def hough_ellipse_params(
    threshold: int = Form(100, ge=0, description="Minimum supporting points (strictly greater)"),
    pair_tolerance: float = Form(2, ge=0, description="Minimum major-axis endpoint separation"),
    center_tolerance: float = Form(4, ge=0, description="Minimum distance of third point from center"),
    max_ellipses: int = Form(2, ge=1, le=20),
    use_vectorized: bool = Form(True, description="Use Numba-accelerated three-point search"),
    max_edge_points: int = Form(400, ge=3, le=20000, description="Reject denser edge maps to avoid cubic-time search"),
    edge_params: dict = Depends(hough_edge_params),
) -> dict:
    return {
        **edge_params,
        "threshold": threshold,
        "pair_tolerance": pair_tolerance,
        "center_tolerance": center_tolerance,
        "max_ellipses": max_ellipses,
        "use_vectorized": use_vectorized,
        "max_edge_points": max_edge_points,
    }


async def run_hough_ellipses(file: UploadFile, params: dict):
    """Return (original image, list of ellipse dictionaries, edge pixel count)."""
    image, edges = await prepare_hough_image(file, params)
    edge_points = int(np.count_nonzero(edges))
    if edge_points > params["max_edge_points"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Ellipse detection received {edge_points} edge pixels, exceeding "
                f"max_edge_points={params['max_edge_points']}. Resize or thin the edge map, "
                "or increase max_edge_points (runtime grows cubically)."
            ),
        )

    # The scalar implementation is cubic and becomes impractical on dense
    # maps. The JIT version has the same complexity but a much smaller constant.
    if edge_points > 120 and (not params["use_vectorized"] or not NUMBA_AVAILABLE):
        raise HTTPException(
            status_code=400,
            detail=(
                "Ellipse detection without Numba acceleration is limited to 120 "
                "edge pixels. Use use_vectorized=true with numba installed "
                "or provide a sparser edge image."
            ),
        )

    try:
        ellipses = await run_in_threadpool(
            DetectEllipses,
            edges,
            pair_tolerance=params["pair_tolerance"],
            center_tolerance=params["center_tolerance"],
            threshold=params["threshold"],
            use_vectorized=params["use_vectorized"],
            max_ellipses=params["max_ellipses"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return image, ellipses, edge_points


@app.post("/hough-ellipses")
async def hough_ellipses(file: Optional[UploadFile] = File(None), params: dict = Depends(hough_ellipse_params)):
    """Return ellipse center, semi-axes, orientation and vote count as JSON."""
    _, ellipses, edge_points = await run_hough_ellipses(file, params)
    return {
        "count": len(ellipses),
        "edge_points": edge_points,
        "ellipses": [
            {
                "center_x": float(e["center_x"]),
                "center_y": float(e["center_y"]),
                "a": float(e["a"]),
                "b": int(e["b"]),
                "alpha": float(e["alpha"]),
                "alpha_degrees": float(np.degrees(e["alpha"])),
                "votes": int(e["count"]),
            }
            for e in ellipses
        ],
    }


@app.post("/hough-ellipses/overlay")
async def hough_ellipses_overlay(file: Optional[UploadFile] = File(None), params: dict = Depends(hough_ellipse_params)):
    """Return the uploaded image with detected ellipses drawn in red (PNG)."""
    image, ellipses, _ = await run_hough_ellipses(file, params)
    overlay = await run_in_threadpool(DrawEllipses, image, ellipses)
    return process_image(overlay)


# ----------------------------------------------------------------------------
# Active contour (greedy snake)
# ----------------------------------------------------------------------------

def snake_params(
    contour: str = Form(..., description='Initial contour drawn by the user, JSON: [[x, y], [x, y], ...] in image pixels'),
    num_points: int = Form(200, ge=3, le=2000),
    alpha: float = Form(1.0, ge=0),
    beta: float = Form(1.0, ge=0),
    gamma: float = Form(2.0, ge=0),
    scales: str = Form("12,6,3", description="Comma-separated attraction sigmas, coarse to fine"),
    max_iterations: int = Form(200, ge=1, le=2000),
    radius: int = Form(2, ge=1, le=5),
    convergence_threshold: float = Form(0.02, ge=0, le=1),
    use_corners: bool = Form(False),
) -> dict:
    """Parse and validate the form fields shared by the snake endpoints."""
    try:
        points = json.loads(contour)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="'contour' must be valid JSON: [[x, y], ...]")

    try:
        parsed_scales = tuple(float(s) for s in scales.split(",") if s.strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="'scales' must be comma-separated numbers, e.g. '12,6,3'")
    if not parsed_scales:
        raise HTTPException(status_code=400, detail="'scales' must contain at least one value.")

    return dict(
        points=points,
        num_points=num_points,
        scales=parsed_scales,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_iterations=max_iterations,
        radius=radius,
        convergence_threshold=convergence_threshold,
        use_corners=use_corners,
    )


async def run_active_contour(file: UploadFile, params: dict, save_as_latest: bool = True):
    """Read the image, run the snake in a worker thread. Returns (image, initial, final)."""
    image = await read_uploaded_image(file, convert_to_grayscale=False, save_as_latest=save_as_latest)

    try:
        # The snake is CPU-bound pure Python, keep it off the event loop.
        initial, final = await run_in_threadpool(segment_active_contour, image, **params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return image, initial, final


@app.post("/active-contour/edge-map")
async def active_contour_edge_map(
    file: Optional[UploadFile] = File(None),
    save_as_latest: bool = Form(True),
):
    """Step 1: the edge map the snake is attracted to (PNG)."""
    image = await read_uploaded_image(file, convert_to_grayscale=False, save_as_latest=save_as_latest)
    edge_map = await run_in_threadpool(compute_edge_map, image)
    return process_image((edge_map * 255).astype(np.uint8))


@app.post("/active-contour")
async def active_contour(
    file: Optional[UploadFile] = File(None),
    params: dict = Depends(snake_params),
    save_as_latest: bool = Form(True),
):
    """Run the snake and return the initial and final contour points as JSON."""
    _, initial, final = await run_active_contour(file, params, save_as_latest)
    return {
        "initial_contour": initial.tolist(),
        "final_contour": final.tolist(),
    }


@app.post("/active-contour/iterations")
async def active_contour_iterations(
    file: Optional[UploadFile] = File(None),
    params: dict = Depends(snake_params),
    save_as_latest: bool = Form(True),
):
    """Stream each snake contour and its Freeman chain code as NDJSON."""
    image = await read_uploaded_image(file, convert_to_grayscale=False, save_as_latest=save_as_latest)
    try:
        initial = await run_in_threadpool(
            prepare_contour, params["points"], image.shape, params["num_points"]
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    edge_map = await run_in_threadpool(compute_edge_map, image)

    def frames():
        contour = initial.copy()
        iteration = 0

        try:
            start_pixel, chain_code = contour_to_chain_code(contour)
        except (ValueError, IndexError, TypeError) as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
            return

        yield json.dumps({
            "type": "frame",
            "iteration": 0,
            "stage": 0,
            "stage_iteration": 0,
            "scale": None,
            "contour": contour.tolist(),
            "start_pixel": list(start_pixel),
            "chain_code": chain_code,
            "moved_points": 0,
            "converged": False,
        }) + "\n"

        snake_options = {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "gamma": params["gamma"],
            "max_iterations": params["max_iterations"],
            "radius": params["radius"],
            "convergence_threshold": params["convergence_threshold"],
            "use_corners": params["use_corners"],
            "edge_map": edge_map,
        }
        all_scales_converged = True

        for stage, scale in enumerate(params["scales"], start=1):
            stage_converged = False
            try:
                for contour, stage_iteration, moved_points, converged in iter_snake(
                    contour,
                    image,
                    attraction_sigma=scale,
                    **snake_options,
                ):
                    iteration += 1
                    stage_converged = converged
                    start_pixel, chain_code = contour_to_chain_code(contour)
                    yield json.dumps({
                        "type": "frame",
                        "iteration": iteration,
                        "stage": stage,
                        "stage_iteration": stage_iteration,
                        "scale": scale,
                        "contour": contour.tolist(),
                        "start_pixel": list(start_pixel),
                        "chain_code": chain_code,
                        "moved_points": moved_points,
                        "converged": converged,
                    }) + "\n"
            except (ValueError, IndexError, TypeError) as e:
                yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
                return
            all_scales_converged = all_scales_converged and stage_converged

        try:
            start_pixel, final_chain_code = contour_to_chain_code(contour)
            final_area = compute_area(final_chain_code, start_pixel)
            final_perimeter = compute_perimeter(final_chain_code)
        except (ValueError, IndexError, TypeError) as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
            return

        yield json.dumps({
            "type": "done",
            "iteration": iteration,
            "final_contour": contour.tolist(),
            "converged": all_scales_converged,
            "area": final_area,
            "perimeter": final_perimeter,
        }) + "\n"

    return StreamingResponse(
        frames(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/active-contour/overlay")
async def active_contour_overlay(
    file: Optional[UploadFile] = File(None),
    params: dict = Depends(snake_params),
    save_as_latest: bool = Form(True),
):
    """Run the snake and return the image with initial (blue) and final (red) contours drawn on it (PNG)."""
    image, initial, final = await run_active_contour(file, params, save_as_latest)
    return process_image(draw_contours(image, initial, final))

# ----------------------------------------------------------------------------
# Freeman 8-direction Chain Code
# ----------------------------------------------------------------------------

@app.post("/chain-code")
async def generate_chain_code(
    contour: str = Form(
        ...,
        description="Contour points in JSON format: [[x, y], [x, y], ...]"
    )
):
    """
    Convert contour points into an 8-direction Freeman chain code.

    Returns:
        start_pixel: Starting pixel of the contour.
        chain_code: List of Freeman direction codes.
    """

    try:
        points = json.loads(contour)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="'contour' must be valid JSON: [[x, y], ...]"
        )

    # Validate contour format
    if not isinstance(points, list) or len(points) < 2:
        raise HTTPException(
            status_code=400,
            detail="Contour must contain at least two points."
        )

    try:
        start_pixel, chain_code = contour_to_chain_code(points)

    except (ValueError, IndexError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    return {
        "start_pixel": list(start_pixel),
        "chain_code": chain_code,
        "chain_code_length": len(chain_code)
    }