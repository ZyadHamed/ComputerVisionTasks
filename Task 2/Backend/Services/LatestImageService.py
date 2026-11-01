
"""Persistent, single-image storage shared by all image-processing endpoints.

The decoded image is stored as a lossless PNG. Atomic file replacement avoids
serving partially written images when requests arrive at the same time.

Set LATEST_IMAGE_PATH to store the image somewhere outside the source tree,
for example on a persistent disk/volume in a deployed environment.
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE_PATH = Path(__file__).resolve().parents[1] / "storage" / "latest_image.png"


def _image_path() -> Path:
    """Read the environment on each call so tests/config changes can override it."""
    return Path(os.environ.get("LATEST_IMAGE_PATH", str(DEFAULT_IMAGE_PATH)))


def save_latest_image(image: np.ndarray) -> None:
    """Save a validated OpenCV image without lossy recompression.

    Image channels use the native OpenCV order (BGR/BGRA). The caller is
    responsible for decoding and validating uploaded image data first.
    """
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("Expected a grayscale, BGR, or BGRA image.")

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Unable to encode the uploaded image as PNG.")

    path = _image_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=".latest_image_",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())

        # Atomic on the same filesystem: existing readers see old or new bytes.
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_latest_image_bytes() -> bytes:
    """Return PNG bytes or raise FileNotFoundError if no upload exists."""
    return _image_path().read_bytes()


def load_latest_image() -> np.ndarray:
    """Return the latest image in OpenCV format; never return a shared array."""
    data = load_latest_image_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError("Stored latest image is unreadable or corrupted.")
    return image
