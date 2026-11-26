"""Media-processing functions shared by the Multimedia Studio web interface."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pywt
from PIL import Image
from scipy import fftpack
from scipy.io import wavfile


def decode_image(payload: bytes, grayscale: bool = False) -> np.ndarray:
    """Decode uploaded image bytes into an OpenCV array."""
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), mode)
    if image is None:
        raise ValueError("The uploaded file is not a supported image.")
    return image


def apply_image_operation(
    image: np.ndarray,
    operation: str,
    brightness: float = 0,
    contrast: float = 1,
) -> np.ndarray:
    """Return an edited copy of an image."""
    if operation == "Convert to Grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if operation == "Equalize Histogram":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    if operation == "Gaussian Blur":
        return cv2.GaussianBlur(image, (7, 7), 1.2)
    if operation == "Canny Edge Detection":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if operation == "Sharpen":
        kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            dtype=np.float32,
        )
        return cv2.filter2D(image, -1, kernel)
    if operation == "Resize to 150%":
        height, width = image.shape[:2]
        return cv2.resize(
            image,
            (int(width * 1.5), int(height * 1.5)),
            interpolation=cv2.INTER_CUBIC,
        )
    if operation == "Center Crop":
        height, width = image.shape[:2]
        y_margin = int(height * 0.15)
        x_margin = int(width * 0.15)
        return image[
            y_margin:max(y_margin + 1, height - y_margin),
            x_margin:max(x_margin + 1, width - x_margin),
        ].copy()
    if operation == "Adjust Brightness & Contrast":
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    raise ValueError(f"Unsupported image operation: {operation}")


def apply_transform(image: np.ndarray, transform: str) -> np.ndarray:
    """Apply a frequency-domain or wavelet transform to a grayscale image."""
    if transform == "DFT":
        spectrum = np.fft.fftshift(np.fft.fft2(image))
        return (20 * np.log(np.abs(spectrum) + 1)).astype(np.float32)

    if transform == "DCT":
        values = image.astype(np.float32)
        transformed = fftpack.dct(
            fftpack.dct(values.T, norm="ortho").T,
            norm="ortho",
        )
        display = np.log(np.abs(transformed) + 1)
        return cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if transform == "Wavelet":
        approximation, (horizontal, vertical, diagonal) = pywt.dwt2(image, "haar")
        height, width = approximation.shape

        def resize_band(band: np.ndarray) -> np.ndarray:
            return cv2.resize(
                band.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )

        top = np.concatenate(
            (resize_band(approximation), resize_band(horizontal)),
            axis=1,
        )
        bottom = np.concatenate(
            (resize_band(vertical), resize_band(diagonal)),
            axis=1,
        )
        combined = np.concatenate((top, bottom), axis=0)
        return cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    raise ValueError(f"Unsupported transform: {transform}")


def load_audio(payload: bytes, filename: str) -> tuple[np.ndarray, int]:
    """Read WAV audio, or decode another supported format when FFmpeg is installed."""
    audio_bytes = payload
    if Path(filename).suffix.lower() != ".wav":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise ValueError(
                "This audio format requires FFmpeg. Install FFmpeg or upload a WAV file."
            )
        try:
            result = subprocess.run(
                [ffmpeg, "-v", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"FFmpeg could not decode this audio file. {detail}") from exc
        audio_bytes = result.stdout

    try:
        sample_rate, data = wavfile.read(io.BytesIO(audio_bytes))
    except (ValueError, EOFError) as exc:
        raise ValueError("The audio file could not be decoded as WAV.") from exc

    if data.size == 0:
        raise ValueError("The audio file contains no samples.")
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        bounds = np.iinfo(data.dtype)
        scale = max(abs(bounds.min), abs(bounds.max))
        samples = data.astype(np.float64) / scale
    else:
        samples = data.astype(np.float64)

    samples -= np.mean(samples)
    return samples, int(sample_rate)


def read_first_video_frame(payload: bytes, filename: str) -> np.ndarray:
    """Extract the first frame from an uploaded video."""
    suffix = Path(filename).suffix.lower() or ".mp4"
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(payload)
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        try:
            if not capture.isOpened():
                raise ValueError("The uploaded video could not be opened.")
            success, frame = capture.read()
        finally:
            capture.release()

        if not success or frame is None:
            raise ValueError("No readable frame was found in the video.")
        return frame
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def create_collage(image_payloads: list[bytes]) -> Image.Image:
    """Create a two-column collage from up to six uploaded images."""
    if not image_payloads:
        raise ValueError("Upload at least one image to create a collage.")

    tiles: list[Image.Image] = []
    for payload in image_payloads[:6]:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                tile = image.convert("RGB").resize((400, 300), Image.Resampling.LANCZOS)
        except OSError as exc:
            raise ValueError("One of the uploaded files is not a valid image.") from exc
        tiles.append(tile)

    rows = (len(tiles) + 1) // 2
    collage = Image.new("RGB", (800, rows * 300), "white")
    for index, tile in enumerate(tiles):
        x = (index % 2) * 400
        y = (index // 2) * 300
        collage.paste(tile, (x, y))
    return collage
