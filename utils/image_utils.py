"""
Image utility helpers — preprocessing before OCR.
"""

from __future__ import annotations
import base64
import io
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance, ImageFilter


def file_to_base64(path: str | Path) -> str:
    """Read an image file and return its base64 encoding."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def bytes_to_base64(data: bytes) -> str:
    """Convert raw bytes to base64 string."""
    return base64.b64encode(data).decode("utf-8")


def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Light preprocessing to improve OCR accuracy on crumpled/low-quality scans:
    - Convert to grayscale
    - Sharpen edges
    - Enhance contrast
    Returns processed image as JPEG bytes.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(2.0)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def detect_mime_type(filename: str) -> str:
    """Detect MIME type from file extension."""
    ext = Path(filename).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/jpeg")
