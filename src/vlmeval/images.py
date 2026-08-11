"""Unified image preparation — identical bytes go to every model (fairness)."""

from __future__ import annotations

import io

from PIL import Image, ImageOps


def prepare_image(img: Image.Image, max_side: int = 1280, quality: int = 90) -> bytes:
    """EXIF-transpose, RGB-convert, LANCZOS-downscale to max_side, JPEG-encode."""
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def image_dims(jpeg: bytes) -> tuple[int, int]:
    """(width, height) of an encoded image without full decode."""
    with Image.open(io.BytesIO(jpeg)) as im:
        return im.size
