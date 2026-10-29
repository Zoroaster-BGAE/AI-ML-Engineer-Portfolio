"""Photos from the technician. Whatever arrives is decoded and re-encoded here, never stored or forwarded as sent:
that checks it really is an image, bounds its size for the vision model, and drops EXIF (GPS included)."""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_EDGE = 2048          # enough to read a panel display; ~5k vision tokens at most
MAX_PIXELS = 40_000_000  # refuse decompression bombs before decoding them
_ALLOWED = {"JPEG", "PNG", "WEBP", "GIF", "MPO"}


class ImageRejected(ValueError):
    pass


@dataclass
class CleanImage:
    data: bytes
    media_type: str
    width: int
    height: int


def clean(raw: bytes) -> CleanImage:
    if len(raw) > MAX_BYTES:
        raise ImageRejected("photo is larger than 10 MB")
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in _ALLOWED:
                raise ImageRejected(f"unsupported image type {im.format}")
            im = ImageOps.exif_transpose(im)  # phones store rotation in EXIF, which is about to be dropped
            im = im.convert("RGB")
            im.thumbnail((MAX_EDGE, MAX_EDGE))
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=85, optimize=True)
            return CleanImage(out.getvalue(), "image/jpeg", im.width, im.height)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ImageRejected("not a readable image") from exc
