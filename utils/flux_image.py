"""FLUX-Kontext image helpers for resolution selection and paired transforms."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

from PIL import Image

_FLUX_RESAMPLING = getattr(Image, "Resampling", Image)

# Canonical FLUX-Kontext resolutions (width, height)
FLUX_KONTEXT_RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
)


def _safe_ratio(size: Tuple[int, int]) -> Optional[float]:
    width, height = size
    if width <= 0 or height <= 0:
        return None
    return float(width) / float(height)


def choose_flux_resolution(
    image_sizes: Sequence[Tuple[int, int]],
    *,
    fallback: Tuple[int, int] = (1024, 1024),
) -> Tuple[int, int]:
    """Pick the FLUX-Kontext resolution with the closest aspect ratio to the inputs."""

    ratios = [ratio for size in image_sizes if (ratio := _safe_ratio(size)) is not None]
    if not ratios:
        return fallback

    target_ratio = sum(ratios) / float(len(ratios))

    def _ratio_distance(candidate: Tuple[int, int]) -> float:
        cand_ratio = _safe_ratio(candidate)
        if cand_ratio is None:
            return math.inf
        # Compare in log-space to treat reciprocal ratios symmetrically
        return abs(math.log(cand_ratio) - math.log(target_ratio))

    return min(FLUX_KONTEXT_RESOLUTIONS, key=_ratio_distance)


def resize_with_aspect(
    image: Image.Image,
    target_size: Tuple[int, int],
    *,
    crop_u: float = 0.5,
    crop_v: float = 0.5,
    resample: int = _FLUX_RESAMPLING.BICUBIC,
) -> Image.Image:
    """Resize an image to cover the target size, then crop using normalized offsets."""

    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target size: {target_size}")

    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source size: {image.size}")

    scale = max(target_w / float(src_w), target_h / float(src_h))
    new_w = max(target_w, int(round(src_w * scale)))
    new_h = max(target_h, int(round(src_h * scale)))

    resized = image.resize((new_w, new_h), resample=resample)

    max_dx = max(new_w - target_w, 0)
    max_dy = max(new_h - target_h, 0)

    crop_u = float(max(0.0, min(1.0, crop_u)))
    crop_v = float(max(0.0, min(1.0, crop_v)))

    left = int(round(crop_u * max_dx))
    top = int(round(crop_v * max_dy))
    left = min(left, max_dx)
    top = min(top, max_dy)

    return resized.crop((left, top, left + target_w, top + target_h))


def prepare_flux_images(
    images: Iterable[Optional[Image.Image]],
    target_size: Tuple[int, int],
    *,
    crop_u: float = 0.5,
    crop_v: float = 0.5,
    flip: bool = False,
) -> Tuple[Optional[Image.Image], ...]:
    """Apply consistent resize/crop/flip to a tuple of images, preserving order."""

    processed = []
    for image in images:
        if image is None:
            processed.append(None)
            continue
        transformed = resize_with_aspect(image, target_size, crop_u=crop_u, crop_v=crop_v)
        if flip:
            transformed = transformed.transpose(Image.FLIP_LEFT_RIGHT)
        processed.append(transformed)
    return tuple(processed)


__all__ = [
    "FLUX_KONTEXT_RESOLUTIONS",
    "choose_flux_resolution",
    "resize_with_aspect",
    "prepare_flux_images",
]
