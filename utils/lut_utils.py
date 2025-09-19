"""Utilities for loading and applying .cube LUTs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import math

import numpy as np
from PIL import Image

_RED_FASTEST = "red_fastest"
_BLUE_FASTEST = "blue_fastest"


@dataclass
class CubeLUT:
    """In-memory representation of a colour lookup table."""

    title: str
    domain_min: np.ndarray
    domain_max: np.ndarray
    table3d: np.ndarray
    table1d: Optional[np.ndarray] = None

    @property
    def size3d(self) -> int:
        return int(self.table3d.shape[0])

    @property
    def size1d(self) -> Optional[int]:
        return None if self.table1d is None else int(self.table1d.shape[0])


def _parse_cube_file(path: Path, order_hint: str) -> CubeLUT:
    title = ""
    domain_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    size1d: Optional[int] = None
    size3d: Optional[int] = None

    floats: List[List[float]] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    header_keys = ("TITLE", "LUT_1D_SIZE", "LUT_3D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX")

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        upper = stripped.upper()
        if upper.startswith("TITLE"):
            match = re.match(r'TITLE\s+"?(.*?)"?$', stripped, flags=re.IGNORECASE)
            if match:
                title = match.group(1)
        elif upper.startswith("LUT_1D_SIZE"):
            size1d = int(stripped.split()[-1])
        elif upper.startswith("LUT_3D_SIZE"):
            size3d = int(stripped.split()[-1])
        elif upper.startswith("DOMAIN_MIN"):
            parts = stripped.split()
            domain_min = tuple(map(float, parts[-3:]))
        elif upper.startswith("DOMAIN_MAX"):
            parts = stripped.split()
            domain_max = tuple(map(float, parts[-3:]))

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or not re.search(r"[0-9]", stripped):
            continue
        upper = stripped.upper()
        if any(upper.startswith(k) for k in header_keys):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) >= 3:
            try:
                floats.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    data = np.asarray(floats, dtype=np.float32)
    cursor = 0
    table1d = None

    if size1d is not None:
        need = size1d
        if cursor + need > data.shape[0]:
            raise ValueError(f"{path.name}: not enough entries for LUT_1D_SIZE={size1d}.")
        table1d = data[cursor: cursor + need]
        cursor += need

    if size3d is None:
        raise ValueError(f"{path.name}: missing LUT_3D_SIZE header.")

    need = size3d ** 3
    if cursor + need > data.shape[0]:
        raise ValueError(f"{path.name}: not enough entries for LUT_3D_SIZE={size3d}.")
    block = data[cursor: cursor + need]
    cursor += need

    if order_hint == _BLUE_FASTEST:
        table3d = block.reshape((size3d, size3d, size3d, 3)).astype(np.float32)
    elif order_hint == _RED_FASTEST:
        tmp = block.reshape((size3d, size3d, size3d, 3)).astype(np.float32)
        table3d = np.transpose(tmp, (2, 1, 0, 3))
    else:
        raise ValueError("order_hint must be 'blue_fastest' or 'red_fastest'")

    return CubeLUT(
        title=title,
        domain_min=np.asarray(domain_min, dtype=np.float32),
        domain_max=np.asarray(domain_max, dtype=np.float32),
        table3d=table3d,
        table1d=table1d,
    )


def load_cube_lut(path: Path, order_hint: str = _RED_FASTEST) -> CubeLUT:
    return _parse_cube_file(path, order_hint)


def load_cube_luts(directory: Path, order_hint: str = _RED_FASTEST) -> List[CubeLUT]:
    cube_paths = sorted(directory.glob("*.cube")) + sorted(directory.glob("*.CUBE"))
    if not cube_paths:
        raise ValueError(f"No LUT files found in {directory}")
    luts: List[CubeLUT] = []
    for cube_path in cube_paths:
        try:
            luts.append(load_cube_lut(cube_path, order_hint=order_hint))
        except Exception as exc:
            raise ValueError(f"Failed to load LUT {cube_path.name}: {exc}") from exc
    return luts


def _apply_1d(table: Optional[np.ndarray], rgb: np.ndarray, domain_min: np.ndarray, domain_max: np.ndarray) -> np.ndarray:
    if table is None:
        return rgb
    count = table.shape[0]
    scale = (count - 1.0) / np.clip(domain_max - domain_min, 1e-8, None)
    xi = (rgb - domain_min) * scale
    xi = np.clip(xi, 0.0, count - 1.0)

    i0 = np.floor(xi).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, count - 1)
    t = xi - i0

    out = np.empty_like(rgb, dtype=np.float32)
    for channel in range(3):
        v0 = table[i0[..., channel], channel]
        v1 = table[i1[..., channel], channel]
        out[..., channel] = v0 * (1.0 - t[..., channel]) + v1 * t[..., channel]
    return out


def _apply_3d(table: np.ndarray, rgb: np.ndarray, domain_min: np.ndarray, domain_max: np.ndarray) -> np.ndarray:
    size = table.shape[0]
    scale = (size - 1.0) / np.clip(domain_max - domain_min, 1e-8, None)
    xyz = (rgb - domain_min) * scale
    xyz = np.clip(xyz, 0.0, size - 1.0)

    r = xyz[..., 0]
    g = xyz[..., 1]
    b = xyz[..., 2]

    r0 = np.floor(r).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, size - 1)
    g0 = np.floor(g).astype(np.int32)
    g1 = np.clip(g0 + 1, 0, size - 1)
    b0 = np.floor(b).astype(np.int32)
    b1 = np.clip(b0 + 1, 0, size - 1)

    fr = r - r0
    fg = g - g0
    fb = b - b0

    c000 = table[r0, g0, b0]
    c100 = table[r1, g0, b0]
    c010 = table[r0, g1, b0]
    c110 = table[r1, g1, b0]
    c001 = table[r0, g0, b1]
    c101 = table[r1, g0, b1]
    c011 = table[r0, g1, b1]
    c111 = table[r1, g1, b1]

    c00 = c000 * (1 - fr)[..., None] + c100 * fr[..., None]
    c01 = c001 * (1 - fr)[..., None] + c101 * fr[..., None]
    c10 = c010 * (1 - fr)[..., None] + c110 * fr[..., None]
    c11 = c011 * (1 - fr)[..., None] + c111 * fr[..., None]

    c0 = c00 * (1 - fg)[..., None] + c10 * fg[..., None]
    c1 = c01 * (1 - fg)[..., None] + c11 * fg[..., None]

    out = c0 * (1 - fb)[..., None] + c1 * fb[..., None]
    return np.clip(out, 0.0, 1.0)


def apply_lut(image: Image.Image, lut: CubeLUT) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = _apply_1d(lut.table1d, arr, lut.domain_min, lut.domain_max)
    arr = _apply_3d(lut.table3d, arr, lut.domain_min, lut.domain_max)
    out = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


class LUTPicker:
    """Helper that holds a list of LUTs and samples them with its own RNG."""

    def __init__(self, luts: Sequence[CubeLUT], *, seed: int) -> None:
        if not luts:
            raise ValueError("LUTPicker requires a non-empty LUT list")
        self._luts: Sequence[CubeLUT] = luts
        self._state = seed & 0xFFFFFFFF

    def _next_random(self) -> float:
        # simple LCG to avoid importing random repeatedly on workers
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state / 0xFFFFFFFF

    def pick(self) -> CubeLUT:
        idx = int(math.floor(self._next_random() * len(self._luts))) % len(self._luts)
        return self._luts[idx]


__all__ = [
    "CubeLUT",
    "LUTPicker",
    "apply_lut",
    "load_cube_lut",
    "load_cube_luts",
]
