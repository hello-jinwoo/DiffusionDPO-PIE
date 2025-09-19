"""Utility helpers for validation metrics on preference pairs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Optional scikit-image dependency for advanced metrics
try:
    from skimage.color import rgb2gray as _sk_rgb2gray
    from skimage.color import rgb2lab as _sk_rgb2lab
    from skimage.color import deltaE_ciede2000 as _sk_delta_e
    from skimage.metrics import niqe as _sk_niqe
    from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
    from skimage.metrics import structural_similarity as _sk_ssim

    _SKIMAGE_AVAILABLE = True
except Exception:  # pragma: no cover - dependency optional
    _SKIMAGE_AVAILABLE = False
    _sk_rgb2gray = None
    _sk_rgb2lab = None
    _sk_delta_e = None
    _sk_niqe = None
    _sk_psnr = None
    _sk_ssim = None


@dataclass
class PairwiseMetrics:
    """Container for similarity metrics against preferred and non-preferred references."""

    metric_name: str
    to_preferred: Optional[float]
    to_non_preferred: Optional[float]

    @property
    def margin(self) -> Optional[float]:
        """Return the margin indicating relative preference when meaningful."""
        if self.to_preferred is None or self.to_non_preferred is None:
            return None
        if any(
            isinstance(val, float) and (math.isinf(val) or math.isnan(val))
            for val in (self.to_preferred, self.to_non_preferred)
        ):
            return None
        if self.metric_name in {"psnr", "ssim"}:  # higher is better similarity
            return self.to_preferred - self.to_non_preferred
        if self.metric_name in {"delta_e", "mse"}:  # lower is better distance
            return self.to_non_preferred - self.to_preferred
        return None


def _to_numpy(image: Image.Image | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert supported image types to float32 numpy array in [0, 1] with shape (H, W, C)."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float32)
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, -1)
    elif torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        arr = tensor.numpy().astype(np.float32)
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.clip(arr, 0.0, 1.0)


def _center_crop(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop an array to the requested spatial size."""
    h, w = arr.shape[:2]
    if h == target_h and w == target_w:
        return arr
    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    return arr[top : top + target_h, left : left + target_w, :]


def _ensure_torch(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    if _SKIMAGE_AVAILABLE and _sk_psnr is not None:
        return float(_sk_psnr(a, b, data_range=1.0))
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: torch.device | None = None) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    if _SKIMAGE_AVAILABLE and _sk_ssim is not None:
        return float(_sk_ssim(a, b, channel_axis=-1, data_range=1.0))
    im1 = _ensure_torch(a)
    im2 = _ensure_torch(b)
    device = im1.device
    kernel = _gaussian_kernel(device=device)
    channel = im1.shape[1]
    kernel = kernel.expand(channel, 1, -1, -1).to(im1.dtype)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.conv2d(im1, kernel, padding=5, groups=channel)
    mu2 = F.conv2d(im2, kernel, padding=5, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(im1 * im1, kernel, padding=5, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(im2 * im2, kernel, padding=5, groups=channel) - mu2_sq
    sigma12 = F.conv2d(im1 * im2, kernel, padding=5, groups=channel) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    return float(ssim_map.mean().item())


def _rgb_to_lab(arr: np.ndarray) -> np.ndarray:
    if _SKIMAGE_AVAILABLE and _sk_rgb2lab is not None:
        return _sk_rgb2lab(arr)

    # Manual RGB -> Lab conversion (D65 reference)
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    rgb_to_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ rgb_to_xyz.T
    xyz_ref = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = xyz / xyz_ref

    def _f(t: np.ndarray) -> np.ndarray:
        delta = 6 / 29
        return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)

    f_xyz = _f(xyz)
    L = 116 * f_xyz[..., 1] - 16
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
    return np.stack([L, a, b], axis=-1)


def _delta_e(a: np.ndarray, b: np.ndarray) -> float:
    lab1 = _rgb_to_lab(a)
    lab2 = _rgb_to_lab(b)
    if _SKIMAGE_AVAILABLE and _sk_delta_e is not None:
        delta = _sk_delta_e(lab1, lab2)
    else:
        delta = np.linalg.norm(lab1 - lab2, axis=-1)
    return float(np.mean(delta))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def compute_pairwise_metrics(
    output_image: Image.Image | np.ndarray | torch.Tensor,
    preferred_image: Image.Image | np.ndarray | torch.Tensor,
    non_preferred_image: Image.Image | np.ndarray | torch.Tensor,
) -> List[PairwiseMetrics]:
    """Compute similarity metrics for a generated image relative to preference references."""
    out_arr = _to_numpy(output_image)
    pref_arr = _to_numpy(preferred_image)
    nonpref_arr = _to_numpy(non_preferred_image)

    # Align spatial shapes via center crop to the smallest height/width among the inputs
    min_h = min(out_arr.shape[0], pref_arr.shape[0], nonpref_arr.shape[0])
    min_w = min(out_arr.shape[1], pref_arr.shape[1], nonpref_arr.shape[1])
    out_arr = _center_crop(out_arr, min_h, min_w)
    pref_arr = _center_crop(pref_arr, min_h, min_w)
    nonpref_arr = _center_crop(nonpref_arr, min_h, min_w)

    metrics: List[PairwiseMetrics] = []
    psnr_pref = _psnr(out_arr, pref_arr)
    psnr_non = _psnr(out_arr, nonpref_arr)
    metrics.append(PairwiseMetrics("psnr", psnr_pref, psnr_non))

    ssim_pref = _ssim(out_arr, pref_arr)
    ssim_non = _ssim(out_arr, nonpref_arr)
    metrics.append(PairwiseMetrics("ssim", ssim_pref, ssim_non))

    delta_pref = _delta_e(out_arr, pref_arr)
    delta_non = _delta_e(out_arr, nonpref_arr)
    metrics.append(PairwiseMetrics("delta_e", delta_pref, delta_non))

    metrics.append(PairwiseMetrics("mse", _mse(out_arr, pref_arr), _mse(out_arr, nonpref_arr)))

    return metrics


def compute_niqe(image: Image.Image | np.ndarray | torch.Tensor) -> Optional[float]:
    """Compute the NIQE no-reference quality score if scikit-image is available."""
    arr = _to_numpy(image)
    if _SKIMAGE_AVAILABLE and _sk_niqe is not None:
        try:
            gray = _sk_rgb2gray(arr) if _sk_rgb2gray is not None else np.dot(arr, [0.2989, 0.5870, 0.1140])
            return float(_sk_niqe(gray))
        except Exception:  # pragma: no cover - NIQE may fail on uniform images
            return None
    # Basic fallback: variance of Laplacian as a proxy when NIQE unavailable
    gray = np.dot(arr, [0.2989, 0.5870, 0.1140]).astype(np.float32)
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)
    laplacian = F.conv2d(tensor, kernel, padding=1)
    return float(laplacian.var().item())


def aggregate_metrics(collections: Sequence[Mapping[str, Optional[float]]]) -> Dict[str, float]:
    """Aggregate metric dictionaries by computing numeric means for each key."""

    stats: Dict[str, List[float]] = {}

    for entry in collections:
        for key, value in entry.items():
            if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                continue
            numeric = float(value)
            stats.setdefault(key, []).append(numeric)

    aggregated: Dict[str, float] = {}
    for key, values in stats.items():
        if not values:
            continue
        aggregated[f"{key}_mean"] = float(np.mean(values))

    return aggregated


def build_metric_record(
    pair_metrics: Iterable[PairwiseMetrics],
    prefix: Optional[str] = None,
    niqe_score: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Flatten pairwise metrics into a dict keyed by `<prefix>/<metric>` naming."""
    base = f"{prefix}/" if prefix else ""
    record: Dict[str, Optional[float]] = {}
    for metric in pair_metrics:
        record[f"{base}{metric.metric_name}_pref"] = metric.to_preferred
        record[f"{base}{metric.metric_name}_nonpref"] = metric.to_non_preferred
        margin = metric.margin
        if margin is not None:
            record[f"{base}{metric.metric_name}_margin"] = margin
    if niqe_score is not None:
        record[f"{base}niqe"] = niqe_score
    return record


def make_difference_image(
    reference: Image.Image | np.ndarray | torch.Tensor,
    candidate: Image.Image | np.ndarray | torch.Tensor,
    *,
    amplify: float = 4.0,
    max_size: int = 512,
) -> Image.Image:
    """Visualize absolute difference between two images for qualitative logging."""

    ref_arr = _to_numpy(reference)
    cand_arr = _to_numpy(candidate)

    min_h = min(ref_arr.shape[0], cand_arr.shape[0])
    min_w = min(ref_arr.shape[1], cand_arr.shape[1])
    ref_arr = _center_crop(ref_arr, min_h, min_w)
    cand_arr = _center_crop(cand_arr, min_h, min_w)

    diff = np.abs(ref_arr - cand_arr)
    if amplify != 1.0:
        diff = np.clip(diff * amplify, 0.0, 1.0)

    diff_img = Image.fromarray((diff * 255.0).astype(np.uint8))
    if max_size is not None:
        largest_dim = max(diff_img.size)
        if largest_dim > max_size:
            scale = max_size / float(largest_dim)
            new_size = (
                max(1, int(round(diff_img.size[0] * scale))),
                max(1, int(round(diff_img.size[1] * scale))),
            )
            resampling = getattr(Image, "Resampling", Image)
            diff_img = diff_img.resize(new_size, resample=resampling.LANCZOS)

    return diff_img


__all__ = [
    "PairwiseMetrics",
    "compute_pairwise_metrics",
    "compute_niqe",
    "build_metric_record",
    "aggregate_metrics",
    "make_difference_image",
]
