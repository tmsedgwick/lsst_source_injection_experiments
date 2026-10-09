"""Peak-finding detection, after LSST's SourceDetectionTask: S/N of the PSF-smoothed image, with the noise from the
variance plane; footprints are the connected regions above threshold, grown by grow_sigmas PSF widths; peaks are the
local maxima of the S/N map above threshold. With several detection bands their smoothed images are combined with
inverse-variance weights."""

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, gaussian_filter, label, maximum_filter

from .config import BANDS


def psf_sigma_pix(kernel):
    """Gaussian-equivalent width (sigma, pixels) of a PSF stamp, from its second moment."""
    kernel = np.clip(np.nan_to_num(np.asarray(kernel, float)), 0.0, None)
    half = kernel.shape[0] // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    total = kernel.sum()
    cx, cy = (kernel * xx).sum() / total, (kernel * yy).sum() / total
    return float(np.sqrt(max((kernel * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / total / 2.0, 1e-12)))


def significance(signal, variance, psf_kernels, detect_bands):
    """S/N map of the PSF-smoothed image in the detection bands, and the smoothing width (pixels) of the first."""
    weighted, inverse_variance, sigma0 = 0.0, 0.0, None
    for band in detect_bands:
        k = BANDS.index(band)
        sigma = psf_sigma_pix(psf_kernels[..., k])
        sigma0 = sigma if sigma0 is None else sigma0
        smoothed = gaussian_filter(np.nan_to_num(signal[k]).astype(float), sigma, mode="constant", truncate=4.0)
        # variance of the smoothed image: the variance convolved with the kernel squared, a Gaussian of width
        # sigma / sqrt(2) and integral 1 / (4 pi sigma^2)
        good_variance = np.where(np.isfinite(variance[k]) & (variance[k] > 0), variance[k], np.inf)
        smoothed_variance = gaussian_filter(np.where(np.isinf(good_variance), 0.0, good_variance).astype(float),
                                            sigma / np.sqrt(2.0), mode="nearest", truncate=4.0) / (4 * np.pi * sigma ** 2)
        smoothed_variance = np.where(smoothed_variance > 0, smoothed_variance, np.inf)
        weighted = weighted + smoothed / smoothed_variance
        inverse_variance = inverse_variance + 1.0 / smoothed_variance
    with np.errstate(invalid="ignore", divide="ignore"):
        sn = np.where(inverse_variance > 0, weighted / np.sqrt(inverse_variance), 0.0)
    return sn, sigma0


def detect(signal, variance, psf_kernels, cfg):
    """(peaks, footprint label map): peaks has x, y (pixels), sn and footprint (label id, from 1)."""
    sn, sigma = significance(signal, variance, psf_kernels, cfg["detect_bands"])
    above = sn >= cfg["detection_sn"]
    radius = max(0, int(np.round(cfg["grow_sigmas"] * sigma)))
    if radius:
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        grown = binary_dilation(above, structure=(xx ** 2 + yy ** 2) <= radius ** 2)
    else:
        grown = above
    labels = np.zeros(grown.shape, np.int32)
    label(grown, structure=np.ones((3, 3), int), output=labels)
    is_peak = above & (sn == maximum_filter(sn, size=3, mode="constant", cval=-np.inf))
    y, x = np.nonzero(is_peak)
    peaks = pd.DataFrame(dict(x=x.astype(float), y=y.astype(float), sn=sn[y, x], footprint=labels[y, x]))
    return peaks.sort_values("sn", ascending=False).reset_index(drop=True), labels


def footprint_bbox(labels, footprint):
    """(x0, y0, x1, y1) bounds of a footprint (x1, y1 exclusive)."""
    y, x = np.nonzero(labels == footprint)
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1
