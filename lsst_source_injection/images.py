"""Images the injections go into, saved as .npz so every step runs without the LSST stack.

  * A coadd: signal and variance (band, y, x) in nJy, PSF stamps (stamp, stamp, band), the bands, the tract pixel of
    the array's corner (origin), and optionally a WCS (FITS header text, in the array's own pixel coordinates) and the
    effective number of exposures per band (n_eff, for the injected light's shot noise). This is also the format
    lsst_unet_detection writes, and what scripts/export_from_butler.py writes on the Rubin Science Platform.
  * An exposure (a calibrated single visit, e.g. a calexp / visit_image): the same for one band (2-D signal and
    variance, one PSF stamp, its band). A multi-band file read as exposures gives one exposure per band.

Injection into a coadd adds the source drawn with the coadd's PSF. Injection into exposures adds it to each exposure
with that exposure's PSF at its own pixel position, and recombines the exposures onto the coadd's pixel grid as an
inverse-variance-weighted mean (each exposure weighted by 1 / its median variance), as coaddition does. Exposures map
onto the coadd grid through their WCSs, or, when neither has one, through their origins (the same tract pixel grid).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .config import BANDS


def wcs_from_text(text):
    text = str(text) if text is not None else ""
    return WCS(fits.Header.fromstring(text)) if text.strip() else None


def wcs_to_text(wcs):
    return wcs.to_header(relax=True).tostring() if wcs is not None else ""


@dataclass
class Coadd:
    signal: np.ndarray  # (band, y, x) nJy
    variance: np.ndarray
    psf_kernels: np.ndarray  # (stamp, stamp, band)
    bands: tuple
    origin: tuple  # tract pixel (x, y) of array pixel (0, 0)
    wcs: WCS | None = None
    n_eff: np.ndarray | None = None  # effective number of exposures per band

    @property
    def shape(self):
        return self.signal.shape[1:]


@dataclass
class Exposure:
    signal: np.ndarray  # (y, x) nJy
    variance: np.ndarray
    psf_kernel: np.ndarray  # (stamp, stamp)
    band: str
    origin: tuple
    wcs: WCS | None = None
    name: str = ""


@dataclass
class Patch:
    """Light added to an image: signal and variance (band, h, w) whose pixel (0, 0) is image pixel (x0, y0)."""
    x0: int
    y0: int
    signal: np.ndarray
    variance: np.ndarray


def load_coadd(path):
    with np.load(path, allow_pickle=False) as data:
        bands = tuple(str(b) for b in data["bands"])
        if bands != BANDS:
            raise ValueError(f"{path}: bands {bands}, expected {BANDS}")
        n_eff = np.asarray(data["n_eff"], float) if "n_eff" in data.files else None
        return Coadd(np.asarray(data["signal"], np.float32), np.asarray(data["variance"], np.float32),
                     np.asarray(data["psf_kernels"], np.float32), bands, tuple(int(v) for v in data["origin"]),
                     wcs_from_text(data["wcs_header"]) if "wcs_header" in data.files else None, n_eff)


def save_coadd(path, coadd):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    extra = {} if coadd.n_eff is None else dict(n_eff=np.asarray(coadd.n_eff, float))
    np.savez(path, signal=coadd.signal, variance=coadd.variance, psf_kernels=coadd.psf_kernels,
             bands=np.array(coadd.bands), origin=np.array(coadd.origin), wcs_header=wcs_to_text(coadd.wcs), **extra)


def load_exposures(paths):
    """Exposures from .npz files: single-band files (2-D signal, a band) or multi-band ones (one exposure per band)."""
    exposures = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            wcs = wcs_from_text(data["wcs_header"]) if "wcs_header" in data.files else None
            origin = tuple(int(v) for v in data["origin"]) if "origin" in data.files else (0, 0)
            signal, variance = np.asarray(data["signal"], np.float32), np.asarray(data["variance"], np.float32)
            if signal.ndim == 2:
                exposures.append(Exposure(signal, variance, np.asarray(data["psf_kernel"], np.float32),
                                          str(data["band"]), origin, wcs, Path(path).stem))
                continue
            for k, band in enumerate(str(b) for b in data["bands"]):
                exposures.append(Exposure(signal[k], variance[k], np.asarray(data["psf_kernels"][..., k], np.float32),
                                          band, origin, wcs, f"{Path(path).stem}_{band}"))
    return exposures


def save_exposure(path, exposure):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, signal=exposure.signal, variance=exposure.variance, psf_kernel=exposure.psf_kernel,
             band=exposure.band, origin=np.array(exposure.origin), wcs_header=wcs_to_text(exposure.wcs))


def place(cube, x, y, shape):
    """(x0, y0, clipped cube) of a cube (band, h, w) centred on pixel (round(x), round(y)) of an image of the given
    shape, cut to the image; None if it misses the image."""
    h, w = cube.shape[-2:]
    x0, y0 = int(np.round(x)) - w // 2, int(np.round(y)) - h // 2
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, shape[1]), min(y0 + h, shape[0])
    if xa >= xb or ya >= yb:
        return None
    return xa, ya, cube[..., ya - y0:yb - y0, xa - x0:xb - x0]


def coadd_injection(coadd, rendered, x, y):
    """The Patch added by injecting a rendered cube (band, h, w) into a coadd at array position (x, y): its light,
    and its shot noise in the variance, divided by each band's effective number of exposures (1 if unknown)."""
    placed = place(rendered, x, y, coadd.shape)
    if placed is None:
        return None
    x0, y0, light = placed
    n_eff = np.ones(len(BANDS)) if coadd.n_eff is None else np.asarray(coadd.n_eff, float)
    return Patch(x0, y0, light.astype(np.float32),
                 (np.clip(light, 0.0, None) / n_eff[:, None, None]).astype(np.float32))


# ------------------------------------------------------------------------------------------ exposures -> coadd

def exposure_weight(exposure):
    variance = exposure.variance[np.isfinite(exposure.variance) & (exposure.variance > 0)]
    return 1.0 / float(np.median(variance)) if variance.size else 0.0


def to_exposure_pixels(coadd, exposure, x, y):
    """The exposure pixel of coadd array pixel (x, y)."""
    if coadd.wcs is not None and exposure.wcs is not None:
        sky = coadd.wcs.pixel_to_world(x, y)
        xe, ye = exposure.wcs.world_to_pixel(sky)
        return float(xe), float(ye)
    if coadd.wcs is not None or exposure.wcs is not None:
        raise ValueError(f"{exposure.name}: give both the coadd and the exposures a WCS, or neither")
    return x + coadd.origin[0] - exposure.origin[0], y + coadd.origin[1] - exposure.origin[1]


def warp_to_coadd(coadd, exposure, image, x0, y0, region):
    """(values, coverage) on the coadd array region (xa, ya, xb, yb) of an exposure-frame image whose pixel (0, 0) is
    exposure pixel (x0, y0); coverage is 1 where the exposure has data."""
    xa, ya, xb, yb = region
    out_shape = (yb - ya, xb - xa)
    if coadd.wcs is None and exposure.wcs is None:
        dx = coadd.origin[0] - exposure.origin[0] - x0
        dy = coadd.origin[1] - exposure.origin[1] - y0
        values, coverage = np.zeros(out_shape, np.float64), np.zeros(out_shape, np.float64)
        sx0, sy0 = xa + dx, ya + dy  # source pixel of the region's first pixel
        cx0, cy0 = max(0, -sx0), max(0, -sy0)
        cx1 = min(out_shape[1], image.shape[1] - sx0)
        cy1 = min(out_shape[0], image.shape[0] - sy0)
        if cx0 < cx1 and cy0 < cy1:
            values[cy0:cy1, cx0:cx1] = image[sy0 + cy0:sy0 + cy1, sx0 + cx0:sx0 + cx1]
            coverage[cy0:cy1, cx0:cx1] = 1.0
        return values, coverage
    from reproject import reproject_interp

    source_wcs = exposure.wcs.slice((slice(y0, y0 + image.shape[0]), slice(x0, x0 + image.shape[1])))
    target_wcs = coadd.wcs.slice((slice(ya, yb), slice(xa, xb)))
    values, coverage = reproject_interp((image, source_wcs), target_wcs, shape_out=out_shape, order="bilinear")
    return np.nan_to_num(values), np.nan_to_num(coverage)


def recombine(coadd, exposures):
    """The coadd rebuilt from exposures on its pixel grid: per band the weighted mean of the exposures covering each
    pixel, and its variance sum(w^2 var) / (sum w)^2. Bands without exposures keep the coadd's own planes."""
    region = (0, 0, coadd.shape[1], coadd.shape[0])
    signal, variance = coadd.signal.copy(), coadd.variance.copy()
    for k, band in enumerate(BANDS):
        in_band = [e for e in exposures if e.band == band]
        if not in_band:
            continue
        numerator, weight_sum, var_sum = (np.zeros(coadd.shape) for _ in range(3))
        for exposure in in_band:
            w = exposure_weight(exposure)
            values, coverage = warp_to_coadd(coadd, exposure, exposure.signal, 0, 0, region)
            var_values, _ = warp_to_coadd(coadd, exposure, exposure.variance, 0, 0, region)
            numerator += w * values
            weight_sum += w * coverage
            var_sum += w ** 2 * var_values
        covered = weight_sum > 0
        signal[k][covered] = (numerator[covered] / weight_sum[covered]).astype(np.float32)
        variance[k][covered] = (var_sum[covered] / weight_sum[covered] ** 2).astype(np.float32)
    return Coadd(signal, variance, coadd.psf_kernels, coadd.bands, coadd.origin, coadd.wcs, None)


def exposure_injection(coadd, exposures, render_band, x, y, size):
    """The Patch the recombined coadd gains when a source is injected into every exposure: render_band(k, psf_kernel)
    draws band k with an exposure's PSF; it is added to the exposure at the source's exposure pixel, then warped and
    weighted onto the coadd region of side size around (x, y), as recombine() would."""
    half = size // 2
    region = (max(int(np.round(x)) - half, 0), max(int(np.round(y)) - half, 0),
              min(int(np.round(x)) + half + 1, coadd.shape[1]), min(int(np.round(y)) + half + 1, coadd.shape[0]))
    shape = (region[3] - region[1], region[2] - region[0])
    signal, variance = np.zeros((len(BANDS), *shape)), np.zeros((len(BANDS), *shape))
    for k, band in enumerate(BANDS):
        in_band = [e for e in exposures if e.band == band]
        if not in_band:
            continue
        numerator, weight_sum, var_sum = (np.zeros(shape) for _ in range(3))
        for exposure in in_band:
            w = exposure_weight(exposure)
            _, coverage = warp_to_coadd(coadd, exposure, np.ones_like(exposure.signal), 0, 0, region)
            weight_sum += w * coverage
            xe, ye = to_exposure_pixels(coadd, exposure, x, y)
            placed = place(render_band(k, exposure.psf_kernel)[None], xe, ye, exposure.signal.shape)
            if placed is None:
                continue
            x0, y0, light = placed
            values, _ = warp_to_coadd(coadd, exposure, light[0], x0, y0, region)
            shot, _ = warp_to_coadd(coadd, exposure, np.clip(light[0], 0.0, None), x0, y0, region)
            numerator += w * values
            var_sum += w ** 2 * shot
        covered = weight_sum > 0
        signal[k][covered] = numerator[covered] / weight_sum[covered]
        variance[k][covered] = var_sum[covered] / weight_sum[covered] ** 2
    return Patch(region[0], region[1], signal.astype(np.float32), variance.astype(np.float32))


def cutout(image_planes, x, y, half_box):
    """(x0, y0, planes[..., y0:y1, x0:x1]) of a box of half side half_box around (x, y), cut to the image."""
    ny, nx = image_planes.shape[-2:]
    x0, y0 = max(int(np.round(x)) - half_box, 0), max(int(np.round(y)) - half_box, 0)
    x1, y1 = min(int(np.round(x)) + half_box, nx), min(int(np.round(y)) + half_box, ny)
    return x0, y0, image_planes[..., y0:y1, x0:x1]


def add_patch(planes, x0, y0, patch, which):
    """A copy of image planes (band, h, w) whose pixel (0, 0) is image pixel (x0, y0), with the patch's signal or
    variance (which) added where they overlap."""
    out = np.array(planes, dtype=np.float32, copy=True)
    added = patch.signal if which == "signal" else patch.variance
    h, w = out.shape[-2:]
    xa, ya = max(patch.x0, x0), max(patch.y0, y0)
    xb, yb = min(patch.x0 + added.shape[-1], x0 + w), min(patch.y0 + added.shape[-2], y0 + h)
    if xa < xb and ya < yb:
        out[..., ya - y0:yb - y0, xa - x0:xb - x0] += added[..., ya - patch.y0:yb - patch.y0,
                                                            xa - patch.x0:xb - patch.x0]
    return out
