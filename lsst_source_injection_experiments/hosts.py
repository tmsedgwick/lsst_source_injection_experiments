"""Real objects the injections are placed next to, with the properties the nearest-neighbour draws are matched to:
position (coadd array pixels), redshift, stellar mass, half-light radius and surface brightness.

Objects come either from the LSST object table (scripts/export_from_butler.py objects) or, without one, are measured
on the coadd itself: detected footprints are split between their peaks (each pixel goes to the nearest peak of its
footprint), and each segment's flux per band and r-band half-light area are summed. Redshifts and stellar masses are
predicted from the magnitudes and size by a random forest trained on COSMOS2025, so they are on the same footing as
the COSMOS properties of the drawn neighbours they are matched to.
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .config import BANDS
from .cosmos import predict_photometric
from .detection import detect
from .photometry import flux_to_mag, surface_brightness


def segment(peaks, labels):
    """Map of segment ids (peak index + 1, 0 outside footprints): each footprint pixel goes to its nearest peak in the
    same footprint."""
    segments = np.zeros(labels.shape, np.int64)
    y, x = np.nonzero(labels)
    if len(peaks) == 0 or len(y) == 0:
        return segments
    tree = KDTree(peaks[["x", "y"]].to_numpy(float))
    k = min(8, len(peaks))
    _, nearest = tree.query(np.column_stack([x, y]), k=k)
    nearest = nearest.reshape(len(x), k)
    pixel_footprint = labels[y, x]
    peak_footprint = peaks["footprint"].to_numpy()
    same = peak_footprint[nearest] == pixel_footprint[:, None]
    has = same.any(axis=1)
    choice = nearest[np.arange(len(x)), np.argmax(same, axis=1)]
    segments[y[has], x[has]] = choice[has] + 1
    return segments


def half_light_npix_per_segment(values, segments, n_segments):
    """Brightest-pixel count holding half of each segment's positive light (0 for segments without light)."""
    seg, value = segments.ravel(), np.clip(values.ravel(), 0.0, None)
    keep = seg > 0
    seg, value = seg[keep], value[keep]
    order = np.lexsort((-value, seg))  # by segment, brightest first
    seg, value = seg[order], value[order]
    cumulative = np.cumsum(value)
    first = np.searchsorted(seg, seg)  # index of each pixel's segment's first pixel
    within = cumulative - np.where(first > 0, cumulative[first - 1], 0.0)
    total = np.bincount(seg, weights=value, minlength=n_segments + 1)
    below = np.bincount(seg, weights=within < 0.5 * total[seg], minlength=n_segments + 1).astype(np.int64)
    has_light = total > 0
    return np.where(has_light, below + 1, 0)[1:]


def measure_objects(coadd, cfg):
    """Objects detected and measured on the coadd: x, y, m_<band>, half_light_radius_arcsec, mu_r."""
    peaks, labels = detect(coadd.signal, coadd.variance, coadd.psf_kernels, cfg)
    segments = segment(peaks, labels)
    n = len(peaks)
    objects = peaks[["x", "y"]].copy()
    for k, band in enumerate(BANDS):
        flux = np.bincount(segments.ravel(), weights=np.nan_to_num(coadd.signal[k]).ravel(), minlength=n + 1)[1:]
        objects[f"m_{band}"] = np.where(flux > 0, flux_to_mag(flux, cfg["zeropoint"]), np.nan)
    npix_half = half_light_npix_per_segment(coadd.signal[BANDS.index("r")], segments, n)
    area = npix_half * cfg["pixel_scale"] ** 2
    objects["half_light_radius_arcsec"] = np.where(npix_half > 0, np.sqrt(npix_half) * cfg["pixel_scale"], np.nan)
    objects["mu_r"] = np.where(npix_half > 0, surface_brightness(objects["m_r"], np.where(area > 0, area, 1.0)), np.nan)
    return objects


def add_redshift_and_mass(objects, forest):
    """objects with z, logmass and log_Re added; rows missing any of them are dropped."""
    out = objects.copy()
    out["z"], out["logmass"] = predict_photometric(forest, out)
    out["log_Re"] = np.log10(out["half_light_radius_arcsec"])
    needed = ["x", "y", "z", "logmass", "log_Re", "mu_r"]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).reset_index(drop=True)
