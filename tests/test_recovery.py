"""Detection, recovery of the injected source and its neighbours, and deblending."""

import numpy as np
import pandas as pd
import pytest

from lsst_source_injection_experiments.config import BANDS
from lsst_source_injection_experiments.deblending import deblend_injection
from lsst_source_injection_experiments.detection import detect
from lsst_source_injection_experiments.recovery import one_to_one_match, recover
from conftest import gaussian


def test_one_to_one_match_pairs_closest_first():
    match, separation = one_to_one_match([[0, 0], [1, 0], [50, 50]], [[0.5, 0], [10, 10]], max_sep=3)
    assert list(match) == [0, -1, -1] or list(match) == [-1, 0, -1]
    assert np.isnan(separation[2])


def test_detect_finds_the_synthetic_galaxies(coadd, cfg):
    peaks, labels = detect(coadd.signal, coadd.variance, coadd.psf_kernels, cfg)
    for x, y in [(60, 60), (150, 140), (110, 50), (60, 170)]:
        assert np.hypot(peaks["x"] - x, peaks["y"] - y).min() <= 1.5
    assert labels.max() >= 4


def injected(coadd, x, y, flux, width):
    added = flux * gaussian(coadd.shape, x, y, width)[None] * np.linspace(0.6, 1.4, len(BANDS))[:, None, None]
    return coadd.signal + added, coadd.variance + np.clip(added, 0, None) / 10.0, added


def test_bright_injection_is_detected_next_to_its_neighbour(coadd, cfg):
    after_signal, after_variance, _ = injected(coadd, 75, 70, 3000, 3.0)
    summary, peaks = recover(coadd.signal, coadd.variance, after_signal, after_variance, coadd.psf_kernels, 75, 70, cfg)
    assert summary["detected"]
    assert summary["nn_exists"] and summary["nn_distance"] == pytest.approx(np.hypot(15, 10), abs=1.5)
    assert peaks.loc[peaks["stage"] == "after", "is_injected"].sum() == 1
    assert summary["rest_n_before"] >= 1


def test_faint_injection_is_missed_and_changes_nothing(coadd, cfg):
    after_signal, after_variance, _ = injected(coadd, 120, 110, 1.0, 6.0)
    summary, _ = recover(coadd.signal, coadd.variance, after_signal, after_variance, coadd.psf_kernels, 120, 110, cfg)
    assert not summary["detected"]
    assert summary["nn_recovered"]
    assert summary["rest_recovered_fraction"] == pytest.approx(1.0)


def test_deblending_measures_the_injected_flux(coadd, cfg):
    x, y = 75, 70
    after_signal, after_variance, added = injected(coadd, x, y, 3000, 3.0)
    summary, peaks = recover(coadd.signal, coadd.variance, after_signal, after_variance, coadd.psf_kernels, x, y, cfg)
    row = pd.Series(dict(summary, half_light_area_arcsec2=1.0,
                         **{f"flux_{band}": added[k].sum() for k, band in enumerate(BANDS)}))
    metrics = deblend_injection(coadd.signal, coadd.variance, after_signal, after_variance, coadd.psf_kernels,
                                peaks, row, cfg)
    assert abs(metrics["inj_bias_r"]) < 15
    assert abs(metrics["nn_bias_r"]) < 15
    assert np.isfinite(metrics["rest_bias_r"])


def test_injection_landing_on_a_real_object_is_not_its_detection(coadd, cfg):
    """A faint injection centred on an existing galaxy leaves that galaxy's peak in place: not a detection."""
    after_signal, after_variance, _ = injected(coadd, 61, 60, 5.0, 3.0)
    summary, _ = recover(coadd.signal, coadd.variance, after_signal, after_variance, coadd.psf_kernels, 61, 60, cfg)
    assert summary["injected_peak_distance"] < cfg["detect_radius_pix"]
    assert not summary["detected"]
    assert summary["nn_recovered"]
