"""Source preparation and injection into a coadd or into exposures."""

import numpy as np
import pytest

from lsst_source_injection_experiments.config import BANDS, R_INDEX
from lsst_source_injection_experiments.cosmos import fit_colour_models, load_reference
from lsst_source_injection_experiments.images import (Coadd, Exposure, coadd_injection, exposure_injection, recombine)
from lsst_source_injection_experiments.photometry import flux_to_mag
from lsst_source_injection_experiments.sources import (core_sersic_image, nh_template_paths, observed_half_light,
                                           prepare_doublesersicbcg, prepare_newhorizon, read_nh_cube, render,
                                           render_band)


def copies(coadd, n):
    """n stand-in exposures per band: the coadd's planes with n times its variance."""
    return [Exposure(coadd.signal[k], coadd.variance[k] * n, coadd.psf_kernels[..., k], band, coadd.origin)
            for _ in range(n) for k, band in enumerate(BANDS)]


@pytest.fixture(scope="module")
def bcg(coadd, cfg):
    return prepare_doublesersicbcg(coadd.psf_kernels, dict(mu_r=23.0, Re_arcsec=3.0), np.random.default_rng(0), cfg)


def test_core_sersic_is_unit_sum_and_centred():
    image = core_sersic_image(5.0, 0.6, 3.9, 0.02, 3.6, 0.7, 0.3, 101, 0.2)
    assert image.sum() == pytest.approx(1.0)
    assert np.unravel_index(image.argmax(), image.shape) == (50, 50)


def test_bcg_has_target_surface_brightness(bcg, coadd, cfg):
    rendered = render(bcg, coadd.psf_kernels)
    assert rendered[R_INDEX].sum() == pytest.approx(bcg.fluxes[R_INDEX], rel=1e-6)
    mu = flux_to_mag(bcg.fluxes[R_INDEX], cfg["zeropoint"]) + 2.5 * np.log10(bcg.info["half_light_area_arcsec2"])
    assert mu == pytest.approx(23.0, abs=1e-6)
    assert bcg.info["mag_u"] - bcg.info["mag_r"] == pytest.approx(cfg["bcg_colours"]["u_minus_r"])


def test_new_horizon_matches_target_size_and_surface_brightness(nh_dir, coadd, cfg):
    cube = read_nh_cube(nh_template_paths(nh_dir, cfg)[0], cfg)
    target = dict(mu_r=24.0, Re_arcsec=3.0, ur=1.3, z=0.1, logmass=8.5)
    prepared = prepare_newhorizon(cube, coadd.psf_kernels, target, fit_colour_models(load_reference()), cfg)
    re, area = observed_half_light(prepared.cube[R_INDEX], coadd.psf_kernels[..., R_INDEX] / coadd.psf_kernels[..., R_INDEX].sum(),
                                   cfg["pixel_scale"])
    assert re == pytest.approx(3.0, rel=0.05)
    mu = flux_to_mag(prepared.fluxes[R_INDEX], cfg["zeropoint"]) + 2.5 * np.log10(area)
    assert mu == pytest.approx(24.0, abs=0.01)
    assert prepared.info["mag_u"] - prepared.info["mag_r"] == pytest.approx(1.3)


def test_small_targets_are_raised_to_the_psf_size(nh_dir, coadd, cfg):
    cube = read_nh_cube(nh_template_paths(nh_dir, cfg)[0], cfg)
    prepared = prepare_newhorizon(cube, coadd.psf_kernels, dict(mu_r=22.0, Re_arcsec=0.05, ur=1.0, z=0.1, logmass=8.0),
                                  fit_colour_models(load_reference()), cfg)
    assert prepared.info["re_target_effective_arcsec"] >= cfg["re_floor_arcsec"]


def test_coadd_injection_adds_flux_and_shot_noise(bcg, coadd):
    with_n_eff = Coadd(coadd.signal, coadd.variance, coadd.psf_kernels, BANDS, coadd.origin, None, np.full(6, 4.0))
    patch = coadd_injection(with_n_eff, render(bcg, coadd.psf_kernels), 110, 110)
    assert patch.signal[R_INDEX].sum() == pytest.approx(bcg.fluxes[R_INDEX], rel=1e-3)
    np.testing.assert_allclose(patch.variance, np.clip(patch.signal, 0, None) / 4.0, rtol=1e-6)


def test_recombined_copies_give_back_the_coadd(coadd):
    rebuilt = recombine(coadd, copies(coadd, 3))
    np.testing.assert_allclose(rebuilt.signal, coadd.signal, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rebuilt.variance, coadd.variance, rtol=1e-5)


def test_injecting_into_stacked_copies_equals_injecting_into_the_coadd(bcg, coadd):
    """Calexp mode on n copies of the coadd (each with n times its variance) must reproduce coadd injection."""
    n = 4
    with_n_eff = Coadd(coadd.signal, coadd.variance, coadd.psf_kernels, BANDS, coadd.origin, None, np.full(6, float(n)))
    direct = coadd_injection(with_n_eff, render(bcg, coadd.psf_kernels), 110.0, 110.0)
    size = max(bcg.cube.shape[-2:]) + 2 * coadd.psf_kernels.shape[0]
    via_exposures = exposure_injection(coadd, copies(coadd, n), lambda k, kernel: render_band(bcg, k, kernel),
                                       110.0, 110.0, size)
    full_direct, full_exposures = np.zeros_like(coadd.signal), np.zeros_like(coadd.signal)
    var_direct, var_exposures = np.zeros_like(coadd.signal), np.zeros_like(coadd.signal)
    for full, var, patch in [(full_direct, var_direct, direct), (full_exposures, var_exposures, via_exposures)]:
        h, w = patch.signal.shape[-2:]
        full[:, patch.y0:patch.y0 + h, patch.x0:patch.x0 + w] += patch.signal
        var[:, patch.y0:patch.y0 + h, patch.x0:patch.x0 + w] += patch.variance
    np.testing.assert_allclose(full_exposures, full_direct, atol=1e-5 * full_direct.max())
    np.testing.assert_allclose(var_exposures, var_direct, atol=1e-5 * var_direct.max())


def test_exposures_with_their_own_psf_and_offset_grid(bcg, coadd):
    """An exposure on a grid shifted by whole pixels, with a wider PSF: the injected light lands at the same sky
    position, carries the same flux, and is spread by the exposure's PSF."""
    from conftest import gaussian_psf

    wide = gaussian_psf(sigma=2.6)
    shifted = [Exposure(coadd.signal[k, 5:, 7:], coadd.variance[k, 5:, 7:], wide, band,
                        (coadd.origin[0] + 7, coadd.origin[1] + 5)) for k, band in enumerate(BANDS)]
    size = max(bcg.cube.shape[-2:]) + 2 * wide.shape[0]
    patch = exposure_injection(coadd, shifted, lambda k, kernel: render_band(bcg, k, kernel), 110.0, 110.0, size)
    r = patch.signal[R_INDEX]
    assert r.sum() == pytest.approx(bcg.fluxes[R_INDEX], rel=1e-3)
    y, x = np.unravel_index(r.argmax(), r.shape)
    assert (x + patch.x0, y + patch.y0) == (110, 110)
    direct = render(bcg, coadd.psf_kernels)[R_INDEX]
    assert r.max() < direct.max()
