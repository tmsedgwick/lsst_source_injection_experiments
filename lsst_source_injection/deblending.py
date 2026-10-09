"""Deblend each injection's blend region with scarlet lite, before and after injecting, and measure flux biases.

Sources are initialised at the detected peaks inside the region and fitted (as ScarletDeblendTask does, without its
deconvolution step); a source's flux in a band is the sum of its model, the modelFlux of DMTN-335. Then
  * injected source (when detected): bias = 100 (F_measured - F_injected) / F_injected, in r and summed over bands,
    and its measured surface brightness, the measured r flux over the injected half-light area;
  * nearest neighbour (when recovered): bias of its flux after injection relative to before;
  * rest-of-blend: bias of the summed flux of every source in the region except the injected one, after relative to
    before.
"""

import numpy as np
from lsst.scarlet.lite import Blend, Box, Observation, default_adaprox_parameterization
from lsst.scarlet.lite.initialization import FactorizedInitialization
from lsst.scarlet.lite.measure import conserve_flux
from lsst.scarlet.lite.operators import Monotonicity
from lsst.scarlet.lite.utils import integrated_circular_gaussian

from .config import BANDS, R_INDEX
from .photometry import flux_to_mag
from .sources import normalised


def source_fluxes(signal, variance, psf_kernels, centres, region, cfg):
    """(n_source, band) model fluxes of sources initialised at centres (x, y) and fitted in the region (x0, y0, x1, y1)
    of the image planes; NaN rows for sources scarlet could not initialise."""
    x0, y0, x1, y1 = region
    images = np.nan_to_num(signal[:, y0:y1, x0:x1]).astype(np.float32)
    var = np.asarray(variance[:, y0:y1, x0:x1], np.float32)
    good = np.isfinite(var) & (var > 0)
    var = np.where(good, var, np.float32(1e10))
    weights = np.where(good, 1.0 / var, 0.0).astype(np.float32)
    psfs = np.stack([normalised(psf_kernels[..., k]) for k in range(len(BANDS))]).astype(np.float32)
    model_psf = integrated_circular_gaussian(sigma=cfg["model_psf_sigma"]).astype(np.float32)[None]
    observation = Observation(images, var, weights, psfs, model_psf, bbox=Box(images.shape[-2:], origin=(y0, x0)),
                              bands=BANDS)
    yx = [(int(np.clip(round(y), y0, y1 - 1)), int(np.clip(round(x), x0, x1 - 1))) for x, y in centres]
    fluxes = np.full((len(centres), len(BANDS)), np.nan)
    if not yx:
        return fluxes
    side = 2 * max(x1 - x0, y1 - y0) + 1
    init = FactorizedInitialization(observation, yx, monotonicity=Monotonicity((side, side)))
    blend = Blend(init.sources, observation).fit_spectra()
    blend.parameterize(default_adaprox_parameterization)
    blend.fit(cfg["deblend_max_iter"], e_rel=cfg["deblend_e_rel"])
    conserve_flux(blend)
    for i, source in enumerate(blend.sources):
        if source.components:
            fluxes[i] = source.get_model().data.sum(axis=(1, 2))
    return fluxes


def percent_bias(measured, reference):
    return 100.0 * (measured - reference) / reference if np.isfinite(reference) and reference > 0 else np.nan


def deblend_injection(before_signal, before_variance, after_signal, after_variance, psf_kernels, peaks, injection, cfg):
    """Flux metrics of one injection from its peaks table (recovery.recover) and injection row (with the injected
    fluxes flux_<band> and half_light_area_arcsec2)."""
    region = tuple(int(injection[k]) for k in ("region_x0", "region_y0", "region_x1", "region_y1"))
    before = peaks[(peaks["stage"] == "before") & peaks["in_region"]].reset_index(drop=True)
    after = peaks[(peaks["stage"] == "after") & peaks["in_region"]].reset_index(drop=True)
    flux_before = source_fluxes(before_signal, before_variance, psf_kernels, before[["x", "y"]].to_numpy(), region, cfg)
    flux_after = source_fluxes(after_signal, after_variance, psf_kernels, after[["x", "y"]].to_numpy(), region, cfg)
    injected_true = np.array([injection[f"flux_{band}"] for band in BANDS], float)

    is_injected = after["is_injected"].to_numpy(bool)
    out = dict(n_sources_before=len(before), n_sources_after=len(after))
    if is_injected.any():
        measured = flux_after[np.argmax(is_injected)]
        out.update(inj_flux_r=measured[R_INDEX], inj_bias_r=percent_bias(measured[R_INDEX], injected_true[R_INDEX]),
                   inj_bias_total=percent_bias(np.nansum(measured), injected_true.sum()),
                   inj_mu_r_measured=float(flux_to_mag(measured[R_INDEX] / injection["half_light_area_arcsec2"],
                                                       cfg["zeropoint"])))
    else:
        out.update(inj_flux_r=np.nan, inj_bias_r=np.nan, inj_bias_total=np.nan, inj_mu_r_measured=np.nan)

    nn_rows = np.flatnonzero(before["is_nn"].to_numpy(bool))
    out.update(nn_flux_r_before=np.nan, nn_flux_r_after=np.nan, nn_bias_r=np.nan)
    if nn_rows.size and injection["nn_recovered"]:
        matched = np.flatnonzero(after["peak"].to_numpy() == before["matched_peak"].iat[nn_rows[0]])
        if matched.size:
            f0, f1 = flux_before[nn_rows[0], R_INDEX], flux_after[matched[0], R_INDEX]
            out.update(nn_flux_r_before=f0, nn_flux_r_after=f1, nn_bias_r=percent_bias(f1, f0))

    rest_after = flux_after[~is_injected]
    r0, r1 = np.nansum(flux_before[:, R_INDEX]), np.nansum(rest_after[:, R_INDEX])
    out.update(rest_flux_r_before=r0, rest_flux_r_after=r1, rest_bias_r=percent_bias(r1, r0),
               rest_bias_total=percent_bias(np.nansum(rest_after), np.nansum(flux_before)))
    return out
