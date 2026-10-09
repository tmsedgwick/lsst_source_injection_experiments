"""The two kinds of injected source, each prepared to target properties as a pre-PSF ugrizy cube plus per-band fluxes.

  * newhorizon: a New Horizon dwarf (Dubois et al. 2021, Martin et al. 2022), resized so its PSF-convolved half-light
    radius matches the target, with r flux set by the target surface brightness, u by u - r, and g, i, z, y by colours
    predicted from (z, log M*, u - r) (DMTN-335 section 5.2).
  * doublesersicbcg: a BCG with its extended envelope, the core-Sersic profile (Graham et al. 2003) of the ICL
    injection notebook, with an old population's colours.

A prepared source is drawn into an image by render(), convolved with that image's PSF. Its size and flux are set
with the reference PSF it was prepared with (the coadd's), so the same galaxy is injected whichever image receives it.
"""

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import fftconvolve

from .config import BANDS, R_INDEX
from .cosmos import predict_colours
from .photometry import half_light_npix, mag_to_flux, stellar_mass


@dataclass
class PreparedSource:
    """A pre-PSF cube (band, y, x), each band normalised to unit sum, and the total flux (nJy) of each band."""
    cube: np.ndarray
    fluxes: np.ndarray
    info: dict = field(default_factory=dict)


def normalised(kernel):
    kernel = np.nan_to_num(np.asarray(kernel, float))
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def convolve_padded(image, kernel):
    """image convolved with kernel, padded first so no light falls off the edge."""
    pad = kernel.shape[0] // 2
    return fftconvolve(np.pad(image, pad), kernel, mode="same")


def render(source, psf_kernels):
    """The source as observed: each band convolved with its PSF (stamp, stamp, band) and scaled to its flux."""
    return np.stack([render_band(source, k, psf_kernels[..., k]) for k in range(len(BANDS))])


def render_band(source, k, kernel):
    """Band k of the source as observed through one PSF stamp (stamp, stamp), scaled to its flux."""
    plane = convolve_padded(source.cube[k], normalised(kernel))
    total = plane.sum()
    return plane * (source.fluxes[k] / total) if total > 0 else plane


def observed_half_light(image_r, psf_r, pixel_scale):
    """(half-light radius arcsec, half-light area arcsec^2) of an r-band image after convolution with the PSF."""
    n_half = half_light_npix(convolve_padded(image_r, psf_r))
    return float(np.sqrt(max(n_half, 1))) * pixel_scale, n_half * pixel_scale ** 2


def psf_half_light_radius(psf_r, pixel_scale):
    return float(np.sqrt(max(half_light_npix(psf_r), 1))) * pixel_scale


def match_observed_radius(make_image, start, target_arcsec, psf_r, cfg):
    """Iterate a size parameter (starting at start) until the PSF-convolved half-light radius of make_image(size) is
    within cfg['re_tolerance'] of the target; returns the size."""
    size = start
    for _ in range(cfg["re_iterations"]):
        image = make_image(size)
        if image is None:
            break
        measured, _ = observed_half_light(image, psf_r, cfg["pixel_scale"])
        if abs(measured - target_arcsec) / target_arcsec < cfg["re_tolerance"]:
            break
        size *= target_arcsec / measured
    return size


def effective_target_radius(target_arcsec, psf_r, cfg):
    """The target half-light radius, raised to the PSF's own (or re_floor_arcsec) when it is smaller than that."""
    return max(target_arcsec, cfg["re_floor_arcsec"], psf_half_light_radius(psf_r, cfg["pixel_scale"]))


# --------------------------------------------------------------------------------------------------- New Horizon

def nh_template_paths(nh_dir, cfg):
    paths = sorted(Path(nh_dir).expanduser().glob(cfg["nh_file_pattern"]))
    if not paths:
        raise FileNotFoundError(f"No New Horizon templates matching {cfg['nh_file_pattern']} in {nh_dir}")
    return paths


def read_nh_cube(path, cfg):
    """The pre-PSF ugrizy image cube of one New Horizon dwarf."""
    key = f"{cfg['nh_projection']}/{cfg['nh_snapshot']}/{cfg['nh_redshift_key']}/image_full"
    with h5py.File(path, "r") as handle:
        dataset = handle[key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{path}: {key} is not an image cube")
        cube = np.asarray(dataset[()], float)
    if cube.shape[0] != len(BANDS):
        raise ValueError(f"{path}: expected {len(BANDS)} bands (ugrizy), found {cube.shape[0]}")
    return np.nan_to_num(np.asarray(cube, float))


def resized(cube, scale):
    """The cube resampled by a linear factor (bilinear, pixel-centred), or None if it would be under 5 pixels."""
    ny, nx = cube.shape[-2:]
    shape = (int(round(ny * scale)), int(round(nx * scale)))
    if min(shape) < 5:
        return None
    factors = (shape[0] / ny, shape[1] / nx)
    planes = cube[None] if cube.ndim == 2 else cube
    out = np.stack([np.asarray(zoom(plane, factors, order=1, grid_mode=True, mode="grid-constant")) for plane in planes])
    return out[0] if cube.ndim == 2 else out


def prepare_newhorizon(nh_cube, psf_kernels, target, colour_models, cfg):
    """The New Horizon cube resized and scaled to target = dict(mu_r, Re_arcsec, ur, z, logmass)."""
    psf_r = normalised(psf_kernels[..., R_INDEX])
    re_target = effective_target_radius(target["Re_arcsec"], psf_r, cfg)
    base_npix = half_light_npix(nh_cube[R_INDEX])
    if base_npix == 0:
        raise ValueError("The New Horizon cube has no light in r")
    base_re = np.sqrt(base_npix) * cfg["pixel_scale"]
    scale = match_observed_radius(lambda s: resized(nh_cube[R_INDEX], s), re_target / base_re, re_target, psf_r, cfg)
    cube = resized(nh_cube, scale)
    if cube is None:
        raise ValueError("The New Horizon cube would be resized below 5 pixels")
    cube = np.stack([plane / plane.sum() if plane.sum() > 0 else plane for plane in np.clip(cube, 0.0, None)])
    re_measured, area = observed_half_light(cube[R_INDEX], psf_r, cfg["pixel_scale"])

    zp = cfg["zeropoint"]
    mag_r = target["mu_r"] - 2.5 * np.log10(area)
    mag_u = mag_r + target["ur"]
    colours = predict_colours(colour_models, target["z"], target["logmass"], target["ur"])
    mag_g = mag_u - colours["u_g"]
    mag_i = mag_g - colours["g_i"]
    mag_z = mag_i - colours["i_z"]
    mag_y = mag_z - colours["z_y"]
    mags = np.array([mag_u, mag_g, mag_r, mag_i, mag_z, mag_y])
    info = dict(re_target_arcsec=target["Re_arcsec"], re_target_effective_arcsec=re_target,
                re_measured_arcsec=re_measured, half_light_area_arcsec2=area, resize_factor=scale,
                logmass_from_photometry=float(stellar_mass(mag_i, mag_g, target["z"])),
                **{f"mag_{band}": float(m) for band, m in zip(BANDS, mags)})
    return PreparedSource(cube, mag_to_flux(mags, zp), info)


# ----------------------------------------------------------------------------------------------- core-Sersic BCG

def core_sersic_image(re_arcsec, rb_arcsec, n, gamma, alpha, axis_ratio, position_angle, size, pixel_scale):
    """A unit-sum elliptical core-Sersic image (Graham et al. 2003, eq. 1, with I_b = 1) of side size pixels."""
    re_pix, rb_pix = re_arcsec / pixel_scale, rb_arcsec / pixel_scale
    b_n = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n ** 2)  # Ciotti & Bertin (1999)
    y, x = np.indices((size, size), dtype=float)
    dx, dy = x - (size - 1) / 2.0, y - (size - 1) / 2.0
    c, s = np.cos(position_angle), np.sin(position_angle)
    radius = np.maximum(np.hypot(dx * c + dy * s, (-dx * s + dy * c) / axis_ratio), 0.5)
    i_prime = 2.0 ** (-gamma / alpha) * np.exp(b_n * 2.0 ** (1.0 / (n * alpha)) * (rb_pix / re_pix) ** (1.0 / n))
    core = (1.0 + (rb_pix / radius) ** alpha) ** (gamma / alpha)
    envelope = np.exp(-b_n * ((radius ** alpha + rb_pix ** alpha) / re_pix ** alpha) ** (1.0 / (n * alpha)))
    image = np.nan_to_num(i_prime * core * envelope)
    return image / image.sum()


def prepare_doublesersicbcg(psf_kernels, target, rng, cfg):
    """A core-Sersic BCG + envelope with target = dict(mu_r, Re_arcsec); random position angle."""
    psf_r = normalised(psf_kernels[..., R_INDEX])
    re = target["Re_arcsec"]
    size = max(101, 3 * psf_r.shape[0], int(np.ceil(8 * re / cfg["pixel_scale"])) | 1)
    position_angle = float(rng.uniform(0.0, np.pi))
    image = core_sersic_image(re, cfg["bcg_rb_over_re"] * re, cfg["bcg_n"], cfg["bcg_gamma"], cfg["bcg_alpha"],
                              cfg["bcg_axis_ratio"], position_angle, size, cfg["pixel_scale"])
    re_measured, area = observed_half_light(image, psf_r, cfg["pixel_scale"])
    colours = cfg["bcg_colours"]
    mag_r = target["mu_r"] - 2.5 * np.log10(area)
    mag_i = mag_r - colours["r_minus_i"]
    mag_z = mag_i - colours["i_minus_z"]
    mags = np.array([mag_r + colours["u_minus_r"], mag_r + colours["g_minus_r"], mag_r, mag_i, mag_z,
                     mag_z - colours["z_minus_y"]])
    info = dict(re_target_arcsec=re, re_measured_arcsec=re_measured, half_light_area_arcsec2=area,
                position_angle_rad=position_angle, **{f"mag_{band}": float(m) for band, m in zip(BANDS, mags)})
    return PreparedSource(np.repeat(image[None], len(BANDS), axis=0), mag_to_flux(mags, cfg["zeropoint"]), info)
