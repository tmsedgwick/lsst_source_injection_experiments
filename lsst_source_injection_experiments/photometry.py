"""Magnitudes, surface brightness, half-light sizes and the stellar-mass prescription."""

import numpy as np
from scipy.integrate import quad

H0, OMEGA_M = 70.0, 0.3  # flat Lambda-CDM, as in DMTN-335
C_KM_S = 299792.458


def flux_to_mag(flux, zeropoint):
    return zeropoint - 2.5 * np.log10(np.clip(np.asarray(flux, float), 1e-30, None))


def mag_to_flux(mag, zeropoint):
    return 10.0 ** (-0.4 * (np.asarray(mag, float) - zeropoint))


def half_light_npix(image):
    """Number of brightest pixels that together hold half the (positive) light of an image; 0 if it has none."""
    flat = np.clip(np.asarray(image, float), 0.0, None).ravel()
    total = flat.sum()
    if total <= 0:
        return 0
    cumulative = np.cumsum(np.sort(flat)[::-1])
    return int(np.searchsorted(cumulative, 0.5 * total) + 1)


def half_light_radius_arcsec(image, pixel_scale):
    """Half-light radius as the side of the half-light area, sqrt(N_half) pixels."""
    return float(np.sqrt(max(half_light_npix(image), 1))) * pixel_scale


def surface_brightness(mag, half_light_area_arcsec2):
    """mu = m + 2.5 log10(A_half): total magnitude spread over the half-light area (see config)."""
    return np.asarray(mag, float) + 2.5 * np.log10(np.asarray(half_light_area_arcsec2, float))


def flux_bias_to_dmag(bias_percent):
    """Magnitude (or surface brightness) bias of a percentage flux bias; NaN where the flux ratio is not positive."""
    ratio = 1.0 + np.asarray(bias_percent, float) / 100.0
    out = np.full(ratio.shape, np.nan)
    good = np.isfinite(ratio) & (ratio > 0)
    out[good] = -2.5 * np.log10(ratio[good])
    return out


def distance_modulus(z):
    z = np.asarray(z, float)
    out = np.full(z.shape, np.nan)
    good = np.isfinite(z) & (z > 0)
    for i in np.flatnonzero(good):
        comoving, _ = quad(lambda u: 1.0 / np.sqrt(OMEGA_M * (1 + u) ** 3 + 1.0 - OMEGA_M), 0.0, z.flat[i])
        luminosity_distance_pc = (1 + z.flat[i]) * comoving * C_KM_S / H0 * 1e6
        out.flat[i] = 5.0 * np.log10(luminosity_distance_pc) - 5.0
    return out


def stellar_mass(mag_i, mag_g, z):
    """log10(M*/Msun) from the i magnitude and g - i colour (Bryant et al. 2015), as in DMTN-335."""
    z = np.asarray(z, float)
    f = 1.104 - 1.687 * z + 9.193 * z ** 2 - 15.15 * z ** 3
    g = 0.8237 + 0.5908 * z - 12.84 * z ** 2 + 26.40 * z ** 3
    return -0.4 * np.asarray(mag_i, float) + 0.4 * distance_modulus(z) + f + g * (np.asarray(mag_g) - mag_i)
