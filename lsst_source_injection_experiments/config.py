"""Settings shared by every step. Every function takes a cfg dict; pass a partial dict to override keys."""

from typing import Any

import numpy as np

BANDS = ("u", "g", "r", "i", "z", "y")
R_INDEX = BANDS.index("r")

CONFIG: dict[str, Any] = dict(
    pixel_scale=0.2,  # arcsec per pixel
    zeropoint=31.4,  # AB zeropoint of nJy fluxes
    seed=20261008,

    # Surface brightness bins: injections are drawn per bin, each centre +- half width (mag / arcsec^2). The surface
    # brightness of a source is its total r magnitude spread over its half-light area, mu_r = m_r + 2.5 log10(A_half),
    # as in the LSST object table and the original injection notebook (0.75 mag brighter than the mean surface
    # brightness of the light inside that area).
    mu_r_centres=np.arange(18.0, 30.01, 0.5),
    mu_r_half_width=0.25,
    n_draws_per_bin=1000,  # property draws per bin, from which the placed injections are taken
    n_injections_per_bin=100,

    # Targeted placement next to a real object matching the drawn nearest neighbour (original notebook values)
    host_max_dz=0.2,
    host_max_dlogmass=0.5,
    host_max_dmu_r=0.5,
    host_max_dre_arcsec=10.0,
    host_candidates=10000,  # best-matching hosts tried per draw
    position_angle_tries=100,  # random position angles tried per host
    min_injection_separation_arcsec=5.0,
    edge_margin_pix=32,  # injections stay this far inside the image

    # Size and flux of an injected source: the observed (PSF-convolved) half-light radius is matched to the target
    # within re_tolerance; targets below the PSF's own half-light radius (or re_floor_arcsec) are raised to it.
    re_floor_arcsec=0.8,
    re_tolerance=0.02,
    re_iterations=10,

    # New Horizon dwarf templates (h5 files with cube[projection][snapshot][redshift]["image_full"], ugrizy, pre-PSF)
    nh_file_pattern="Rvir_mocks_V2_*.h5",
    nh_projection="xy",
    nh_snapshot="904",
    nh_redshift_key="0.05",

    # BCG + envelope: core-Sersic profile (Graham et al. 2003) of the ICL injection notebook. Re is drawn log-uniform
    # in bcg_re_arcsec; the break radius scales with Re as in Abell 2261; the colours are those of an old population.
    bcg_re_arcsec=(5.0, 60.0),
    bcg_n=3.9,
    bcg_rb_over_re=3.63 / 30.0,
    bcg_gamma=0.02,
    bcg_alpha=3.6,
    bcg_axis_ratio=0.7,
    bcg_colours=dict(u_minus_r=2.2, g_minus_r=0.65, r_minus_i=0.35, i_minus_z=0.25, z_minus_y=0.10),

    # Each injection is analysed in a square cutout of side 2 * half_box around it
    half_box=300,

    # Detection: peaks of the PSF-smoothed image at S/N >= detection_sn, in footprints grown by grow_sigmas PSF widths
    detect_bands=("r",),
    detection_sn=5.0,
    grow_sigmas=2.4,

    # Matching (pixels): the injection counts as detected with a peak within detect_radius_pix; objects before and
    # after injection are tracked one to one within track_radius_pix; the nearest neighbour counts as recovered within
    # nn_radius_pix.
    detect_radius_pix=5.0,
    track_radius_pix=5.0,
    nn_radius_pix=3.0,

    # Deblending with scarlet lite
    model_psf_sigma=0.8,  # pixels
    deblend_max_iter=300,
    deblend_e_rel=1e-2,

    psf_stamp=25,  # side of the PSF stamps written by the Butler export
)

SOURCES = ("newhorizon", "doublesersicbcg")
MODES = ("deepcoadd", "calexp")
