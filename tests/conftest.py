"""Shared fixtures: a small synthetic six-band coadd with a few galaxies and noise, New Horizon-like template files,
and hosts drawn from the COSMOS2025 reference neighbours."""

import h5py
import numpy as np
import pandas as pd
import pytest

from lsst_source_injection_experiments.config import BANDS, CONFIG
from lsst_source_injection_experiments.cosmos import load_reference
from lsst_source_injection_experiments.images import Coadd, save_coadd

PSF_SIGMA_PIX = 1.8
SIZE = 220


def gaussian_psf(sigma=PSF_SIGMA_PIX, size=25):
    yy, xx = np.mgrid[:size, :size] - size // 2
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian(shape, x0, y0, sigma):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    image = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return image / image.sum()


@pytest.fixture(scope="session")
def cfg():
    return dict(CONFIG, half_box=70, n_injections_per_bin=2, n_draws_per_bin=200, mu_r_centres=np.array([21.0, 25.0]),
                edge_margin_pix=40, host_candidates=2000)


@pytest.fixture(scope="session")
def coadd():
    rng = np.random.default_rng(3)
    shape = (SIZE, SIZE)
    sky_sigma = np.array([4.0, 1.4, 1.2, 2.3, 5.0, 17.0]) * 0.1
    signal = rng.normal(0.0, sky_sigma[:, None, None], (len(BANDS), *shape))
    for x0, y0, flux, width in [(60, 60, 4000, 3.0), (150, 140, 2500, 4.0), (110, 50, 1500, 2.5), (60, 170, 900, 2.0)]:
        signal += flux * gaussian(shape, x0, y0, width)[None]
    variance = np.broadcast_to((sky_sigma ** 2)[:, None, None], signal.shape).astype(np.float32)
    psf = np.repeat(gaussian_psf()[..., None], len(BANDS), axis=-1)
    return Coadd(signal.astype(np.float32), variance.copy(), psf.astype(np.float32), BANDS, (1000, 2000), None, None)


@pytest.fixture(scope="session")
def coadd_path(coadd, tmp_path_factory):
    path = tmp_path_factory.mktemp("coadd") / "coadd.npz"
    save_coadd(path, coadd)
    return path


@pytest.fixture(scope="session")
def nh_dir(tmp_path_factory, cfg):
    """Three template files in the New Horizon layout: a clumpy disc, pre-PSF, ugrizy."""
    folder = tmp_path_factory.mktemp("nh")
    rng = np.random.default_rng(5)
    for k in range(3):
        cube = np.zeros((len(BANDS), 81, 81))
        disc = gaussian((81, 81), 40, 40, 8 + 2 * k)
        for b in range(len(BANDS)):
            cube[b] = disc * (1 + 0.2 * b)
            for _ in range(5):
                cube[b] += 0.05 * gaussian((81, 81), *rng.uniform(25, 55, 2), 1.5)
        with h5py.File(folder / f"Rvir_mocks_V2_{k:05d}.h5", "w") as handle:
            handle.create_dataset(f"{cfg['nh_projection']}/{cfg['nh_snapshot']}/{cfg['nh_redshift_key']}/image_full",
                                  data=cube)
    return folder


@pytest.fixture(scope="session")
def hosts(cfg):
    """Hosts with the properties of COSMOS reference neighbours, scattered over the synthetic coadd."""
    reference = load_reference()
    rng = np.random.default_rng(7)
    picks = reference.sample(n=400, random_state=1)
    return pd.DataFrame(dict(x=rng.uniform(10, SIZE - 10, len(picks)), y=rng.uniform(10, SIZE - 10, len(picks)),
                             z=picks["z_NN"].to_numpy(), logmass=picks["logmass_NN"].to_numpy(),
                             log_Re=picks["log_Re_NN"].to_numpy(), mu_r=picks["mu_r_NN"].to_numpy()))
