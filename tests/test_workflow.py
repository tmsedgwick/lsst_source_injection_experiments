"""The whole workflow on the synthetic coadd: draws, placement, injection (both sources, both modes), recovery,
deblending, the per-bin summary and the figure."""

import numpy as np
import pandas as pd
import pytest

from lsst_source_injection_experiments.config import BANDS
from lsst_source_injection_experiments.cosmos import DRAW_COLUMNS, draw_properties, load_reference
from lsst_source_injection_experiments.images import Exposure, save_exposure
from lsst_source_injection_experiments.performance import plot_performance, summarise
from lsst_source_injection_experiments.pipeline import deblend_all, draw_injections, inject, load_results, recover_all


def test_draws_stay_in_their_surface_brightness_bin():
    draws = draw_properties(load_reference(), 24.0, 0.25, 500, np.random.default_rng(0))
    assert draws is not None
    assert list(draws.columns) == DRAW_COLUMNS and len(draws) == 500
    assert draws["mu_r"].between(23.5, 24.5).mean() > 0.9


def test_targeted_placement_puts_each_injection_next_to_its_host(coadd, hosts, nh_dir, cfg):
    injections = draw_injections("newhorizon", coadd, cfg, hosts=hosts, nh_dir=nh_dir)
    assert len(injections) > 0
    host_xy = hosts[["x", "y"]].to_numpy()
    for _, row in injections.iterrows():
        distances = np.hypot(host_xy[:, 0] - row["x"], host_xy[:, 1] - row["y"])
        assert np.argmin(distances) == row["host"]
        assert abs(row["z_NN"] - hosts["z"].iat[row["host"]]) <= cfg["host_max_dz"]
    assert injections["template"].str.startswith("Rvir_mocks_V2_").all()


@pytest.mark.parametrize("source, mode, n_workers", [("newhorizon", "deepcoadd", 1), ("doublesersicbcg", "calexp", 2)])
def test_workflow(source, mode, n_workers, coadd, coadd_path, hosts, nh_dir, cfg, tmp_path):
    injections = draw_injections(source, coadd, cfg, hosts=hosts, nh_dir=nh_dir)
    exposure_paths = []
    if mode == "calexp":
        for copy in range(2):
            for k, band in enumerate(BANDS):
                path = tmp_path / "exposures" / f"copy{copy}_{band}.npz"
                save_exposure(path, Exposure(coadd.signal[k], coadd.variance[k] * 2, coadd.psf_kernels[..., k], band,
                                             coadd.origin))
                exposure_paths.append(path)
    injected = inject(injections, coadd_path, mode, tmp_path / "run", cfg, exposure_paths=exposure_paths,
                      nh_dir=nh_dir, n_workers=n_workers)
    tmp_path = tmp_path / "run"
    assert injected["injected"].to_numpy(bool).all()
    assert (tmp_path / "injected").exists()
    recovery = recover_all(tmp_path, cfg, n_workers)
    assert recovery["detected"].to_numpy(bool).any()
    deblend = deblend_all(tmp_path, cfg, n_workers)
    assert (deblend["deblend_error"] == "").all()
    results = load_results(tmp_path)
    summary = summarise(results)
    assert list(summary["mu_r_bin"]) == list(cfg["mu_r_centres"])
    assert {"det_frac_inj_percent", "med_bias_r", "med_rest_bias_r"} <= set(summary.columns)
    out = tmp_path / "figure.png"
    plot_performance(summary, source, out)
    assert out.stat().st_size > 10_000


def test_summary_fractions(tmp_path):
    table = pd.DataFrame(dict(mu_r_bin=[20.0] * 4, detected=[True, True, False, True], nn_exists=[True] * 3 + [False],
                              nn_recovered=[True, False, True, False], rest_recovered_fraction=[1.0, 0.5, np.nan, 1.0]))
    summary = summarise(table)
    assert summary["det_frac_inj_percent"].iat[0] == pytest.approx(75.0)
    assert summary["det_frac_nn_percent"].iat[0] == pytest.approx(200 / 3)
    assert summary["det_frac_rest_percent"].iat[0] == pytest.approx(250 / 3)
