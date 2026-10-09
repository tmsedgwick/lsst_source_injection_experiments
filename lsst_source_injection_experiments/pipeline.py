"""The steps of an injection run, each reading and writing files in one run folder:

  draw_injections  -> injections.parquet      (targets and positions; the source type is chosen here)
  inject           -> injected/inj_NNNNN.npz  (the light each injection adds to the coadd), run.json, and in calexp
                      mode baseline_coadd.npz (the exposures recombined without injections)
  recover          -> recovery.parquet, peaks.parquet
  deblend          -> deblend.parquet
  summarise / plot -> summary.csv, detection_and_deblending_vs_mu_r.png

Every injection is analysed on its own: the image it is compared with is the coadd (or the recombined exposures)
without it, and nothing else is injected at the same time.
"""

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BANDS, MODES, SOURCES
from .cosmos import draw_properties, fit_colour_models, load_reference
from .deblending import deblend_injection
from .images import (Coadd, Patch, add_patch, coadd_injection, cutout, exposure_injection, load_coadd, load_exposures,
                     recombine, save_coadd)
from .placement import place_random, place_targeted
from .recovery import recover
from .sources import (nh_template_paths, prepare_doublesersicbcg, prepare_newhorizon, read_nh_cube, render,
                      render_band)

RUN_FILE, INJECTIONS_FILE, BASELINE_FILE = "run.json", "injections.parquet", "baseline_coadd.npz"
RECOVERY_FILE, PEAKS_FILE, DEBLEND_FILE = "recovery.parquet", "peaks.parquet", "deblend.parquet"


def patch_path(run_dir, injection_id):
    return Path(run_dir) / "injected" / f"inj_{int(injection_id):05d}.npz"


# --------------------------------------------------------------------------------------------------- draw

def draw_injections(source, coadd, cfg, hosts=None, nh_dir=None, reference=None):
    """Targets and positions of every injection: per surface-brightness bin, n_injections_per_bin of them."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    rng = np.random.default_rng(cfg["seed"])
    tables = []
    for centre in cfg["mu_r_centres"]:
        if source == "newhorizon":
            draws = draw_properties(reference if reference is not None else load_reference(), centre,
                                    cfg["mu_r_half_width"], cfg["n_draws_per_bin"], rng)
            if draws is None:
                print(f"mu_r {centre:.1f}: no COSMOS galaxies in this bin, skipped")
                continue
            placed = place_targeted(draws, hosts, coadd.shape, cfg["n_injections_per_bin"], rng, cfg)
            if placed.empty:
                print(f"mu_r {centre:.1f}: no draw could be placed next to a matching host, skipped")
                continue
            templates = nh_template_paths(nh_dir, cfg)
            placed = placed.assign(Re_arcsec=10 ** placed["log_Re"],
                                   template=[templates[i].name for i in rng.integers(len(templates), size=len(placed))])
        else:
            n = cfg["n_injections_per_bin"]
            low, high = np.log10(cfg["bcg_re_arcsec"])
            placed = place_random(n, coadd.shape, rng, cfg).assign(
                mu_r=rng.uniform(centre - cfg["mu_r_half_width"], centre + cfg["mu_r_half_width"], n),
                Re_arcsec=10 ** rng.uniform(low, high, n))
        tables.append(placed.assign(mu_r_bin=float(centre), source=source))
        print(f"mu_r {centre:.1f}: {len(placed)} injections")
    injections = pd.concat(tables, ignore_index=True)
    return injections.assign(injection_id=np.arange(len(injections)))


# --------------------------------------------------------------------------------------------------- inject

def prepare(row, coadd, cfg, nh_dir=None, colour_models=None):
    """The PreparedSource of one injection row, sized and scaled with the coadd's PSF."""
    target = dict(mu_r=row["mu_r"], Re_arcsec=max(1e-3, row["Re_arcsec"]))
    if row["source"] == "newhorizon":
        if nh_dir is None:
            raise ValueError("New Horizon injections need the template folder (nh_dir)")
        target.update(ur=row["ur"], z=row["z"], logmass=row["logmass"])
        cube = read_nh_cube(Path(nh_dir).expanduser() / row["template"], cfg)
        return prepare_newhorizon(cube, coadd.psf_kernels, target, colour_models, cfg)
    rng = np.random.default_rng([cfg["seed"], int(row["injection_id"])])
    return prepare_doublesersicbcg(coadd.psf_kernels, target, rng, cfg)


_INJECT: dict[str, Any] = {}


def _load_inject_inputs(coadd_path, exposure_paths, need_colours):
    _INJECT.update(coadd=load_coadd(coadd_path), exposures=load_exposures(exposure_paths) if exposure_paths else None,
                   colour_models=fit_colour_models(load_reference()) if need_colours else None)


def _inject_one(args):
    row, mode, run_dir, cfg, nh_dir = args
    coadd, exposures = _INJECT["coadd"], _INJECT["exposures"]
    prepared = prepare(row, coadd, cfg, nh_dir, _INJECT["colour_models"])
    if mode == "deepcoadd":
        patch = coadd_injection(coadd, render(prepared, coadd.psf_kernels), row["x"], row["y"])
    else:
        size = max(prepared.cube.shape[-2:]) + 2 * coadd.psf_kernels.shape[0]
        patch = exposure_injection(coadd, exposures, lambda k, kernel: render_band(prepared, k, kernel),
                                   row["x"], row["y"], size)
    if patch is not None:
        np.savez_compressed(patch_path(run_dir, row["injection_id"]), x0=patch.x0, y0=patch.y0,
                            signal=patch.signal, variance=patch.variance)
    return dict(row, injected=patch is not None, **{f"flux_{b}": f for b, f in zip(BANDS, prepared.fluxes)},
                **prepared.info)


def inject(injections, coadd_path, mode, run_dir, cfg, exposure_paths=None, nh_dir=None, n_workers=1):
    """Inject every row (each on its own) into the coadd, or into the exposures and recombine; write the added light
    of each, the run description and, in calexp mode, the recombined coadd without injections."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "calexp" and not exposure_paths:
        raise ValueError("calexp mode needs exposures")
    run_dir = Path(run_dir)
    (run_dir / "injected").mkdir(parents=True, exist_ok=True)
    exposure_paths = [str(p) for p in exposure_paths or []] if mode == "calexp" else None
    need_colours = bool((injections["source"] == "newhorizon").any())
    _load_inject_inputs(coadd_path, exposure_paths, need_colours)
    coadd = _INJECT["coadd"]
    if mode == "deepcoadd" and coadd.n_eff is None:
        print("The coadd has no n_eff (effective number of visits per band): the injected light's shot noise is added "
              "to the variance as for a single exposure")
    baseline_path = Path(coadd_path).resolve()
    if mode == "calexp":
        baseline_path = run_dir / BASELINE_FILE
        save_coadd(baseline_path, recombine(coadd, _INJECT["exposures"]))
    tasks = [(row, mode, run_dir, cfg, nh_dir) for _, row in injections.iterrows()]
    if n_workers <= 1:
        rows = [_inject_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn"), initializer=_load_inject_inputs,
                                 initargs=(coadd_path, exposure_paths, need_colours)) as pool:
            rows = list(pool.map(_inject_one, tasks, chunksize=2))
    injected = pd.DataFrame(rows)
    injected.to_parquet(run_dir / INJECTIONS_FILE)
    description = dict(mode=mode, source=str(injections["source"].iat[0]), coadd=str(Path(coadd_path).resolve()),
                       baseline=str(baseline_path), exposures=exposure_paths or [],
                       nh_dir=str(Path(nh_dir).expanduser().resolve()) if nh_dir else None)
    (run_dir / RUN_FILE).write_text(json.dumps(description, indent=2))
    return injected


# --------------------------------------------------------------------------------------- recover / deblend

_BASELINE: Coadd | None = None


def _baseline():
    """The image injections are compared with, loaded by _load_baseline in this process."""
    if _BASELINE is None:
        raise RuntimeError("The baseline coadd is not loaded")
    return _BASELINE


def _cutouts(run_dir, row, cfg):
    """(before signal, before variance, after signal, after variance, x, y) of one injection's cutout."""
    with np.load(patch_path(run_dir, row["injection_id"])) as data:
        patch = Patch(int(data["x0"]), int(data["y0"]), data["signal"], data["variance"])
    x0, y0, signal = cutout(_baseline().signal, row["x"], row["y"], cfg["half_box"])
    _, _, variance = cutout(_baseline().variance, row["x"], row["y"], cfg["half_box"])
    return (signal, variance, add_patch(signal, x0, y0, patch, "signal"), add_patch(variance, x0, y0, patch, "variance"),
            row["x"] - x0, row["y"] - y0)


def _recover_one(args):
    run_dir, row, cfg = args
    before_s, before_v, after_s, after_v, x, y = _cutouts(run_dir, row, cfg)
    summary, peaks = recover(before_s, before_v, after_s, after_v, _baseline().psf_kernels, x, y, cfg)
    return dict(injection_id=row["injection_id"], **summary), peaks.assign(injection_id=row["injection_id"])


def _deblend_one(args):
    run_dir, row, peaks, cfg = args
    before_s, before_v, after_s, after_v, _, _ = _cutouts(run_dir, row, cfg)
    try:
        metrics = deblend_injection(before_s, before_v, after_s, after_v, _baseline().psf_kernels, peaks, row, cfg)
        return dict(injection_id=row["injection_id"], deblend_error="", **metrics)
    except Exception as error:  # a failed fit is recorded, and the other injections still run
        return dict(injection_id=row["injection_id"], deblend_error=f"{type(error).__name__}: {error}")


def _load_baseline(run_dir):
    global _BASELINE
    _BASELINE = load_coadd(json.loads((Path(run_dir) / RUN_FILE).read_text())["baseline"])


def _map(function, tasks, run_dir, n_workers):
    """function over tasks, in n_workers fresh processes that each load the baseline once (forking is not safe with
    the deblender's numerical libraries on macOS)."""
    if n_workers <= 1:
        _load_baseline(run_dir)
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn"), initializer=_load_baseline,
                             initargs=(run_dir,)) as pool:
        return list(pool.map(function, tasks, chunksize=4))


def recover_all(run_dir, cfg, n_workers=1):
    run_dir = Path(run_dir)
    injections = pd.read_parquet(run_dir / INJECTIONS_FILE)
    injections = injections[injections["injected"]]
    results = _map(_recover_one, [(run_dir, row, cfg) for _, row in injections.iterrows()], run_dir, n_workers)
    recovery = injections.merge(pd.DataFrame([r[0] for r in results]), on="injection_id")
    recovery.to_parquet(run_dir / RECOVERY_FILE)
    pd.concat([r[1] for r in results], ignore_index=True).to_parquet(run_dir / PEAKS_FILE)
    return recovery


def deblend_all(run_dir, cfg, n_workers=1, batch=24):
    """Deblend every recovered injection, saving deblend.parquet after each batch; injections already in it are
    skipped, so an interrupted run carries on where it stopped."""
    run_dir = Path(run_dir)
    recovery = pd.read_parquet(run_dir / RECOVERY_FILE)
    peaks = pd.read_parquet(run_dir / PEAKS_FILE)
    done = pd.read_parquet(run_dir / DEBLEND_FILE) if (run_dir / DEBLEND_FILE).exists() else pd.DataFrame()
    todo = recovery.loc[~recovery["injection_id"].isin(done["injection_id"] if len(done) else [])]
    rows_of = peaks.groupby("injection_id").indices  # injection id -> row positions of its peaks
    tasks = [(run_dir, row, peaks.iloc[rows_of[injection_id]], cfg)
             for injection_id, (_, row) in zip(todo["injection_id"].to_numpy(), todo.iterrows())]
    for start in range(0, len(tasks), batch):
        done = pd.concat([done, pd.DataFrame(_map(_deblend_one, tasks[start:start + batch], run_dir, n_workers))],
                         ignore_index=True)
        done.to_parquet(run_dir / DEBLEND_FILE)
        print(f"deblended {len(done)} of {len(recovery)}", flush=True)
    return done


def load_results(run_dir):
    """Per-injection table: injection, recovery and (if run) deblending columns."""
    run_dir = Path(run_dir)
    table = pd.read_parquet(run_dir / RECOVERY_FILE)
    if (run_dir / DEBLEND_FILE).exists():
        table = table.merge(pd.read_parquet(run_dir / DEBLEND_FILE), on="injection_id", how="left")
    return table
