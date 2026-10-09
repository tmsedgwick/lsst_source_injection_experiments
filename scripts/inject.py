"""Draw injections and inject them, each on its own, into a deep coadd or into the exposures it is built from.

    # New Horizon dwarfs placed next to matching real objects, injected into the deep coadd
    python scripts/inject.py --source newhorizon --mode deepcoadd --coadd coadd.npz --hosts hosts.parquet \
        --nh-dir ~/Rvir_mocks --run-dir runs/nh_coadd
    # core-Sersic BCGs at random positions, injected into each exposure and recombined
    python scripts/inject.py --source doublesersicbcg --mode calexp --coadd coadd.npz \
        --exposures visits/*.npz --run-dir runs/bcg_calexp

Per surface-brightness bin (0.5 mag/arcsec^2 from 18 to 30) --n-per-bin injections are drawn: New Horizon dwarfs take
their properties and their nearest neighbour's from the COSMOS2025 joint distribution of that bin and are placed next
to the real object (from --hosts, see build_hosts.py) best matching that neighbour; BCGs are placed at random. In
calexp mode each injection is added to every exposure with that exposure's PSF and the exposures are recombined onto
the coadd's grid; the coadd supplies the grid, WCS and reference PSF.

Writes the run folder that recover.py, deblend.py and plot_performance.py read.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lsst_source_injection_experiments.config import CONFIG, MODES, SOURCES
from lsst_source_injection_experiments.images import load_coadd
from lsst_source_injection_experiments.pipeline import INJECTIONS_FILE, draw_injections, inject


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=SOURCES, required=True, help="what to inject")
    parser.add_argument("--mode", choices=MODES, required=True,
                        help="deepcoadd: into the coadd; calexp: into each exposure, then recombined")
    parser.add_argument("--coadd", type=Path, required=True, help="the coadd (.npz)")
    parser.add_argument("--exposures", type=Path, nargs="+", help="calexp mode: the exposures (.npz)")
    parser.add_argument("--hosts", type=Path, help="newhorizon: real objects to place injections next to")
    parser.add_argument("--nh-dir", type=Path, help="newhorizon: folder of New Horizon templates (Rvir_mocks_V2_*.h5)")
    parser.add_argument("--run-dir", type=Path, required=True, help="output folder")
    parser.add_argument("--n-per-bin", type=int, default=CONFIG["n_injections_per_bin"],
                        help="injections per surface-brightness bin (default %(default)s)")
    parser.add_argument("--mu-r-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="only the bins with centres in this range (default 18 to 30)")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()
    if args.source == "newhorizon" and (args.hosts is None or args.nh_dir is None):
        parser.error("--source newhorizon needs --hosts and --nh-dir")
    if args.mode == "calexp" and not args.exposures:
        parser.error("--mode calexp needs --exposures")

    cfg = dict(CONFIG, n_injections_per_bin=args.n_per_bin, seed=args.seed)
    if args.mu_r_range:
        centres = cfg["mu_r_centres"]
        cfg["mu_r_centres"] = centres[(centres >= args.mu_r_range[0]) & (centres <= args.mu_r_range[1])]
    coadd = load_coadd(args.coadd)
    hosts = pd.read_parquet(args.hosts) if args.hosts else None
    injections = draw_injections(args.source, coadd, cfg, hosts=hosts, nh_dir=args.nh_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    injected = inject(injections, args.coadd, args.mode, args.run_dir, cfg, exposure_paths=args.exposures,
                      nh_dir=args.nh_dir, n_workers=args.n_workers)
    print(f"{int(injected['injected'].sum())} of {len(injected)} injections written to {args.run_dir} "
          f"({INJECTIONS_FILE}, injected/); median observed Re {np.nanmedian(injected['re_measured_arcsec']):.2f}\"")


if __name__ == "__main__":
    main()
