"""Was each injected source recovered by peak finding, and were its nearest neighbour and the rest of its blend?

    python scripts/recover.py --run-dir runs/nh_coadd --n-workers 4

Detects peaks in each injection's cutout before and after injecting (S/N >= 5 in the PSF-smoothed r band, noise
from the variance plane) and writes recovery.parquet (one row per injection) and peaks.parquet (the peaks, with
their roles and matches, used by deblend.py). See lsst_source_injection/recovery.py for the definitions.
"""

import argparse
from pathlib import Path

from lsst_source_injection.config import CONFIG
from lsst_source_injection.pipeline import recover_all


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="folder written by inject.py")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()
    recovery = recover_all(args.run_dir, CONFIG, args.n_workers)
    by_bin = recovery.groupby("mu_r_bin").agg(n=("detected", "size"), injected=("detected", "mean"),
                                             nearest_neighbour=("nn_recovered", "mean"),
                                             rest_of_blend=("rest_recovered_fraction", "mean"))
    print("Recovered fractions per mu_r bin:")
    print(by_bin.round(3).to_string())


if __name__ == "__main__":
    main()
