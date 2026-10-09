"""Deblend each injection's blend with scarlet lite, before and after injecting, and measure flux biases.

    python scripts/deblend.py --run-dir runs/nh_coadd --n-workers 4

Run after recover.py. Writes deblend.parquet: the injected source's r-band and total flux bias and measured surface
brightness (when detected), the nearest neighbour's flux bias (when recovered) and the rest-of-blend flux bias. See
lsst_source_injection/deblending.py for the definitions.
"""

import argparse
from pathlib import Path

from lsst_source_injection.config import CONFIG
from lsst_source_injection.pipeline import deblend_all


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="folder written by inject.py and recover.py")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()
    deblend = deblend_all(args.run_dir, CONFIG, args.n_workers)
    failed = deblend["deblend_error"].astype(bool)
    print(f"Deblended {int((~failed).sum())} of {len(deblend)} injections"
          + (f"; {int(failed.sum())} failed, e.g. {deblend.loc[failed, 'deblend_error'].iat[0]}" if failed.any() else ""))


if __name__ == "__main__":
    main()
