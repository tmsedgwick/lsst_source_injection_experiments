"""Stand-in exposures for testing calexp mode without calexps: n copies of a coadd, each with n times its variance.

    python scripts/make_test_exposures.py --coadd coadd.npz --n 4 --out-dir test_exposures

Each copy is written as six single-band exposures on the coadd's own pixel grid. Recombining the n copies gives back
the coadd (signal and variance), so injecting into them must give the same result as injecting into the coadd, which
tests the injection into exposures and their recombination. For real data, export the visits with
export_from_butler.py visits and pass those files instead.
"""

import argparse
from pathlib import Path

from lsst_source_injection.config import BANDS
from lsst_source_injection.images import Exposure, load_coadd, save_exposure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coadd", type=Path, required=True)
    parser.add_argument("--n", type=int, default=4, help="number of copies (default %(default)s)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    coadd = load_coadd(args.coadd)
    for copy in range(args.n):
        for k, band in enumerate(BANDS):
            exposure = Exposure(coadd.signal[k], coadd.variance[k] * args.n, coadd.psf_kernels[..., k], band,
                                coadd.origin, coadd.wcs, f"copy{copy}_{band}")
            save_exposure(args.out_dir / f"exposure_copy{copy}_{band}.npz", exposure)
    print(f"{args.n * len(BANDS)} exposures -> {args.out_dir}")


if __name__ == "__main__":
    main()
