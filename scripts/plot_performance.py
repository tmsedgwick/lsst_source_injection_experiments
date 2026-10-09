"""Detection and deblending performance vs surface brightness: the three-panel figure of DMTN-335.

    python scripts/plot_performance.py --run-dir runs/nh_coadd

Left: detection fraction of the injected source, its nearest neighbour and the rest of its blend. Centre: median
r-band flux bias (%) of the injected source (when detected), the nearest neighbour (when the injection was detected or
missed) and the rest of the blend. Right: the same as a surface brightness bias. Writes summary.csv and
detection_and_deblending_vs_mu_r.png in the run folder (or --out).
"""

import argparse
from pathlib import Path

from lsst_source_injection.performance import plot_performance, summarise
from lsst_source_injection.pipeline import load_results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="folder with recovery.parquet (and deblend.parquet)")
    parser.add_argument("--out", type=Path, help="figure path (default: in the run folder)")
    args = parser.parse_args()
    results = load_results(args.run_dir)
    summary = summarise(results)
    summary.to_csv(args.run_dir / "summary.csv", index=False)
    out = args.out or args.run_dir / "detection_and_deblending_vs_mu_r.png"
    plot_performance(summary, results["source"].iat[0], out)
    print(summary.round(2).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
