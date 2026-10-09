"""The real objects New Horizon injections are placed next to, with redshift, stellar mass, size and surface brightness.

    # from the LSST object table (exported on the RSP by export_from_butler.py objects)
    python scripts/build_hosts.py --objects objects.parquet --out hosts.parquet
    # without an object table: detect and measure the objects on the coadd itself
    python scripts/build_hosts.py --coadd coadd.npz --out hosts.parquet

Objects need x, y (coadd array pixels), m_u .. m_y, half_light_radius_arcsec and mu_r; redshifts and stellar masses
come from a random forest trained on COSMOS2025 magnitudes and sizes.
"""

import argparse
from pathlib import Path

import pandas as pd

from lsst_source_injection.config import CONFIG
from lsst_source_injection.cosmos import fit_photometric, load_photometric_training
from lsst_source_injection.hosts import add_redshift_and_mass, measure_objects
from lsst_source_injection.images import load_coadd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--objects", type=Path, help="object table (.parquet) from export_from_butler.py objects")
    source.add_argument("--coadd", type=Path, help="measure the objects on this coadd (.npz) instead")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    objects = pd.read_parquet(args.objects) if args.objects else measure_objects(load_coadd(args.coadd), CONFIG)
    hosts = add_redshift_and_mass(objects, fit_photometric(load_photometric_training(), CONFIG["seed"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hosts.to_parquet(args.out)
    print(f"{len(hosts):,} hosts of {len(objects):,} objects -> {args.out}")
    print(hosts[["z", "logmass", "log_Re", "mu_r"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
