"""Build the COSMOS2025 tables shipped in data/ from the full catalogue (cosmos2025_cat.npz).

    python scripts/build_cosmos_reference.py --catalogue ~/cosmos2025_cat.npz

Writes
  * data/cosmos2025_injection_reference.parquet: galaxies at 0 < z < max-z, each with its nearest neighbour (the
    closest other galaxy on the sky brighter than neighbour-max-mag-r): their stellar mass, redshift, u - r, half-light
    radius and surface brightness, and their separation. These are the joint distributions injections are drawn from.
  * data/cosmos2025_photometric_training.parquet: magnitudes, half-light radius, redshift and stellar mass of galaxies at
    all redshifts, to give real LSST objects a photometric redshift and stellar mass.

Galaxies are LePhare type 0 objects not flagged as stars, with a finite position, redshift, mass, Sersic radius and
ugrizy magnitudes. The surface brightness is mu_r = m_r + 2.5 log10(pi Re^2) (see lsst_source_injection.config).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from lsst_source_injection.cosmos import PHOTOMETRIC_PATH, REFERENCE_PATH

MAG_SOURCES = {"m_u": "mag_model_cfht-u", "m_g": "mag_model_hsc-g", "m_r": "mag_model_hsc-r",
               "m_i": "mag_model_hsc-i", "m_z": "mag_model_hsc-z", "m_y": "mag_model_hsc-y"}
SENTINELS = (-999.0, -998.0, -99.9, -99.0, 998.0, 999.0)


def read_galaxies(path):
    columns = ["ra", "dec", "zfinal", "type", "flag_star", "mass_med", "radius_sersic", *MAG_SOURCES.values()]
    with np.load(path, allow_pickle=True) as npz:
        table = pd.DataFrame({column: npz[column] for column in columns})
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            values = table[column].to_numpy(float)
            bad = np.isin(np.round(values, 3), SENTINELS)
            table[column] = np.where(bad, np.nan, values)
    galaxies = pd.DataFrame(dict(ra=table["ra"], dec=table["dec"], z=table["zfinal"], logmass=table["mass_med"],
                                 half_light_radius_arcsec=table["radius_sersic"] * 3600.0))
    for name, source in MAG_SOURCES.items():
        galaxies[name] = table[source]
    keep = (table["type"] == 0) & (table["flag_star"] == 0)
    galaxies = galaxies[keep].replace([np.inf, -np.inf], np.nan).dropna()
    galaxies = galaxies[galaxies["half_light_radius_arcsec"] > 0].reset_index(drop=True)
    galaxies["ur"] = galaxies["m_u"] - galaxies["m_r"]
    galaxies["log_Re"] = np.log10(galaxies["half_light_radius_arcsec"])
    galaxies["mu_r"] = galaxies["m_r"] + 2.5 * np.log10(np.pi * galaxies["half_light_radius_arcsec"] ** 2)
    return galaxies


def nearest_neighbours(sample, neighbours):
    """Index into neighbours of each sample galaxy's nearest other neighbour on the sky, and the distance (arcsec)."""
    dec0 = np.radians(neighbours["dec"].mean())
    plane = lambda t: np.column_stack([t["ra"] * np.cos(dec0), t["dec"]]) * 3600.0
    distance, index = cKDTree(plane(neighbours)).query(plane(sample), k=2)
    # the closest hit is the galaxy itself whenever it is in the neighbour pool
    itself = distance[:, 0] < 1e-6
    return np.where(itself, index[:, 1], index[:, 0]), np.where(itself, distance[:, 1], distance[:, 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogue", type=Path, required=True, help="the full COSMOS2025 catalogue (.npz)")
    parser.add_argument("--max-z", type=float, default=0.4, help="injected galaxies' redshift limit (default 0.4)")
    parser.add_argument("--neighbour-max-mag-r", type=float, default=27.5,
                        help="faintest neighbour (r mag; default 27.5, the LSST 10-year depth)")
    parser.add_argument("--training-rows", type=int, default=200_000, help="rows kept for training the photometric redshift and mass model")
    args = parser.parse_args()

    galaxies = read_galaxies(args.catalogue)
    neighbours = galaxies[galaxies["m_r"] <= args.neighbour_max_mag_r].reset_index(drop=True)
    sample = neighbours[(neighbours["z"] > 0) & (neighbours["z"] < args.max_z)].reset_index(drop=True)
    index, distance = nearest_neighbours(sample, neighbours)
    reference = sample.copy()
    for column in ["logmass", "z", "ur", "log_Re", "mu_r", "half_light_radius_arcsec"]:
        reference[f"{column}_NN"] = neighbours[column].to_numpy()[index]
    reference["log_d_NN_arcsecs"] = np.log10(distance)
    reference["log_d_NN_radii"] = np.log10(distance / reference["half_light_radius_arcsec_NN"])
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference.astype({c: np.float32 for c in reference.columns if c not in ("ra", "dec")}).to_parquet(REFERENCE_PATH)

    training = galaxies.sample(n=min(args.training_rows, len(galaxies)), random_state=0)
    columns = ["m_u", "m_g", "m_r", "m_i", "m_z", "m_y", "half_light_radius_arcsec", "z", "logmass"]
    training[columns].astype(np.float32).reset_index(drop=True).to_parquet(PHOTOMETRIC_PATH)
    print(f"{len(galaxies):,} galaxies; {len(neighbours):,} neighbours (r <= {args.neighbour_max_mag_r}); "
          f"{len(reference):,} reference galaxies at z < {args.max_z} -> {REFERENCE_PATH}; "
          f"{len(training):,} photometric training rows -> {PHOTOMETRIC_PATH}")


if __name__ == "__main__":
    main()
