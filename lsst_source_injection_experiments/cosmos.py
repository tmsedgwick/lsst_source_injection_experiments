"""COSMOS2025 reference population: joint draws of a galaxy's properties and its nearest neighbour's, the colour models
that complete an injected galaxy's SED, and the photometric model that gives real LSST objects a redshift and stellar
mass (on the same footing as the COSMOS masses the drawn neighbours have).

The tables are built from the full COSMOS2025 catalogue by scripts/build_cosmos_reference.py and shipped in data/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REFERENCE_PATH = DATA_DIR / "cosmos2025_injection_reference.parquet"
PHOTOMETRIC_PATH = DATA_DIR / "cosmos2025_photometric_training.parquet"

MAG_COLUMNS = [f"m_{band}" for band in "ugrizy"]
# The joint distribution drawn per surface-brightness bin: the galaxy, its nearest neighbour and their separation
# (in neighbour half-light radii and in arcsec).
PROPERTIES = ["logmass", "z", "ur", "log_Re", "mu_r"]
DRAW_COLUMNS = PROPERTIES + [f"{p}_NN" for p in PROPERTIES] + ["log_d_NN_radii", "log_d_NN_arcsecs"]
COLOUR_TARGETS = dict(u_g=("m_u", "m_g"), g_i=("m_g", "m_i"), i_z=("m_i", "m_z"), z_y=("m_z", "m_y"))


def load_reference(path=REFERENCE_PATH):
    return pd.read_parquet(path)


def load_photometric_training(path=PHOTOMETRIC_PATH):
    return pd.read_parquet(path)


def draw_properties(reference, mu_centre, half_width, n, rng):
    """n joint draws of DRAW_COLUMNS for galaxies with mu_r within mu_centre +- half_width: a Gaussian KDE in
    standardised coordinates (Scott's bandwidth), or, with too few galaxies for a KDE, the galaxies themselves
    jittered by Scott's factor. Returns None if the bin is empty."""
    in_bin = reference["mu_r"].between(mu_centre - half_width, mu_centre + half_width, inclusive="left")
    values = reference.loc[in_bin, DRAW_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(values) == 0:
        return None
    n_dim = values.shape[1]
    if len(values) == 1:
        everyone = reference[DRAW_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna()
        scott = len(everyone) ** (-1.0 / (n_dim + 4.0))
        draws = values[0] + rng.normal(scale=everyone.std(ddof=0).to_numpy() * scott, size=(n, n_dim))
        return pd.DataFrame(draws, columns=DRAW_COLUMNS)
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    draws = None
    if len(values) > n_dim:
        try:
            draws = gaussian_kde(scaled.T, bw_method="scott").resample(n, seed=rng).T
        except np.linalg.LinAlgError:
            draws = None
    if draws is None:
        scott = len(values) ** (-1.0 / (n_dim + 4.0))
        draws = scaled[rng.integers(len(values), size=n)] + rng.normal(scale=scott, size=(n, n_dim))
    return pd.DataFrame(scaler.inverse_transform(draws), columns=DRAW_COLUMNS)


def fit_colour_models(reference):
    """{u_g, g_i, i_z, z_y: model} predicting each colour from (z, logmass, u - r): 10 nearest neighbours, weighted
    by distance in standardised coordinates."""
    table = reference.dropna(subset=["z", "logmass", "ur", *MAG_COLUMNS])
    features = table[["z", "logmass", "ur"]].to_numpy(float)
    models = {}
    for name, (blue, red) in COLOUR_TARGETS.items():
        model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance"))
        models[name] = model.fit(features, (table[blue] - table[red]).to_numpy(float))
    return models


def predict_colours(models, z, logmass, u_minus_r):
    features = np.array([[z, logmass, u_minus_r]], float)
    return {name: float(model.predict(features)[0]) for name, model in models.items()}


def fit_photometric(training, seed=0):
    """Random forest predicting (z, log M*) from the six magnitudes and the half-light radius (arcsec)."""
    table = training.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*MAG_COLUMNS, "half_light_radius_arcsec", "z", "logmass"])
    forest = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
    return forest.fit(table[MAG_COLUMNS + ["half_light_radius_arcsec"]].to_numpy(np.float32),
                      table[["z", "logmass"]].to_numpy())


def predict_photometric(forest, table):
    """(z, log M*) for each row of a table with m_<band> and half_light_radius_arcsec; NaN where an input is missing."""
    features = table[MAG_COLUMNS + ["half_light_radius_arcsec"]].to_numpy(np.float32)
    good = np.isfinite(features).all(axis=1)
    out = np.full((len(table), 2), np.nan)
    if good.any():
        out[good] = forest.predict(features[good])
    return out[:, 0], out[:, 1]
