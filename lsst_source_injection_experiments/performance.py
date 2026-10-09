"""Detection fractions and flux biases per surface-brightness bin, and the three-panel figure of DMTN-335
(Figure: detection and deblending performance vs surface brightness)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .photometry import flux_bias_to_dmag  # noqa: E402

TITLES = {
    "newhorizon": ("Targeted New Horizon galaxy injection", "Galaxy properties sampled from COSMOS-2025 PDF"),
    "doublesersicbcg": ("Double-Sérsic BCG injection", "Core-Sérsic BCG + envelope, Re log-uniform"),
}


def _median(values):
    values = np.asarray(values, float)
    return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan


def _mad(values):
    values = np.asarray(values, float)
    return float(np.nanmedian(np.abs(values - np.nanmedian(values)))) if np.isfinite(values).any() else np.nan


def _percent(flags):
    flags = np.asarray(flags, float)
    return 100.0 * float(np.nanmean(flags)) if np.isfinite(flags).any() else np.nan


def summarise(injections):
    """One row per mu_r bin from the per-injection table (injection, recovery and, if run, deblending columns)."""
    rows = []
    for centre, group in injections.groupby("mu_r_bin"):
        detected = group["detected"].to_numpy(bool)
        nn_defined = group["nn_exists"].to_numpy(bool)
        row = dict(mu_r_bin=centre, n_injections=len(group),
                   det_frac_inj_percent=_percent(detected),
                   det_frac_nn_percent=_percent(group.loc[nn_defined, "nn_recovered"].astype(float)),
                   det_frac_rest_percent=_percent(group["rest_recovered_fraction"]))
        if "inj_bias_r" in group:
            nn_detected, nn_missed = group[detected], group[~detected]
            row.update(
                med_bias_r=_median(group.loc[detected, "inj_bias_r"]), mad_bias_r=_mad(group.loc[detected, "inj_bias_r"]),
                med_bias_total=_median(group.loc[detected, "inj_bias_total"]),
                med_mu_r_measured=_median(group.loc[detected, "inj_mu_r_measured"]),
                med_nn_bias_r_det=_median(nn_detected["nn_bias_r"]), mad_nn_bias_r_det=_mad(nn_detected["nn_bias_r"]),
                med_nn_bias_r_miss=_median(nn_missed["nn_bias_r"]), mad_nn_bias_r_miss=_mad(nn_missed["nn_bias_r"]),
                med_rest_bias_r=_median(group["rest_bias_r"]), mad_rest_bias_r=_mad(group["rest_bias_r"]),
                med_rest_bias_total=_median(group["rest_bias_total"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mu_r_bin").reset_index(drop=True)


def plot_performance(summary, source, out_path):
    """The three panels: detection fractions; r-band flux bias (%, symlog); r-band surface brightness bias."""
    x = summary["mu_r_bin"].to_numpy(float)
    column = lambda name: summary[name].to_numpy(float) if name in summary else np.full(len(x), np.nan)
    c_inj, c_nn, c_rest = "tab:blue", "tab:orange", "tab:green"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

    ax1.plot(x, column("det_frac_inj_percent"), lw=2, label="Injected source", color=c_inj)
    ax1.plot(x, column("det_frac_nn_percent"), lw=2, label="Nearest neighbour", color=c_nn)
    ax1.plot(x, column("det_frac_rest_percent"), lw=2, label="Rest-of-blend", color=c_rest)
    ax1.set_xlabel(r"$\mu_{r}$ (mag arcsec$^{-2}$)")
    ax1.set_ylabel("Detection fraction (%)")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, loc="best")
    ax1.set_xlim(18, 30)
    ax1.set_ylim(-1, 101)

    biases = [("med_bias_r", "Injected (when detected)", c_inj, "-"),
              ("med_nn_bias_r_det", "NN (when inj detected)", c_nn, "-"),
              ("med_nn_bias_r_miss", "NN (when inj missed)", c_nn, "--"),
              ("med_rest_bias_r", "Rest-of-blend", c_rest, "-")]
    for name, label, colour, style in biases:
        ax2.plot(x, column(name), lw=2, label=label, color=colour, linestyle=style)
    ax2.axhline(0, color="k", linestyle=":", lw=1)
    ax2.set_xlabel(r"$\mu_{r}$ (mag arcsec$^{-2}$)")
    ax2.set_ylabel("r-band flux bias (%)")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False, loc="best")
    linthresh = 5.0
    ax2.set_yscale("symlog", linthresh=linthresh, base=10)
    all_bias = np.concatenate([column(name) for name, *_ in biases])
    finite = all_bias[np.isfinite(all_bias)]
    if finite.size:
        vmax = max(np.nanpercentile(np.abs(finite), 99) * 1.2, linthresh * 2.0, 1.0)
        ax2.set_ylim(-vmax, vmax)
    else:
        ax2.set_ylim(-200, 200)

    dmu = []
    for name, label, colour, style in biases:
        dmu.append(flux_bias_to_dmag(column(name)))
        ax3.plot(x, dmu[-1], lw=2, label=label, color=colour, linestyle=style)
    ax3.axhline(0, color="k", linestyle=":", lw=1)
    ax3.set_xlabel(r"$\mu_{r}$ (mag arcsec$^{-2}$)")
    ax3.set_ylabel(r"$\Delta \mu_{r}$ (mag arcsec$^{-2}$)")
    ax3.grid(alpha=0.3)
    ax3.legend(frameon=False, loc="best")
    ax3.set_xlim(18, 30)
    finite = np.concatenate(dmu)[np.isfinite(np.concatenate(dmu))]
    if finite.size:
        lo, hi = np.nanpercentile(finite, [1, 99])
        pad = 0.15 * max(hi - lo, 0.2)
        ax3.set_ylim(lo - pad, hi + pad)
    else:
        ax3.set_ylim(-3, 0.5)

    method, properties = TITLES[source]
    n = int(np.median(summary["n_injections"])) if len(summary) else 0
    fig.suptitle(f"Detection and Deblending Performance vs Surface Brightness\n{method} – {n} iterations\n{properties}",
                 fontsize=14)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
