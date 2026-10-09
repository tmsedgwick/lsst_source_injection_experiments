"""Where injections go, in coadd array pixels.

Targeted placement (New Horizon dwarfs, DMTN-335 section 5.5): each draw holds the properties of the injected galaxy
and of its nearest neighbour (NN) and their separation. Real objects (hosts) are ranked by how well they match the
drawn NN (summed absolute differences in log Re, mu_r, z and log M*, each over the median drawn value), hosts outside
the hard cuts are skipped, and the injection is put at the drawn separation (in the host's half-light radii) at a
random position angle, provided the host stays the nearest object to it, the position is inside the image, and it is
at least min_injection_separation_arcsec from earlier injections. A host is used once.

Random placement (BCGs): uniform inside the image.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MATCH_COLUMNS = ["log_Re", "mu_r", "z", "logmass"]


def inside_image(x, y, shape, margin):
    return margin <= x < shape[1] - margin and margin <= y < shape[0] - margin


def place_targeted(draws, hosts, shape, n_wanted, rng, cfg, used_hosts=None, used_positions=None):
    """The first n_wanted draws that can be placed, with x, y and host (index into hosts). used_hosts and
    used_positions (lists, updated in place) carry the hosts and positions taken by earlier bins."""
    used_hosts = set() if used_hosts is None else used_hosts
    used_positions = [] if used_positions is None else used_positions
    host_xy = hosts[["x", "y"]].to_numpy(float)
    tree = cKDTree(host_xy)
    host_values = {c: hosts[c].to_numpy(float) for c in MATCH_COLUMNS}
    host_re = 10 ** host_values["log_Re"]
    scale = {c: max(abs(float(np.median(draws[f"{c}_NN"]))), 1e-9) for c in MATCH_COLUMNS}
    min_sep = cfg["min_injection_separation_arcsec"] / cfg["pixel_scale"]
    margin = cfg["edge_margin_pix"]
    rows = []
    for _, draw in draws.iterrows():
        if len(rows) >= n_wanted:
            break
        mismatch = sum(np.abs(draw[f"{c}_NN"] - host_values[c]) / scale[c] for c in MATCH_COLUMNS)
        candidates = np.argsort(np.nan_to_num(mismatch, nan=np.inf))[:cfg["host_candidates"]]
        placed = None
        for host in candidates:
            if host in used_hosts or not np.isfinite(mismatch[host]):
                continue
            if (abs(draw["z_NN"] - host_values["z"][host]) > cfg["host_max_dz"]
                    or abs(draw["logmass_NN"] - host_values["logmass"][host]) > cfg["host_max_dlogmass"]
                    or abs(draw["mu_r_NN"] - host_values["mu_r"][host]) > cfg["host_max_dmu_r"]
                    or abs(10 ** draw["log_Re_NN"] - host_re[host]) > cfg["host_max_dre_arcsec"]):
                continue
            separation = 10 ** draw["log_d_NN_radii"] * host_re[host] / cfg["pixel_scale"]
            for _ in range(cfg["position_angle_tries"]):
                angle = rng.uniform(0.0, 2.0 * np.pi)
                x = host_xy[host, 0] + separation * np.cos(angle)
                y = host_xy[host, 1] + separation * np.sin(angle)
                if not inside_image(x, y, shape, margin):
                    continue
                if used_positions and np.min(np.hypot(*(np.array(used_positions) - (x, y)).T)) < min_sep:
                    continue
                if tree.query((x, y))[1] == host:
                    placed = (x, y, host)
                    break
            if placed:
                break
        if placed is None:
            continue
        x, y, host = placed
        used_hosts.add(host)
        used_positions.append((x, y))
        rows.append(dict(draw, x=x, y=y, host=int(host)))
    return pd.DataFrame(rows)


def place_random(n, shape, rng, cfg):
    margin = cfg["edge_margin_pix"]
    return pd.DataFrame(dict(x=rng.uniform(margin, shape[1] - margin, n), y=rng.uniform(margin, shape[0] - margin, n)))
