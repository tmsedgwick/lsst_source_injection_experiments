"""Was the injected source detected, and what did its injection do to the real objects around it?

Each injection is analysed in its cutout, with peak-finding detection run before and after injecting:
  * injected source: detected when a peak lies within detect_radius_pix of the injection position;
  * nearest neighbour (NN): the object detected before injection that is closest to the injection position (the real
    object the injection was placed next to); recovered when, after injection, an object matched to it one to one
    (within track_radius_pix, the injected source's own peak excluded) lies within nn_radius_pix;
  * rest-of-blend: the objects detected before injection inside the blend region, the bounding box of the footprint
    the injection lands in (the one holding the peak nearest the injection position) and of the NN's footprint; their
    recovered fraction is the share still matched one to one after injection.
The peaks, with their roles and matches, are kept for the deblending step.
"""

import numpy as np
import pandas as pd

from .detection import detect, footprint_bbox


def one_to_one_match(xy0, xy1, max_sep):
    """For each point of xy0, the index of its match in xy1 (-1 if none) and the separation: pairs are taken closest
    first, each point used at most once, up to max_sep."""
    xy0, xy1 = np.asarray(xy0, float).reshape(-1, 2), np.asarray(xy1, float).reshape(-1, 2)
    match, separation = np.full(len(xy0), -1), np.full(len(xy0), np.nan)
    if len(xy0) == 0 or len(xy1) == 0:
        return match, separation
    distance = np.hypot(*(xy0[:, None, :] - xy1[None, :, :]).transpose(2, 0, 1))
    used0, used1 = np.zeros(len(xy0), bool), np.zeros(len(xy1), bool)
    for flat in np.argsort(distance, axis=None):
        i, j = divmod(int(flat), len(xy1))
        if distance[i, j] > max_sep:
            break
        if not (used0[i] or used1[j]):
            used0[i] = used1[j] = True
            match[i], separation[i] = j, distance[i, j]
    return match, separation


def union_bbox(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def inside(peaks, bbox):
    x0, y0, x1, y1 = bbox
    return ((peaks["x"] >= x0) & (peaks["x"] < x1) & (peaks["y"] >= y0) & (peaks["y"] < y1)).to_numpy()


def injected_light_detected(before, after, peak, cfg):
    """Whether an after-injection peak is the injected source's own detection: it is new (no peak before within
    track_radius_pix), or its S/N rose by at least the detection threshold, so the injected light alone would have
    been detected there. Otherwise it is a real object the injection merely landed on."""
    distance = np.hypot(before["x"] - after["x"].iat[peak], before["y"] - after["y"].iat[peak]).to_numpy()
    if len(distance) == 0 or distance.min() > cfg["track_radius_pix"]:
        return True
    return bool(after["sn"].iat[peak] - before["sn"].iat[int(np.argmin(distance))] >= cfg["detection_sn"])


def recover(before_signal, before_variance, after_signal, after_variance, psf_kernels, x, y, cfg):
    """(summary dict, peaks table) for one injection at cutout position (x, y)."""
    before, labels_before = detect(before_signal, before_variance, psf_kernels, cfg)
    after, labels_after = detect(after_signal, after_variance, psf_kernels, cfg)

    distance_after = np.hypot(after["x"] - x, after["y"] - y).to_numpy()
    nearest_after = int(np.argmin(distance_after)) if len(after) else -1
    detected = bool(nearest_after >= 0 and distance_after[nearest_after] < cfg["detect_radius_pix"]
                    and injected_light_detected(before, after, nearest_after, cfg))
    injected_peak = nearest_after if detected else -1
    region = (footprint_bbox(labels_after, after["footprint"].iat[nearest_after]) if nearest_after >= 0 else None)

    distance_before = np.hypot(before["x"] - x, before["y"] - y).to_numpy()
    nn = int(np.argmin(distance_before)) if len(before) else -1
    if nn >= 0:
        region = union_bbox(region, footprint_bbox(labels_before, before["footprint"].iat[nn]))
    if region is None:
        r = int(np.ceil(cfg["detect_radius_pix"]))
        region = (max(int(x) - r, 0), max(int(y) - r, 0), int(x) + r + 1, int(y) + r + 1)

    others = np.flatnonzero(np.arange(len(after)) != injected_peak)
    match_sub, separation = one_to_one_match(before[["x", "y"]].to_numpy(), after[["x", "y"]].to_numpy()[others],
                                             cfg["track_radius_pix"])
    match = np.full(len(before), -1)
    match[match_sub >= 0] = others[match_sub[match_sub >= 0]]
    nn_recovered = bool(nn >= 0 and match[nn] >= 0 and separation[nn] <= cfg["nn_radius_pix"])
    rest = inside(before, region)
    n_rest = int(rest.sum())

    summary = dict(
        detected=detected, injected_peak_distance=float(distance_after[nearest_after]) if nearest_after >= 0 else np.nan,
        nn_exists=nn >= 0, nn_recovered=nn_recovered,
        nn_distance=float(distance_before[nn]) if nn >= 0 else np.nan,
        nn_separation_after=float(separation[nn]) if nn >= 0 else np.nan,
        rest_n_before=n_rest, rest_n_after=int(inside(after, region).sum()) - int(detected),
        rest_n_matched=int((match[rest] >= 0).sum()),
        rest_recovered_fraction=float((match[rest] >= 0).mean()) if n_rest else np.nan,
        region_x0=region[0], region_y0=region[1], region_x1=region[2], region_y1=region[3],
        n_peaks_before=len(before), n_peaks_after=len(after))
    before_table = before.assign(stage="before", peak=np.arange(len(before)), in_region=rest,
                                 is_nn=np.arange(len(before)) == nn, matched_peak=match, is_injected=False)
    after_table = after.assign(stage="after", peak=np.arange(len(after)), in_region=inside(after, region),
                               is_nn=False, matched_peak=-1, is_injected=np.arange(len(after)) == injected_peak)
    return summary, pd.concat([before_table, after_table], ignore_index=True)
