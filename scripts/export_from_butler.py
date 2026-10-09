"""Export what an injection run needs from the Butler (needs the LSST stack, e.g. the Rubin Science Platform), so the
rest of the workflow runs anywhere.

    # a deep coadd region (ugrizy): around a position, or a whole patch
    python scripts/export_from_butler.py coadd --ra 59.5 --dec -0.75 --size 3072 --out region/coadd.npz
    python scripts/export_from_butler.py coadd --tract 9812 --patch 50 --out region/coadd.npz
    # the visit images overlapping it (for --mode calexp), cut to the region
    python scripts/export_from_butler.py visits --coadd region/coadd.npz --out-dir region/visits
    # the object table in it (for build_hosts.py)
    python scripts/export_from_butler.py objects --coadd region/coadd.npz --out region/objects.parquet

Each image is saved with a WCS fitted (TAN-SIP) to the stack's WCS over its own array pixels, so the exposures and the
coadd can be mapped onto each other without the stack. Images are in nJy.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points

from lsst_source_injection.config import BANDS, CONFIG
from lsst_source_injection.images import Coadd, Exposure, load_coadd, save_coadd, save_exposure
from lsst_source_injection.photometry import flux_to_mag

DEFAULTS = dict(repo="dp2", collections="dp2", skymap="lsst_cells_v1")


def fitted_wcs(stack_wcs, bbox, step=64):
    """An astropy TAN-SIP WCS in the array pixels of bbox (pixel (0, 0) = bbox corner) fitted to the stack's WCS."""
    import lsst.geom as geom

    xs = np.linspace(0, bbox.getWidth() - 1, max(2, bbox.getWidth() // step + 1))
    ys = np.linspace(0, bbox.getHeight() - 1, max(2, bbox.getHeight() // step + 1))
    gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
    sky = [stack_wcs.pixelToSky(geom.Point2D(x + bbox.getMinX(), y + bbox.getMinY())) for x, y in zip(gx, gy)]
    coords = SkyCoord([s.getRa().asDegrees() for s in sky], [s.getDec().asDegrees() for s in sky], unit="deg")
    return fit_wcs_from_points((gx, gy), coords, projection="TAN", sip_degree=3)


def psf_stamp(psf, point, size):
    """The PSF kernel at point, cropped or zero-padded to size x size and normalised."""
    kernel = np.asarray(psf.computeKernelImage(point).array, float)
    out = np.zeros((size, size))
    h, w = kernel.shape
    cy, cx, half = h // 2, w // 2, size // 2
    ry, rx = min(half, cy, h - cy - 1), min(half, cx, w - cx - 1)
    out[half - ry:half + ry + 1, half - rx:half + rx + 1] = kernel[cy - ry:cy + ry + 1, cx - rx:cx + rx + 1]
    return out / out.sum()


def effective_visits(exposure):
    """Effective number of visits (sum w)^2 / sum w^2 of a coadd's inputs."""
    weights = np.array([row["weight"] for row in exposure.getInfo().getCoaddInputs().visits], float)
    return float(weights.sum() ** 2 / (weights ** 2).sum()) if weights.size and (weights ** 2).sum() > 0 else 1.0


def export_coadd(butler, args):
    import lsst.geom as geom

    skymap = butler.get("skyMap", skymap=args.skymap)
    if args.tract is not None:
        tract_info = skymap[args.tract]
        patch_info = tract_info[args.patch]
        bbox = patch_info.getOuterBBox()
    else:
        point = geom.SpherePoint(args.ra, args.dec, geom.degrees)
        tract_info = skymap.findTract(point)
        patch_info = tract_info.findPatch(point)
        centre = tract_info.getWcs().skyToPixel(point)
        half = args.size // 2
        bbox = geom.Box2I(geom.Point2I(int(centre.getX()) - half, int(centre.getY()) - half),
                          geom.Extent2I(args.size, args.size))
        bbox.clip(patch_info.getOuterBBox())
    data_id = dict(skymap=args.skymap, tract=tract_info.getId(), patch=tract_info.getSequentialPatchIndex(patch_info))
    exposures = [butler.get("deep_coadd", dataId=dict(data_id, band=band), parameters={"bbox": bbox}) for band in BANDS]
    centre = geom.Point2D(bbox.getCenter())
    coadd = Coadd(np.stack([e.image.array for e in exposures]).astype(np.float32),
                  np.stack([e.variance.array for e in exposures]).astype(np.float32),
                  np.stack([psf_stamp(e.getPsf(), centre, CONFIG["psf_stamp"]) for e in exposures], axis=-1),
                  BANDS, (bbox.getMinX(), bbox.getMinY()), fitted_wcs(exposures[0].getWcs(), bbox),
                  np.array([effective_visits(e) for e in exposures]))
    save_coadd(args.out, coadd)
    print(f"deep_coadd {data_id}, {bbox} -> {args.out}")


def region_sky_corners(coadd):
    ny, nx = coadd.shape
    return coadd.wcs.pixel_to_world([0, nx - 1, nx - 1, 0], [0, 0, ny - 1, ny - 1])


def export_visits(butler, args):
    import lsst.geom as geom

    coadd = load_coadd(args.coadd)
    ny, nx = coadd.shape
    centre = coadd.wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
    corners = region_sky_corners(coadd)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for band in BANDS:
        where = (f"band='{band}' AND visit_detector_region.region OVERLAPS "
                 f"POINT({centre.ra.deg}, {centre.dec.deg})")
        refs = sorted(butler.query_datasets(args.dataset, where=where), key=lambda ref: str(ref.dataId))
        for ref in refs[:args.max_visits]:
            full_bbox = butler.get(f"{args.dataset}.bbox", dataId=ref.dataId)
            wcs = butler.get(f"{args.dataset}.wcs", dataId=ref.dataId)
            pixels = [wcs.skyToPixel(geom.SpherePoint(c.ra.deg, c.dec.deg, geom.degrees)) for c in corners]
            bbox = geom.Box2I(geom.Point2I(int(min(p.getX() for p in pixels)) - 8, int(min(p.getY() for p in pixels)) - 8),
                              geom.Point2I(int(max(p.getX() for p in pixels)) + 8, int(max(p.getY() for p in pixels)) + 8))
            bbox.clip(full_bbox)
            if bbox.isEmpty():
                continue
            exposure = butler.get(args.dataset, dataId=ref.dataId, parameters={"bbox": bbox})
            scale = exposure.getPhotoCalib().getCalibrationMean()  # 1 for images already in nJy
            name = f"{args.dataset}_{band}_{ref.dataId['visit']}_{ref.dataId['detector']}"
            save_exposure(args.out_dir / f"{name}.npz", Exposure(
                (exposure.image.array * scale).astype(np.float32),
                (exposure.variance.array * scale ** 2).astype(np.float32),
                psf_stamp(exposure.getPsf(), geom.Point2D(bbox.getCenter()), CONFIG["psf_stamp"]).astype(np.float32),
                band, (bbox.getMinX(), bbox.getMinY()), fitted_wcs(exposure.getWcs(), bbox), name))
            n_saved += 1
    print(f"{n_saved} visit images -> {args.out_dir}")


def object_properties(table, pixel_scale, zeropoint):
    """m_<band> (PSF flux for point sources, cModel otherwise), half-light radius (geometric mean of the Sersic radii)
    and mu_r = m_r + 2.5 log10(pi a b) over the Sersic half-light ellipse, as in the original notebook."""
    out = pd.DataFrame(index=table.index)
    for band in BANDS:
        point_like = table[f"{band}_extendedness"].to_numpy(float) < 0.5
        flux = np.where(point_like, table[f"{band}_psfFlux"], table[f"{band}_cModelFlux"]).astype(float)
        out[f"m_{band}"] = np.where(flux > 0, flux_to_mag(flux, zeropoint), np.nan)
    reff_x, reff_y = table["sersic_reff_x"].to_numpy(float), table["sersic_reff_y"].to_numpy(float)
    out["half_light_radius_arcsec"] = np.sqrt(reff_x * reff_y) * pixel_scale
    area = np.pi * reff_x * reff_y * pixel_scale ** 2
    out["mu_r"] = np.where(area > 0, out["m_r"] + 2.5 * np.log10(np.where(area > 0, area, 1.0)), np.nan)
    return out


def export_objects(butler, args):
    coadd = load_coadd(args.coadd)
    corners = region_sky_corners(coadd)
    skymap = butler.get("skyMap", skymap=args.skymap)
    import lsst.geom as geom

    tract = skymap.findTract(geom.SpherePoint(float(np.mean(corners.ra.deg)), float(np.mean(corners.dec.deg)),
                                              geom.degrees)).getId()
    columns = ["coord_ra", "coord_dec", "sersic_reff_x", "sersic_reff_y"] + [
        f"{band}_{name}" for band in BANDS for name in ("psfFlux", "cModelFlux", "extendedness")]
    table = butler.get(args.dataset, dataId=dict(skymap=args.skymap, tract=tract), parameters={"columns": columns})
    table = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
    x, y = coadd.wcs.world_to_pixel(SkyCoord(table["coord_ra"].to_numpy(), table["coord_dec"].to_numpy(), unit="deg"))
    ny, nx = coadd.shape
    inside = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    objects = object_properties(table[inside], CONFIG["pixel_scale"], CONFIG["zeropoint"])
    objects.insert(0, "y", y[inside])
    objects.insert(0, "x", x[inside])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    objects.reset_index(drop=True).to_parquet(args.out)
    print(f"{len(objects):,} objects of tract {tract} in the region -> {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULTS["repo"])
    parser.add_argument("--collections", default=DEFAULTS["collections"])
    parser.add_argument("--skymap", default=DEFAULTS["skymap"])
    sub = parser.add_subparsers(dest="what", required=True)
    coadd = sub.add_parser("coadd", help="a deep coadd region, ugrizy")
    coadd.add_argument("--tract", type=int)
    coadd.add_argument("--patch", type=int)
    coadd.add_argument("--ra", type=float)
    coadd.add_argument("--dec", type=float)
    coadd.add_argument("--size", type=int, default=3072, help="side in pixels around --ra/--dec")
    coadd.add_argument("--out", type=Path, required=True)
    visits = sub.add_parser("visits", help="the visit images overlapping an exported coadd region")
    visits.add_argument("--coadd", type=Path, required=True)
    visits.add_argument("--dataset", default="visit_image", help="default %(default)s (calexp for older repos)")
    visits.add_argument("--max-visits", type=int, default=None, help="per band (default: all)")
    visits.add_argument("--out-dir", type=Path, required=True)
    objects = sub.add_parser("objects", help="the object table inside an exported coadd region")
    objects.add_argument("--coadd", type=Path, required=True)
    objects.add_argument("--dataset", default="object_all", help="default %(default)s")
    objects.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.what == "coadd" and (args.tract is None) == (args.ra is None):
        parser.error("give --tract and --patch, or --ra and --dec")

    from lsst.daf.butler import Butler  # LSST stack

    butler = Butler(args.repo, collections=args.collections)
    dict(coadd=export_coadd, visits=export_visits, objects=export_objects)[args.what](butler, args)


if __name__ == "__main__":
    main()
