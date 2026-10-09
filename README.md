# lsst_source_injection_experiments

Inject synthetic galaxies into LSST deep coadds, or into the single exposures they are built from, and measure how
well detection and deblending recover them and the real objects around them, as a function of surface brightness.
This is the injection experiment of DMTN-335 (*Quantifying LSST detection and deblender performance for LSB science*)
as a reusable workflow.
Not to be confused with the LSST pipelines' own injection package, `lsst.source.injection` (`source_injection`): this repo
runs injection *experiments* (draws, placement, recovery, deblending and their statistics) around its own injection.

Two kinds of injected source:

- **`newhorizon`**: New Horizon dwarf galaxies (Dubois et al. 2021; Martin et al. 2022), each resized and rescaled to
  properties drawn from COSMOS2025, and placed next to a real object that matches the drawn nearest neighbour
  ("targeted injection").
- **`doublesersicbcg`**: a BCG with its extended envelope, the core-Sérsic profile (Graham et al. 2003) of the ICL
  injection notebook, at random positions.

Two places to inject:

- **`deepcoadd`**: into the deep coadd, drawn with the coadd's PSF.
- **`calexp`**: into every exposure (calexp / `visit_image`), each with its own PSF at its own pixel position, then
  recombined onto the coadd's pixel grid (inverse-variance-weighted mean).

Every injection is analysed on its own, against the same image without it.

## Install

```bash
git clone https://github.com/tmsedgwick/lsst_source_injection_experiments.git
cd lsst_source_injection_experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The deblender, scarlet lite, is part of the LSST stack (e.g. the Rubin Science Platform) but not on PyPI. Elsewhere,
build it from source with Eigen headers (on macOS: `brew install eigen`):

```bash
EIGEN_INCLUDE=$(brew --prefix)/include/eigen3 pip install git+https://github.com/lsst/scarlet_lite.git
```

Only `scripts/export_from_butler.py` needs the rest of the LSST stack; every other step runs anywhere.

## Workflow

### 1. Export the data (Rubin Science Platform)

```bash
python scripts/export_from_butler.py coadd --ra 59.5 --dec -0.75 --size 3072 --out region/coadd.npz
python scripts/export_from_butler.py visits --coadd region/coadd.npz --out-dir region/visits   # for --mode calexp
python scripts/export_from_butler.py objects --coadd region/coadd.npz --out region/objects.parquet
```

A coadd is saved as `.npz` (signal and variance in nJy, PSF stamps, bands, origin, a WCS fitted to the stack's,
and the effective number of visits per band); each visit image the same for one band. A coadd saved by
`lsst_unet_detection` (`--save-coadd`) works too: without a WCS, positions stay in its pixel coordinates.

### 2. Hosts

The real objects New Horizon injections are placed next to, with redshift, stellar mass, size and surface brightness
(redshift and mass from a random forest trained on COSMOS2025):

```bash
python scripts/build_hosts.py --objects region/objects.parquet --out region/hosts.parquet
# or, without an object table, measured on the coadd itself:
python scripts/build_hosts.py --coadd region/coadd.npz --out region/hosts.parquet
```

### 3. Inject

```bash
python scripts/inject.py --source newhorizon --mode deepcoadd --coadd region/coadd.npz \
    --hosts region/hosts.parquet --nh-dir ~/Rvir_mocks --run-dir runs/nh_deepcoadd
python scripts/inject.py --source doublesersicbcg --mode calexp --coadd region/coadd.npz \
    --exposures region/visits/*.npz --run-dir runs/bcg_calexp
```

`--source` picks what is injected, `--mode` where. Per surface-brightness bin (0.5 mag arcsec⁻², 18 to 30)
`--n-per-bin` injections (default 100) are drawn; `--mu-r-range 22 26` runs only some bins. `--nh-dir` holds the New
Horizon templates (`Rvir_mocks_V2_<id>.h5`, cube `xy/904/0.05/image_full`), one picked at random per injection.

### 4. Recovery, by peak finding

```bash
python scripts/recover.py --run-dir runs/nh_deepcoadd --n-workers 4
```

Peaks are found before and after injecting (S/N ≥ 5 in the PSF-smoothed r band, noise from the variance plane;
footprints grown by 2.4 PSF widths), and:

- the **injected source** is detected when a peak within 5 px of it is new, or rose in S/N by at least the detection
  threshold (so a real object the injection lands on does not count as its detection);
- its **nearest neighbour**, the object closest to it before injection (the host it was placed next to), is recovered
  when matched one to one within 3 px after injection;
- the **rest-of-blend**, the objects before injection in the footprint the injection lands in (and the NN's), are
  tracked one to one within 5 px; their recovered fraction is reported.

### 5. Deblending

```bash
python scripts/deblend.py --run-dir runs/nh_deepcoadd --n-workers 4
```

scarlet lite is fitted to the blend region before and after injecting, initialised at the detected peaks; fluxes are
model fluxes (`modelFlux`). Reported: the injected source's flux bias, `100 (F_measured − F_injected) / F_injected`,
and measured surface brightness (when detected); the nearest neighbour's flux bias (when recovered); and the
rest-of-blend's summed flux bias.

### 6. The figure

```bash
python scripts/plot_performance.py --run-dir runs/nh_deepcoadd
```

Writes `summary.csv` and `detection_and_deblending_vs_mu_r.png`: detection fraction of the injected source, nearest
neighbour and rest-of-blend; median r-band flux bias (symlog); and the same as a surface brightness bias Δμ_r.

## Definitions

- **Surface brightness**: `μ_r = m_r + 2.5 log10(A_half)`, the total r magnitude over the half-light area (of the
  PSF-convolved image for injections, of the Sérsic half-light ellipse for LSST objects, `π Re²` for COSMOS). This
  is the convention of the LSST object table and the original notebooks; the mean surface brightness of the light
  inside the half-light area is 0.75 mag fainter.
- **Half-light radius**: `sqrt(N_half) × 0.2″`, with `N_half` the number of brightest pixels holding half the light.
- **Injected New Horizon dwarf** (DMTN-335 §5.2): resized until its PSF-convolved half-light radius matches the target
  (at least the PSF's and 0.8″); r flux from the target μ_r; u from the target u − r; g, i, z, y from colours
  predicted (10 nearest COSMOS neighbours in z, log M*, u − r).
- **COSMOS draws** (DMTN-335 §5.3): per μ_r bin, a Gaussian KDE (Scott bandwidth, standardised) over the galaxy's and
  its nearest neighbour's log M*, z, u − r, log Re and μ_r, and their separation, from COSMOS2025 galaxies at z < 0.4;
  neighbours are galaxies with r ≤ 27.5. Rebuild the shipped tables with `scripts/build_cosmos_reference.py`.
- **Placement** (DMTN-335 §5.5, with the original notebook's cuts): hosts within Δz 0.2, Δlog M* 0.5, Δμ_r 0.5 and
  ΔRe 10″ of the drawn neighbour, ranked by summed normalised differences; the injection goes at the drawn separation
  and a random angle where the host stays its nearest object, inside the image and ≥ 5″ from other injections.
- **BCG** (ICL notebook): core-Sérsic n = 3.9, γ = 0.02, α = 3.6, b/a = 0.7, break radius 0.121 Re, Re log-uniform in
  5–60″, colours u−r 2.2, g−r 0.65, r−i 0.35, i−z 0.25, z−y 0.10.
- **Variance**: injected light adds its shot noise, divided by the coadd's effective number of visits (calexp mode:
  per exposure, then recombined).

All settings are in `lsst_source_injection_experiments/config.py`.

## Testing calexp mode without calexps

Stand-in exposures: copies of a coadd, each with the coadd's variance times the number of copies, so recombining them
gives back the coadd and calexp mode must reproduce deepcoadd mode exactly (the tests check this):

```bash
python scripts/make_test_exposures.py --coadd region/coadd.npz --n 4 --out-dir region/test_exposures
python scripts/inject.py --source doublesersicbcg --mode calexp --coadd region/coadd.npz \
    --exposures region/test_exposures/*.npz --run-dir runs/bcg_calexp_test
```

For real data, pass the exported visit images (`region/visits/*.npz`) instead.

## Tests

```bash
python -m pytest tests
```

The tests use a small synthetic coadd and New Horizon-like template files; they run without the LSST stack.
`scripts/export_from_butler.py` is not covered by them.
