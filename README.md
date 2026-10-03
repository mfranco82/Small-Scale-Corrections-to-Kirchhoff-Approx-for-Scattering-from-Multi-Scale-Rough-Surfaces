# Small-scale corrections to the Kirchhoff approximation: code for the figures and validation

<!-- Replace 12345678 (twice) with the Zenodo concept DOI ("Cite all versions") once the first release is archived.
     The concept DOI always resolves to the latest version; the version DOIs point to a fixed release. -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12345678.svg)](https://doi.org/10.5281/zenodo.12345678)

Python code that generates the figures of the Results section (Figs. 2–4) of

> M. Franco, *Small-Scale Corrections to the Kirchhoff Approximation for Scattering from Multi-Scale Rough Dielectric Surfaces*.

together with the scripts that validate the numerical reference and produce the other numbers quoted in the text (`validation/`).

The paper treats a 1D dielectric interface `z = a cos(Kx) + b cos(ΛKx)`, with `K = 2π/L`, in two steps. The smooth
periodic profile is solved with the Kirchhoff approximation (KA), and the small scale is added as a perturbation with the
method of variation of boundaries (MVB) taken **about the periodic surface** instead of a flat one, so the order-`n`
Taylor coefficients `d±_{n,r}` are coupled across all Floquet orders `r`. TE and TM polarizations are supported.
The reference solution is the Rayleigh (Fourier–Galerkin) solution of the full profile, checked against an independent
C-method solver.

Figure 1 of the paper is a schematic of the geometry and is not produced by this code.

## Requirements

* Python 3 with `numpy`, `scipy` and `matplotlib` (see `requirements.txt`; tested with Python 3.12.7, numpy 1.26.4,
  scipy 1.13.1, matplotlib 3.9.2).
* A LaTeX installation with the `mathpazo` and `amsmath` packages (e.g. TeX Live): Figs. 2–4 are drawn with
  `text.usetex`, so that their typography matches the paper.
* The computations use `multiprocessing.Pool(8)`. They run on fewer cores (only slower); keep the 8 workers if you want to
  reproduce the same Monte Carlo realization of Fig. 2(d), whose seeds are tied to the 8 chunks.

## Contents

| file | role |
|---|---|
| `scattering.py` | KA amplitudes of the smooth profile (reflected and transmitted), MVB recursion about the smooth surface (`mvb_coefficients`; `SmoothOperator` factorizes the system matrix once and reuses it for any roughness), Rayleigh reference solver, efficiencies |
| `ensemble.py` | analytic average over random small-scale roughness (Eq. (40) of the paper) and the Monte Carlo reference |
| `make_final_figures.py` | Fig. 2 (`spectra`) and Fig. 3 (`robust`) |
| `dielectric_scan.py` | scan over the dielectric constant that feeds Fig. 3(c); prints the statistics quoted in the text |
| `backscatter_map.py` | Fig. 4: backscattering at the Littrow condition as a function of `L/λ` and `ε` |
| `data/` | cached results, so the figures can be redrawn without recomputing (see below) |
| `validation/` | tests of the solvers against the independent C-method and the scripts behind the other numbers of the text; see `validation/README.md` |

Conventions: incident wave `exp(i(αx − βz))`, `α_r = α + rK`; TE has `ν² = 1` and TM `ν² = 1/ε`; the error of an
approximation is the relative `L¹` distance between its efficiencies and the reference ones over the propagating orders.

## Reproducing the figures

Run the commands from this folder. The figures are written to `figures/`, which is created if it does not exist.

| figure | command | time (8 cores) |
|---|---|---|
| Fig. 2, spectra (a–d) | `python make_final_figures.py spectra` | ~45 s (includes the Monte Carlo over 2000 exact solutions) |
| Fig. 3, robustness (a–c) | `python dielectric_scan.py`, then `python make_final_figures.py robust` | ~6 s + ~9 s |
| Fig. 4, backscattering | `python backscatter_map.py compute`, then `python backscatter_map.py plot` | up to ~50 min (5808 cells; ~1 s per cell on one core) |

Both `make_final_figures.py` and `backscatter_map.py` print the numbers quoted in the text (`backscatter_map.py summary`
prints the statistics of the map).

Cached results in `data/`:

* `dielectric_scan.npy` is read by `make_final_figures.py robust` for Fig. 3(c). It is produced by `dielectric_scan.py`.
* `final_robust_cache.npz` holds the sweeps of Fig. 3(a,b). `python make_final_figures.py robust --replot` redraws the
  figure from it; without `--replot` the sweeps are recomputed and the cache is overwritten.
* `backscatter_map.npz` holds the full map of Fig. 4. `python backscatter_map.py plot` draws from it; `compute` overwrites it.
* Fig. 2 has no cache: `spectra` recomputes everything.

Side outputs not used in the paper: `dielectric_scan.py` also draws `figures/fig_eps.pdf`, and `backscatter_map.py plot`
also draws `figures/fig_backscatter.pdf` (TE and TM) and `figures/fig_backscatter_phi0.pdf` (fixed relative phase).

## Validation

The scripts in `validation/` check the Rayleigh reference against the C-method, the recursion and the ensemble average, and
reproduce the numbers of the text that are not in a figure (stability limits, role of the zeroth order, cost). For example:

```
cd validation
python test_scattering.py
python test_reference.py
```

## License

The code is released under the MIT license (see `LICENSE`).
