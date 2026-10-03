# Small-scale corrections to the Kirchhoff approximation: code for the figures and validation

[![DOI](https://zenodo.org/badge/1403194780.svg)](https://zenodo.org/badge/latestdoi/1403194780)

Python code that generates the figures of the Results section (Figs. 2–4) and the numbers quoted in the text of

> M. Franco, *Small-Scale Corrections to the Kirchhoff Approximation for Scattering from Multi-Scale Rough Dielectric Surfaces*.

together with the scripts that validate the numerical reference (`validation/`).

The Kirchhoff approximation (KA) is an expansion in the **curvature** of a surface (valid when `C = max|F''|/(2k cos³θ_i) ≪ 1`),
and the method of variation of boundaries is an expansion in its **height** (`kb ≪ 1`). The paper combines them for a 1D dielectric
interface made of a smooth periodic profile, for which KA holds, carrying a small-scale roughness of low height but of wavelength
comparable to the radiation, for which it does not. The boundary variations are taken **about the periodic surface** instead of a flat
one, so the Taylor coefficients `d±_{n,r}` of order `n` are coupled across all Floquet orders `r`. The zeroth order of the series is either
the Kirchhoff amplitudes of the smooth profile or its exact (Rayleigh) solution. TE and TM polarizations are supported. The reference
solution is an independent C-method solver; the baseline is KA applied to the whole profile.

Figure 1 of the paper is a schematic of the geometry and is not produced by this code.

## Requirements

* Python 3 with `numpy`, `scipy` and `matplotlib` (see `requirements.txt`; tested with Python 3.12.7, numpy 1.26.4,
  scipy 1.13.1, matplotlib 3.9.2).
* A LaTeX installation with the `mathpazo` and `amsmath` packages (e.g. TeX Live): the figures are drawn with `text.usetex`, so that their
  typography matches the paper.
* The cache-producing scripts use `multiprocessing.Pool(8)`. Every case is seeded independently, so the results do not depend on the
  number of cores (only the running time does).

## Contents

| file | role |
|---|---|
| `scattering.py` | KA amplitudes of the smooth profile (reflected and transmitted), MVB recursion about the smooth surface (`mvb_coefficients`; `SmoothOperator` factorizes the system matrix once and reuses it for any roughness), Rayleigh solver (the exact zeroth order and a reference), efficiencies |
| `cmethod.py` | independent C-method (Chandezon et al.) solver, which does not use the Rayleigh hypothesis; it is the reference of all the figures |
| `make_results_figures.py` | Figs. 2–4 and the numbers quoted in the text |
| `phase_map.py`, `robustness.py`, `random_stats.py` | studies that fill the caches in `data/` (see below) |
| `data/` | cached results of the three studies, so the figures can be redrawn without recomputing |
| `validation/` | tests of the solvers and of the recursion, stability scan and timings; see `validation/README.md` |

Conventions: incident wave `exp(i(αx − βz))`, `α_r = α + rK`; TE has `ν² = 1` and TM `ν² = 1/ε`; the error of a model is the relative
`L¹` distance between its efficiencies and the reference ones over the propagating orders.

## Reproducing the figures

Run the commands from this folder. The figures are written to `figures/`, which is created if it does not exist.

| figure | command | what it reads | time |
|---|---|---|---|
| Fig. 2, spectra and convergence | `python make_results_figures.py spectra` | recomputes three cases; random surface from `data/random_stats.npz` | ~5 s |
| Fig. 3, error versus `C` | `python make_results_figures.py caxis` | `data/robustness.npz`, `data/random_stats.npz`, `data/phase_map.npz` | seconds |
| Fig. 4, plane `(Λ, kb)` | `python make_results_figures.py phasemap` | `data/phase_map.npz` | seconds |
| numbers of Secs. 4 and 5 | `python make_results_figures.py numbers` | all of the above | ~10 s |

The studies behind the caches can be rerun (they overwrite the files in `data/`):

| cache | command | content | time (8 cores) |
|---|---|---|---|
| `phase_map.npz` | `python phase_map.py compute` | 9 values of `Λ` × 7 of `kb`, TE and TM (Fig. 4) | ~1.5 min |
| `robustness.npz` | `python robustness.py compute` | cosine profiles, one parameter at a time: `L/λ`, incidence angle, `ε`, `aK` (Fig. 3) | ~7 min |
| `random_stats.npz` | `python random_stats.py compute` | 40 random flat-band surfaces for each of 12 configurations, TE and TM (Figs. 2c and 3) | ~11 min |

`python phase_map.py table`, `python robustness.py summary` and `python random_stats.py summary` print the contents of the caches.

## Validation

The scripts in `validation/` check the Rayleigh solution against the C-method, the recursion, and the stability limits, and time the
solvers. For example:

```
cd validation
python test_scattering.py
python test_reference.py
```

## Versions

`v0.1.0` accompanied an earlier version of the paper (KA applied to the smooth profile as the baseline; an analytic average over random
roughness). `v0.2.0` corresponds to the present approach, with KA of the full profile as the baseline, the validity parameter `C` as
the organizing axis, and both zeroth orders. The DOI badge above always points to the latest version.

## License

The code is released under the MIT license (see `LICENSE`).
