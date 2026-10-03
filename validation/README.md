# Validation

Scripts that check the numerical machinery of the paper and reproduce the numbers quoted in its text. They import
`scattering.py`, `ensemble.py` and `make_final_figures.py` from the parent folder and use `cmethod.py`, an independent
C-method (Chandezon et al.) solver that does not rely on the Rayleigh hypothesis. Run them from this folder, for example
`python test_scattering.py`.

The `test_*.py` files and `check_backscatter_map.py` end with assertions and exit with a non-zero status if a check fails.
The other scripts print the numbers of the paper.

Why a second solver: the reference used throughout the paper is the Rayleigh (Fourier–Galerkin) solution of the full profile,
which shares the Rayleigh expansion with the perturbative scheme and so is not independent of it. The C-method is.

## Checks of the solvers

| command | what it checks | result | time |
|---|---|---|---|
| `python test_scattering.py` | flat interface (Rayleigh and KA give the Fresnel coefficient); energy conservation `R+T=1` of the Rayleigh solution; MVB started from the exact zeroth order converges to the exact solution with error `∝ b^(n+1)` (TE and TM) | `\|R+T−1\| ~ 1e-15`; error scaling confirmed | 2 s |
| `python test_reference.py` | Rayleigh reference vs C-method on the profiles of Fig. 2 (periodic and random) and on strongly asymmetric profiles (a slope-sign error would not show on even profiles) | agreement `≲ 5e-9` relative in all orders with `e_r > 1e-9` | 2 s |
| `python test_reference_sweeps.py` | same comparison on samples of the sweeps of Figs. 3 and 4 (`ε` up to 16, incidence 5°–58°, `kb` up to 1; TE and TM) | `L¹` distance `≲ 1.1e-6`; per order (`e_r > 1e-7`) `≲ 1.1e-4` | 2 s |
| `python check_backscatter_map.py` | Rayleigh vs C-method for the backscattered order on 40 cells of the map of Fig. 4 (corners included) | relative difference `≲ 7e-8` | 3 s |
| `python test_ensemble.py` | the analytic ensemble average (Eq. (40) of the paper) agrees with the numerical average over the random phase up to terms of order `A⁴` | difference ratio 13.9 when `A` is doubled (16 expected) | 1 s |

## Numbers quoted in the paper

| command | what it prints | where it is quoted | time |
|---|---|---|---|
| `python stability_scan.py` | condition number of the system matrix and error of KA, of KA+MVB (`n ≤ 3`) and of MVB started from the exact zeroth order, for growing `aK` and several truncations `\|r\| ≤ r_max`, against the C-method | limits of validity (instability for `aK ≳ 0.4`) | 14 s |
| `python zeroth_order_comparison.py` | error of the series started from the KA amplitudes vs from the exact Rayleigh solution of the smooth profile, in the cases of Figs. 2 and 3 | role of KA as zeroth order | ~1 min (8 cores) |
| `python ensemble_accuracy.py` | error of the analytic average vs a Monte Carlo over 2000 exact solutions, as a function of the strength of the small scale | Discussion | ~2.5 min (8 cores) |
| `OMP_NUM_THREADS=1 python benchmark_cost.py` | timings of the C-method, the Rayleigh method, KA+MVB and the analytic average, for one profile and for a Monte Carlo of 2000 realizations | Discussion | ~10 s |

Timings are hardware dependent: run `benchmark_cost.py` on an otherwise idle machine. Repeated runs differ by about 10 %;
the ratios between methods are what matters.

Like the figure scripts, the Monte Carlo parts use `Pool(8)` with seeds tied to the 8 chunks, so they run on fewer cores
(more slowly) and give the same numbers.
