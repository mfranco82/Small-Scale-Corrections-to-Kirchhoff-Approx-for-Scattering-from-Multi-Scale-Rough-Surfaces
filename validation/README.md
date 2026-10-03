# Validation

Scripts that check the numerical machinery of the paper and reproduce the numbers quoted in its text. They import `scattering.py` and
`cmethod.py` from the parent folder. `cmethod.py` is an independent C-method (Chandezon et al.) solver that does not rely on the Rayleigh
hypothesis. Run them from this folder, for example `python test_scattering.py`.

The `test_*.py` files end with assertions and exit with a non-zero status if a check fails. The other scripts print the numbers of the paper.

Why a second solver: the Rayleigh (Fourier–Galerkin) solution of the full profile shares the Rayleigh expansion with the perturbative
scheme and so is not independent of it. The C-method is, and it is the reference of all the figures.

## Checks of the solvers

| command | what it checks | result | time |
|---|---|---|---|
| `python test_scattering.py` | flat interface (Rayleigh and KA give the Fresnel coefficient); energy conservation `R+T=1` of the Rayleigh solution; MVB started from the exact zeroth order converges to the exact solution with error `∝ b^(n+1)` (TE and TM) | `\|R+T−1\| ~ 1e-15`; error scaling confirmed | 2 s |
| `python test_reference.py` | Rayleigh vs C-method on two-scale cosine, random multi-scale and strongly asymmetric profiles (a slope-sign error would not show on even profiles) | agreement `≲ 5e-9` relative in all orders with `e_r > 1e-9` | 2 s |
| `python test_reference_sweeps.py` | same comparison on samples over `ε` up to 16, incidence 5°–58° and `kb` up to 1 (TE and TM) | `L¹` distance `≲ 1.1e-6`; per order (`e_r > 1e-7`) `≲ 1.1e-4` | 2 s |

The agreement of the Rayleigh solution with the C-method over all the cases of the figures (91 % below `1e-4`, 97 % below `1e-3` for
`C ≤ 0.8`) is printed by `python ../make_results_figures.py numbers`.

## Numbers quoted in the paper

| command | what it prints | where it is quoted | time |
|---|---|---|---|
| `python stability_scan.py` | condition number of the system matrix and error of KA, of KA+MVB (`n ≤ 3`) and of MVB started from the exact zeroth order, for growing `aK` and several truncations `\|r\| ≤ r_max`, against the C-method | Sec. 4, stability of the Kirchhoff zeroth order | 14 s |
| `python benchmark_cost.py` | timings of the Rayleigh and C-method solutions, KA of the full profile and the series, for the profile of Fig. 2(a) | Sec. 5, Table 1 | seconds |

Timings are hardware dependent: run `benchmark_cost.py` on an otherwise idle machine (it forces single-thread BLAS). Repeated runs agree
to a few per cent on one machine; the ratios between methods are what matters.
