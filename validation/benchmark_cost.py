"""
Cost of the different routes for one profile (Table 1 of the paper; timings: best of REP runs, one core; run it on an otherwise idle machine).

Profile of Fig. 2(a): z = a cos(Kx) + b cos(Lambda K x), a = 0.03 L, Lambda = 15, kb = 0.25, L = 10 lambda, sin(theta_i) = 0.35, eps = 2.25, TE.
  * exact reference solutions of the full profile: Rayleigh (direct) and C-method
  * KA applied to the full profile (the two-scale Kirchhoff model)
  * series KA + MVB (n <= 3): zeroth order (KA or exact) + factorization of the smooth-profile operator (both done once) + recursion
usage: python benchmark_cost.py          (forces single-thread BLAS; run it on an otherwise idle machine)
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"          # single-core timings (set before importing numpy)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, cmethod.py, ... live in the parent folder
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative

REP, LAM, KB = 3, 15, 0.25


def best(fn, rep=REP):
    t = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


if __name__ == "__main__":
    th = np.arcsin(0.35)
    p0 = Problem(1.0, 0.1, th, 2.25, "TE", 30, N=4096)
    rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * LAM + 12)
    p = Problem(1.0, 0.1, th, 2.25, "TE", rmax, N=4096)
    f, f1 = cos_surface(p, 0.03); z, z1 = cos_surface(p, KB / p.k, LAM)
    F, F1 = f + z, f1 + z1
    t_ray = best(lambda: rayleigh_exact(p, F, F1))
    t_c = best(lambda: c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8), 2)
    t_kaf = best(lambda: ka_amplitudes(p, F, F1))
    t_ka0 = best(lambda: ka_amplitudes(p, f, f1)); t_ex0 = best(lambda: rayleigh_exact(p, f, f1))
    t_op = best(lambda: SmoothOperator(p, f, f1)); op = SmoothOperator(p, f, f1)
    Bk, Bx = ka_amplitudes(p, f, f1), rayleigh_exact(p, f, f1)
    t_ska = best(lambda: mvb_coefficients(p, f, f1, z, z1, Bk, 3, smooth=op)); t_sex = best(lambda: mvb_coefficients(p, f, f1, z, z1, Bx, 3, smooth=op))
    print(f"profile of Fig. 2(a): Lambda={LAM}, kb={KB}, r_max={rmax} ({2 * rmax + 1} orders), N=4096")
    print(f"  Rayleigh, full profile  {1e3 * t_ray:6.0f} ms")
    print(f"  C-method, full profile  {1e3 * t_c:6.0f} ms")
    print(f"  KA, full profile        {1e3 * t_kaf:6.0f} ms")
    print(f"  series n<=3, KA zeroth order   : recursion {1e3 * t_ska:5.0f} ms  (+ once: zeroth order {1e3 * t_ka0:.0f} ms, factorization {1e3 * t_op:.0f} ms)")
    print(f"  series n<=3, exact zeroth order: recursion {1e3 * t_sex:5.0f} ms  (+ once: zeroth order {1e3 * t_ex0:.0f} ms, factorization {1e3 * t_op:.0f} ms)")
