"""
Cost of the different routes for the same problem (timings: best of REP runs; run it on an otherwise idle machine).

Profile: z = a cos(Kx) + b cos(Lambda K x) (a = 0.03 L, b = 0.004 L, Lambda = 5), eps = 2.25, theta_i = 20 deg, TE,
for L/lambda = 10 and 20 (number of retained orders = 2 rmax + 1, rmax = r_trans + 12).

 * single deterministic profile: C-method, Rayleigh (direct), KA + MVB (n <= 3, smooth operator already factorised)
 * statistics of random small-scale roughness (7 harmonics): Monte Carlo with K exact solutions
   (Rayleigh or C-method) versus the analytic average of ensemble.py (3 recursion runs per harmonic)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, ensemble.py, ... live in the parent folder
import time
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative
from ensemble import *

REP, K_MC = 3, 2000
HARM = [(m, 0.003 * (m / 4) ** (-0.5)) for m in range(4, 11)]


def best(fn, rep=REP):
    t = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


def run(L_rel):
    wl = 1.0 / L_rel
    p0 = Problem(1.0, wl, np.deg2rad(20), 2.25, "TE", 30, N=2048)
    rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30)
    N = 2048 if rmax < 60 else 4096
    p = Problem(1.0, wl, np.deg2rad(20), 2.25, "TE", rmax, N=N)
    f, f1 = cos_surface(p, 0.03); z, z1 = cos_surface(p, 0.004, 5)
    F, F1 = f + z, f1 + z1
    F2 = spectral_derivative(p, F, 2)
    Bka = ka_amplitudes(p, f, f1)
    op = SmoothOperator(p, f, f1)
    t_c = best(lambda: c_method(p, F, F1, F2, M=rmax, delta=1e-8))
    t_r = best(lambda: rayleigh_exact(p, F, F1))
    t_op = best(lambda: SmoothOperator(p, f, f1))
    t_ka = best(lambda: ka_amplitudes(p, f, f1))
    t_m = best(lambda: mvb_coefficients(p, f, f1, z, z1, Bka, 3, smooth=op))
    t_an = best(lambda: analytic_average(p, f, f1, Bka, HARM, smooth=op), rep=2)
    print(f"L/lambda = {L_rel}: orders retained 2*rmax+1 = {2 * rmax + 1}, grid N = {N}")
    print(f"  single profile : C-method {t_c * 1e3:8.0f} ms | Rayleigh {t_r * 1e3:8.0f} ms | KA+MVB(n<=3, factorisation reused) {t_m * 1e3:8.0f} ms"
          f"  (one-off: KA {t_ka * 1e3:.0f} ms, factorisation {t_op * 1e3:.0f} ms)")
    print(f"  random roughness ({len(HARM)} harmonics), mean efficiencies: analytic {t_an:7.2f} s ({3 * len(HARM)} recursion runs)")
    print(f"     Monte Carlo, K={K_MC}: Rayleigh {K_MC * t_r:8.0f} s  ({K_MC * t_r / t_an:6.0f}x slower) | "
          f"C-method {K_MC * t_c:8.0f} s  ({K_MC * t_c / t_an:6.0f}x slower)")
    return dict(L=L_rel, rmax=rmax, t_c=t_c, t_r=t_r, t_m=t_m, t_an=t_an)




def run_fig_ensemble(K=2000):
    """Exactly the configuration of Fig. 2(d): random multi-scale smooth profile, theta_i = 25 deg, rmax = 40."""
    from make_final_figures import random_profile, WL as WL_, EPS0
    p = Problem(1.0, WL_, np.deg2rad(25), EPS0, "TE", 40, N=2048)
    f, f1, z, z1 = random_profile(p)
    F, F1 = f + z, f1 + z1
    F2 = spectral_derivative(p, F, 2)
    Bka = ka_amplitudes(p, f, f1)
    op = SmoothOperator(p, f, f1)
    t_c = best(lambda: c_method(p, F, F1, F2, M=40, delta=1e-8))
    t_r = best(lambda: rayleigh_exact(p, F, F1))
    t_m = best(lambda: mvb_coefficients(p, f, f1, z, z1, Bka, 3, smooth=op))
    t_an = best(lambda: analytic_average(p, f, f1, Bka, HARM, smooth=op), rep=3)
    print(f"Fig. 2(d) configuration: {2 * 40 + 1} orders retained, N = 2048")
    print(f"  one solution: C-method {t_c * 1e3:.0f} ms | Rayleigh {t_r * 1e3:.0f} ms | KA+MVB(n<=3) {t_m * 1e3:.0f} ms")
    print(f"  analytic average (21 runs): {t_an:.2f} s | Monte Carlo K={K}: Rayleigh {K * t_r:.0f} s ({K * t_r / t_an:.0f}x), "
          f"C-method {K * t_c:.0f} s ({K * t_c / t_an:.0f}x)")


if __name__ == "__main__":
    for L_rel in (10, 20):
        run(L_rel)
    run_fig_ensemble()
