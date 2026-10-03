"""
Accuracy of the analytic ensemble average (Eq. (ensemble_avg)) versus the strength of the small scale, in the configuration of
Fig. 2(d): random multi-scale smooth profile, theta_i = 25 deg, eps = 2.25, TE, roughness harmonics m = 4..10 with
A_m = A0 (m/4)^(-1/2) and random phases.  Reference: Monte Carlo over K exact (Rayleigh) solutions.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, ensemble.py, ... live in the parent folder
import time
import numpy as np
from multiprocessing import Pool
from scattering import *
from ensemble import *
from make_final_figures import random_profile, WL, EPS0, N

K_MC = 2000


def chunk(args):
    seed, k, harm = args
    p = Problem(1.0, WL, np.deg2rad(25), EPS0, "TE", 40, N=N)
    f, f1, _, _ = random_profile(p)
    return monte_carlo(p, f, f1, harm, k, seed=seed)


if __name__ == "__main__":
    p = Problem(1.0, WL, np.deg2rad(25), EPS0, "TE", 40, N=N)
    f, f1, _, _ = random_profile(p)
    Bka = ka_amplitudes(p, f, f1)
    op = SmoothOperator(p, f, f1)
    for A0 in (0.0015, 0.0022, 0.003):
        harm = [(m, A0 * (m / 4) ** (-0.5)) for m in range(4, 11)]
        ksig = p.k * np.sqrt(sum(A ** 2 / 2 for _, A in harm))
        t = time.perf_counter(); (Ip, Im), _ = analytic_average(p, f, f1, Bka, harm, smooth=op); t_an = time.perf_counter() - t
        ep, em = to_efficiency(p, Ip, Im)
        with Pool(8) as pool:
            res = pool.map(chunk, [(100 + i, K_MC // 8, harm) for i in range(8)])
        mp = np.mean([x[0] for x in res], axis=0); mm = np.mean([x[1] for x in res], axis=0)
        err = np.sqrt(np.mean([x[2] ** 2 for x in res], axis=0) / 8)
        m = p.prop_p & (mp > 1e-7)
        e_ka = efficiencies(p, *Bka)[0]
        print(f"A0={A0}: k*sigma_zeta = {ksig:.2f} | L1 error of the analytic mean spectrum {np.abs(ep - mp)[m].sum() / mp[m].sum():.3f} "
              f"(KA alone {np.abs(e_ka - mp)[m].sum() / mp[m].sum():.3f}; MC statistical error {np.sqrt((err[m] ** 2).sum()) / mp[m].sum():.4f}) "
              f"| transmitted {np.abs(em - mm)[p.prop_m].sum() / mm[p.prop_m].sum():.4f} | analytic time {t_an:.2f} s")
