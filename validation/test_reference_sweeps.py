"""
Reference (Rayleigh) vs independent C-method on samples of the parameter sweeps of Figs. 3 and 4:
dielectric constant (eps = 1.2 ... 16), incidence angle (5 ... 58 deg) and roughness amplitude (kb up to 1), TE and TM.
Prints (i) the relative L1 distance between the efficiencies of both solvers and (ii) the worst per-order relative difference
among the orders with e > 1e-7 (reflected and transmitted).  Both solvers retain rmax = r_trans + 25 orders.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, ensemble.py, ... live in the parent folder
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative

WL, A, B, LAM = 0.1, 0.03, 0.004, 5


def compare(eps, th, b, pol, margin=25, N=2048):
    p0 = Problem(1.0, WL, np.deg2rad(th), eps, pol, 30, N=N)
    rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + margin, 30)
    p = Problem(1.0, WL, np.deg2rad(th), eps, pol, rmax, N=N)
    F = A * np.cos(p.K * p.x) + b * np.cos(LAM * p.K * p.x)
    F1, F2 = spectral_derivative(p, F, 1), spectral_derivative(p, F, 2)
    er = efficiencies(p, *rayleigh_exact(p, F, F1))
    ec = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, F2, M=rmax, delta=1e-8)])
    l1, worst = [], []
    for k, m in ((0, p.prop_p), (1, p.prop_m)):
        l1.append(np.abs(ec[k] - er[k])[m].sum() / er[k][m].sum())
        big = m & (er[k] > 1e-7)
        worst.append((np.abs(ec[k] - er[k])[big] / er[k][big]).max())
    return max(l1), max(worst)


if __name__ == "__main__":
    w1 = w2 = 0
    print(f"{'sweep':>10} {'value':>7} {'pol':>3} | L1 distance | worst order (e>1e-7)")
    for pol in ("TE", "TM"):
        cases = [("eps", e_, (e_, 20, B)) for e_ in (1.2, 5.0, 16.0)] + \
                [("theta_i", t_, (2.25, t_, B)) for t_ in (5.0, 35.0, 58.0)] + \
                [("kb", 2 * np.pi / WL * b_, (2.25, 20, b_)) for b_ in (0.001, 0.008, 0.016)]
        for name, val, args in cases:
            d1, d2 = compare(*args, pol); w1 = max(w1, d1); w2 = max(w2, d2)
            print(f"{name:>10} {val:7.2f} {pol:>3} | {d1:11.1e} | {d2:.1e}")
    print(f"worst: L1 {w1:.1e}, per order {w2:.1e}")
    assert w1 < 2e-6 and w2 < 2e-4
