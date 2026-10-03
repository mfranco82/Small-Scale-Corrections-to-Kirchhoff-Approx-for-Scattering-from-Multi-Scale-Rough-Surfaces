"""
Numerical-stability scan quoted in the limitation paragraph of the Results section.

Two-scale profile a cos(Kx) + b cos(Lambda K x), b = 0.004 L, Lambda = 5, lambda = 0.1 L, theta_i = 20 deg, TE,
eps = 2.25.  For growing amplitude a of the smooth component and for several truncations |r| <= rmax of the
Rayleigh orders it prints the condition number of the system matrix and the relative L1 error (over the
propagating orders) of KA, of KA+MVB (n <= 3) and of MVB (n <= 3) started from the EXACT zeroth-order
coefficients, all measured against the independent C-method (cmethod.py, M = 40).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, ensemble.py, ... live in the parent folder
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative

WL, B, LAM, TH = 0.1, 0.004, 5, 20.0


def eff(p, B_):
    return efficiencies(p, B_, np.zeros_like(B_))[0]


def l1(p, Bp, eref, pref):
    r = np.arange(-13, 14)                    # contains every propagating order of this configuration
    e = eff(p, Bp)[p.rmax + r]; er = eref[pref.rmax + r]
    m = p.prop_p[p.rmax + r]
    return np.abs(e - er)[m].sum() / er[m].sum()


if __name__ == "__main__":
    pref = Problem(1.0, WL, np.deg2rad(TH), 2.25, "TE", 40)
    print(f"{'a':>6} {'aK':>5} {'rmax':>5} {'cond':>8} | {'KA':>6} {'KA+MVB3':>9} {'exact d0 + MVB3':>16}")
    for a in (0.03, 0.045, 0.06, 0.07, 0.085, 0.10):
        f, f1 = cos_surface(pref, a); z, z1 = cos_surface(pref, B, LAM)
        F, F1 = f + z, f1 + z1
        eref = eff(pref, np.nan_to_num(c_method(pref, F, F1, spectral_derivative(pref, F, 2), M=40)[0]))
        for rmax in (30, 40, 60):
            p = Problem(1.0, WL, np.deg2rad(TH), 2.25, "TE", rmax)
            f, f1 = cos_surface(p, a); z, z1 = cos_surface(p, B, LAM)
            Bk, Be = ka_amplitudes(p, f, f1), rayleigh_exact(p, f, f1)
            dk, dmk = mvb_coefficients(p, f, f1, z, z1, Bk, 3)
            de, dme = mvb_coefficients(p, f, f1, z, z1, Be, 3)
            cond = np.linalg.cond(system_matrix(p, f + z, f1 + z1))
            print(f"{a:6.3f} {a * p.K:5.2f} {rmax:5d} {cond:8.1e} | {l1(p, Bk[0], eref, pref):6.3f} "
                  f"{l1(p, mvb_sum(dk, dmk, 1, 3)[0], eref, pref):9.3g} "
                  f"{l1(p, mvb_sum(de, dme, 1, 3)[0], eref, pref):16.3g}")
