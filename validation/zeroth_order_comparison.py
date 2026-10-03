"""Role of the Kirchhoff zeroth order (the paragraph before the limits of validity in the Results section).
Error (relative L1 over propagating orders, n <= 3) of the KA+MVB series started from the KA amplitudes (KA-d0) versus
started from the exact Rayleigh solution of the SMOOTH profile (exact-d0), against the Rayleigh solution of the full
profile.  Prints: periodic TE/TM (Fig. 2a,b), random profile (Fig. 2c), ensemble average (Fig. 2d), the angle sweep and
the kb sweep (Fig. 3a,b) and a few dielectric constants (Fig. 3c), reflected (R) and transmitted (T) orders.
R0/T0 = zeroth order alone, R3/T3 = n <= 3.   Usage:  python zeroth_order_comparison.py   (~1 min, 8 cores)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, ensemble.py, ... live in the parent folder
import sys, time
import numpy as np
from multiprocessing import Pool
from scattering import *
from ensemble import *
import make_final_figures as M

WL, A, B, LAM, EPS0, N = M.WL, M.A, M.B, M.LAM, M.EPS0, M.N
l1 = M.l1

def both(p, f, f1, z, z1, Bex, n=3):
    """returns reflected L1 and transmitted L1 errors: (KA, KA+MVB, exactd0+MVB, smooth-exact alone)"""
    d0 = {"ka": ka_amplitudes(p, f, f1), "sm": rayleigh_exact(p, f, f1)}
    ex = efficiencies(p, *Bex)
    out = {}
    for key, B0 in d0.items():
        dp, dm = mvb_coefficients(p, f, f1, z, z1, B0, n)
        e0 = efficiencies(p, *B0); e3 = efficiencies(p, *mvb_sum(dp, dm, 1, n))
        out[key] = dict(R0=l1(e0[0], ex[0], p.prop_p), R3=l1(e3[0], ex[0], p.prop_p),
                        T0=l1(e0[1], ex[1], p.prop_m), T3=l1(e3[1], ex[1], p.prop_m))
    return out

def ang(th):
    p = Problem(1.0, WL, np.deg2rad(th), EPS0, "TE", 40, N=N)
    f, f1 = cos_surface(p, A); z, z1 = cos_surface(p, B, LAM)
    return both(p, f, f1, z, z1, rayleigh_exact(p, f + z, f1 + z1))

def kbj(b):
    p = Problem(1.0, WL, np.deg2rad(20), EPS0, "TE", 40, N=N)
    f, f1 = cos_surface(p, A); z, z1 = cos_surface(p, b, LAM)
    return both(p, f, f1, z, z1, rayleigh_exact(p, f + z, f1 + z1))

def epsj(args):
    eps, pol = args
    p0 = Problem(1.0, WL, np.deg2rad(20), eps, pol, 30, N=N)
    rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30)
    p = Problem(1.0, WL, np.deg2rad(20), eps, pol, rmax, N=N)
    f, f1 = cos_surface(p, A); z, z1 = cos_surface(p, B, LAM)
    return both(p, f, f1, z, z1, rayleigh_exact(p, f + z, f1 + z1))

if __name__ == "__main__":
    fmt = lambda o: "KA-d0: R0 %.4f R3 %.4f T0 %.4f T3 %.4f | exact-d0: R0 %.4f R3 %.4f T0 %.4f T3 %.4f" % (
        o["ka"]["R0"], o["ka"]["R3"], o["ka"]["T0"], o["ka"]["T3"], o["sm"]["R0"], o["sm"]["R3"], o["sm"]["T0"], o["sm"]["T3"])
    for pol in ("TE", "TM"):
        p = Problem(1.0, WL, np.deg2rad(20), EPS0, pol, 40, N=N)
        f, f1 = cos_surface(p, A); z, z1 = cos_surface(p, B, LAM)
        print("periodic", pol, fmt(both(p, f, f1, z, z1, rayleigh_exact(p, f + z, f1 + z1))))
    p = Problem(1.0, WL, np.deg2rad(25), EPS0, "TE", 40, N=N)
    f, f1, z, z1 = M.random_profile(p)
    print("random   TE", fmt(both(p, f, f1, z, z1, rayleigh_exact(p, f + z, f1 + z1))))
    # ensemble (Fig 2d): analytic average with KA vs exact d0, vs the MC reference (recomputed with fewer samples is too slow; use exact numerical phase average with 200 samples)
    harm = [(m, 0.003 * (m / 4) ** (-0.5)) for m in range(4, 11)]
    op = SmoothOperator(p, f, f1)
    s = p.prop_p
    ref = None
    with Pool(8) as pool:
        res = pool.map(M.mc_chunk, [(100 + i, 250, harm) for i in range(8)])
    mp = np.mean([x[0] for x in res], axis=0)
    for name, d0 in (("KA", ka_amplitudes(p, f, f1)), ("exact", rayleigh_exact(p, f, f1))):
        (Ip, Im), _ = analytic_average(p, f, f1, d0, harm, smooth=op)
        ep, em = to_efficiency(p, Ip, Im)
        m = s & (mp > 1e-7)
        print("ensemble analytic with", name, "d0: L1 vs MC(2000) = %.4f ; d0 alone: %.4f" % (l1(ep, mp, m), l1(efficiencies(p, *d0)[0], mp, m)))
    ths = np.linspace(2.13, 60.07, 59)
    with Pool(8) as pool:
        A_ = pool.map(ang, ths); K_ = pool.map(kbj, [0.001, 0.002, 0.004, 0.008, 0.016])
        E_ = pool.map(epsj, [(e, pol) for pol in ("TE", "TM") for e in (1.2, 2.25, 6.0, 12.0, 16.0)])
    for key in ("ka", "sm"):
        print("angle median R3(%s) = %.4f  max %.4f" % (key, np.median([a[key]["R3"] for a in A_]), np.max([a[key]["R3"] for a in A_])))
    kb = 2 * np.pi / WL * np.array([0.001, 0.002, 0.004, 0.008, 0.016])
    for k, o in zip(kb, K_):
        print("kb %.3f  R3 KA-d0 %.4f  exact-d0 %.4f" % (k, o["ka"]["R3"], o["sm"]["R3"]))
    i = 0
    for pol in ("TE", "TM"):
        for e in (1.2, 2.25, 6.0, 12.0, 16.0):
            o = E_[i]; i += 1
            print("eps %5.2f %s  R3 %.4f/%.4f  T3 %.4f/%.4f  (T0 KA %.4f, smooth-exact T0 %.4f)  [KA-d0 / exact-d0]" % (
                e, pol, o["ka"]["R3"], o["sm"]["R3"], o["ka"]["T3"], o["sm"]["T3"], o["ka"]["T0"], o["sm"]["T0"]))
