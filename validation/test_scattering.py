"""Validation of the analytical recursion and of the KA amplitudes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, ensemble.py, ... live in the parent folder
import numpy as np
from scattering import *

def setup(pol="TE", a=0.04, b=0.003, Lam=5, wl=0.1, th=20.0, eps=2.25, rmax=60):
    p = Problem(L=1.0, wavelength=wl, theta_i=np.deg2rad(th), eps=eps, pol=pol, rmax=rmax)
    f, f1 = cos_surface(p, a)
    z, z1 = cos_surface(p, b, Lam)
    return p, f, f1, z, z1

def test_flat():
    p = Problem(rmax=5)
    z0 = np.zeros(p.N)
    Bp, Bm = rayleigh_exact(p, z0, z0)
    Bk, Tk = ka_amplitudes(p, z0, z0)
    R = fresnel_local(p, np.zeros(1))[0]
    assert abs(Bp[p.rmax] - R) < 1e-12 and abs(Bk[p.rmax] - R) < 1e-12
    print("flat: OK  R =", R)

def test_energy(pol):
    p, f, f1, z, z1 = setup(pol)
    Bp, Bm = rayleigh_exact(p, f + z, f1 + z1)
    ep, em = efficiencies(p, Bp, Bm)
    print(f"{pol} exact Rayleigh  R+T-1 = {ep.sum() + em.sum() - 1:.2e}")
    assert abs(ep.sum() + em.sum() - 1) < 1e-6

def test_recursion(pol):
    """With exact d_0 the MVB series must converge to the exact solution for f+delta*zeta,
    the error of order n scaling as b^(n+1)."""
    errs = {}
    for b in (0.002, 0.001):
        p, f, f1, z, z1 = setup(pol, b=b, rmax=40)
        d0 = rayleigh_exact(p, f, f1)
        dp, dm = mvb_coefficients(p, f, f1, z, z1, d0, nmax=4)
        Bex = rayleigh_exact(p, f + z, f1 + z1)
        errs[b] = [np.abs(mvb_sum(dp, dm, 1.0, n)[0] - Bex[0])[p.prop_p].max() for n in range(5)]
    print(f"{pol} MVB(d0 exact) error of propagating B+ vs order (b=0.002):",
          ["%.1e" % e for e in errs[0.002]])
    for n in range(5):
        ratio = errs[0.002][n] / errs[0.001][n]
        assert abs(np.log2(ratio) - (n + 1)) < 0.35, (n, ratio)
    print(f"{pol} error scaling  ~ b^(n+1): OK")

def test_ka_smooth(pol):
    p, f, f1, z, z1 = setup(pol, a=0.02)
    Be = rayleigh_exact(p, f, f1)
    Bk = ka_amplitudes(p, f, f1)
    ee = efficiencies(p, *Be); ek = efficiencies(p, *Bk)
    print(f"{pol} KA smooth: e+_0 exact {ee[0][p.rmax]:.4f}  KA {ek[0][p.rmax]:.4f}; "
          f"R_KA+T_KA = {ek[0].sum()+ek[1].sum():.4f}")

if __name__ == "__main__":
    test_flat()
    for pol in ("TE", "TM"):
        test_energy(pol); test_recursion(pol); test_ka_smooth(pol)
    print("all passed")
