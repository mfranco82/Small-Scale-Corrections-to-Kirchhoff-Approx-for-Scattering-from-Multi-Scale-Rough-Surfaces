"""
Validation of the reference ("exact") solution used in the Results section.

The reference is the Rayleigh (Fourier-Galerkin) solution of the full profile f+zeta (scattering.rayleigh_exact).
It shares the Rayleigh expansion with the MVB recursion, so it is NOT an independent benchmark by itself.
Here it is compared with the C-method (cmethod.py), which does not use the Rayleigh hypothesis.

Result (see printed table): for all the profiles of the figures the two agree to ~1e-9 relative
(limited by the tiny loss, 1e-10, used by the C-method to select outgoing modes).  Outside this regime
(pure cosine, a*K >~ 0.4) the Rayleigh solution degrades (1e-6 ... 1e-2 at a*K = 0.4 ... 0.63) and the
Rayleigh matrix becomes extremely ill-conditioned (cond ~ 1e20 at rmax=60), which makes the KA+MVB
recursion numerically unstable unless rmax is reduced (see the notes at the bottom of the printout).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, ensemble.py, ... live in the parent folder
import numpy as np
from scattering import *
from cmethod import c_method, spectral_derivative


def eff(p, B):
    return efficiencies(p, B, np.zeros_like(B))[0]


def compare(label, p, F, tol=1e-7):
    F1, F2 = spectral_derivative(p, F, 1), spectral_derivative(p, F, 2)
    Br = rayleigh_exact(p, F, F1)
    Bc = c_method(p, F, F1, F2, M=p.rmax)
    er, ec = efficiencies(p, *Br), efficiencies(p, *Bc)
    sp, sm = p.prop_p, p.prop_m
    big = sp & (er[0] > 1e-9)
    rel_p = (np.abs(ec[0] - er[0])[big] / er[0][big]).max()
    abs_p = np.abs(ec[0] - er[0])[sp].max()
    abs_m = np.abs(ec[1] - er[1])[sm].max()
    print(f"{label:34s} max rel diff e+ (orders > 1e-9): {rel_p:.1e} | max abs diff e+: {abs_p:.1e}, e-: {abs_m:.1e}")
    assert rel_p < tol and abs_p < 1e-9 and abs_m < 1e-8, label


if __name__ == "__main__":
    # flat interface: both solvers must give the Fresnel coefficients
    p = Problem(1.0, 0.1, np.deg2rad(20), 2.25, "TE", rmax=20)
    z0 = np.zeros(p.N)
    Bp, Bm = c_method(p, z0, z0, z0, M=20)
    R = fresnel_local(p, np.zeros(1))[0]
    assert abs(Bp[p.rmax] - R) < 1e-9 and abs(Bm[p.rmax] - (1 + R)) < 1e-9
    print("flat interface: C-method reproduces Fresnel")

    for pol in ("TE", "TM"):
        p = Problem(1.0, 0.1, np.deg2rad(20), 2.25, pol, 60)
        compare(f"two-scale cos+cos (Fig. 2a,b) {pol}", p, 0.03 * np.cos(p.K * p.x) + 0.004 * np.cos(5 * p.K * p.x))

    # random asymmetric profile of Fig. 2(c) (same generator / seed as make_final_figures.random_profile)
    for pol in ("TE", "TM"):
        p = Problem(1.0, 0.1, np.deg2rad(25), 2.25, pol, 70)
        rng = np.random.default_rng(3)
        def rp(kmin, kmax, amp, decay):
            g = np.zeros(p.N)
            for m in range(kmin, kmax + 1):
                am = amp * (m / kmin) ** (-decay) * rng.normal(); ph = rng.uniform(0, 2 * np.pi)
                g += am * np.cos(m * p.K * p.x + ph)
            return g
        F = rp(1, 3, 0.012, 1.0) + rp(4, 10, 0.003, 0.5)
        compare(f"random profile (Fig. 2c) {pol}", p, F)

    # strongly asymmetric profiles (a mirror-sign error in the slope would show up here, not for even profiles)
    for th in (10, 35, 50):
        p = Problem(1.0, 0.1, np.deg2rad(th), 2.25, "TE", 60)
        F = 0.03 * np.cos(p.K * p.x) + 0.02 * np.sin(2 * p.K * p.x + 0.7) + 0.004 * np.cos(5 * p.K * p.x + 1.1)
        compare(f"asymmetric profile, theta_i={th}", p, F)
    print("reference validated against the independent C-method")
