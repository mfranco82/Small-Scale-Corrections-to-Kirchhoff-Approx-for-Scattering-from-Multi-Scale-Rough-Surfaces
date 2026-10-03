"""
Phase map of the three models in the plane (small-scale wavelength / lambda, k b).
Profile: z = a cos(Kx) + b cos(Lambda K x),  L = 10 lambda, a = 0.03 L (aK = 0.19), theta_i = 20 deg, eps = 2.25.
Reference: C-method (valid beyond the Rayleigh hypothesis).  Models: KA of the smooth profile, KA of the FULL profile,
KA+MVB (n<=3) with KA zeroth order, and with the exact (Rayleigh) zeroth order of the smooth profile.
Error = relative L1 distance of the reflected efficiencies over the propagating orders.
usage: python phase_map.py compute | table   (cache for Fig. 4 of the paper; ~1.5 min on 8 cores)
"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)               # cached results
from scattering import *
from cmethod import c_method, spectral_derivative

WLr, A_L, TH, EPS = 0.1, 0.03, 20.0, 2.25
LAMS = [3, 5, 8, 10, 12, 15, 18, 20, 25]
KBS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
POLS = ("TE", "TM")
FILE = os.path.join(DATA, "phase_map.npz")

def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()

def cell(args):
    ip, iL, ik = args
    pol, Lam, kb = POLS[ip], LAMS[iL], KBS[ik]
    out = np.full(6, np.nan)           # KA_smooth, KA_full, MVB_KA(n<=3), MVB_exact(n<=3), cond, ref check (Rayleigh vs C)
    try:
        with np.errstate(all="ignore"):
            b = kb / (2 * np.pi / WLr)
            p0 = Problem(1.0, WLr, np.deg2rad(TH), EPS, pol, 30, N=4096)
            rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
            p = Problem(1.0, WLr, np.deg2rad(TH), EPS, pol, rmax, N=4096)
            f, f1 = cos_surface(p, A_L); z, z1 = cos_surface(p, b, Lam); F, F1 = f + z, f1 + z1
            Bc = c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)
            ref = efficiencies(p, *[np.nan_to_num(x) for x in Bc])[0]; s = p.prop_p
            Bs, Bf, Bx = ka_amplitudes(p, f, f1), ka_amplitudes(p, F, F1), rayleigh_exact(p, f, f1)
            ef = lambda B_: l1(efficiencies(p, *B_)[0], ref, s)
            out[0], out[1] = ef(Bs), ef(Bf)
            op = SmoothOperator(p, f, f1)
            dp, dm = mvb_coefficients(p, f, f1, z, z1, Bs, 3, smooth=op); out[2] = ef(mvb_sum(dp, dm, 1, 3))
            dp, dm = mvb_coefficients(p, f, f1, z, z1, Bx, 3, smooth=op); out[3] = ef(mvb_sum(dp, dm, 1, 3))
            out[4] = np.linalg.cond(system_matrix(p, f, f1))
            out[5] = l1(efficiencies(p, *rayleigh_exact(p, F, F1))[0], ref, s)
    except Exception as e:
        pass
    return ip, iL, ik, out

def compute():
    tasks = [(ip, iL, ik) for ip in range(2) for iL in range(len(LAMS)) for ik in range(len(KBS))]
    data = np.full((2, len(LAMS), len(KBS), 6), np.nan); t0 = time.time()
    with Pool(8) as pool:
        for ip, iL, ik, out in pool.imap_unordered(cell, tasks):
            data[ip, iL, ik] = out
    np.savez(FILE, data=data, lams=LAMS, kbs=KBS)
    print(f"saved {FILE} in {time.time()-t0:.0f} s")

def table():
    d = np.load(FILE)["data"]
    for ip, pol in enumerate(POLS):
        print(f"\n=== {pol}:  L1 error (%) as  KA_full | MVB(KA d0) | MVB(exact d0) | [Rayleigh-vs-C ref check]   rows: kb, columns: Lambda (lambda_s/lambda = 10/Lambda)")
        print("kb\\Lam " + "".join(f"{L:>26d}" for L in LAMS))
        for ik, kb in enumerate(KBS):
            row = f"{kb:5.2f}  "
            for iL in range(len(LAMS)):
                v = d[ip, iL, ik]
                fmt = lambda x: ("   --" if not np.isfinite(x) else (f"{100*x:5.1f}" if x < 9.99 else " >999"))
                row += f"{fmt(v[1])}|{fmt(v[2])}|{fmt(v[3])}|{v[5]:6.0e}  "
            print(row)

if __name__ == "__main__":
    {"compute": compute, "table": table}[sys.argv[1]]()
