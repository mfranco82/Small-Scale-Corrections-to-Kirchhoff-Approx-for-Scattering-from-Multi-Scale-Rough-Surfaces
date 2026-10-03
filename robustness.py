"""
Robustness of the (lambda_s/lambda, kb) window of KA+MVB versus KA of the FULL profile, one parameter at a time around the
base case  L = 10 lambda, theta_i ~ 20 deg (sin = 0.35), eps = 2.25, a = 0.03 L (aK = 0.19):
  * L/lambda = 5, 20          * incidence angle (sin theta = 0.05 ... 0.85)        * eps = 1.5, 4, 9        * aK = 0.10, 0.28, 0.38
Angles are chosen at half-way between Wood anomalies.  Reference: C-method.  n <= 3, KA zeroth order (and exact zeroth order).
Metrics: relative L1 error of the reflected (R) and transmitted (T) efficiencies over the propagating orders.
usage: python robustness.py compute | summary   (cache for Fig. 3 of the paper; ~7 min on 8 cores)
"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)               # cached results
from scattering import *
from cmethod import c_method, spectral_derivative

KBS = [0.1, 0.2, 0.3, 0.5]
LAMS = {5: [2, 3, 4, 5, 6, 8, 10], 10: [4, 5, 8, 10, 12, 15, 20], 20: [8, 10, 16, 20, 24, 30, 40]}   # lambda_s/lambda = L/(lambda*Lam)
POLS = ("TE", "TM")
FILE = os.path.join(DATA, "robustness.npz")

def safe_sin(Lrel, s):
    step = 1.0 / Lrel                                       # Wood anomalies at sin(theta) = integer multiples of lambda/L (mod 1)
    return (np.round(s / step - 0.5) + 0.5) * step

def configs():
    C = {"base": dict(L=10, sin=safe_sin(10, 0.342), eps=2.25, a=0.03)}
    C["L=5"] = dict(L=5, sin=0.35, eps=2.25, a=0.03)      # safe_sin(5, .) would sit exactly on a transmitted Wood anomaly (sqrt(eps)=1.5 = 7.5 lambda/L)
    C["L=20"] = dict(L=20, sin=safe_sin(20, 0.342), eps=2.25, a=0.03)
    for s in (0.05, 0.15, 0.55, 0.75, 0.85):
        C[f"sin={s:.2f}"] = dict(L=10, sin=safe_sin(10, s), eps=2.25, a=0.03)
    for e in (1.5, 4.0, 9.0):
        C[f"eps={e}"] = dict(L=10, sin=safe_sin(10, 0.342), eps=e, a=0.03)
    for a in (0.016, 0.045, 0.06):
        C[f"aK={2*np.pi*a:.2f}"] = dict(L=10, sin=safe_sin(10, 0.342), eps=2.25, a=a)
    return C
CONF = configs(); NAMES = list(CONF)

def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()

def cell(args):
    ic, ip, iL, ik = args
    name, c = NAMES[ic], CONF[NAMES[ic]]
    Lrel = c["L"]; Lam = LAMS[Lrel][iL]; kb = KBS[ik]; pol = POLS[ip]
    out = np.full(8, np.nan)       # eR: KAfull, MVB(KA), MVB(exact) ; eT: same ; cond ; Rayleigh-vs-C check
    try:
        with np.errstate(all="ignore"):
            wl = 1.0 / Lrel; th = np.arcsin(c["sin"]); b = kb * wl / (2 * np.pi)
            p0 = Problem(1.0, wl, th, c["eps"], pol, 30, N=4096)
            rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
            p = Problem(1.0, wl, th, c["eps"], pol, rmax, N=4096)
            f, f1 = cos_surface(p, c["a"]); z, z1 = cos_surface(p, b, Lam); F, F1 = f + z, f1 + z1
            Bc = c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)
            ref = efficiencies(p, *[np.nan_to_num(x) for x in Bc])
            Bs, Bf, Bx = ka_amplitudes(p, f, f1), ka_amplitudes(p, F, F1), rayleigh_exact(p, f, f1)
            def e2(B_):
                e = efficiencies(p, *B_); return l1(e[0], ref[0], p.prop_p), l1(e[1], ref[1], p.prop_m)
            op = SmoothOperator(p, f, f1)
            dpk, dmk = mvb_coefficients(p, f, f1, z, z1, Bs, 3, smooth=op); dpx, dmx = mvb_coefficients(p, f, f1, z, z1, Bx, 3, smooth=op)
            (out[0], out[3]), (out[1], out[4]), (out[2], out[5]) = e2(Bf), e2(mvb_sum(dpk, dmk, 1, 3)), e2(mvb_sum(dpx, dmx, 1, 3))
            out[6] = np.linalg.cond(system_matrix(p, f, f1))
            out[7] = e2(rayleigh_exact(p, F, F1))[0]
    except Exception:
        pass
    return ic, ip, iL, ik, out

def compute():
    tasks = [(ic, ip, iL, ik) for ic in range(len(NAMES)) for ip in range(2) for iL in range(7) for ik in range(len(KBS))]
    data = np.full((len(NAMES), 2, 7, len(KBS), 8), np.nan); t0 = time.time()
    with Pool(8) as pool:
        for k, (ic, ip, iL, ik, out) in enumerate(pool.imap_unordered(cell, tasks, chunksize=4)):
            data[ic, ip, iL, ik] = out
            if k % 200 == 0: print(f"{k}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    np.savez(FILE, data=data, names=NAMES, kbs=KBS)
    print(f"saved {FILE} in {time.time()-t0:.0f} s; failed cells: {int(np.isnan(data[..., 0]).sum())}")

def summary():
    d = np.load(FILE)["data"]
    print("CORE = kb in {0.2, 0.3} and lambda_s/lambda <= 1.25 (curvature of the small scale large).  gain = error KA(full) / error KA+MVB(n<=3, KA zeroth order)")
    print("EDGE = kb = 0.5, same lambda_s.   SMOOTH = lambda_s/lambda >= 2 (small scale gentle), kb in {0.2,0.3}: KA(full) error there.")
    hdr = f"{'config':>10} {'pol':>3} | {'core gain R: med [min]':>24} | {'core MVB err R %: med / max':>28} | {'core gain T: med':>16} | {'core MVB err T % max':>20} | {'edge MVB err R %: med':>22} | {'smooth KAfull R %':>17} | {'exact-d0 core R %':>17} | fails"
    print(hdr)
    for ic, name in enumerate(NAMES):
        Lrel = CONF[name]["L"]; lams = LAMS[Lrel]; rho = np.array([Lrel / L for L in lams])
        core_i = [i for i, r in enumerate(rho) if r <= 1.25]; smooth_i = [i for i, r in enumerate(rho) if r >= 2.0]
        for ip, pol in enumerate(POLS):
            D = d[ic, ip]                       # [Lam, kb, 8]
            core = D[np.ix_(core_i, [1, 2])]    # kb = 0.2, 0.3
            gR = core[..., 0] / core[..., 1]; gT = core[..., 3] / core[..., 4]
            edge = D[np.ix_(core_i, [3])]
            sm = D[np.ix_(smooth_i, [1, 2])]
            fails = int(np.isnan(D[..., 0]).sum())
            nm = np.nanmedian
            print(f"{name:>10} {pol:>3} | {nm(gR):10.1f} [{np.nanmin(gR):6.1f}]      | {100*nm(core[...,1]):8.2f} / {100*np.nanmax(core[...,1]):8.2f}       | {nm(gT):16.1f} | {100*np.nanmax(core[...,4]):20.2f} | {100*nm(edge[...,1]):22.1f} | {100*nm(sm[...,0]):17.2f} | {100*nm(core[...,2]):17.2f} | {fails}")

def compute_one(name):
    """Recompute a single configuration and merge it into the saved file."""
    d = np.load(FILE); data, names = d["data"].copy(), list(d["names"]); ic = names.index(name)
    tasks = [(ic, ip, iL, ik) for ip in range(2) for iL in range(7) for ik in range(len(KBS))]
    with Pool(8) as pool:
        for ic_, ip, iL, ik, out in pool.imap_unordered(cell, tasks, chunksize=2):
            data[ic_, ip, iL, ik] = out
    np.savez(FILE, data=data, names=names, kbs=KBS); print("recomputed", name)

if __name__ == "__main__":
    if sys.argv[1] == "compute_one": compute_one(sys.argv[2])
    else: {"compute": compute, "summary": summary}[sys.argv[1]]()
