"""
Statistics over random two-scale surfaces with FIXED amplitudes, UNIFORM RANDOM PHASES and FLAT-BAND spectra:
    smooth part   f(x)    = sum_{m=1..3}      A_s cos(mKx + phi_m)            (A_s fixed, flat band)
    small scale   zeta(x) = sum_{m=m1..m2}    A_b cos(mKx + psi_m)            (A_b fixed by the target k*sigma_zeta, flat band)
L = 10 lambda, sin(theta_i) = 0.35, eps = 2.25.  Phases are drawn independently for every realisation (both scales).
Reference: C-method.  Models: KA of the full profile, KA+MVB (n<=3) with KA zeroth order, and with exact zeroth order.
Error = relative L1 distance of the reflected / transmitted efficiencies over the propagating orders.
usage: python random_stats.py compute | summary   (cache for Figs. 2c and 3 of the paper; ~11 min on 8 cores)
"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)               # cached results
from scattering import *
from cmethod import c_method, spectral_derivative

WLr, SIN, EPS = 0.1, 0.35, 2.25
A_S, M_S = 0.008, (1, 2, 3)                              # smooth band, flat amplitude (units of L)
BANDS = {"B1: m=5-10": (5, 10), "B2: m=8-14": (8, 14), "B3: m=12-20": (12, 20)}
KSIG = [0.05, 0.1, 0.2, 0.3]                             # target k*sigma_zeta of the small scale
NREAL = 40
POLS = ("TE", "TM")
FILE = os.path.join(DATA, "random_stats.npz")
def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()

def build(p, band, ksig, rng):
    m1, m2 = BANDS[band]; ms = np.arange(m1, m2 + 1); M = len(ms)
    A_b = ksig / p.k * np.sqrt(2.0 / M)                  # sigma_zeta = A_b sqrt(M/2)
    ph, ps = rng.uniform(0, 2 * np.pi, len(M_S)), rng.uniform(0, 2 * np.pi, M)
    f = sum(A_S * np.cos(m * p.K * p.x + a) for m, a in zip(M_S, ph)); f1 = sum(-A_S * m * p.K * np.sin(m * p.K * p.x + a) for m, a in zip(M_S, ph))
    z = sum(A_b * np.cos(m * p.K * p.x + a) for m, a in zip(ms, ps)); z1 = sum(-A_b * m * p.K * np.sin(m * p.K * p.x + a) for m, a in zip(ms, ps))
    return f, f1, z, z1, m2

def one(args):
    ib, ik, ip, ir = args
    band, ksig, pol = list(BANDS)[ib], KSIG[ik], POLS[ip]
    out = np.full(14, np.nan)   # eR: KAfull, MVB_KA, MVB_ex | eT: same | cond | C_tot(actual) | C_smooth | max slope | k sigma_zeta(actual) | Rayleigh-vs-C | rmax
    try:
        with np.errstate(all="ignore"):
            th = np.arcsin(SIN)
            rng = np.random.default_rng(10_000 * (ib + 1) + 100 * ik + ir)         # same phases for TE and TM
            p0 = Problem(1.0, WLr, th, EPS, pol, 30, N=4096)
            m2 = BANDS[band][1]; rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * m2 + 12)
            p = Problem(1.0, WLr, th, EPS, pol, rmax, N=4096)
            f, f1, z, z1, _ = build(p, band, ksig, rng); F, F1 = f + z, f1 + z1
            ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)])
            Bs, Bf, Bx = ka_amplitudes(p, f, f1), ka_amplitudes(p, F, F1), rayleigh_exact(p, f, f1)
            def e2(B_):
                e = efficiencies(p, *B_); return l1(e[0], ref[0], p.prop_p), l1(e[1], ref[1], p.prop_m)
            op = SmoothOperator(p, f, f1)
            dpk, dmk = mvb_coefficients(p, f, f1, z, z1, Bs, 3, smooth=op); dpx, dmx = mvb_coefficients(p, f, f1, z, z1, Bx, 3, smooth=op)
            (out[0], out[3]), (out[1], out[4]), (out[2], out[5]) = e2(Bf), e2(mvb_sum(dpk, dmk, 1, 3)), e2(mvb_sum(dpx, dmx, 1, 3))
            out[6] = np.linalg.cond(system_matrix(p, f, f1))
            c3 = 2 * p.k * np.cos(th) ** 3
            out[7] = np.abs(spectral_derivative(p, F, 2)).max() / c3; out[8] = np.abs(spectral_derivative(p, f, 2)).max() / c3
            out[9] = np.abs(F1).max(); out[10] = p.k * z.std(); out[11] = e2(rayleigh_exact(p, F, F1))[0]; out[12] = rmax
    except Exception:
        pass
    return ib, ik, ip, ir, out

def compute():
    tasks = [(ib, ik, ip, ir) for ib in range(len(BANDS)) for ik in range(len(KSIG)) for ip in range(2) for ir in range(NREAL)]
    data = np.full((len(BANDS), len(KSIG), 2, NREAL, 14), np.nan); t0 = time.time()
    with Pool(8) as pool:
        for k, (ib, ik, ip, ir, out) in enumerate(pool.imap_unordered(one, tasks, chunksize=4)):
            data[ib, ik, ip, ir] = out
            if k % 200 == 0: print(f"{k}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    np.savez(FILE, data=data); print(f"saved in {time.time()-t0:.0f} s; failed: {int(np.isnan(data[..., 0]).sum())}")

def summary():
    d = np.load(FILE)["data"]; pc = lambda x, q: 100 * np.nanpercentile(x, q)
    print(f"{NREAL} random surfaces per cell; error % as median [p10 - p90];  gain = per-realisation ratio KA_full/series (median, min);  'wins' = fraction of surfaces where series < KA_full;  'stable' = series error < 5 %")
    print(f"{'band':>12} {'k*sig':>5} {'pol':>3} | {'C_tot med':>9} {'slope max':>9} | {'KA full R':>18} | {'series KA d0 R':>18} | {'series exact d0 R':>18} | {'gain med (min)':>15} | wins | stable | {'series KA d0 T':>16}")
    for ib, band in enumerate(BANDS):
        for ik, ks in enumerate(KSIG):
            for ip, pol in enumerate(POLS):
                D = d[ib, ik, ip]; eK, eM, eX, tK, tM = D[:, 0], D[:, 1], D[:, 2], D[:, 3], D[:, 4]
                g = eK / eM
                fm = lambda x: f"{pc(x,50):6.2f} [{pc(x,10):5.2f}-{pc(x,90):6.2f}]"
                print(f"{band:>12} {ks:5.2f} {pol:>3} | {np.nanmedian(D[:,7]):9.2f} {np.nanmedian(D[:,9]):9.2f} | {fm(eK)} | {fm(eM)} | {fm(eX)} | {np.nanmedian(g):7.1f} ({np.nanmin(g):5.2f}) | {100*np.mean(eM<eK):3.0f}% | {100*np.mean(eM<0.05):4.0f}% | {pc(tM,50):6.2f} [{pc(tM,10):5.2f}-{pc(tM,90):6.2f}]")

if __name__ == "__main__":
    {"compute": compute, "summary": summary}[sys.argv[1]]()
