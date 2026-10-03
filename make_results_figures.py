"""
Results figures of the paper, built from the complementarity of the two expansions:
  KA            : expansion in the CURVATURE of the surface (valid for C = max|F''| / (2 k cos^3 theta_i) << 1)
  variation of boundaries about the smooth periodic surface : expansion in the HEIGHT of the small scale (k b << 1)
Both zeroth orders of the series are shown: the Kirchhoff amplitudes of the smooth profile ("KA zeroth order") and the exact (Rayleigh) solution
of the smooth profile ("exact zeroth order").  The baseline is KA applied to the FULL profile (the two-scale KA model); reference = C-method.
  figures/fig_spectra.pdf   (Fig. 2): spectra of a periodic and a random flat-band surface, deviation per order, convergence with the order n
  figures/fig_Caxis.pdf     (Fig. 3): error of each model versus the KA validity parameter C of the full profile (central figure)
  figures/fig_phasemap.pdf  (Fig. 4): error in the plane (Lambda, kb)
  numbers                   : prints the numbers quoted in Secs. 4 and 5 of the paper
usage: python make_results_figures.py spectra | caxis | phasemap | numbers
spectra recomputes three cases (~20 s); caxis, phasemap and numbers read the caches in data/ (robustness.npz, random_stats.npz, phase_map.npz),
which are produced by robustness.py, random_stats.py and phase_map.py.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from scattering import *
from cmethod import c_method, spectral_derivative
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "text.latex.preamble": r"\usepackage{mathpazo}\usepackage{amsmath}",
                     "font.size": 8, "axes.grid": True, "grid.alpha": .3, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
OUT = os.path.join(HERE, "figures") + os.sep; os.makedirs(OUT, exist_ok=True)
C_KAF, C_KA0, C_EX0, C_REF, C_Z0 = "tab:blue", "tab:red", "tab:green", "k", "0.55"
WL, SIN, EPS, A_SMOOTH = 0.1, 0.35, 2.25, 0.03          # lambda/L (L = 10 lambda), sin(theta_i), eps, smooth amplitude a/L
def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()


def case(pol, builder, Lam_max):
    th = np.arcsin(SIN)
    p0 = Problem(1.0, WL, th, EPS, pol, 30, N=4096)
    rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam_max + 12)
    p = Problem(1.0, WL, th, EPS, pol, rmax, N=4096)
    f, f1, z, z1 = builder(p)
    F, F1 = f + z, f1 + z1
    ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)])
    B_ka0, B_ex0, B_kaf = ka_amplitudes(p, f, f1), rayleigh_exact(p, f, f1), ka_amplitudes(p, F, F1)
    op = SmoothOperator(p, f, f1)
    S = {}
    for key, B0 in (("ka", B_ka0), ("ex", B_ex0)):
        dp, dm = mvb_coefficients(p, f, f1, z, z1, B0, 4, smooth=op)
        S[key] = [efficiencies(p, *mvb_sum(dp, dm, 1, n)) for n in range(5)]
    return dict(p=p, ref=ref, kaf=efficiencies(p, *B_kaf), ka0=efficiencies(p, *B_ka0), ex0=efficiencies(p, *B_ex0), S=S)


def cos_builder(Lam, kb):
    def b(p):
        bb = kb / p.k
        f, f1 = cos_surface(p, A_SMOOTH); z, z1 = cos_surface(p, bb, Lam)
        return f, f1, z, z1
    return b


def random_builder(ib, ik, ir):
    import random_stats as RS
    def b(p):
        rng = np.random.default_rng(10_000 * (ib + 1) + 100 * ik + ir)
        f, f1, z, z1, _ = RS.build(p, list(RS.BANDS)[ib], RS.KSIG[ik], rng)
        return f, f1, z, z1
    return b


def pick_median_surface(ib=1, ik=2):
    """Index of the random surface (band ib, k*sigma index ik, TE) whose errors are closest to the medians of the 40 realisations."""
    import random_stats as RS
    d = np.load(RS.FILE)["data"][ib, ik, 0]
    e = np.log(d[:, [0, 1]]); med = np.median(e, axis=0)
    return int(np.argmin(((e - med) ** 2).sum(axis=1)))


def spectra(save=True):
    LAM, KB = 15, 0.25
    cases = {"cosTE": case("TE", cos_builder(LAM, KB), LAM), "cosTM": case("TM", cos_builder(LAM, KB), LAM)}
    ir = pick_median_surface(); import random_stats as RS
    cases["rndTE"] = case("TE", random_builder(1, 2, ir), max(RS.BANDS["B2: m=8-14"]))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.2))
    titles = {"cosTE": r"(a) TE, periodic profile", "rndTE": r"(c) TE, random flat-band profile"}
    stats = {}
    for key in ("cosTE", "cosTM", "rndTE"):
        c = cases[key]; p = c["p"]; ref = c["ref"][0]; s = p.prop_p
        stats[key] = dict(ka0=l1(c["ka0"][0], ref, s), kaf=l1(c["kaf"][0], ref, s), ser_ka=[l1(c["S"]["ka"][n][0], ref, s) for n in range(5)],
                          ser_ex=[l1(c["S"]["ex"][n][0], ref, s) for n in range(5)], Ttr=[l1(c["kaf"][1], c["ref"][1], p.prop_m), l1(c["S"]["ka"][3][1], c["ref"][1], p.prop_m), l1(c["S"]["ex"][3][1], c["ref"][1], p.prop_m)])
    from matplotlib.ticker import MaxNLocator
    for ax, key in zip((axs[0, 0], axs[1, 0]), ("cosTE", "rndTE")):
        c = cases[key]; p = c["p"]; s = p.prop_p; r = p.r[s]
        ref = c["ref"][0]
        ax.semilogy(r, np.maximum(c["ka0"][0][s], 1e-14), ":", color=C_Z0, lw=1.0)
        ax.semilogy(r, c["kaf"][0][s], "--", color=C_KAF, lw=1.1)
        ax.semilogy(r, c["S"]["ka"][3][0][s], "-", color=C_KA0, lw=1.2)
        ax.semilogy(r, c["S"]["ex"][3][0][s], "-.", color=C_EX0, lw=1.2)
        ax.semilogy(r, np.maximum(ref[s], 1e-14), "o", color=C_REF, ms=2.6, mfc="none")
        ax.set_title(titles[key]); ax.set_ylim(1e-8, 0.1); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # (b): deviation from the reference order by order, for the periodic TE profile of (a)
    ax = axs[0, 1]; c = cases["cosTE"]; p = c["p"]; s = p.prop_p; r = p.r[s]; ref = c["ref"][0]
    dev = lambda e: np.maximum(np.abs(e[s] / ref[s] - 1), 1e-5)
    ax.semilogy(r, dev(c["kaf"][0]), "--s", color=C_KAF, lw=1.0, ms=2.6)
    ax.semilogy(r, dev(c["S"]["ka"][3][0]), "-o", color=C_KA0, lw=1.1, ms=2.6)
    ax.semilogy(r, dev(c["S"]["ex"][3][0]), "-.^", color=C_EX0, lw=1.1, ms=2.6)
    ax.set_title(r"(b) TE, periodic profile: $|e_r/e_r^{\rm ref}-1|$"); ax.set_ylim(1e-4, 30); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(r"Floquet order $r$")
    for ax in (axs[0, 0], axs[1, 0]):
        ax.set_xlabel(r"Floquet order $r$")
    axs[0, 0].set_ylabel(r"efficiency $e^+_r$"); axs[1, 0].set_ylabel(r"efficiency $e^+_r$")
    ax = axs[1, 1]
    n_ = np.arange(5)
    for key, ls, lab in (("cosTE", "-", "TE"), ("cosTM", "--", "TM")):
        ax.semilogy(n_, 100 * np.array(stats[key]["ser_ka"]), ls, color=C_KA0, marker="o", ms=3, lw=1.1)
        ax.semilogy(n_, 100 * np.array(stats[key]["ser_ex"]), ls, color=C_EX0, marker="^", ms=3, lw=1.1)
        ax.axhline(100 * stats[key]["kaf"], color=C_KAF, ls=ls, lw=0.9)
    ax.set_xlabel(r"perturbative order $n$ (sum up to $n$)"); ax.set_ylabel("relative error (\\%)"); ax.set_xticks(n_)
    ax.set_title(r"(d) convergence: periodic TE (solid), TM (dashed)"); ax.set_ylim(3e-3, 300)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=C_REF, marker="o", mfc="none", ls="", ms=3.5), Line2D([0], [0], color=C_KAF, ls="--"), Line2D([0], [0], color=C_Z0, ls=":"),
         Line2D([0], [0], color=C_KA0), Line2D([0], [0], color=C_EX0, ls="-.")]
    fig.legend(h, ["reference (C-method)", "KA, full profile", r"zeroth order alone (KA, smooth profile)", r"KA+MVB, $n\leq3$, KA zeroth order", r"KA+MVB, $n\leq3$, exact zeroth order"],
               loc="upper center", ncol=3, frameon=False, handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    if save: fig.savefig(OUT + "fig_spectra.pdf")
    plt.close(fig)
    print("random surface used in (c): realisation index", ir)
    for k_, v in stats.items():
        print(f"  {k_}: KA(smooth, zeroth order alone) {100*v['ka0']:.1f}% | KA(full) {100*v['kaf']:.2f}% | series KA d0 n=0..4: {['%.2f' % (100*x) for x in v['ser_ka']]} | exact d0: {['%.3f' % (100*x) for x in v['ser_ex']]} | transmitted L1 (KA full, KA d0 n<=3, exact d0 n<=3): {['%.2f' % (100*x) for x in v['Ttr']]} %")
    return cases, stats


# ------------------------------------------------------------------------------------------------------------------------------------------------
def gather(cond_max=1e15):
    """All cells/surfaces with their KA validity parameter C of the full profile.  Excluded: grazing incidence (sin>=0.75), aK=0.38 and cond(M)>cond_max."""
    import robustness as R, phase_map as P, random_stats as RS
    rows = []        # (pol, family, C_tot, C_smooth, eKAfull, eKA0, eEX0, cond)
    d = np.load(R.FILE)["data"]
    for ic, name in enumerate(R.NAMES):
        if name in ("sin=0.75", "sin=0.85", "aK=0.38"): continue
        c = R.CONF[name]; Lr = c["L"]; cs = np.sqrt(1 - c["sin"] ** 2) ** 3
        for ip, pol in enumerate(R.POLS):
            for iL, Lam in enumerate(R.LAMS[Lr]):
                for ik, kb in enumerate(R.KBS):
                    v = d[ic, ip, iL, ik]; Csm = np.pi * c["a"] / Lr / cs; Cl = np.pi * kb / (2 * np.pi * Lr) * Lam ** 2 / Lr / cs
                    rows.append((pol, "cos", Csm + Cl, Csm, v[0], v[1], v[2], v[6], v[7]))
    pm = np.load(P.FILE)["data"]; cs = np.cos(np.deg2rad(P.TH)) ** 3
    for ip, pol in enumerate(P.POLS):
        for iL, Lam in enumerate(P.LAMS):
            for ik, kb in enumerate(P.KBS):
                v = pm[ip, iL, ik]; Csm = np.pi * P.A_L * P.WLr / cs; Cl = np.pi * (kb * P.WLr / (2 * np.pi)) * Lam ** 2 * P.WLr / cs
                rows.append((pol, "cos", Csm + Cl, Csm, v[1], v[2], v[3], v[4], v[5]))
    rd = np.load(RS.FILE)["data"]
    for ib in range(rd.shape[0]):
        for ik in range(rd.shape[1]):
            for ip, pol in enumerate(RS.POLS):
                for ir in range(rd.shape[3]):
                    v = rd[ib, ik, ip, ir]; rows.append((pol, "rnd", v[7], v[8], v[0], v[1], v[2], v[6], v[11]))
    pol = np.array([r[0] for r in rows]); fam = np.array([r[1] for r in rows]); X = np.array([r[2:] for r in rows], float)
    ok = np.isfinite(X[:, :6]).all(axis=1) & (X[:, 5] < cond_max) & (X[:, 0] > 0)
    return pol[ok], fam[ok], X[ok], int((~ok).sum()), len(rows)


def caxis(save=True):
    pol, fam, X, n_excl, n_all = gather()
    C, eK, eM, eX = X[:, 0], X[:, 2], X[:, 3], X[:, 4]
    bins = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 4.0]); mid = np.sqrt(bins[:-1] * bins[1:])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    out = {}
    for ax, pl in zip(axs, ("TE", "TM")):
        m = pol == pl
        for e, col, mk in ((eK, C_KAF, "s"), (eM, C_KA0, "o"), (eX, C_EX0, "^")):
            for fm, size, al in (("cos", 5, 0.30), ("rnd", 4, 0.22)):
                s_ = m & (fam == fm)
                ax.scatter(C[s_], np.clip(100 * e[s_], 2e-3, 1e3), s=size, marker=mk if fm == "cos" else ".", color=col, alpha=al, lw=0, rasterized=True)
            med = [np.median(100 * e[m & (fam == "cos") & (C >= lo) & (C < hi)]) if (m & (fam == "cos") & (C >= lo) & (C < hi)).sum() > 3 else np.nan for lo, hi in zip(bins[:-1], bins[1:])]
            ax.plot(mid, med, "-" + mk, color=col, lw=1.6, ms=4, mec="w", mew=0.4)
        sel = m & (fam == "cos") & (C > 0.03)
        k = np.polyfit(np.log(C[sel]), np.log(eK[sel]), 1)
        cc = np.logspace(np.log10(0.03), np.log10(4), 40); ax.plot(cc, 100 * np.exp(k[1]) * cc ** k[0], "k--", lw=0.9)
        out[pl] = (np.exp(k[1]), k[0], m.sum())
        ax.axvspan(0.01, 0.05, color="0.88", alpha=0.7, lw=0); ax.axhline(1, color="k", lw=0.5, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.01, 4); ax.set_ylim(2e-3, 1e3)
        ax.set_xlabel(r"KA validity parameter $C=\max|F''|/(2k\cos^3\theta_i)$"); ax.set_title(f"({'a' if pl == 'TE' else 'b'}) {pl}", fontsize=8, loc="left")
        ax.text(0.0105, 3e2, "both scales\nKA-valid", fontsize=6.5, va="top")
    axs[0].set_ylabel("relative $L^1$ error (\\%)")
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=C_KAF, marker="s", ms=4), Line2D([0], [0], color=C_KA0, marker="o", ms=4), Line2D([0], [0], color=C_EX0, marker="^", ms=4),
         Line2D([0], [0], color="k", ls="--", lw=0.9)]
    fig.legend(h, ["KA, full profile", r"KA+MVB ($n\leq3$), KA zeroth order", r"KA+MVB ($n\leq3$), exact zeroth order", r"fit of KA, full profile"], loc="upper center", ncol=4, frameon=False, fontsize=6.5, handlelength=1.8, columnspacing=1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save: fig.savefig(OUT + "fig_Caxis.pdf")
    plt.close(fig)
    print(f"cells/surfaces used: {len(C)} of {n_all} ({n_excl} excluded: grazing incidence, aK=0.38 or cond(M)>1e15); per polarisation TE {np.sum(pol=='TE')}, TM {np.sum(pol=='TM')}")
    for pl in ("TE", "TM"): print(f"  KA(full) fit, cosine profiles, {pl}: error = {out[pl][0]:.2f} * C^{out[pl][1]:.2f}")
    print("median error (%) by C bin  [KA full | KA+MVB KA d0 | KA+MVB exact d0 | median gain | fraction series<KA full | n]")
    for pl in ("TE", "TM"):
        print(" ", pl)
        for lo, hi in zip(bins[:-1], bins[1:]):
            s_ = (pol == pl) & (C >= lo) & (C < hi)
            if s_.sum() > 3: print(f"    C {lo:5.2f}-{hi:5.2f}: {100*np.median(eK[s_]):7.2f} | {100*np.median(eM[s_]):6.2f} | {100*np.median(eX[s_]):6.2f} | {np.median(eK[s_]/eM[s_]):5.1f} | {100*np.mean(eM[s_]<eK[s_]):3.0f}% | {s_.sum()}")
    return out


def phasemap():
    import phase_map as P
    from matplotlib.colors import LogNorm
    pm = np.load(P.FILE)["data"][0]                       # TE:  [Lam, kb, (KAsmooth, KAfull, MVB_KA, MVB_ex, cond, check)]
    Lams, kbs = np.array(P.LAMS, float), np.array(P.KBS, float)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    cs = np.cos(np.deg2rad(P.TH)) ** 3
    Cfun = lambda Lam, kb: np.pi * P.WLr / cs * (P.A_L + (kb * P.WLr / (2 * np.pi)) * Lam ** 2)
    LL, KK = np.meshgrid(np.geomspace(3, 25, 200), np.geomspace(0.05, 1.0, 200)); CC = Cfun(LL, KK)
    titles = ["(a) KA, full profile", r"(b) KA+MVB, KA zeroth order", r"(c) KA+MVB, exact zeroth order"]
    for ax, j, t in zip(axs, (1, 2, 3), titles):
        val = 100 * pm[:, :, j].T
        im = ax.pcolormesh(Lams, kbs, np.where(val > 0, val, np.nan), shading="nearest", cmap="magma_r", norm=LogNorm(0.1, 100), rasterized=True)
        cs_ = ax.contour(LL, KK, CC, levels=[0.05, 0.4], colors=["c", "0.3"], linewidths=1.1, linestyles=["-", "--"])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([3, 5, 8, 12, 20]); ax.set_xticklabels(["3", "5", "8", "12", "20"])
        ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"]); ax.minorticks_off()
        ax.set_xlabel(r"$\Lambda$  ($\lambda_s/\lambda=10/\Lambda$)"); ax.set_title(t, fontsize=8, loc="left")
    axs[0].set_ylabel(r"$kb$")
    cb = fig.colorbar(im, ax=axs, pad=0.015, fraction=0.03); cb.set_label("error (\\%)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    from matplotlib.lines import Line2D
    axs[0].legend([Line2D([0], [0], color="c", lw=1.1), Line2D([0], [0], color="0.3", lw=1.1, ls="--")], [r"$C=0.05$", r"$C=0.4$"], loc="lower left", fontsize=6.5, framealpha=0.9, handlelength=1.6)
    fig.savefig(OUT + "fig_phasemap.pdf", bbox_inches="tight"); plt.close(fig)
    print("saved fig_phasemap.pdf")


def per_family_tables():
    """Median errors by bin of C, separately for the cosine profiles and the random surfaces (Sec. 4)."""
    pol, fam, X, nex, nall = gather()
    C, eK, eM, eX = X[:, 0], X[:, 2], X[:, 3], X[:, 4]
    bins = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    for fm, name in (("cos", "cosine profiles"), ("rnd", "random flat-band surfaces")):
        print(f"  {name}: median error (%) KA full | series KA zeroth | series exact zeroth | median gain | n")
        for pl in ("TE", "TM"):
            for lo, hi in zip(bins[:-1], bins[1:]):
                s_ = (pol == pl) & (fam == fm) & (C >= lo) & (C < hi)
                if s_.sum() > 3: print(f"    {pl} C {lo:4.2f}-{hi:4.2f}: {100*np.median(eK[s_]):7.2f} | {100*np.median(eM[s_]):6.2f} | {100*np.median(eX[s_]):6.3f} | {np.median(eK[s_]/eM[s_]):5.1f} | {s_.sum()}")
    ok = (C <= 0.8) & np.isfinite(X[:, 6])
    d = X[ok, 6]
    print(f"  Rayleigh vs C-method, {ok.sum()} cases with C<=0.8 (cond<1e15): median {np.median(d):.1e}, 95th percentile {np.percentile(d, 95):.1e}, max {d.max():.1e}; below 1e-4: {100*np.mean(d<1e-4):.0f}%, below 1e-3: {100*np.mean(d<1e-3):.0f}%")


def stability_table():
    """Fraction of cases with series error < 5 % versus the condition number of the smooth-profile matrix (kb <= 0.3, no grazing incidence)."""
    import robustness as R, phase_map as P, random_stats as RS
    rows = []
    d = np.load(R.FILE)["data"]
    for ic, name in enumerate(R.NAMES):
        if name in ("sin=0.75", "sin=0.85"): continue
        for ip in range(2):
            for iL in range(7):
                for ik, kb in enumerate(R.KBS):
                    if kb <= 0.3: rows.append(d[ic, ip, iL, ik][[6, 1, 2]])
    pm = np.load(P.FILE)["data"]
    for ip in range(2):
        for iL in range(len(P.LAMS)):
            for ik, kb in enumerate(P.KBS):
                if kb <= 0.3: rows.append(pm[ip, iL, ik][[4, 2, 3]])
    rd = np.load(RS.FILE)["data"]
    for ib in range(3):
        for ik, ks in enumerate(RS.KSIG):
            if ks <= 0.2:
                for ip in range(2):
                    for ir in range(rd.shape[3]): rows.append(rd[ib, ik, ip, ir][[6, 1, 2]])
    a = np.array(rows, float); a = a[np.isfinite(a).all(axis=1)]
    print(f"  {len(a)} cases; error < 5 % (KA zeroth | exact zeroth) by condition number of the zeroth-order matrix")
    for lo, hi in zip([1, 1e8, 1e11, 1e14, 1e15, 1e16, 1e18], [1e8, 1e11, 1e14, 1e15, 1e16, 1e18, 1e30]):
        s_ = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if s_.sum(): print(f"    {lo:7.0e}..{hi:7.0e}: n={s_.sum():4d}  {100*np.mean(a[s_, 1] < 0.05):4.0f}% | {100*np.mean(a[s_, 2] < 0.05):4.0f}%")
    s_ = a[:, 0] < 1e16
    print(f"    below 1e16: n={s_.sum()}, KA zeroth {100*np.mean(a[s_, 1] < 0.05):.0f}%, exact zeroth {100*np.mean(a[s_, 2] < 0.05):.0f}%")


def floor_table():
    """Floor of the series with the Kirchhoff zeroth order = error of KA on the smooth profile alone (parametric study, Sec. 4)."""
    import robustness as R
    d = np.load(R.FILE)["data"]
    print("  config, pol: KA vs exact on the smooth profile alone | floor of the series (KA zeroth, kb=0.1) | median exact-zeroth error in the core (kb 0.2-0.3, lambda_s<=1.25 lambda)")
    for ic, name in enumerate(R.NAMES):
        c = R.CONF[name]; Lr = c["L"]; th = np.arcsin(c["sin"])
        for ip, pol in enumerate(R.POLS):
            p0 = Problem(1.0, 1.0 / Lr, th, c["eps"], pol, 30, N=4096)
            rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30)
            p = Problem(1.0, 1.0 / Lr, th, c["eps"], pol, rmax, N=4096); f, f1 = cos_surface(p, c["a"])
            es = l1(efficiencies(p, *ka_amplitudes(p, f, f1))[0], efficiencies(p, *rayleigh_exact(p, f, f1))[0], p.prop_p)
            D = d[ic, ip]; core = [i for i, L_ in enumerate(R.LAMS[Lr]) if Lr / L_ <= 1.25]
            print(f"    {name:>9} {pol}: {100*es:6.2f} % | {100*np.nanmedian(D[:3, 0, 1]):6.2f} % | {100*np.nanmedian(D[np.ix_(core, [1, 2])][..., 2]):6.3f} %")


def numbers():
    """Print the numbers quoted in Secs. 4 and 5 of the paper."""
    print("=== Fig. 2 (periodic profile Lambda=15, kb=0.25; median random surface)")
    cases, stats = spectra(save=False)
    c = cases["cosTE"]; p = c["p"]; s = p.prop_p; r = p.r[s]; ref = c["ref"][0][s]
    dev = lambda e: np.abs(e[s] / ref - 1); big = ref > 1e-4
    for name, e in (("KA full", c["kaf"][0]), ("series, KA zeroth", c["S"]["ka"][3][0]), ("series, exact zeroth", c["S"]["ex"][3][0])):
        dd = dev(e); print(f"  per-order deviation, {name}: median {100*np.median(dd[big]):.1f} %, max {100*dd[big].max():.1f} % (e>1e-4), max overall {100*dd.max():.1f} % at r={r[np.argmax(dd)]}")
    dk = dev(c["kaf"][0]); print(f"  KA full: {100*dk[r >= -8].min():.0f}-{100*dk[r >= -8].max():.0f} % for r>=-8; model/reference at r=-13: {c['kaf'][0][s][r == -13][0] / ref[r == -13][0]:.2f}")
    cs3 = np.cos(np.arcsin(SIN)) ** 3; Cs = np.pi * WL * A_SMOOTH / cs3
    print(f"  C_smooth = {Cs:.3f}, C_small = kb (lambda/lambda_s)^2 / (2 cos^3 theta) = {0.25 * (15 * WL) ** 2 / (2 * cs3):.3f}")
    print("\n=== Fig. 3 (error versus C)")
    caxis(save=False); per_family_tables()
    print("\n=== Fig. 4 (plane Lambda-kb, TE and TM)")
    import phase_map as P
    pm = np.load(P.FILE)["data"]
    for ip, pol in enumerate(P.POLS):
        v = pm[ip]; ka, mk, mx = 100 * v[:, :, 1], 100 * v[:, :, 2], 100 * v[:, :, 3]
        g = (ka / mk)[3:, 2:5]
        print(f"  {pol}: gain KA full / series (KA zeroth), kb=0.2-0.3, Lambda>=10: min {g.min():.1f}, median {np.median(g):.1f}, max {g.max():.1f}")
        for ik, kb in enumerate(P.KBS):
            print(f"      kb={kb:4.2f}: series KA zeroth {mk[:, ik].min():6.2f}-{mk[:, ik].max():6.2f} %, exact zeroth {mx[:, ik].min():6.3f}-{mx[:, ik].max():6.2f} %, KA full {ka[:, ik].min():6.2f}-{ka[:, ik].max():6.1f} %")
    print("\n=== floor of the Kirchhoff zeroth order and incidence angle (parametric study)")
    floor_table()
    print("\n=== stability: condition number")
    stability_table()


if __name__ == "__main__":
    cmds = {"spectra": spectra, "caxis": caxis, "phasemap": phasemap, "numbers": numbers}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds: sys.exit("usage: python make_results_figures.py " + " | ".join(cmds))
    cmds[sys.argv[1]]()
