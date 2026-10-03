"""
Ensemble averages over random small-scale roughness, computed from the MVB coefficients (no Monte Carlo).

Roughness model: zeta(x) = sum_m A_m cos(m K x + phi_m) with independent phases phi_m ~ U[0, 2 pi)
(equivalently complex amplitudes c_m = (A_m/2) e^{i phi_m}, <|c_m|^2> = (A_m/2)^2).

Because the Taylor coefficients d_n are polynomials of degree n in zeta, with
    zeta = sum_m (c_m E_m + c_m^* E_-m),   E_m = exp(i m K x),
    d_1 = sum_m (c_m P_m + c_m^* P_-m),
    d_2 = sum_m (c_m^2 Q_m + c_m^{*2} Q_-m + |c_m|^2 X_m) + (cross terms between different m),
the average of the intensity |B|^2 up to second order in the roughness amplitude is
    <|B|^2> = |d_0|^2 + sum_m <|c_m|^2> (|P_m|^2 + |P_-m|^2) + 2 Re[ d_0^* sum_m <|c_m|^2> X_m ] + O(zeta^4),
with P_{+-m} = d_1[E_{+-m}] and X_m = d_2[E_m + E_-m] - d_2[E_m] - d_2[E_-m]  (polarisation identity).
Three runs of the recursion per roughness harmonic (one factorisation of the smooth-surface matrix) give the
average for ANY set of amplitudes A_m, whereas an exact solver needs one full solution per realisation.
"""
import numpy as np
from scattering import *


def analytic_average(prob, f, f1, d0, harmonics, smooth=None):
    """Return (<|B+|^2>, <|B-|^2>) averaged over the random phases, up to O(zeta^2).
    `harmonics` is a list of (m, A_m).  d0 = zeroth-order amplitudes (KA or exact)."""
    op = SmoothOperator(prob, f, f1) if smooth is None else smooth
    x, K = prob.x, prob.K
    I_p = np.abs(d0[0]) ** 2
    I_m = np.abs(d0[1]) ** 2
    S1p = np.zeros(len(prob.r)); S1m = np.zeros(len(prob.r))
    S2p = np.zeros(len(prob.r), complex); S2m = np.zeros(len(prob.r), complex)
    for m, A in harmonics:
        Ep, Em = np.exp(1j * m * K * x), np.exp(-1j * m * K * x)
        run = {}
        for key, (z, z1) in {"p": (Ep, 1j * m * K * Ep), "m": (Em, -1j * m * K * Em),
                             "s": (Ep + Em, 1j * m * K * (Ep - Em))}.items():
            run[key] = mvb_coefficients(prob, f, f1, z, z1, d0, 2, smooth=op)
        sig2 = (A / 2) ** 2
        for (S1, S2, j) in ((S1p, S2p, 0), (S1m, S2m, 1)):
            R_p, R_m = run["p"][j][1], run["m"][j][1]
            X = run["s"][j][2] - run["p"][j][2] - run["m"][j][2]
            S1 += sig2 * (np.abs(R_p) ** 2 + np.abs(R_m) ** 2)
            S2 += sig2 * X
    return (I_p + S1p + 2 * np.real(np.conj(d0[0]) * S2p),
            I_m + S1m + 2 * np.real(np.conj(d0[1]) * S2m)), (S1p, S1m)


def to_efficiency(prob, Ip, Im):
    ep = np.where(prob.prop_p, prob.bp.real / prob.beta * Ip, 0.0)
    em = np.where(prob.prop_m, prob.nu2 * prob.bm.real / prob.beta * Im, 0.0)
    return ep, em


def realisation(prob, harmonics, rng):
    """One random realisation of the roughness (and its derivative)."""
    z = np.zeros(prob.N); z1 = np.zeros(prob.N)
    for m, A in harmonics:
        ph = rng.uniform(0, 2 * np.pi)
        z += A * np.cos(m * prob.K * prob.x + ph)
        z1 += -A * m * prob.K * np.sin(m * prob.K * prob.x + ph)
    return z, z1


def monte_carlo(prob, f, f1, harmonics, K, seed=0):
    """Reference: mean efficiencies over K realisations, each solved with the exact (Rayleigh) solver."""
    rng = np.random.default_rng(seed)
    n = len(prob.r)
    sp = np.zeros(n); sm = np.zeros(n); sp2 = np.zeros(n)
    for _ in range(K):
        z, z1 = realisation(prob, harmonics, rng)
        ep, em = efficiencies(prob, *rayleigh_exact(prob, f + z, f1 + z1))
        sp += ep; sm += em; sp2 += ep ** 2
    mean_p, mean_m = sp / K, sm / K
    err_p = np.sqrt(np.maximum(sp2 / K - mean_p ** 2, 0) / K)
    return mean_p, mean_m, err_p
