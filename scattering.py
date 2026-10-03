"""
Scattering from a 1D two-scale periodic dielectric surface z = f(x) + delta*zeta(x).

Implements
  * Kirchhoff (tangent-plane) amplitudes for the smooth surface f   (d_{0,r})
  * the method of variation of boundaries (MVB) about the smooth periodic surface:
    recursion for the Taylor coefficients d^{+-}_{n,r}  (eqs. 1ercc / 2daec of the paper,
    written in x-space and with the nu^2 factor of the 2nd boundary condition)
  * an exact Rayleigh-method solver (reference / validation)

Conventions (as in the paper)
  incident   phi_i = exp(i(alpha x - beta z))
  reflected  u+ = sum_r B+_r exp(i(alpha_r x + beta+_r z)),  alpha_r = alpha + r K, K = 2 pi / L
  transmitted u- = sum_r B-_r exp(i(alpha_r x - beta-_r z))
  BC on z = F(x):  u+ + phi_i - u- = 0 ;  d_n(u+ + phi_i) - nu^2 d_n u- = 0,
  nu^2 = 1 (TE) or 1/eps (TM),  d_n = -F' d_x + d_z
"""
import numpy as np
from math import factorial
from scipy.linalg import lu_factor, lu_solve


# ----------------------------------------------------------------------------- setup
class Problem:
    """Geometry/medium/incidence and the truncated set of Rayleigh orders."""

    def __init__(self, L=1.0, wavelength=0.1, theta_i=np.deg2rad(20.0), eps=2.25,
                 pol="TE", rmax=60, N=4096):
        self.L, self.k, self.eps = L, 2 * np.pi / wavelength, eps
        self.K = 2 * np.pi / L
        self.theta_i = theta_i
        self.nu2 = 1.0 if pol.upper() == "TE" else 1.0 / eps
        self.alpha = self.k * np.sin(theta_i)
        self.beta = self.k * np.cos(theta_i)
        self.rmax = rmax
        self.r = np.arange(-rmax, rmax + 1)
        self.alpha_r = self.alpha + self.r * self.K
        self.bp = np.sqrt((self.k ** 2 - self.alpha_r ** 2).astype(complex))   # beta+_r
        self.bm = np.sqrt((eps * self.k ** 2 - self.alpha_r ** 2).astype(complex))  # beta-_r
        # choose Im >= 0 (evanescent decaying) -- sqrt already gives that for negative args
        self.N = N
        self.x = L * np.arange(N) / N

    @property
    def prop_p(self):
        return self.k ** 2 - self.alpha_r ** 2 > 0

    @property
    def prop_m(self):
        return self.eps * self.k ** 2 - self.alpha_r ** 2 > 0


def cos_surface(prob, amp, harmonic=1):
    """amp*cos(harmonic*K*x) and its derivative on the grid."""
    th = harmonic * prob.K * prob.x
    return amp * np.cos(th), -amp * harmonic * prob.K * np.sin(th)


# ----------------------------------------------------------------------------- KA
def fresnel_local(prob, f1):
    """Local (tangent-plane) reflection coefficient from  cos(theta_l)=(beta+alpha f')/(k sqrt(1+f'^2))."""
    k, eps = prob.k, prob.eps
    cl = (prob.beta + prob.alpha * f1) / (k * np.sqrt(1 + f1 ** 2))
    q = np.sqrt(eps - (1 - cl ** 2) + 0j)
    return (cl - prob.nu2 * q) / (cl + prob.nu2 * q)


def ka_amplitudes(prob, f, f1):
    """Kirchhoff d_{0,r}^{+-}: reflected and transmitted amplitudes for the surface f."""
    R = fresnel_local(prob, f1)
    al, be, r = prob.alpha, prob.beta, prob.r
    inc = np.exp(1j * ((al - prob.alpha_r[None, :]) * prob.x[:, None]
                       - (be + prob.bp[None, :]) * f[:, None]))                    # reflected phase
    incm = np.exp(1j * ((al - prob.alpha_r[None, :]) * prob.x[:, None]
                        + (prob.bm[None, :] - be) * f[:, None]))                   # transmitted phase
    f1c, Rc = f1[:, None], R[:, None]
    brk_p = (prob.bp[None, :] - prob.alpha_r[None, :] * f1c) * (1 + Rc) - (be + al * f1c) * (1 - Rc)
    brk_m = (be + al * f1c) * (1 - Rc) / prob.nu2 + (prob.bm[None, :] + prob.alpha_r[None, :] * f1c) * (1 + Rc)
    Bp = (inc * brk_p).mean(axis=0) / (2 * prob.bp)
    Bm = (incm * brk_m).mean(axis=0) / (2 * prob.bm)
    return Bp, Bm


# ----------------------------------------------------------------------------- Rayleigh / MVB matrices
def _coeff_matrix(prob, cols):
    """Given cols[:, j] = g_j(x) (one function per Rayleigh order r_j), return the matrix
    M[r', r] = (1/L) int g_r(x) exp(-i (r'-r) K x) dx."""
    c = np.fft.fft(cols, axis=0) / prob.N
    idx = (prob.r[:, None] - prob.r[None, :]) % prob.N
    return c[idx, np.arange(len(prob.r))[None, :]]


def _fields(prob, F, F1):
    """Columns exp(+i beta+_r F) and exp(-i beta-_r F)."""
    Ep = np.exp(1j * prob.bp[None, :] * F[:, None])
    Em = np.exp(-1j * prob.bm[None, :] * F[:, None])
    return Ep, Em


def system_matrix(prob, F, F1):
    """Matrix of the order-n problem; unknowns [d+_n ; d-_n] (identical for all n)."""
    Ep, Em = _fields(prob, F, F1)
    P_p = _coeff_matrix(prob, Ep)
    P_m = _coeff_matrix(prob, Em)
    Q_p = _coeff_matrix(prob, Ep * (prob.bp[None, :] - prob.alpha_r[None, :] * F1[:, None]))
    Q_m = _coeff_matrix(prob, Em * (prob.bm[None, :] + prob.alpha_r[None, :] * F1[:, None]))
    return np.block([[P_p, -P_m], [Q_p, prob.nu2 * Q_m]])


def _incident_rhs_exact(prob, F, F1):
    """RHS for the exact (all-orders) problem on the surface F:   M d = rhs."""
    Ei = np.exp(-1j * prob.beta * F)
    n = len(prob.r)
    r0 = prob.rmax
    # incident wave only in the r = 0 column
    c1 = np.fft.fft(Ei) / prob.N
    c2 = np.fft.fft(Ei * (prob.beta + prob.alpha * F1)) / prob.N
    ii = (prob.r - 0) % prob.N
    return np.concatenate([-c1[ii], c2[ii]])


def rayleigh_exact(prob, F, F1):
    """Reference solution (Rayleigh hypothesis) for the full profile F: returns B+_r, B-_r."""
    M = system_matrix(prob, F, F1)
    d = np.linalg.solve(M, _incident_rhs_exact(prob, F, F1))
    n = len(prob.r)
    return d[:n], d[n:]


# ----------------------------------------------------------------------------- MVB recursion
def _power_terms(prob, F, F1, Z, Z1, m, sign, E0=None):
    """Columns for delta^m coefficient of  e^{i s b (F+dZ)}  and of the BC2 prefactor times it.
    sign=+1: reflected (b=beta+), sign=-1: transmitted (b=beta-, phase -i b).
    E0 = exp(i s b F) may be passed in (it only depends on the smooth profile)."""
    b = prob.bp if sign > 0 else prob.bm
    ph = 1j * sign * b[None, :]
    if E0 is None:
        E0 = np.exp(ph * F[:, None])
    # (i s b Z)^m / m!
    t_m = (ph * Z[:, None]) ** m / factorial(m)
    t_m1 = (ph * Z[:, None]) ** (m - 1) / factorial(m - 1)
    col1 = E0 * t_m
    if sign > 0:
        col2 = E0 * (t_m * (b[None, :] - prob.alpha_r[None, :] * F1[:, None])
                     - prob.alpha_r[None, :] * Z1[:, None] * t_m1)
    else:
        col2 = E0 * (t_m * (b[None, :] + prob.alpha_r[None, :] * F1[:, None])
                     + prob.alpha_r[None, :] * Z1[:, None] * t_m1)
    return col1, col2


class SmoothOperator:
    """Everything of the recursion that depends only on the smooth profile f (not on the roughness zeta):
    the LU factorisation of the Rayleigh matrix and the exponentials exp(+-i beta_r f).
    Build it once and pass it to mvb_coefficients(..., smooth=op) to treat many roughness realisations."""

    def __init__(self, prob, f, f1):
        self.f, self.f1 = f, f1
        self.lu = lu_factor(system_matrix(prob, f, f1))
        self.E0p = np.exp(1j * prob.bp[None, :] * f[:, None])
        self.E0m = np.exp(-1j * prob.bm[None, :] * f[:, None])


def mvb_coefficients(prob, f, f1, z, z1, d0, nmax, smooth=None):
    """Taylor coefficients d^{+-}_{n,r}, n=0..nmax.
    d0 = (Bp0, Bm0): zeroth-order amplitudes (KA, or the exact smooth-surface solution).
    smooth: optional SmoothOperator(prob, f, f1) to reuse the factorisation.  z, z1 may be complex
    (the d_n are polynomials in zeta).  Returns lists dp[n], dm[n]."""
    n_r = len(prob.r)
    if smooth is None:
        lu, E0p, E0m = lu_factor(system_matrix(prob, f, f1)), None, None
    else:
        lu, E0p, E0m = smooth.lu, smooth.E0p, smooth.E0m
    dp, dm = [d0[0]], [d0[1]]

    # coupling matrices for each power m of zeta (cached)
    C = {}
    for m in range(1, nmax + 1):
        c1p, c2p = _power_terms(prob, f, f1, z, z1, m, +1, E0p)
        c1m, c2m = _power_terms(prob, f, f1, z, z1, m, -1, E0m)
        C[m] = (_coeff_matrix(prob, c1p), _coeff_matrix(prob, c2p),
                _coeff_matrix(prob, c1m), _coeff_matrix(prob, c2m))

    # incident wave contributions for each order n:  e^{-i beta (f+dz)} and BC2 prefactor
    inc = {}
    for n in range(1, nmax + 1):
        ph = -1j * prob.beta
        E0 = np.exp(ph * f)
        t_n = (ph * z) ** n / factorial(n)
        t_n1 = (ph * z) ** (n - 1) / factorial(n - 1)
        g1 = E0 * t_n
        g2 = E0 * (t_n * (prob.beta + prob.alpha * f1) + prob.alpha * z1 * t_n1)
        c1 = np.fft.fft(g1) / prob.N
        c2 = np.fft.fft(g2) / prob.N
        ii = prob.r % prob.N
        inc[n] = (c1[ii], c2[ii])

    for n in range(1, nmax + 1):
        s1 = np.zeros(n_r, complex)
        s2 = np.zeros(n_r, complex)
        for p in range(n):
            m = n - p
            P1p, P2p, P1m, P2m = C[m]
            s1 += P1p @ dp[p] - P1m @ dm[p]
            s2 += P2p @ dp[p] + prob.nu2 * (P2m @ dm[p])
        s1 += inc[n][0]
        s2 -= inc[n][1]          # incident term enters BC2 with a minus sign
        d = lu_solve(lu, np.concatenate([-s1, -s2]))
        dp.append(d[:n_r])
        dm.append(d[n_r:])
    return dp, dm


def mvb_sum(dp, dm, delta=1.0, order=None):
    """Partial sums  sum_{n<=order} d_n delta^n."""
    order = len(dp) - 1 if order is None else order
    Bp = sum(dp[n] * delta ** n for n in range(order + 1))
    Bm = sum(dm[n] * delta ** n for n in range(order + 1))
    return Bp, Bm


# ----------------------------------------------------------------------------- efficiencies
def efficiencies(prob, Bp, Bm):
    """e+_r = (beta+_r/beta)|B+_r|^2 ; e-_r = nu^2 (beta-_r/beta)|B-_r|^2 (propagating orders only)."""
    ep = np.where(prob.prop_p, prob.bp.real / prob.beta * np.abs(Bp) ** 2, 0.0)
    em = np.where(prob.prop_m, prob.nu2 * prob.bm.real / prob.beta * np.abs(Bm) ** 2, 0.0)
    return ep, em
