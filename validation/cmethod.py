"""
Independent reference solver: C-method (Chandezon et al.) for the same scalar transmission problem.

It does NOT use the Rayleigh hypothesis nor any of the matrices of scattering.py: the Helmholtz equation is
written in the curvilinear coordinates (x', y') = (x, z - F(x)), which turns the corrugated interface into the
plane y' = 0, and the fields in each medium are expanded in the eigenmodes of the resulting quadratic
eigenproblem

    gamma^2 [1+F'^2] a  -  gamma (2 [F'] D - i [F'']) a  +  (D^2 - k_j^2) a = 0 ,   D = diag(alpha + nK)

([g] = Toeplitz matrix of the Fourier coefficients of g).  Boundary conditions at y' = 0:
    U_up + U_inc = U_dn ,     N U_up + N U_inc = nu^2 N U_dn ,    N = (1+F'^2) d_y' - F' d_x'
Same conventions as scattering.py: incident exp(i(alpha x - beta z)), nu^2 = 1 (TE) or 1/eps (TM).
"""
import numpy as np


def _toeplitz_from_grid(g, n_idx):
    c = np.fft.fft(g) / len(g)
    return c[(n_idx[:, None] - n_idx[None, :]) % len(g)]


def _modes(k2, A2, TF1, TF2, D, delta):
    """Eigenpairs (gamma, a) of the quadratic eigenproblem; tiny loss delta selects the outgoing direction."""
    n = len(D)
    A1 = -(2 * TF1 @ np.diag(D) - 1j * TF2)
    A0 = np.diag(D ** 2) - k2 * (1 + 1j * delta) * np.eye(n)
    A2inv = np.linalg.inv(A2)
    Lin = np.block([[np.zeros((n, n)), np.eye(n)], [-A2inv @ A0, -A2inv @ A1]])
    g, v = np.linalg.eig(Lin)
    a = v[:n, :]
    a = a / np.linalg.norm(a, axis=0)[None, :]
    return g, a


def c_method(prob, F, F1, F2, M=None, delta=1e-10):
    """Return B+_r, B-_r (r = -rmax..rmax of `prob`; only |r|<=M are computed, M defaults to prob.rmax)
    for the profile z = F(x) sampled on prob.x.  F1, F2: first and second derivatives."""
    M = prob.rmax if M is None else M
    n_idx = np.arange(-M, M + 1)
    N = 2 * M + 1
    al_n = prob.alpha + n_idx * prob.K
    D = al_n.astype(complex)
    T2 = _toeplitz_from_grid(1 + F1 ** 2, n_idx)
    TF1 = _toeplitz_from_grid(F1, n_idx)
    TF2 = _toeplitz_from_grid(F2, n_idx)
    I = np.eye(N)

    k2_up, k2_dn = prob.k ** 2, prob.eps * prob.k ** 2
    g1, a1 = _modes(k2_up, T2, TF1, TF2, D, delta)
    g2, a2 = _modes(k2_dn, T2, TF1, TF2, D, delta)
    up = g1.imag > 0
    dn = g2.imag < 0
    if up.sum() != N or dn.sum() != N:
        raise RuntimeError(f"mode classification failed: up={up.sum()}, down={dn.sum()}, expected {N} "
                           "(Wood anomaly? change incidence angle)")
    g1, Phi = g1[up], a1[:, up]
    g2, Psi = g2[dn], a2[:, dn]

    def conormal(gam, A):
        return 1j * (T2 @ (A * gam[None, :]) - TF1 @ (D[:, None] * A))

    Nup, Ndn = conormal(g1, Phi), conormal(g2, Psi)
    gi = np.fft.fft(np.exp(-1j * prob.beta * F)) / len(F)
    gi = gi[n_idx % len(F)]
    Ninc = -1j * (prob.beta * (T2 @ gi) + TF1 @ (D * gi))
    Msys = np.block([[Phi, -Psi], [Nup, -prob.nu2 * Ndn]])
    rhs = np.concatenate([-gi, -Ninc])
    sol = np.linalg.solve(Msys, rhs)
    c, e = sol[:N], sol[N:]

    # --- far-field amplitudes: evaluate the exterior fields on z = Fmax (upper) and z = Fmin (lower)
    x = prob.x
    Fmax, Fmin = F.max(), F.min()

    def synth(A):   # columns: sum_n A[n, m] exp(i n K x)
        return np.exp(1j * np.outer(x * prob.K, n_idx)) @ A

    up_field = np.zeros(len(x), complex)
    dn_field = np.zeros(len(x), complex)
    SPhi, SPsi = synth(Phi), synth(Psi)
    for m in range(N):
        up_field += c[m] * SPhi[:, m] * np.exp(1j * g1[m] * (Fmax - F))
        dn_field += e[m] * SPsi[:, m] * np.exp(1j * g2[m] * (Fmin - F))
    cu = np.fft.fft(up_field) / len(x)
    cd = np.fft.fft(dn_field) / len(x)
    # field = e^{i alpha x} sum_r coeff_r e^{i r K x}
    r = prob.r
    Bp = np.full(len(r), np.nan, complex)
    Bm = np.full(len(r), np.nan, complex)
    ok = np.abs(r) <= M
    ri = r[ok]
    Bp[ok] = cu[ri % len(x)] * np.exp(-1j * prob.bp[ok] * Fmax)
    Bm[ok] = cd[ri % len(x)] * np.exp(1j * prob.bm[ok] * Fmin)
    return Bp, Bm


def spectral_derivative(prob, g, order=1):
    """Derivative of a periodic band-limited function sampled on prob.x (FFT based)."""
    k = np.fft.fftfreq(prob.N, d=1.0 / prob.N) * prob.K
    return np.real(np.fft.ifft((1j * k) ** order * np.fft.fft(g)))
