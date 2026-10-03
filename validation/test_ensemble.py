"""
Check of the analytic ensemble average (ensemble.py, Eq. (ensemble_avg) of the paper).
For one roughness harmonic with random phase, the average of |d_0 + d_1 + d_2|^2 over the phase equals the analytic expression
up to terms of order A^4 (|d_2|^2 etc.), so the difference must scale as A^4 when the amplitude A is doubled.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, ensemble.py, ... live in the parent folder
import numpy as np
from scattering import *
from ensemble import *


def numeric_phase_average(p, f, f1, Bka, m, A, op, nphase=16):
    out_p = np.zeros(len(p.r)); out_m = np.zeros(len(p.r))
    for ph in 2 * np.pi * np.arange(nphase) / nphase:
        z = A * np.cos(m * p.K * p.x + ph); z1 = -A * m * p.K * np.sin(m * p.K * p.x + ph)
        dp, dm = mvb_coefficients(p, f, f1, z, z1, Bka, 2, smooth=op)
        Bp, Bm = mvb_sum(dp, dm, 1.0, 2)
        out_p += np.abs(Bp) ** 2 / nphase; out_m += np.abs(Bm) ** 2 / nphase
    return out_p, out_m


if __name__ == "__main__":
    p = Problem(1.0, 0.1, np.deg2rad(20), 2.25, "TE", 30, N=1024)
    f, f1 = cos_surface(p, 0.03)
    Bka = ka_amplitudes(p, f, f1)
    op = SmoothOperator(p, f, f1)
    m = 5
    diffs = []
    for A in (4e-3, 8e-3):
        (Ip, Im), _ = analytic_average(p, f, f1, Bka, [(m, A)], smooth=op)
        Np, Nm = numeric_phase_average(p, f, f1, Bka, m, A, op)
        sel = p.prop_p & (Np > 1e-9)
        d = np.abs(Ip - Np)[sel].sum() / Np[sel].sum()
        diffs.append(d)
        print(f"A = {A:.0e} (k A = {p.k * A:.2f}): relative L1 difference between analytic and numerical phase average = {d:.2e}")
    ratio = diffs[1] / diffs[0]
    print(f"ratio for doubled amplitude = {ratio:.1f}  (A^4 scaling -> 16)")
    assert 10 < ratio < 22
    print("ensemble average: OK")
