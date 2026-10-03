"""
Validation of the reference (Rayleigh) efficiency of the backscattered order against the independent C-method, on cells of
the map of Fig. 4 (corners included): both solvers are run on the same profile and the relative difference of e_back is printed.
The grid, the profile and the retained orders are those of backscatter_map.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, backscatter_map.py, ... live in the parent folder
import numpy as np
from scattering import *
from backscatter_map import make_problem, A_L, B_L, LAM, POLS
from cmethod import c_method, spectral_derivative


def check():
    cells = [(3.213, 1.5), (3.213, 10.0), (5.713, 4.0), (8.313, 2.25), (8.313, 9.0), (10.113, 1.5),
             (12.313, 6.0), (14.913, 10.0), (15.013, 1.5), (6.413, 7.0)]
    worst = 0.0
    print(f"{'pol':>3} {'L/lam':>6} {'eps':>5} {'phi':>5} {'rmax':>4} | {'e_back Rayleigh':>15} {'e_back C-method':>15} {'rel diff':>9}")
    for pol in POLS:
        for Lrel, eps in cells:
            for ph in (0.0, np.pi / 2):
                p, th = make_problem(Lrel, eps, pol)
                f, f1 = cos_surface(p, A_L)
                z = B_L * np.cos(LAM * p.K * p.x + ph)
                z1 = -B_L * LAM * p.K * np.sin(LAM * p.K * p.x + ph)
                F, F1 = f + z, f1 + z1
                i = p.rmax - LAM
                e_r = abs(rayleigh_exact(p, F, F1)[0][i]) ** 2
                e_c = abs(c_method(p, F, F1, spectral_derivative(p, F, 2), M=p.rmax, delta=1e-8)[0][i]) ** 2
                rel = abs(e_r - e_c) / e_c
                worst = max(worst, rel)
                print(f"{pol:>3} {Lrel:6.2f} {eps:5.2f} {ph:5.2f} {p.rmax:4d} | {e_r:15.6e} {e_c:15.6e} {rel:9.1e}")
    print("worst relative difference:", worst)
    assert worst < 1e-6


if __name__ == "__main__":
    check()
