import fmm
import groundtruth as gt
import numpy as np

M = 1000
N = 1000

X = sorted(np.random.uniform(0, 1, N))
Y = sorted(np.random.uniform(0, 1, M))
U = np.random.normal(0, 1, N)
L = 5
p = 5

Phi = gt.get_phi_direct(X, U)

V_gt = Phi(Y)
V_fmm = fmm.fmm1d_cauchy_double(X, Y, U, L, p)
