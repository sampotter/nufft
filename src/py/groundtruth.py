import itertools as _itertools
import numpy as _np

def K(X, Y, n_min, n_max):
    N = len(X)
    M = len(Y)
    X = _np.array(X, dtype=_np.float64)
    Y = _np.array(Y, dtype=_np.float64)
    ns = _np.array(range(n_min, n_max + 1))
    K = _np.zeros((M, N), dtype=_np.complex128)
    for j, k in _itertools.product(range(M), range(N)):
        K[j, k] = (1/N)*_np.exp(1j*ns*(Y[j] - X[k])).sum()
    return K

def get_phi_direct(X, U):
    N = len(X)
    assert(N == len(U))
    def phi(y):
        def kernel(x):
            return 1/(y - x)
        return sum([U[n]*kernel(X[n]) for n in range(N)])
    return _np.vectorize(phi)
