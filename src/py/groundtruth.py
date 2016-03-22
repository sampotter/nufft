import fmm as _fmm
import itertools as _itertools
import numpy as _np

_twopi = 2*_np.pi

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

def inufft_radial_approximation(F, K, Y, L, p, q):
    '''
    Approximates the inverse NUFFT by approximating multiplication by
    the bandlimited interpolation matrix by truncating the infinite
    series in its definition and using the FMM to compute the
    truncated summation. This is as opposed to applying the periodic
    summation algorithm to the infinite series.

    Arguments:
        F: test function evaluated at N equally spaced points 
           along [0, 2pi).
        K: bandlimit parameter.
        Y: sorted list of target points in [0, 2pi).
        L: depth of FMM.
        p: truncation parameter of FMM.
        q: term at which to truncate infinite series -- i.e., the 
           approx_interp_fmm will sum over terms -q, -q+1, ..., q.

    Returns: a numpy array containing the results.
    '''
    
    N = len(F)
    X = _np.linspace(0, _twopi, N, endpoint=False)

    # Create a linearly spaced grid of source points for the 2q + 1
    # terms of the truncated series.
    def shifted_X(m):
        return X + _twopi*m
    X_per = _np.concatenate([shifted_X(m) for m in range(-q, q + 1)])

    # Compute F with alternating signs and extend it to match X.
    Fas = [F[n]*(-1)**n for n in range(N)]
    Fas_sum = sum(Fas)
    Fas_per = _np.tile(Fas, 2*q + 1)

    V = _fmm.fmm1d_cauchy_double(X_per, Y, Fas_per, L, p,
                                scaled_domain=(-_twopi*q, _twopi*(q + 1)))
    def g(j):
        return (-Fas_sum*_np.cos(K*Y[j]) + 2*_np.sin(K*Y[j])*V[j])/N
    G = [g(j) for j in range(len(Y))]

    index_ratio = len(Y)/(2*K)
    
    def test(y):
        '''
        This test is necessary to check if we've passed a point that we
        essentially already know the value of (i.e. an evaluation
        point that's nearly equal to a source point). We could
        probably handle this correctly in the FMM itself, but for now,
        a test like this should give us approximately what we
        want... That is, no interpolates that are NaN, +/-Inf, or
        incorrectly equal to zero.
        '''
        return _np.abs(_np.mod(y*(len(Y)/_twopi), index_ratio)) < 1e-13

    for i, y in enumerate(Y):
        if test(y):
            j = int(i/index_ratio)
            G[i] = F[j]
            
    return G
