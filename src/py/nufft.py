import fmm as _fmm
import numpy as _np

_twopi = 2*_np.pi

def _transpose_argsort_indices(I):
    '''
    Compute the inverse permutation of I, where I is a list of indices
    (a permutation) computed using np.argsort.
    '''
    J = _np.zeros(len(I), dtype=_np.int)
    for i, j in enumerate(I):
        J[j] = i
    return J

def _test(y, N, index_ratio):
    '''
    This test is necessary to check if we've passed a point that we
    essentially already know the value of (i.e. an evaluation
    point that's nearly equal to a source point). We could
    probably handle this correctly in the FMM itself, but for now,
    a test like this should give us approximately what we
    want... That is, no interpolates that are NaN, +/-Inf, or
    incorrectly equal to zero.
    '''
    return _np.abs(_np.mod(y*(N/_twopi), index_ratio)) < 1e-13
    
def inufft(F, K, Y, L, p, n, q):
    '''
    Arguments:
        F: samples of a K-bandlimited function spaced equally along [0, 2pi).
        K: the bandlimit of the sampled function.
        Y: a list of target points in [0, 2pi).
        L: the depth of the FMM used in interpolation.
        p: the truncation number of the FMM.
        n: the 'radius' of the neighborhood around [0, 2pi) -- i.e. determining
           the intervals [-2pi*n, 0) and [2pi, 2pi(n+1)).
        q: the number of checkpoint pairs.
    '''

    # Compute N equally spaced gridpoints that lie in [0, 2pi).
    N = len(F)
    X = _np.linspace(0, _twopi, N, endpoint=False)

    # Extend X to the radial neighborhood of [0, 2pi) -- the intervals
    # [-2pin, 0) and [2pi, 2pi(n+1)).
    def translate_X(l):
        return X + _twopi*l
    X_per = _np.concatenate([translate_X(l) for l in range(-n, n + 1)])

    # Extend (F), but with (a)lternating (s)igns, to the radial
    # neighborhood.
    Fas = [F[n]*(-1)**n for n in range(N)]
    Fas_per = _np.tile(Fas, 2*n + 1)

    # Compute uniformly distributed checkpoints.
    Yc = _np.concatenate(
        (_np.random.uniform(-n*_twopi, 0, int(_np.ceil(q/2))),
         _np.random.uniform(_twopi, (n + 1)*_twopi, int(_np.floor(q/2)))))
    Yc_tilde = _np.mod(Yc, _twopi)

    # Join together actual targets and checkpoint targets and compute
    # the sorting permutation and its inverse for use with FMM.
    Ycat = _np.concatenate((Y, Yc, Yc_tilde))
    I = _np.argsort(Ycat)
    J = _transpose_argsort_indices(I)

    # Use the FMM to evaluate the truncated periodic summation at the
    # evaluation points Y.
    dom = (-_twopi*n, _twopi*(n + 1))
    V = _fmm.fmm1d_cauchy_double(
        X_per, Ycat[I], Fas_per, L, p, scaled_domain=dom)

    # Permute V so that it is in the same order as Ycat.
    V = V[J]

    # Extract checkpoint evaluates for computing far summation.
    Vc_start = len(Y)
    Vc_tilde_start = Vc_start + len(Yc)
    Vc = V[Vc_start:Vc_start + len(Yc)]
    Vc_tilde = V[Vc_tilde_start:Vc_tilde_start + len(Yc_tilde)]

    # Compute far summation using least squares collocation.
    f = Vc_tilde - Vc
    A = _np.zeros((q, p))
    def R(Y, m):
        return _np.power(Y - _np.pi, m)
    for m in range(p):
        A[:, m] = R(Yc, m) - R(Yc_tilde, m)
    C = _np.linalg.lstsq(A, f)[0]
    
    phifar = _np.zeros(Y.shape, dtype=Y.dtype)
    for j in range(len(Y)):
        phifar[j] = _np.sum([C[m]*(Y[j] - _np.pi)**m for m in range(p)])

    # Extract near summation from evaluates computed using FMM and
    # compute final V from phinear and phifar.
    V = V[:len(Y)] + phifar
    
    # Use the interpolation formula with the values of V to compute G.
    Fas_sum = sum(Fas)
    def g(j):
        return (-Fas_sum*_np.cos(K*Y[j]) + 2 * _np.sin(K*Y[j])*V[j])/N
    G = [g(j) for j in range(len(Y))]

    # Update the values of any of the points that might have coincided
    # with the grid points.
    # TODO: it might be cheaper to do this if we do it first?
    # Although, then we need to futz with Y, which might be too
    # expensive...
    index_ratio = len(Y)/(2*K)
    for i, y in enumerate(Y):
        if _test(y, len(Y), index_ratio):
            G[i] = F[int(i/index_ratio)]

    return G
