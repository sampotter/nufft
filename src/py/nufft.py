import fmm as _fmm
import matplotlib.pyplot as _plt
import numpy as _np

from sanitycheck import cest

_twopi = 2*_np.pi


def _transpose_argsort_indices(I):
    '''Compute the inverse permutation of I, where I is a list of indices
    (a permutation) computed using np.argsort.

    '''
    J = _np.zeros(len(I), dtype=_np.int)
    for i, j in enumerate(I):
        J[j] = i
    return J


def _test(y, N, index_ratio):
    '''This test is necessary to check if we've passed a point that we
    essentially already know the value of (i.e. an evaluation point
    that's nearly equal to a source point). We could probably handle
    this correctly in the FMM itself, but for now, a test like this
    should give us approximately what we want... That is, no
    interpolates that are NaN, +/-Inf, or incorrectly equal to zero.

    '''
    return _np.abs(_np.mod(y*(N/_twopi), index_ratio)) < 1e-13


def _extend_X(X, n):
    '''Periodically extend the grid points in X that lie in [0, 2*pi) to
    [-2*pi*n, 2*pi*(n + 1)).

    '''
    return _np.concatenate([X + _twopi*l for l in range(-n, n + 1)])


def _get_extended_alt_sign_F_and_sum(F, n, N):
    '''Modulate F with (-1)^n and periodically extend the result to
    [-2*pi*n, 2*pi*(n + 1)). Also, return the sum of one period of the
    modulated F as the second return value.

    '''
    Fas = _np.multiply(F, _np.power(-1, range(N)))
    return _np.tile(Fas, 2*n + 1), _np.sum(Fas)


def _get_checkpoints(q):
    '''Compute q checkpoint pairs. The first point in each checkpoint pair
    is uniformly distributed in [2*pi, 4*pi). The second is simply
    that point but shifted into [0, 2*pi).

    '''
    Yc = _np.random.uniform(_twopi, 2*_twopi, q)
    return Yc, Yc - _twopi


def _R(Y, m):
    'Evaluate the mth Cauchy regular basis function R.'
    return _np.power(Y - _np.pi, m)


def _get_phinear_and_f(Y, Yc, Yc_tilde, n, X_per, Fas_per, L, p):
    '''This is a workhorse function that uses the FMM to compute both
    phinear and the difference in the checkpoint values. These are
    returned as the first and second return values.

    '''
    Ycat = _np.concatenate((Y, Yc, Yc_tilde))
    I = _np.argsort(Ycat)
    J = _transpose_argsort_indices(I)
    dom = (-_twopi*n, _twopi*(n + 1))
    fmm = _fmm.fmm1d_cauchy_double
    V = fmm(X_per, Ycat[I], Fas_per, L, p, scaled_domain=dom)[J]
    i = len(Y)
    j = i + len(Yc)
    k = j + len(Yc_tilde)
    return V[:i], V[j:k] - V[i:j]


def _get_phifar(Y, Yc, Yc_tilde, f, p, q):
    '''Use least squares collocation to compute phifar. TODO: this needs
    to be updated and replaced with the Vandermonde approach.

    '''
    A = _np.zeros((q, p))
    for m in range(p):
        A[:, m] = _R(Yc, m) - _R(Yc_tilde, m)
    C = _np.linalg.lstsq(A, f)[0]
    C[0] = cest(1e-6)
    phifar = _np.zeros(Y.shape, dtype=Y.dtype)
    for j in range(len(Y)):
        phifar[j] = _np.sum([C[m]*(Y[j] - _np.pi)**m for m in range(p)])
    return phifar


def _finish_interpolation(Y, F, phi, K, N, Fas_sum):
    '''This function takes care of the last couple steps: it modulates the
    result by the sine-based factor and sets any values that were too
    close source points to the corresponding weight (i.e. function
    value).

    '''
    G = [(-Fas_sum*_np.cos(K*Y[j]) + 2*_np.sin(K*Y[j])*phi[j])/N
         for j in range(len(Y))]
    index_ratio = len(Y)/(2*K)
    for i, y in enumerate(Y):
        if _test(y, len(Y), index_ratio):
            G[i] = F[int(i/index_ratio)]
    return G


def inufft(F, K, Y, L, p, n, q):
    '''Arguments:

        F: samples of a K-bandlimited function spaced equally along [0, 2pi).
        K: the bandlimit of the sampled function.
        Y: a list of target points in [0, 2pi).
        L: the depth of the FMM used in interpolation.
        p: the truncation number of the FMM.
        n: the 'radius' of the neighborhood around [0, 2pi) -- i.e. determining
           the intervals [-2pi*n, 0) and [2pi, 2pi(n+1)).
        q: the number of checkpoint pairs.

    '''
    N = len(F)
    X = _np.linspace(0, _twopi, N, endpoint=False)
    X_per = _extend_X(X, n)
    Fas_per, Fas_sum = _get_extended_alt_sign_F_and_sum(F, n, N)
    Yc, Yc_tilde = _get_checkpoints(q)
    phinear, f = _get_phinear_and_f(Y, Yc, Yc_tilde, n, X_per, Fas_per, L, p)
    phifar = _get_phifar(Y, Yc, Yc_tilde, f, p, q)
    phi = phinear + phifar
    return _finish_interpolation(Y, F, phi, K, N, Fas_sum)


if __name__ == '__main__':
    from testseries import semicircle
    from util import get_X

    K = 10
    J = 2*K
    X = get_X(K)
    Y = _np.sort(_np.random.uniform(0, _twopi, J))
    F = semicircle(X, K).real
    L = 4
    n = 3
    p = 4
    q = 2*J
    G = inufft(F, K, Y, L, p, n, q)

    _plt.plot(Y, G)
    _plt.xlim(0, _twopi)
    _plt.ylim(0, _np.pi)
