import ctypes.util as _util
import fmm as _fmm
import numpy as _np
import time
import unittest as _unittest

from _nufft import ffi as _ffi
_lib = _ffi.dlopen(_util.find_library('nufft'))

_twopi = 2*_np.pi


def inufft_cpp(F, K, Y, L, p, n):
    num_values = F.size
    values = _ffi.new(
        'double[%d]' % (2*num_values),
        list(_np.array([F.real, F.imag]).T.reshape(2*num_values)))
    num_nodes = Y.size
    nodes = _ffi.new('double[%d]' % num_nodes, list(Y))
    fmm_depth = L
    truncation_number = p
    neighborhood_radius = n
    output = _ffi.new('double[%d]' % (2*num_nodes))
    t0 = time.clock()
    _lib.compute_P_ddi(
        values, nodes, num_values, num_nodes, fmm_depth, truncation_number,
        neighborhood_radius, output)
    t = time.clock() - t0
    output = _np.array(list(output)).reshape(num_nodes, 2)
    return output[:, 0] + 1j*output[:, 1], t


# TODO: implement c0 estimate
# TODO: implement NEW c0 estimate
# TODO: rewrite this as a class to simplify code a bit


# ALL THIS IS TEMPORARY (for cest implementation)...


def _phinear(F, X, y, n):
    if y in X:
        return 0
    else:
        tmp = 0
        for l in _np.arange(-n, n + 1):
            numer = _np.multiply((-1.0)**_np.arange(F.size), F)
            denom = y - X - 2*_np.pi*l
            tmp += _np.sum(_np.divide(numer, denom))
        return tmp


def _s(K, X, k, eps):
    return _np.sin(K*(X[k] + eps))/K


def _p(F, X, n, k, eps):
    return _phinear(F, X, X[k] + eps, n)


def _cest(F, X, K, n, eps=1e-5):
    A = _np.sum([_s(K, X, k, eps)*(F[k] - _s(K, X, k, eps)*_p(F, X, n, k, eps))
                 for k in _np.arange(2*K)])
    B = _np.sum([_s(K, X, k, eps)*_s(K, X, k, eps) for k in _np.arange(2*K)])
    return A/B


#### ... UP THROUGH HERE.


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


def _get_phifar(X, Y, Yc, Yc_tilde, F, K, f, p, n, q):
    '''Use least squares collocation to compute phifar. TODO: this needs
    to be updated and replaced with the Vandermonde approach.

    '''
    A = _np.zeros((q, p), dtype=Y.dtype)
    for m in range(p):
        A[:, m] = _R(Yc, m) - _R(Yc_tilde, m)
    c0 = _cest(F, X, K, n)
    C = _np.zeros(p, dtype=c0.dtype)
    C[0] = c0
    C[1:] = _np.linalg.lstsq(A, f)[0][1:]
    phifar = _np.zeros(Y.shape, dtype=C.dtype)
    for j in range(len(Y)):
        phifar[j] = _np.sum([C[m]*(Y[j] - _np.pi)**m for m in range(p)])
    return phifar


def _finish_interpolation(Y, F, phi, K, N, Fas_sum):
    '''This function takes care of the last couple steps: it modulates the
    result by the sine-based factor and sets any values that were too
    close to source points to the corresponding weight (i.e. function
    value).

    '''
    G = [(-Fas_sum*_np.cos(K*Y[j]) + 2*_np.sin(K*Y[j])*phi[j])/N
         for j in range(len(Y))]
    index_ratio = len(Y)/(2*K)
    for i, y in enumerate(Y):
        if _test(y, len(Y), index_ratio):
            G[i] = F[int(i/index_ratio)]
    return _np.array(G)


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
    phifar = _get_phifar(X, Y, Yc, Yc_tilde, F, K, f, p, n, q)
    phi = phinear + phifar
    return _finish_interpolation(Y, F, phi, K, N, Fas_sum)


class inufft_test(_unittest.TestCase):
    def test(self):
        Y = _np.array([
            0.06144733,
            0.12742735,
            0.19382462,
            0.76838064,
            0.81637296,
            0.97971646,
            1.04936551,
            1.21592974,
            1.25491235,
            1.45162833,
            1.53560262,
            1.59485713,
            1.75060987,
            1.79657517,
            1.8218675,
            2.54129315,
            2.6901106,
            2.87683413,
            3.01196423,
            3.10353108,
            3.23054149,
            3.59794781,
            3.83246419,
            4.30807601,
            4.61761236,
            5.31786545,
            5.62484061,
            5.66583099,
            5.67175258,
            6.07424511,
            6.11556149,
            6.19334216])
        coefs = _np.array([
            0.+0.j,
            -0.+0.09094568j,
            0.+0.j,
            -0.+0.12732395j,
            0.+0.j,
            -0.+0.21220659j,
            0.+0.j,
            -0.+0.63661977j,
            0.+0.j,
            -0.-0.63661977j,
            0.+0.j,
            -0.-0.21220659j,
            0.+0.j,
            -0.-0.12732395j,
            0.+0.j,
            -0.-0.09094568j])
        F = _np.fft.ifft(_np.fft.fftshift(coefs))
        K = int(F.size/2)
        L = 4
        p = 4
        n = 3
        q = Y.size
        expected = _np.array([
            0.01930708 + 4.67176242e-19j,  0.03833431 + 8.43065927e-19j,
            0.05414096 + 9.89540130e-19j,  0.05579820 - 1.34327825e-19j,
            0.05594726 + 2.42754205e-19j,  0.06196288 + 9.89611364e-19j,
            0.06507520 + 8.48430386e-19j,  0.06753474 - 2.95002823e-19j,
            0.06680686 - 5.70652687e-19j,  0.05968726 - 8.06992194e-19j,
            0.05779020 - 2.74994500e-19j,  0.05769407 + 1.89337641e-19j,
            0.06196792 + 9.81094362e-19j,  0.06382237 + 9.62438121e-19j,
            0.06479719 + 8.96407345e-19j,  0.06311348 + 9.85735660e-19j,
            0.07273939 + 4.48473340e-19j,  0.06605620 - 8.45187551e-19j,
            0.03886123 - 8.52064588e-19j,  0.01203484 - 2.96733713e-19j,
            -0.02750106 + 6.46340413e-19j, -0.07253002 - 4.82520351e-19j,
            -0.05780036 - 6.79132847e-19j, -0.06775652 + 9.18257837e-20j,
            -0.05897387 - 6.80571405e-19j, -0.06127630 - 9.81206947e-19j,
            -0.05936333 + 8.41511326e-19j, -0.06188170 + 9.64474493e-19j,
            -0.06228141 + 9.73915329e-19j, -0.05715692 - 9.84725529e-19j,
            -0.04839742 - 9.63722365e-19j, -0.02781498 - 6.51686509e-19j])

        # actual = inufft(F, K, Y, L, p, n, q)
        # for i, val in enumerate(actual):
        #     try:
        #         self.assertTrue(_np.abs(val - expected[i]) <= 1e-5)
        #     except:
        #         print('failure at i = %d' % i)
        #         print('expected: %g + j*%g'
        #               % (expected[i].real, expected[i].imag))
        #         print('actual: %g + j*%g' % (val.real, val.imag))
        #         self.assertTrue(False)

        actual_cpp = inufft_cpp(F, K, Y, L, p, n)
        for i, val in enumerate(actual_cpp):
            try:
                self.assertTrue(_np.abs(val - expected[i]) <= 1e-5)
            except:
                print('failure at i = %d' % i)
                print('expected: %g + j*%g'
                      % (expected[i].real, expected[i].imag))
                print('actual: %g + j*%g' % (val.real, val.imag))
                self.assertTrue(False)


if __name__ == '__main__':
    _unittest.main()
