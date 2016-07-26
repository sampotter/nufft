import numpy as _np
import unittest as _unittest
from ctypes.util import find_library as _find_library
from _nfft import ffi as _ffi

_lib = _ffi.dlopen(_find_library('nfft_timer'))
_nfft_timer = _lib.nfft_timer


def nfft_timer(coefs, interp_pts):
    N = coefs.size
    M = interp_pts.size

    f_hat_real = _ffi.new('double[%d]' % N)
    f_hat_imag = _ffi.new('double[%d]' % N)
    for i in range(N):
        f_hat_real[i] = -coefs[i].real
        f_hat_imag[i] = -coefs[i].imag

    pts = _np.mod(interp_pts.copy(), 2*_np.pi)
    pts = _np.divide(pts, 2*_np.pi)
    I = pts >= 0.5
    pts[I] = pts[I] - 1
    x = _ffi.new('double[%d]' % M)
    for i in range(M):
        x[i] = pts[i]

    f_real = _ffi.new('double[%d]' % M)
    f_imag = _ffi.new('double[%d]' % M)

    time = _ffi.new('double *')

    code = _nfft_timer(N, M, f_hat_real, f_hat_imag, x, f_real, f_imag, time)
    if code != 0:
        raise Exception('nfft_timer error: code ' + str(code))

    result = (_np.array(list(f_real)) + 1j*_np.array(list(f_imag)))/N
    return result, time[0]


class nfft_timer_test(_unittest.TestCase):
    def test(self):
        from testfunc import SquareTestSeries
        square = SquareTestSeries()
        coefs = square.get_coefs(16)
        interp_pts = _np.linspace(0, 2*_np.pi, 24, endpoint=False)
        output, time = nfft_timer(coefs, interp_pts)
        expected_output = _np.array([
            -3.28335909e-17+0.j,
            -1.05130864e+00+0.j,
            -1.09741123e+00+0.j,
            -8.91741875e-01+0.j,
            -1.03964877e+00+0.j,
            -1.04273420e+00+0.j,
            -9.21582909e-01+0.j,
            -1.04273420e+00+0.j,
            -1.03964877e+00+0.j,
            -8.91741875e-01+0.j,
            -1.09741123e+00+0.j,
            -1.05130864e+00+0.j,
            3.28335909e-17+0.j,
            1.05130864e+00+0.j,
            1.09741123e+00+0.j,
            8.91741875e-01+0.j,
            1.03964877e+00+0.j,
            1.04273420e+00+0.j,
            9.21582909e-01+0.j,
            1.04273420e+00+0.j,
            1.03964877e+00+0.j,
            8.91741875e-01+0.j,
            1.09741123e+00+0.j,
            1.05130864e+00+0.j])
        print(output)
        print(expected_output)
        print(output == expected_output)


if __name__ == '__main__':
    _unittest.main()
