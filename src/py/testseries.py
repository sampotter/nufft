import abc as _abc
import numpy as _np

from scipy.special import j1 as _j1


def _from_coef_func(coef_func, X, N):
    Ns = _np.array(range(1 - N, N))
    C = _np.vectorize(coef_func)(Ns)
    return _np.vectorize(lambda x: _np.dot(C, _np.exp(1j*Ns*x)))(X)


def sawtooth(X, N):
    def c(n):
        if n == 0:
            return 0.5
        else:
            return 1j/(2*_np.pi*n)
    return _from_coef_func(c, X, N)


def square(X, N):
    def c(n):
        if n % 2 == 0:
            return complex(0.0)
        else:
            return -2j/(_np.pi*n)
    return _from_coef_func(c, X, N)


def triangle(X, N):
    def c(n):
        if n % 2 == 0:
            return complex(0.0)
        else:
            return -(4j*(-1)**((n - 1)/2))/(4*(_np.pi**2)*(n**2))
    return _from_coef_func(c, X, N)


def semicircle(X, N):
    def c(n):
        if n == 0:
            return (_np.pi**2)/4
        else:
            return _np.pi*((-1.0)**n)*_j1(_np.pi*n)/(2*n)
    return _from_coef_func(c, X, N)


def _analysis_coefs_to_numpy_coefs(coefs):
    n = int(coefs.size/2)
    C = _np.zeros(coefs.shape, dtype=coefs.dtype)
    C[0] = coefs[n]
    C[1:n+1] = coefs[n+1:]
    C[n+1:] = coefs[:n]
    return C


def _numpy_coefs_to_analysis_coefs(coefs):
    n = int(coefs.size/2)
    C = _np.zeros(coefs.shape, dtype=coefs.dtype)
    C[:n] = coefs[n+1:]
    C[n] = coefs[0]
    C[n+1:] = coefs[1:n+1]
    return C


class TestSeries(metaclass=_abc.ABCMeta):
    _update_func_map = {
        ('analysis', 'numpy'): _analysis_coefs_to_numpy_coefs,
        ('numpy', 'analysis'): _numpy_coefs_to_analysis_coefs}

    _dft_forms = ['analysis', 'numpy']

    def __init__(self, dft_form='analysis'):
        self._coefs = dict()
        self._dft_form = dft_form

    @property
    def dft_form(self):
        return self._dft_form

    @dft_form.setter
    def dft_form(self, value):
        if value not in TestSeries._dft_forms:
            forms_str = ', '.join(TestSeries._dft_forms)
            raise Exception('dft_form must be one of: ' + forms_str)
        pair = (self.dft_form, value)
        if pair in TestSeries._update_func_map:
            for K in self._coefs:
                C = self._coefs[K]
                self._coefs[K] = TestSeries._update_func_map[pair](C)
        self._dft_form = value

    @_abc.abstractmethod
    def get_coef(self, k):
        pass

    def get_coefs(self, K):
        if K not in self._coefs:
            Ks = self._get_Ks(K)
            coefs = _np.array([self.get_coef(k) for k in Ks])
            if self.dft_form == 'numpy':
                coefs = _analysis_coefs_to_numpy_coefs(coefs)
            self._coefs[K] = coefs
        return self._coefs[K]

    def __call__(self, X, K):
        Ks = self._get_Ks(K)

        @_np.vectorize
        def func(x):
            return _np.dot(self.get_coefs(K), _np.exp(1j*_np.multiply(Ks, x)))

        return func(X)

    def _get_Ks(self, K):
        return _np.arange(1 - K, K)


class SawtoothTestSeries(TestSeries):
    def get_coef(self, k):
        if k == 0:
            return 0.5
        else:
            return 1j/(2*_np.pi*k)


class SquareTestSeries(TestSeries):
    def get_coef(self, k):
        if k % 2 == 0:
            return 0j
        else:
            return -2j/(_np.pi*k)


class TriangleTestSeries(TestSeries):
    def get_coef(self, k):
        if k % 2 == 0:
            return 0j
        else:
            return -(4j*(-1)**((k - 1)/2))/(4*(_np.pi**2)*(k**2))


class SemicircleTestSeries(TestSeries):
    def get_coef(self, k):
        if k == 0:
            return (_np.pi**2)/4
        else:
            return _np.pi*((-1.0)**k)*_j1(_np.pi*k)/(2*k)
