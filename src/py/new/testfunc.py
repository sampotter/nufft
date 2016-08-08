import abc as _abc
import numpy as _np

from scipy.special import j1 as _j1


class TestSeries(metaclass=_abc.ABCMeta):
    _update_func_map = {
        ('analysis', 'numpy'): _np.fft.ifftshift,
        ('numpy', 'analysis'): _np.fft.fftshift}

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
                coefs = _np.fft.ifftshift(coefs)
            self._coefs[K] = coefs
        return self._coefs[K]

    def __call__(self, X, K):
        Ks = self._get_Ks(K)

        @_np.vectorize
        def func(x):
            return _np.dot(self.get_coefs(K), _np.exp(1j*_np.multiply(Ks, x)))
        return func(X)/K

    def _get_Ks(self, K):
        return _np.arange(-_np.floor(K/2), _np.ceil(K/2))


class SawtoothTestSeries(TestSeries):
    def get_coef(self, k):
        if k == 0:
            return 0
        else:
            return 1j*_np.power(-1, k)/(_np.pi*k)


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
            return -(1j*(-1)**((k - 1)/2))/((_np.pi**2)*(k**2))


class SemicircleTestSeries(TestSeries):
    def get_coef(self, k):
        if k == 0:
            return (_np.pi**2)/4
        else:
            return _np.pi*((-1.0)**k)*_j1(_np.pi*k)/(2*k)
