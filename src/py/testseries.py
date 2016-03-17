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
