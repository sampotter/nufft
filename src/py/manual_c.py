import itertools as _itertools
import numpy as _np

def a(x, xstar, m):
    return -(x - xstar)**(-(m + 1))

def c(m, X, f, n, N, K):
    tmp = 0
    for l in _itertools.chain(range(-N + 1, -n + 1), range(n, N)):
        tmp += sum([(-1)**k * f[k] * a(X[k] + 2*_np.pi*l, _np.pi, m) for k in range(2*K)])
    return tmp

def c_precise(m, X, f, n, K):
    '''TODO: this may or may not work correctly.'''
    def am(x):
        return _np.power(-(x - _np.pi), -(m + 1))
    c_prev = _np.inf
    c = 0
    i = 0
    l = n + 1
    Ks = _np.arange(2*K)
    Fas = [(-1)**k * f[k] for k in Ks]
    while abs(c - c_prev) > 1e-15:
        c_prev = c
        c += _np.vdot(Fas, am(X + 2*_np.pi*l))
        c += _np.vdot(Fas, am(X - 2*_np.pi*l))
        l += 1
        i += 1
        if i % 1000 == 0: print(c)
    return c, i

def shanks(A, N):
    ANm1 = A(N - 1)
    ANp1 = A(N + 1)
    AN = A(N)
    return (ANp1*ANm1 - AN*AN)/(ANp1 - 2*AN + ANm1)

def repshanks(A, N, R=1):
    A = [A(n) for n in range(N - R, N + R + 1)]
    def S(B, M):
        return (B[M+1]*B[M-1] - B[M]**2)/(B[M+1] - 2*B[M] + B[M-1])
    for r in range(R):
        A = [S(A, n) for n in range(1, len(A) - 1)]
    return A[0]

def c_shanks(m, X, f, n, N, K):
    return shanks(lambda N: c(m, X, f, n, N, K), N)

def c_shanks_2(m, X, f, n, N, K):
    return shanks(lambda N: c_shanks(m, X, f, n, N, K), N)

def c_shanks_3(m, X, f, n, N, K):
    return shanks(lambda N: c_shanks_2(m, X, f, n, N, K), N)
