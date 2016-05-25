import groundtruth
import matplotlib.pyplot as plt
import numpy as np
import testseries
import util

def get_ground_truth_interp(X, F, Y, K):
    KK = groundtruth.K(X, Y, 1 - K, K - 1)
    return np.matrix(KK) * np.matrix(F).transpose()

K = 4
X = util.get_X(K)
n = 4
p = 4
Ls = list(map(int, np.logspace(2, 4, 2*(4 - 2) + 1)))
F = testseries.sawtooth(X, K).real
Y = np.sort(np.random.uniform(0, 2*np.pi, 500))
G_eval = testseries.sawtooth(Y, K).real
G_gt = np.array(get_ground_truth_interp(X, F, Y, K).real.T)[0]

@np.vectorize
def c(m, L):
    tmp = 0
    I = np.arange(F.size)
    Fas = np.multiply((-1.0)**I, F)
    expt = -(m + 1)
    for l in range(n + 1, L + 1):
        tmp += np.sum(np.multiply(Fas, -(X + (2*l - 1)*np.pi)**expt))
        tmp += np.sum(np.multiply(Fas, -(X - (2*l + 1)*np.pi)**expt))
    return tmp

@np.vectorize
def phinear(y):
    if y in X:
        return 0
    else:
        tmp = 0
        for l in range(-n, n + 1):
            numer = np.multiply((-1.0)**np.arange(F.size), F)
            denom = y - X - 2*np.pi*l
            tmp += np.sum(np.divide(numer, denom))
        return tmp

@np.vectorize
def phifar(y, L, use_c0=True):
    P = range(0 if use_c0 else 1, p)
    if (L, p) not in phifar.coefs[use_c0]:
        phifar.coefs[use_c0][L, p] = c(P, L)
    coefs = phifar.coefs[use_c0][L, p]
    R = np.power(y - np.pi, P)
    return np.sum(np.multiply(coefs, R))

phifar.coefs = {True: {}, False: {}}

@np.vectorize
def phi(y, L, use_c0=True):
    return phinear(y) + phifar(y, L, use_c0)

@np.vectorize
def g(y, L, use_c0=True):
    return (1/K)*np.multiply(np.sin(K*y), phi(y, L, use_c0))

@np.vectorize
def g_only_phinear(y):
    return (1/K)*np.multiply(np.sin(K*y), phinear(y))

@np.vectorize
def _s(k, eps=1e-3):
	return np.sin(K*(X[k] + eps))/K

@np.vectorize
def _p(k, eps=1e-3):
	return phinear(X[k] + eps)

@np.vectorize
def cest(eps=1e-3):
    A = np.sum([_s(k, eps)*(F[k] - _s(k, eps)*_p(k, eps)) for k in range(2*K)])
    B = np.sum([_s(k, eps)*_s(k, eps) for k in range(2*K)])
    return A/B

@np.vectorize
def g_c0(y, c0):
    return (1/K)*np.multiply(np.sin(K*y), c0 + phinear(y))

if __name__ == '__main__':
    G_only_phinear = g_only_phinear(Y).real

    err_persum = []
    for L in Ls:
        print(L)
        G_persum = g(Y, L, use_c0=False).real
        err_persum.append(-np.log10(np.abs(G_persum - G_eval)))
    err_only_phinear = -np.log10(np.abs(G_only_phinear - G_eval))
    err_gt = np.minimum(15, -np.log10(np.abs(G_gt - G_eval)))

    # plotting number of correct digits for phifar, groundtruth, etc

    plt.figure()
    for i, L in enumerate(Ls):
        plt.plot(Y, err_persum[i], 'k--', label='Periodic Summation (L = %d)' % L)
    plt.plot(Y, err_only_phinear, 'k:', label='Only phi_near')
    plt.plot(Y, err_gt, 'k', label='Groundtruth')
    plt.legend()
    plt.xlim(0, 2*np.pi)
    plt.show()
