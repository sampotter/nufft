import groundtruth as _groundtruth
import matplotlib.pyplot as _plt
import nufft as _nufft
import numpy as _np
import testseries as _testseries

_twopi = 2*_np.pi

def _get_X(K):
    return _np.linspace(0, _twopi, 2*K, endpoint=False)

def _get_ground_truth_interp(X, F, Y, K):
    KK = _groundtruth.K(X, Y, 1 - K, K - 1)
    return _np.matrix(KK) * _np.matrix(F).transpose()

def vary_neighborhood_size():
    K = 20
    X = _get_X(K)
    J = 100
    Y = _np.sort(_np.random.uniform(0, _twopi, J))
    F = _testseries.semicircle(X, K).real
    L = 4
    p = 4
    q = 100
    min_n = 1
    max_n = 20
    Ns = range(min_n, max_n + 1)
    G_gt = _get_ground_truth_interp(X, F, Y, K)
    norms = []
    for n in Ns:
        G = _nufft.inufft(F, K, Y, L, p, n, q)
        norms.append(_np.log10(_np.linalg.norm(G - G_gt, 2)))
    _plt.figure()
    _plt.plot(Ns, norms, 'o')
    _plt.show()
    return Ns, norms

# TODO: try different check point distributions

# TODO: vary truncation number

# TODO: FMM depth vs speed, optimal depth

# TODO: radial neighboorhood approx vs per sum

# TODO: different test series -- with respect to what, though...? do this last
