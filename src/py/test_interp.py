import matplotlib.pyplot as plt

import groundtruth
import nufft
import testseries

twopi = 2*np.pi

if __name__ == '__main__':
    J = 100
    Y = np.linspace(0, twopi, J, endpoint=False)
    K = 10

    def compute_G_fmm():
        X = np.linspace(0, twopi, 2*K, endpoint=False)
        F = testseries.semicircle(X, K).real
        L = 4
        p = 4
        q = 4
        G = groundtruth.inufft_radial_approximation(F, K, Y, L, p, q)
        return G
        
    def compute_G_persum():
        X = np.linspace(0, twopi, 2*K, endpoint=False)
        F = testseries.semicircle(X, K).real
        L = 4
        p = 4
        n = 4
        q = 10
        return nufft.inufft(F, K, Y, L, p, n, q)
        
    def compute_G_gt():
        N = 30
        X = np.linspace(0, twopi, N, endpoint=False)
        KK = groundtruth.K(X, Y, 1-K, K-1)
        F = testseries.semicircle(X, K).real
        return np.matrix(KK)*np.matrix(F).transpose()

    G_fmm = compute_G_fmm()
    G_persum = compute_G_persum()
    G_gt = compute_G_gt().real
    
    plt.figure()
    plt.plot(Y, G_fmm)
    plt.plot(Y, G_persum)
    plt.plot(Y, G_gt)
    plt.xlim(0, twopi)
    plt.show()
