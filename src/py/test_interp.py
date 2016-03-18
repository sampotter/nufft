import fmm
import groundtruth
import testseries

import functools
import matplotlib.pyplot as plt
import numpy as np

twopi = 2*np.pi

def approx_interp_fmm(F, K, Y, L, p, q):
    '''Approximates multiplication by the bandlimited interpolation matrix
    by truncating the infinite series in its definition and using the FMM
    to compute the truncated summation. This is as opposed to applying the
    periodic summation algorithm to the infinite series.

    Arguments:
        F: test function evaluated at N equally spaced points 
           along [0, 2pi).
        K: bandlimit parameter.
        Y: sorted list of target points in [0, 2pi).
        L: depth of FMM.
        p: truncation parameter of FMM.
        q: term at which to truncate infinite series -- i.e., the 
           approx_interp_fmm will sum over terms -q, -q+1, ..., q.

    Returns: a numpy array containing the results.
    '''
    N = len(F)
    X = np.linspace(0, twopi, N, endpoint=False)

    # Create a linearly spaced grid of source points for the 2q + 1
    # terms of the truncated series.
    def shifted_X(m):
        return X + twopi*m
    X_per = np.concatenate([shifted_X(m) for m in range(-q, q + 1)])

    # Compute F with alternating signs and extend it to match X.
    Fas = [F[n]*(-1)**n for n in range(N)]
    Fas_sum = sum(Fas)
    Fas_per = np.tile(Fas, 2*q + 1)

    V = fmm.fmm1d_cauchy_double(X_per, Y, Fas_per, L, p,
                                scaled_domain=(-twopi*q, twopi*(q + 1)))
    def g(j):
        return (-Fas_sum*np.cos(K*Y[j]) + 2*np.sin(K*Y[j])*V[j])/N
    return [g(j) for j in range(len(Y))]

def transpose_argsort_indices(I):
    J = np.zeros(len(I), dtype=np.int)
    for i, j in enumerate(I):
        J[j] = i
    return J

def approx_interp_persum(F, K, Y, L, p, q, J):
    '''
    q is the truncation number for the periodic summation method --
    i.e. the number that chooses the number of terms in phi_near
    '''
    N = len(F)
    X = np.linspace(0, twopi, N, endpoint=False)

    # Create a linearly spaced grid of source points for the 2q + 1
    # terms of the truncated series.
    def shifted_X(m):
        return X + twopi*m
    X_per = np.concatenate([shifted_X(m) for m in range(-q, q + 1)])

    # Compute F with alternating signs and extend it to match X.
    Fas = [F[n]*(-1)**n for n in range(N)]
    Fas_sum = sum(Fas)
    Fas_per = np.tile(Fas, 2*q + 1)

    # Compute checkpoints...
    Yc = np.concatenate((np.random.uniform(-twopi, 0, np.ceil(J/2)),
                         np.random.uniform(0, twopi, np.floor(J/2))))
    Yc_tilde = np.mod(Yc, twopi)

    # Join together actual targets and checkpoint targets and compute
    # sorting and "unsorting" permutations for use with FMM.
    Ycat = np.concatenate((Y, Yc, Yc_tilde))
    I = np.argsort(Ycat)
    Irev = transpose_argsort_indices(I)

    # Use FMM to evaluate at targets...
    V = fmm.fmm1d_cauchy_double(X_per, Ycat[I], Fas_per, L, p,
                                scaled_domain=(-twopi*q, twopi*(q + 1)))

    # Unsort V so that it's possible to extract the different "types"
    # of evaluates.
    V_unsorted = V[Irev]
    
    # Extract checkpoint evaluates for computing far summation.
    Vc_start = len(Y)
    Vc_tilde_start = Vc_start + len(Yc)
    Vc = V_unsorted[Vc_start:Vc_start + len(Yc)]
    Vc_tilde = V_unsorted[Vc_tilde_start:Vc_tilde_start + len(Yc_tilde)]

    # Compute far summation using least squares collocation.
    f = Vc_tilde - Vc
    A = np.zeros((J, p))
    def R(Y, m):
        return np.power(Y - np.pi, m)
    for m in range(p):
        A[:, m] = R(Yc, m) - R(Yc_tilde, m)
    C = np.linalg.lstsq(A, f)[0]

    phifar = np.zeros(Y.shape, dtype=Y.dtype)
    for j in range(len(Y)):
        phifar[j] = np.sum([C[m]*(Y[j] - np.pi)**m for m in range(p)])

    # Extract near summation from evaluates computed using FMM.
    phinear = V_unsorted[:len(Y)]

    # Compute final V from phinear and phifar.
    V = phinear + phifar

    # Compute and return interpolation using (10).
    def g(j):
        return (-Fas_sum*np.cos(K*Y[j]) + 2*np.sin(K*Y[j])*V[j])/N
    return [g(j) for j in range(len(Y))]

if __name__ == '__main__':
    J = 100
    # Y = np.array(sorted(np.random.uniform(0, 2*np.pi, J)))
    Y = np.linspace(0, twopi, J, endpoint=False)
    K = 10
    XY_index_ratio = J/(2*K)

    def test(y):
        '''
        This test is necessary to check if we've passed a point that we
        essentially already know the value of (i.e. an evaluation
        point that's nearly equal to a source point). We could
        probably handle this correctly in the FMM itself, but for now,
        a test like this should give us approximately what we
        want... That is, no interpolates that are NaN, +/-Inf, or
        incorrectly equal to zero.
        '''
        return np.abs(np.mod(y*(J/twopi), XY_index_ratio)) < 1e-13

    def compute_G_fmm():
        X = np.linspace(0, twopi, 2*K, endpoint=False)
        F = testseries.semicircle(X, K).real
        L = 4
        p = 4
        q = 4
        G = approx_interp_fmm(F, K, Y, L, p, q)
        for i, y in enumerate(Y):
            if test(y):
                j = int(i/XY_index_ratio)
                G[i] = F[j]
        return G
        
    def compute_G_persum():
        X = np.linspace(0, twopi, 2*K, endpoint=False)
        F = testseries.semicircle(X, K).real
        L = 4
        p = 4
        q = 4
        num_cps = 10
        G = approx_interp_persum(F, K, Y, L, p, q, num_cps)
        for i, y in enumerate(Y):
            if test(y):
                j = int(i/XY_index_ratio)
                G[i] = F[j]
        return G
        
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
