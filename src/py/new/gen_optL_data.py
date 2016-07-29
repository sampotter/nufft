import argparse
import nufft
import numpy as np
import testfunc
import util


def get_nufft_cpp_timing(pts, coefs, L):
    F = np.fft.ifft(np.fft.fftshift(coefs))
    K = int(F.size/2)
    Y = pts
    p = 4
    n = 3
    output, t = nufft.inufft_cpp(F, K, Y, L, p, n)
    return t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''compute a table of timings for the C++-based NUFFT
        comparing choice of FMM depth and problem size (number of sources and
        targets in the resulting FMM)''')
    parser.add_argument('min_level', type=int, help='minimum FMM depth value')
    parser.add_argument('max_level', type=int, help='maximum FMM depth value')
    parser.add_argument('min_N_power', type=int)
    parser.add_argument('max_N_power', type=int)
    parser.add_argument('num_Ns', type=int)
    args = parser.parse_args()
    min_level = args.min_level
    max_level = args.max_level
    min_N_power = args.min_N_power
    max_N_power = args.max_N_power
    num_Ns = args.num_Ns
    if min_level < 2:
        raise Exception('min FMM depth must be at least 2')
    if min_level > max_level:
        raise Exception('min FMM depth must be less than or equal to max')
    if min_N_power < 0:
        raise Exception('min prob. size power must be nonnegative')
    if min_N_power > max_N_power:
        raise Exception('min prob. size power must be at least max size')
    if num_Ns < 1:
        raise Exception('number of problem sizes must be positive')

    square = testfunc.SquareTestSeries()
    Ks = np.array([
        int(K) if K % 2 == 0 else int(K + 1)
        for K in map(np.round, np.logspace(min_N_power, max_N_power, num_Ns))])
    Ls = np.arange(min_level, max_level + 1)
    timings = np.zeros((len(Ks), len(Ls)))

    for i, K in enumerate(Ks):
        print('K = %d' % K)
        coefs = square.get_coefs(K)
        pts = np.sort(np.random.uniform(0, 2*np.pi, K))
        for j, L in enumerate(Ls):
            T = util.time_adaptive(lambda: get_nufft_cpp_timing(pts, coefs, L))
            print('  L = %d, t = %g (%d trials)' % (L, T[0], len(T)))
            timings[i, j] = T[0]

    np.savez_compressed('optL.npz', Ks, Ls, timings)
