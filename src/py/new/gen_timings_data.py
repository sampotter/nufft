import groundtruth
import irt
import nfft
import nufft_greengard
import nufft
import numpy as np
import testfunc
import time
import util


include_gt = False

timing_ratio = 0.998
max_K_power = 4
p = 4
n = 7


# TODO: need to play with this a bit to figure out good parameters
def get_irt_timing(pts, coefs):
    return get_irt_timing.irt.inufft(pts, np.fft.fftshift(coefs),
                                     coefs.size, 4, 2*coefs.size)[1]
get_irt_timing.irt = irt.irt()


def get_potts_timing(pts, coefs):
    return nfft.nfft_timer(coefs, pts)[1]


def get_greengard_timing(pts, coefs):
    return get_greengard_timing.ng.inufft(pts, coefs)[1]
get_greengard_timing.ng = nufft_greengard.nufft_greengard()


def find_optimal_L(pts, coefs):
    F = np.fft.ifft(np.fft.fftshift(coefs))
    K = int(F.size/2)
    Y = pts

    L = 2
    output, t = nufft.inufft_cpp(F, K, Y, L, p, n)
    while True:
        t_prev = t
        L += 1
        output, t = nufft.inufft_cpp(F, K, Y, L, p, n)
        if t > t_prev:
            return L - 1


def get_nufft_cpp_timing(pts, coefs, L):
    F = np.fft.ifft(np.fft.fftshift(coefs))
    K = int(F.size/2)
    Y = pts
    output, t = nufft.inufft_cpp(F, K, Y, L, p, n)
    return t


def get_ifft_timing(pts, coefs):
    t0 = time.clock()
    np.fft.ifft(coefs)
    return time.clock() - t0


def get_gt_timing(pts, coefs):
    t0 = time.clock()
    groundtruth.groundtruth_interp_using_np_ifft(coefs, pts)
    return time.clock() - t0


if __name__ == '__main__':
    square = testfunc.SquareTestSeries()
    Ks = np.array([int(K) if K % 2 == 0 else int(K + 1)
                   for K in map(np.round, np.logspace(1, 4, 30))])
    num_sizes = len(Ks)

    timing_methods = [
        get_irt_timing,
        get_potts_timing,
        get_greengard_timing,
        get_nufft_cpp_timing,
        get_ifft_timing
    ]
    if include_gt:
        timing_methods += [get_gt_timing]
    timing_method_names = ['irt', 'potts', 'greengard', 'nufft', 'ifft']
    if include_gt:
        timing_method_names += ['gt']
    num_timing_methods = len(timing_methods)

    timings = np.zeros((num_sizes, num_timing_methods))
    for i, K in enumerate(Ks):
        print('K = %d' % K)
        coefs = square.get_coefs(K)
        pts = np.sort(np.random.uniform(0, 2*np.pi, K))
        for j, timing_method in enumerate(timing_methods):
            if timing_method == get_nufft_cpp_timing:
                L = find_optimal_L(pts, coefs)
                times = util.time_adaptive(lambda: timing_method(pts, coefs, L),
                                           timing_ratio)
                print('  method: %s, time: %g (using L = %d, out of %d trials)'
                      % (timing_method_names[j],
                         times[0],
                         L,
                         len(times)))
            else:
                times = util.time_adaptive(lambda: timing_method(pts, coefs),
                                           timing_ratio)
                print('  method: %s, time: %g (out of %d trials)'
                      % (timing_method_names[j],
                         times[0],
                         len(times)))
            timings[i, j] = times[0]

    for i, method_name in enumerate(timing_method_names):
        np.savetxt('data_timings_%s.dat' % method_name,
                   np.array([Ks, timings[:, i]]).T)
