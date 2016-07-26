import bisect
import groundtruth
import irt
import nfft
import nufft_greengard
import nufft
import numpy as np
import testfunc
import time


include_gt = False


def time_adaptive(timer, ratio=0.9965, minreps=10, maxreps=100):
    times = []
    reps = 0
    cond = True
    while cond:
        time = timer()
        bisect.insort_left(times, time)
        reps += 1
        good_enough = times[0]/times[1] > ratio if reps > 1 else False
        stop = reps >= maxreps or good_enough
        cond = reps < minreps or not stop
    return times


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


def get_nufft_timing(pts, coefs):
    K = int(coefs.size/2)
    Y = pts
    L = 4
    p = 4
    n = 3
    q = Y.size
    shifted_coefs = np.fft.fftshift(coefs)
    t0 = time.clock()
    nufft.inufft(np.fft.ifft(shifted_coefs), K, Y, L, p, n, q)
    return time.clock() - t0


def get_ifft_timing(pts, coefs):
    t0 = time.clock()
    np.fft.ifft(coefs)
    return time.clock() - t0


def get_gt_timing(pts, coefs):
    t0 = time.clock()
    groundtruth.groundtruth_interp_using_np_ifft(coefs, pts)
    return time.clock() - t0


square = testfunc.SquareTestSeries()
Ks = np.array([int(K) if K % 2 == 0 else int(K + 1)
               for K in map(np.round, np.logspace(1, 3, 20))])

num_sizes = len(Ks)

timing_methods = [
    get_irt_timing,
    get_potts_timing,
    get_greengard_timing,
    get_nufft_timing,
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
        times = time_adaptive(lambda: timing_method(pts, coefs))
        print('  method: %s, time: %g (out of %d trials)'
              % (timing_method_names[j],
                 times[0],
                 len(times)))
        timings[i, j] = times[0]


np.savez_compressed('timings.npz', Ks, timings)
