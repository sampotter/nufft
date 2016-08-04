import argparse
import groundtruth
import irt
import nfft
import nufft_greengard
import nufft
import numpy as np
import testfunc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=''' compute a table of data consisting of the log10
        difference along [0, 2pi] between each method and the groundtruth
        interpolation method.''')
    parser.add_argument('bandlimit', type=int,
                        help='bandlimit of the trig. poly. (K)')
    parser.add_argument('resolution', type=int,
                        help='number of points to divide [0, 2pi] into')
    args = parser.parse_args()


def get_irt_diff(gt, pts, coefs):
    result = get_irt_diff.irt.inufft(pts, np.fft.fftshift(coefs),
                                     coefs.size, 2, 2*coefs.size)[0]
    print('irt: %g (l2)', np.linalg.norm(gt - result))
    diff = np.abs(gt - result)
    for i in range(len(diff)):
        diff[i] = max(1e-16, diff[i])
    return diff
get_irt_diff.irt = irt.irt()


def get_potts_diff(gt, pts, coefs):
    result = nfft.nfft_timer(coefs, pts)[0]
    print('nfft: %g (l2)', np.linalg.norm(gt - result))
    diff = np.abs(gt - result)
    for i in range(len(diff)):
        diff[i] = max(1e-16, diff[i])
    return diff


def get_greengard_diff(gt, pts, coefs):
    result = get_greengard_diff.ng.inufft(pts, coefs)[0]
    print('greengard: %g (l2)', np.linalg.norm(gt - result))
    diff = np.abs(gt - result)
    for i in range(len(diff)):
        diff[i] = max(1e-16, diff[i])
    return diff
get_greengard_diff.ng = nufft_greengard.nufft_greengard()


def get_nufft_diff(gt, pts, coefs):
    F = np.fft.ifft(np.fft.fftshift(coefs))
    K = int(F.size/2)
    Y = pts
    L = 4
    p = 4
    n = 3
    result = nufft.inufft_cpp(F, K, Y, L, p, n)[0]
    print('nufft: %g (l2)', np.linalg.norm(gt - result))
    diff = np.abs(gt - result)
    for i in range(len(diff)):
        diff[i] = max(1e-16, diff[i])
    return diff


if __name__ == '__main__':
    K = args.bandlimit
    if K < 1:
        raise Exception('bandlimit must be positive')
    N = args.resolution
    if N < 2:
        raise Exception('resolution must be at least 2')

    funcs = [
        testfunc.SemicircleTestSeries,
        testfunc.TriangleTestSeries,
        testfunc.SquareTestSeries,
        testfunc.SawtoothTestSeries]
    func_names = ['semi', 'tri', 'square', 'saw']

    for i, func in enumerate(funcs):
        print('func: %s' % func_names[i])
        pts = np.sort(np.random.uniform(0, 2*np.pi, N))
        coefs = func().get_coefs(K)
        gt = groundtruth.groundtruth_interp_using_np_ifft(coefs, pts)

        methods = [get_irt_diff, get_potts_diff, get_greengard_diff,
                   get_nufft_diff]
        method_names = ['irt', 'potts', 'greengard', 'nufft']
        for j, method in enumerate(methods):
            diff = method(gt, pts, coefs)
            np.savetxt('data_radian_diff_%s_%s.dat' % (method_names[j],
                                                       func_names[i]),
                       np.array([pts, diff]).T)
