import groundtruth
import irt
import nfft
import nufft_greengard
import nufft
import numpy as np
import testfunc
import time
import util


def get_irt_errors(gt, pts, coefs):
    result = get_irt_errors.irt.inufft(pts, np.fft.fftshift(coefs),
                                       coefs.size, 2, 2*coefs.size)[0]
    return np.linalg.norm(gt - result, 2), np.linalg.norm(gt - result, np.inf)
get_irt_errors.irt = irt.irt()


def get_potts_errors(gt, pts, coefs):
    result = nfft.nfft_timer(coefs, pts)[0]
    return np.linalg.norm(gt - result, 2), np.linalg.norm(gt - result, np.inf)


def get_greengard_errors(gt, pts, coefs):
    result = get_greengard_errors.ng.inufft(pts, coefs)[0]
    return np.linalg.norm(gt - result, 2), np.linalg.norm(gt - result, np.inf)
get_greengard_errors.ng = nufft_greengard.nufft_greengard()


def get_nufft_errors(gt, pts, coefs):
    F = np.fft.ifft(np.fft.fftshift(coefs))
    K = int(F.size/2)
    Y = pts
    L = 4
    p = 4
    n = 3
    result = nufft.inufft_cpp(F, K, Y, L, p, n)[0]
    return np.linalg.norm(gt - result, 2), np.linalg.norm(gt - result, np.inf)


if __name__ == '__main__':
    funcs = [testfunc.TriangleTestSeries,
             testfunc.SemicircleTestSeries,
             testfunc.SawtoothTestSeries,
             testfunc.SquareTestSeries]
    func_names = ['tri', 'semi', 'saw', 'square']

    for m, func in enumerate(funcs):
        func_name = func_names[m]
        print('func: %s' % func_name)
        Ks = np.array([int(K) if K % 2 == 0 else int(K + 1)
                       for K in map(np.round, np.logspace(1, 3, 20))])
        num_sizes = len(Ks)
        methods = [
            get_irt_errors,
            get_potts_errors,
            get_greengard_errors,
            get_nufft_errors]
        method_names = ['irt', 'potts', 'greengard', 'nufft']
        num_methods = len(methods)

        N = 2*np.max(Ks)

        l2_errors = np.zeros((num_sizes, num_methods))
        linf_errors = np.zeros((num_sizes, num_methods))
        for i, K in enumerate(Ks):
            print('K = %d' % K)
            coefs = func().get_coefs(K)
            pts = np.sort(np.random.uniform(0, 2*np.pi, N))
            gt = groundtruth.groundtruth_interp_using_np_ifft(coefs, pts)
            for j, method in enumerate(methods):
                l2_error, linf_error = method(gt, pts, coefs)
                l2_errors[i, j] = l2_error
                linf_errors[i, j] = linf_error
                print('  method: %s, l2 error: %g, linf error: %g' % (
                    method_names[j],
                    l2_error,
                    linf_error))

        for i, method_name in enumerate(method_names):
            np.savetxt('data_l2_error_%s_%s.dat' % (method_name, func_name),
                       np.array([Ks, l2_errors[:, i]]).T)
            np.savetxt('data_linf_error_%s_%s.dat' % (method_name, func_name),
                       np.array([Ks, linf_errors[:, i]]).T)
