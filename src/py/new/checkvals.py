import numpy as np

from groundtruth import *
from irt import *
from nfft import *
from nufft_greengard import *
from nufft import inufft
from testfunc import *

def compute_irt_inufft(pts, coefs):
    return irt().inufft(pts, np.fft.fftshift(coefs), coefs.size, 4, 4*coefs.size)[0].T

def compute_potts_inufft(pts, coefs):
    return nfft_timer(coefs, pts)[0]

def compute_greengard_inufft(pts, coefs):
    return nufft_greengard().inufft(pts, coefs)[0]

def compute_gt_inufft(pts, coefs):
    return groundtruth_interp_using_np_ifft(coefs, pts)

def compute_inufft(pts, coefs):
    F = np.fft.ifft(np.fft.fftshift(coefs))
    K = int(F.size/2)
    Y = pts
    L = 4
    p = 4
    n = 3
    q = Y.size
    return inufft(F, K, Y, L, p, n, q)

if __name__ == '__main__':
    coefs = SquareTestSeries().get_coefs(16)
    pts = np.linspace(0, 2*np.pi, 25, endpoint=False)
    print('irt')
    print(compute_irt_inufft(pts, coefs))
    print('potts')
    print(compute_potts_inufft(pts, coefs))
    print('greengard')
    print(compute_greengard_inufft(pts, coefs))
    print('groundtruth')
    print(compute_gt_inufft(pts, coefs))
    print('mine')
    print(compute_inufft(pts, coefs))
