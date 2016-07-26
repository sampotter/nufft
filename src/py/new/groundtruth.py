import numpy as _np


# TODO: get this working
def groundtruth_inv_nufft(xs, Xs):
    '''Compute the nonuniform inverse DFT directly (as from (38) in
    Fessler 2003 at xs (points in the time domain) from equally spaced
    frequencies (Xs). The frequencies Xs are assumed to be laid out in
    'FFT form'.'''

    @_np.vectorize
    def func(x):
        return _np.dot(Xs, _np.exp((2j*_np.pi*x/Xs.size)*_np.arange(Xs.size)))
    return func(xs)


def groundtruth_interp_using_np_ifft(coefs, interp_pts, dft_form='analysis'):
    if dft_form not in {'fft', 'analysis'}:
        raise Exception("dft_form must be one of 'fft' or 'analysis'")
    if dft_form == 'analysis':
        coefs = _np.fft.ifftshift(coefs)
    xs = _np.fft.ifft(coefs)
    return groundtruth_interp(xs, interp_pts)


# TODO: optimize this for fairness and reasonableness
def groundtruth_interp(xs, interp_pts):
    K = xs.size
    J = interp_pts.size
    grid = _np.linspace(0, 2*_np.pi, K, endpoint=False)
    Ks = _np.arange(-_np.floor(K/2), _np.ceil(K/2))
    ys = _np.zeros(J, dtype=xs.dtype)
    for j in range(J):
        for i, k in enumerate(Ks):
            ys[j] += xs[i]*_np.sum(_np.exp(1j*Ks*(interp_pts[j] - grid[i])))
        ys[j] /= K
    return ys
