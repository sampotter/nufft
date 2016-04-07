import inspect as _inspect
import groundtruth as _groundtruth
import matplotlib.pyplot as _plt
import nufft as _nufft
import numpy as _np
import testseries as _testseries
import util as _util

from matplotlib import cm as _cm

_twopi = 2*_np.pi

def _get_X(K):
    return _np.linspace(0, _twopi, 2*K, endpoint=False)

def _get_ground_truth_interp(X, F, Y, K):
    KK = _groundtruth.K(X, Y, 1 - K, K - 1)
    return _np.matrix(KK) * _np.matrix(F).transpose()

# TODO: plot of error vs location of evaluation point... This would be
# a useful graph.

# TODO: try different check point distributions

# TODO: different test series -- with respect to what, though...? do this last

def radial_error_vs_target_location(save=False, preamble=False):
    if preamble:
        print('#' * 80)
        print('# ' + _inspect.stack()[0][3])
        print('#')

    K = 2
    X = _get_X(K).real
    J = 500
    L = 10
    p = 50
    q = 10
    Y = _np.linspace(0, _twopi, J, endpoint=False)
    F = _testseries.semicircle(X, K).real
    G_gt = _get_ground_truth_interp(X, F, Y, K).real
    n_min_power = 1
    n_max_power = 4
    num_Ns = 500
    Ns = _np.logspace(n_min_power, n_max_power, num_Ns)

    digits = _np.matrix(_np.zeros((J, num_Ns), dtype=_np.float64))
    for i, n in enumerate(Ns):
        print('n = %d' % n)
        G_rad = _np.matrix(_groundtruth.inufft_radial_approximation(F, K, Y, L, p, int(n))).T
        digits[:, i] = -_np.log10(_np.abs(G_gt - G_rad))

    fig = _plt.figure()
    imshow_ax = _plt.imshow(digits, interpolation='nearest', cmap=_cm.cubehelix)
    _plt.xticks(_np.linspace(1, num_Ns, n_max_power - n_min_power + 1))
    _plt.yticks(_np.linspace(1, J, 5))
    ax = _plt.gca()
    xticklabels = ['1e%d' % power for power in range(n_min_power, n_max_power + 1)]
    ax.set_xticklabels(xticklabels)
    yticklabels = ['0', u'π/2', u'π', u'3π/2', u'2π']
    ax.set_yticklabels(yticklabels)
    fig.colorbar(imshow_ax)
    _plt.title('Number of Correct Digits by Radial Approx. INUFFT (K = %d)' % K)
    _plt.xlabel('Neighborhood Radius (n)')
    _plt.ylabel('Evaluation Point (y)')
    if save:
        _plt.savefig(_inspect.stack()[0][3] + '.pdf')
    else:
        _plt.show()

def per_error_vs_target_location(save=False, preamble=False):
    if preamble:
        print('#' * 80)
        print('# ' + _inspect.stack()[0][3])
        print('#')

    K = 2
    X = _get_X(K).real
    J = 500
    L = 10
    p = 50
    q = 10
    Y = _np.linspace(0, _twopi, J, endpoint=False)
    F = _testseries.semicircle(X, K).real
    G_gt = _get_ground_truth_interp(X, F, Y, K).real
    n_min_power = 1
    n_max_power = 4
    num_Ns = 500
    Ns = _np.logspace(n_min_power, n_max_power, num_Ns)

    digits = _np.matrix(_np.zeros((J, num_Ns), dtype=_np.float64))
    for i, n in enumerate(Ns):
        print('n = %d' % n)
        G_per = _np.matrix(_nufft.inufft(F, K, Y, L, p, int(n), q)).T
        digits[:, i] = -_np.log10(_np.abs(G_gt - G_per))

    fig = _plt.figure()
    imshow_ax = _plt.imshow(digits, interpolation='nearest', cmap=_cm.cubehelix)
    _plt.xticks(_np.linspace(1, num_Ns, n_max_power - n_min_power + 1))
    _plt.yticks(_np.linspace(1, J, 5))
    ax = _plt.gca()
    xticklabels = ['1e%d' % power for power in range(n_min_power, n_max_power + 1)]
    ax.set_xticklabels(xticklabels)
    yticklabels = ['0', u'π/2', u'π', u'3π/2', u'2π']
    ax.set_yticklabels(yticklabels)
    fig.colorbar(imshow_ax)
    _plt.title('Number of Correct Digits by Per. Sum. INUFFT (K = %d)' % K)
    _plt.xlabel('Neighborhood Radius (n)')
    _plt.ylabel('Evaluation Point (y)')
    if save:
        _plt.savefig(_inspect.stack()[0][3] + '.pdf')
    else:
        _plt.show()

def rad_vs_per_wrt_n_with_fixed_Y(save=False, preamble=False):
    if preamble:
        print('#' * 80)
        print('# ' + _inspect.stack()[0][3])
        print('#')

    K = 20
    X = _get_X(K)
    J = 100
    L = 4
    p = 4
    q = 10
    Y = _np.sort(_np.random.uniform(0, _twopi, J))
    F = _testseries.semicircle(X, K).real
    G_gt = _get_ground_truth_interp(X, F, Y, K)
        
    min_n = 1
    max_n = 20
    Ns = range(min_n, max_n + 1)
    
    mses = {
        'per': _np.zeros((len(Ns), 1), dtype=_np.float64),
        'rad': _np.zeros((len(Ns), 1), dtype=_np.float64)
    }
    for i, n in enumerate(Ns):
        print('n = %d' % n)
        G_per = _nufft.inufft(F, K, Y, L, p, n, q)
        G_rad = _groundtruth.inufft_radial_approximation(F, K, Y, L, p, n)
        mses['per'][i] = _np.log10(_np.linalg.norm(G_per - G_gt, 2)**2/len(G_gt))
        mses['rad'][i] = _np.log10(_np.linalg.norm(G_rad - G_gt, 2)**2/len(G_gt))

    _plt.figure()
    _plt.plot(Ns, mses['per'], label='INUFFT')
    _plt.plot(Ns, mses['rad'], label='Radial Approx.')
    _plt.xlim((min_n - 1, max_n + 1))
    _plt.title('Error vs. Neighborhood Size with Fixed Targets')
    _plt.xlabel('Neighborhood Size')
    _plt.ylabel('Log10 MSE')
    _plt.legend()
    if save:
        _plt.savefig(_inspect.stack()[0][3] + '.pdf')
    else:
        _plt.show()
    
    return Ns, mses
    
def rad_vs_per_for_mse_wrt_n(save=False, preamble=False):
    # TODO: this plot indicates that the error due to the position of
    # the evaluation points dwarfs the error due to ignoring the use
    # of least squares collocation.

    if preamble:
        print('#' * 80)
        print('# ' + _inspect.stack()[0][3])
        print('#')

    K = 20
    X = _get_X(K)
    J = 100
    L = 4
    p = 4
    q = 10
    min_n = 1
    max_n = 20
    Ns = range(min_n, max_n + 1)

    trials = 50
    mses = {
        'per': _np.zeros((len(Ns), 1), dtype=_np.float64),
        'rad': _np.zeros((len(Ns), 1), dtype=_np.float64)
    }
    for i, n in enumerate(Ns):
        print('n = %d' % n)
        _mses = _np.zeros((trials, 2), dtype=_np.float64)
        for t in range(trials):
            Y = _np.sort(_np.random.uniform(0, _twopi, J))
            F = _testseries.semicircle(X, K).real
            G_gt = _get_ground_truth_interp(X, F, Y, K)
            G_per = _nufft.inufft(F, K, Y, L, p, n, q)
            G_rad = _groundtruth.inufft_radial_approximation(F, K, Y, L, p, n)
            _mses[t, 0] = _np.log10(_np.linalg.norm(G_per - G_gt, 2)**2/len(G_gt))
            _mses[t, 1] = _np.log10(_np.linalg.norm(G_rad - G_gt, 2)**2/len(G_gt))
        mses['per'][i] = _np.median(_mses[:, 0])
        mses['rad'][i] = _np.median(_mses[:, 1])
        
    _plt.figure()
    _plt.plot(Ns, mses['per'], label='INUFFT')
    _plt.plot(Ns, mses['rad'], label='Radial Approx.')
    _plt.xlim((min_n - 1, max_n + 1))
    _plt.xlabel('Neighborhood Size')
    _plt.ylabel('Median of Log10 of MSE (50 trials)')
    _plt.legend()
    if save:
        _plt.savefig(_inspect.stack()[0][3] + '.pdf')
    else:
        _plt.show()
    
    return Ns, mses

def p_vs_n_wrt_mse(save=False, preamble=False):
    if preamble:
        print('#' * 80)
        print('# ' + _inspect.stack()[0][3])
        print('#')

    K = 10
    X = _get_X(K)
    J = 100
    q = 100
    L = 4
    p_min = 1
    p_max = 20
    Ps = range(p_min, p_max + 1)
    n_min = 1
    n_max = 20
    Ns = range(n_min, n_max + 1)

    trials = 50
    mses = _np.zeros((len(Ps), len(Ns)), dtype=_np.float64)
    for i, p in enumerate(Ps):
        for j, n in enumerate(Ns):
            print('(p, n) = (%d, %d)' % (n, p))
            _mses = _np.zeros((trials, 1), dtype=_np.float64)
            for t in range(trials):
                Y = _np.sort(_np.random.uniform(0, _twopi, J))
                F = _testseries.semicircle(X, K).real
                G_gt = _get_ground_truth_interp(X, F, Y, K)
                G = _nufft.inufft(F, K, Y, L, p, n, q)
                diff = (G - G_gt).real
                _mses[t] = _np.log10(_np.vdot(diff, diff))
            mses[i, j] = _np.median(_mses)

    fig = _plt.figure()
    imshow_ax = _plt.imshow(mses, interpolation='nearest', cmap=_cm.cubehelix)
    ax = _plt.gca()
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_yticklabels([str(p) for p in Ps])
    fig.colorbar(imshow_ax)
    _plt.title('Log10 of MSE (median of 50 trials)')
    _plt.xlabel('Neighborhood Radius')
    _plt.ylabel('Truncation Number')
    _plt.xticks(range(len(Ns)))
    _plt.yticks(range(len(Ps)))
    if save:
        _plt.savefig(_inspect.stack()[0][3] + '.pdf')
    else:
        _plt.show()
    
    return Ps, Ns, mses

def L_vs_p_wrt_time(save=False, preamble=False):
    if preamble:
        print('#' * 80)
        print('# ' + _inspect.stack()[0][3])
        print('#')

    K = 10
    X = _get_X(K)
    J = 100
    Y = _np.sort(_np.random.uniform(0, _twopi, J))
    F = _testseries.semicircle(X, K).real
    q = 100
    n = 10
    p_min = 1
    p_max = 10
    Ps = range(p_min, p_max + 1)
    L_min = 2
    L_max = 11
    Ls = range(L_min, L_max + 1)

    times = _np.zeros((len(Ls), len(Ps)), dtype=_np.float64)
    for i, L in enumerate(Ls):
        for j, p in enumerate(Ps):
            print('(L, p) = (%d, %d)' % (L, p))
            def f():
                _nufft.inufft(F, K, Y, L, p, n, q)
            timer = _util.Timer(f)
            times[i, j] = _np.log10(timer.median)

    fig = _plt.figure()
    imshow_ax = _plt.imshow(times, interpolation='nearest', cmap=_cm.cubehelix)
    ax = _plt.gca()
    ax.set_xticklabels([str(p) for p in Ps])
    ax.set_yticklabels([str(L) for L in Ls])
    fig.colorbar(imshow_ax)
    _plt.title('Log10 of Execution Time')
    _plt.xlabel('Truncation Number')
    _plt.ylabel('FMM Depth')
    _plt.xticks(range(len(Ps)))
    _plt.yticks(range(len(Ls)))
    if save:
        _plt.savefig(_inspect.stack()[0][3] + '.pdf')
    else:
        _plt.show()
    
    return Ls, Ps, times

def rad_vs_persum_with_n_wrt_mse(save=False, preamble=False):
    if preamble:
        print('#' * 80)
        print('# ' + _inspect.stack()[0][3])
        print('#')

    K = 10
    X = _get_X(K)
    J = 100
    q = 100
    L = 4
    p = 4
    n_min = 1
    n_max = 20
    Ns = range(n_min, n_max + 1)

    trials = 10
    mses = _np.zeros((len(Ns), 2), dtype=_np.float64)
    for i, n in enumerate(Ns):
        print('n = %d' % n)
        _mses = _np.zeros((trials, 2), dtype=_np.float64)
        for t in range(trials):
            Y = _np.sort(_np.random.uniform(0, _twopi, J))
            F = _testseries.semicircle(X, K).real
            G_gt = _get_ground_truth_interp(X, F, Y, K).real
            G_persum = _nufft.inufft(F, K, Y, L, p, n, q)
            G_radial = _groundtruth.inufft_radial_approximation(F, K, Y, L, p, n)
            diff_persum = G_persum - G_gt
            diff_radial = G_radial - G_gt
            digits_persum = -_np.log10(_np.abs(diff_persum))
            digits_radial = -_np.log10(_np.abs(diff_radial))
            _mses[t, 0] = digits_persum.mean()
            _mses[t, 1] = digits_radial.mean()
        mses[i, 0] = _np.median(_mses[:, 0])
        mses[i, 1] = _np.median(_mses[:, 1])

    _plt.figure()
    plotrange = range(len(Ns))
    _plt.plot(plotrange, mses[:, 0]/mses[:, 1], 'o')
    _plt.xticks(plotrange)
    _plt.xlim((plotrange[0] - 1, plotrange[-1] + 1))
    _plt.ylabel('Log10 MSE of Per. Sum. over Log10 MSE of Radial')
    _plt.xlabel('Neighborhood Size')
    ax = _plt.gca()
    ax.set_xticklabels([''] + [str(n) for n in Ns] + [''])
    if save:
        _plt.savefig(_inspect.stack()[0][3] + '.pdf')
    else:
        _plt.show()

    return Ns, mses

def gt_vs_rad_vs_per_with_bandlimit_wrt_time(save=False, preamble=False):
    if preamble:
        print('#' * 80)
        print('# ' + _inspect.stack()[0][3])
        print('#')

    q = 100
    L = 4
    p = 4
    n = 10
    
    K_min = 5
    K_max = 50
    Ks = range(K_min, K_max + 1)

    times = {
        'gt': _np.zeros((len(Ks), 1), dtype=_np.float64),
        'rad': _np.zeros((len(Ks), 1), dtype=_np.float64),
        'per': _np.zeros((len(Ks), 1), dtype=_np.float64)
    }
    for i, K in enumerate(Ks):
        print('K = %d' % K)
        
        J = 2*K
        X = _get_X(K)
        Y = _np.sort(_np.random.uniform(0, _twopi, J))
        F = _testseries.semicircle(X, K).real
        
        timer_gt = _util.Timer(lambda: _get_ground_truth_interp(X, F, Y, K))
        timer_rad = _util.Timer(lambda: _groundtruth.inufft_radial_approximation(F, K, Y, L, p, n))
        timer_per = _util.Timer(lambda: _nufft.inufft(F, K, Y, L, p, n, q))

        times['gt'][i] = _np.log10(timer_gt.median)
        times['rad'][i] = _np.log10(timer_rad.median)
        times['per'][i] = _np.log10(timer_per.median)

    _plt.figure()
    _plt.plot(Ks, times['gt'], label='Ground Truth')
    _plt.plot(Ks, times['rad'], label='Radial Approx.')
    _plt.plot(Ks, times['per'], label='Periodic Sum')
    _plt.ylabel('Log10 Execution Time (sec.)')
    _plt.xlabel('Bandlimit (determines problem size)')
    _plt.legend()
    if save:
        _plt.savefig(_inspect.stack()[0][3] + '.pdf')
    else:
        _plt.show()

