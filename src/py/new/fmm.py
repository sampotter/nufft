'''This is a quickly put together test module. The goal here is to
expose the optimized C++ FMM to Python (in particular, numpy) for
rapidly testing the rest of the NUFFT algorithm. On OS X, libfmm.dylib
will need to be built and copied to this directory before importing
this module.

'''

import ctypes.util
import numpy as np

from _fmm import ffi as _ffi

_lib = _ffi.dlopen(ctypes.util.find_library('fmm'))


def _issorted(lst):
    it = iter(lst)
    next(it)
    return all(b >= a for a, b in zip(lst, it))


def _normalize(lst, L, R):
    return (lst - L)/(R - L)


def fmm1d_cauchy_double(sources, targets, weights, max_level, p,
                        scaled_domain=None):
    '''Nota bene: this is super inefficient!'''

    num_sources = len(sources)
    num_targets = len(targets)
    num_weights = len(weights)

    assert(len(sources) == len(weights))
    assert(_issorted(sources))
    assert(_issorted(targets))

    scale_factor = 1
    if scaled_domain:
        assert(isinstance(scaled_domain, tuple))
        assert(len(scaled_domain) == 2)
        L = scaled_domain[0]
        R = scaled_domain[1]
        assert(L < R)
        assert(min(sources) >= L)
        assert(max(sources) < R)
        assert(min(targets) >= L)
        assert(max(targets) < R)
        sources = _normalize(sources, L, R)
        targets = _normalize(targets, L, R)
        scale_factor = 1/(R - L)

    assert(min(sources) >= 0.0)
    assert(max(sources) < 1.0)
    assert(min(targets) >= 0.0)
    assert(max(targets) < 1.0)

    output = _ffi.new('double[%d]' % num_targets)
    sources = _ffi.new('double[%d]' % num_sources, list(sources))
    weights = _ffi.new('double[%d]' % num_weights, list(weights))
    targets = _ffi.new('double[%d]' % num_targets, list(targets))

    _lib.fmm1d_cauchy_double(output, sources, num_sources, targets,
                             num_targets, weights, num_weights,
                             max_level, p)

    return scale_factor*np.array(list(output))
