import numpy as _np
from time import perf_counter as _perf_counter

def get_X(K):
    return _np.linspace(0, 2*_np.pi, 2*K, endpoint=False)

class Timer(object):

    def __init__(self, func, trials=100):
        self._trials = trials
        self._func = func
        self._has_been_timed = False

    @property
    def times(self):
        if not self._has_been_timed:
            self.time()
        return self._times

    def time(self):
        def trial():
            start = _perf_counter()
            self._func()
            return _perf_counter() - start
        self._times = [trial() for _ in range(self._trials)]
        self._has_been_timed = True

    @property
    def median(self):
        return _np.median(self.times)
