import matlab.engine as _engine
import numpy as _np
import os as _os
import os.path as _path
import tempfile as _tempfile

from matlab import double as _double

default_nufft_path = '/Volumes/Molly/Dropbox/Research/nufft/' + \
                     'thirdparty/nufftall-1.33'


class nufft_greengard(object):
    def __init__(self, engine=None, nufft_path=None):
        self._engine = engine if engine else _engine.start_matlab()
        nufft_path = nufft_path if nufft_path else default_nufft_path
        self._engine.path(nufft_path, self._engine.path())

    def inufft(self, interp_pts, coefs, eps=1e-15):
        nj = float(len(interp_pts))
        xj = _double(list(map(float, interp_pts)))
        iflag = float(1)
        ms = float(len(coefs))
        fk = _double(list(map(complex, coefs)), is_complex=True)
        fd, filename = _tempfile.mkstemp(suffix='.m',
                                         dir=_path.curdir, text=True)
        funcname = _path.splitext(_path.basename(filename))[0]
        funcstr = '''
function out = %s(nj, xj, iflag, tol, ms, fk)
    tic();
    out.Y = nufft1d2(nj, xj, iflag, tol, ms, fk)/ms;
    out.t = toc();
end
''' % funcname
        _os.write(fd, bytes(funcstr, 'UTF-8'))
        _os.close(fd)
        try:
            out = getattr(self._engine, funcname)(nj, xj, iflag, eps, ms, fk)
        except:
            _os.remove(filename)
            return
        else:
            _os.remove(filename)
            return _np.array(out['Y']), out['t']
