import matlab.engine
import numpy as np
import os.path
import tempfile

default_irt_path = '/Volumes/Molly/Dropbox/research/nufft/build/irt'

# TODO: add unit tests to make sure our wrapper doesn't diverge!
# import unittest


class irt(object):
    def __init__(self, engine=None, irt_path=None):
        self._engine = engine if engine else matlab.engine.start_matlab()
        irt_path = irt_path if irt_path else default_irt_path
        self._engine.path(irt_path, self._engine.path())
        self._engine.workspace['irtdir'] = irt_path
        self._engine.setup(nargout=0)

    def nufft(self, om, X, N, J, K):
        om = matlab.double(list(map(float, om)))
        X = matlab.double(list(map(complex, X)), is_complex=True)
        Nd = matlab.double([N])
        Jd = matlab.double([J])
        Kd = matlab.double([K])
        fd, filename = tempfile.mkstemp(suffix='.m',
                                        dir=os.path.curdir, text=True)
        funcname = os.path.splitext(os.path.basename(filename))[0]
        funcstr = '''function out = %s(X, om, Nd, Jd, Kd)
    tic();
    st = nufft_init(om, Nd, Jd, Kd);
    out.Y = nufft(X, st);
    out.t = toc();
end
''' % funcname
        os.write(fd, bytes(funcstr, 'UTF-8'))
        os.close(fd)
        try:
            out = getattr(self._engine, funcname)(
                self._engine.transpose(X),
                self._engine.transpose(om),
                self._engine.transpose(Nd),
                self._engine.transpose(Jd),
                self._engine.transpose(Kd))
        except:
            os.remove(filename)
            return
        else:
            os.remove(filename)
            return np.array(out['Y']), out['t']

    def inufft(self, interp_pts, coefs, N, J, K):
        '''This method uses the nufft method to compute the inufft. To get the
        correct output, this just involves flipping the sign of the
        frequency coefficients for the IDFT and scaling the output by
        the right factor (i.e. the DFT size).'''
        Y, t = self.nufft(interp_pts, -coefs, N, J, K)
        return Y/coefs.size, t
