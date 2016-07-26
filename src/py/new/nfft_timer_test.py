import nfft
import numpy as np
import testfunc

square = testfunc.SquareTestSeries()
coefs = square.get_coefs(16)
interp_pts = np.linspace(0, 2*np.pi, 24, endpoint=False)
output, time = nfft.nfft_timer(coefs, interp_pts)
print(output)
print(time)
