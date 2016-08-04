import numpy as np
import testfunc

if __name__ == '__main__':
    funcs = [
        testfunc.SemicircleTestSeries,
        testfunc.TriangleTestSeries,
        testfunc.SquareTestSeries,
        testfunc.SawtoothTestSeries]
    func_names = ['semi', 'tri', 'square', 'saw']

    Ks = np.arange(4, 11)
    N = 129

    for i, func in enumerate(funcs):
        pts = np.linspace(0, 2*np.pi, N)
        output = np.zeros((len(pts), len(Ks) + 1))
        output[:, 0] = pts
        for j, K in enumerate(Ks):
            output[:, j + 1] = func()(pts, K)
        np.savetxt('data_testfunc_%s.dat' % func_names[i], output,
                   header='x ' + ' '.join(map(lambda K: 'K' + str(K), Ks)),
                   comments='')
