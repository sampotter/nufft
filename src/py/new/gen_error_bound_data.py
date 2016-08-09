import argparse
import numpy as np

from scipy.special import expi as Ei

def error(n, p):
    tmp = Ei(-p*np.log(2*n + 3)) + Ei(-p*np.log(2*n + 1))
    tmp /= -2*np.pi
    return tmp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''compute a table of data for the periodic summation error
        bounds''')
    parser.add_argument('min_p', type=int, help='minimum truncation number')
    parser.add_argument('max_p', type=int, help='maximum truncation number')
    parser.add_argument('min_n', type=int, help='minimum neighborhood radius')
    parser.add_argument('max_n', type=int, help='maximum neighborhood radius')
    args = parser.parse_args()
    if args.min_p < 1:
        raise Exception('min_p must be positive')
    if args.max_p < 1:
        raise Exception('max_p must be positive')
    if args.min_n < 1:
        raise Exception('min_n must be positive')
    if args.max_n < 1:
        raise Exception('max_n must be positive')
    args = parser.parse_args()
    P = np.arange(args.min_p, args.max_p + 1)
    N = np.arange(args.min_n, args.max_n + 1)

    errors = np.zeros((len(N), len(P)))

    with open('data_error_bound.dat', 'w') as f:
        for i, n in enumerate(N):
            for j, p in enumerate(P):
                errors[i, j] = error(n, p)
                f.write('%d %d %g\n' % (n, p, errors[i, j]))
            f.write('\n')

    with open('data_error_bound_thresholds1.dat', 'w') as f:
        for i, n in enumerate(N):
            for j, p in enumerate(P):
                if errors[i, j] < 1e-7:
                    f.write('%d %d %g\n' % (n, p, errors[i, j]))
                    break

    with open('data_error_bound_thresholds2.dat', 'w') as f:
        for i, n in enumerate(N):
            for j, p in enumerate(P):
                if errors[i, j] < 1e-15:
                    f.write('%d %d %g\n' % (n, p, errors[i, j]))
                    break
