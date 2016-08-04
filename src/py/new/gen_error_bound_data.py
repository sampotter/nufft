import argparse
import numpy as np

from scipy.special import expi as Ei

def A(n, p):
    num1 = 2*(p + n + 1) - 1
    denom1 = 2*p*np.power(2*(n + 1) - 1, p + 1)
    num2 = 2*(p + n + 1) + 1
    denom2 = 2*p*np.power(2*(n - 1) + 1, p + 1)
    return num1/denom1 + num2/denom2

def B(n, p):
    num1 = np.power(2*(n + 1) - 1, -p - 1)
    denom1 = 2*np.log(2*(n + 1) - 1)
    num2 = np.power(2*(n + 1) + 1, -p - 1)
    denom2 = 2*np.log(2*(n + 1) + 1)
    return num1/denom1 + num2/denom2

def C(n, p):
    term1 = Ei(-p*np.log(2*(n + 1) - 1))
    term2 = Ei(-p*np.log(2*(n + 1) + 1))
    return -0.25*(term1 + term2)

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
                errors[i, j] = A(n, p) + B(n, p) + C(n, p)
                f.write('%d %d %g\n' % (n, p, errors[i, j]))
            f.write('\n')

    with open('data_error_bound_thresholds.dat', 'w') as f:
        for i, n in enumerate(N):
            for j, p in enumerate(P):
                if errors[i, j] < 1e-6:
                    f.write('%d %d %g\n' % (n, p, errors[i, j]))
                    break
