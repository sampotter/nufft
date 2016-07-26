import cffi


def read_file(path):
    with open(path, 'r') as f:
        return ''.join(f.readlines())

if __name__ == '__main__':
    ffi = cffi.FFI()
    ffi.set_source("_nfft", None)
    ffi.cdef('''
int nfft_timer(int N, int M, double * f_hat_real, double * f_hat_imag,
               double * x, double * f_real, double * f_imag, double * time);
''')
    ffi.compile()
