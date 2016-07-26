from cffi import FFI

SHARED_LIB_PATH = '/Volumes/Molly/Dropbox/Research/nufft/Build/' + \
                  'nufftall-1.33/nufft1df90.so'

if __name__ == '__main__':
    ffi = FFI()
    ffi.set_source('_nufft_greengard', None)
    ffi.cdef('''
typedef struct { double real, imag; } complex;
void _nufft1d2f90_(
    int nj,       // number of output values
    double * xj,  // location of output values
    complex * cj, // output values
    int iflag,    // sign of FFT
    double * eps, // precision request
    int ms,       // number of Fourier modes given
    complex * fk, // Fourier coefficient values
    int ier       // error return code (0: succes, 1: eps out of range)
    );
''')
    ffi.dlopen(SHARED_LIB_PATH)
    ffi.compile(verbose=True)
