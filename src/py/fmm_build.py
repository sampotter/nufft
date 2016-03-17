import cffi

ffi = cffi.FFI()
ffi.set_source("_fmm", None)
ffi.cdef('''
void fmm1d_cauchy_double(
    double * const output,
    double const * const sources,
    uintmax_t num_sources,
    double const * const targets,
    uintmax_t num_targets,
    double const * const weights,
    uintmax_t num_weights,
    uintmax_t max_level,
    intmax_t p);
''')

if __name__ == '__main__':
    ffi.compile()
