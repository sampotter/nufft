import cffi

if __name__ == '__main__':
    ffi = cffi.FFI()
    ffi.set_source('_nufft', None)
    ffi.cdef('''
int compute_P_ddi(
    double const * const values,
    double const * const nodes,
    int const num_values,
    int const num_nodes,
    int const fmm_depth,
    int const truncation_number,
    int const neighborhood_radius,
    double * const output);
''')
    ffi.compile()
