#ifndef __NUFFT_FMM1D_HPP__
#define __NUFFT_FMM1D_HPP__

#include "types.hpp"

namespace nufft {
    template <typename kernel_type>
    struct fmm1d {
        static vector_type<domain_elt_type>
        get_multipole_coefs(vector_type<domain_elt_type> const & sources,
                            vector_type<domain_elt_type> const & weights,
                            domain_elt_type x_star,
                            integer_type p);

        static vector_type<range_elt_type>
        evaluate_regular(vector_type<domain_elt_type> const & targets,
                         vector_type<range_elt_type> const & coefs,
                         domain_elt_type x_star,
                         integer_type p);
    };
}

#include "fmm1d.impl.hpp"

#endif // __NUFFT_FMM1D_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
