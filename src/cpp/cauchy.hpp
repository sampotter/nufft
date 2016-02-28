#ifndef __NUFFT_CAUCHY_HPP__
#define __NUFFT_CAUCHY_HPP__

#include "types.hpp"

namespace nufft {
    struct cauchy {
        static inline range_elt_type phi(domain_elt_type y, domain_elt_type x) {
            return 1.0/(y - x);
        }
                
        static range_elt_type R(integer_type m, domain_elt_type x);
        static range_elt_type S(integer_type m, domain_elt_type x);
        static range_elt_type a(integer_type m, domain_elt_type x);
        static range_elt_type b(integer_type m, domain_elt_type x);

        static void
        apply_SS_translation(vector_type<range_elt_type> const & input,
                             vector_type<range_elt_type> & output,
                             domain_elt_type delta,
                             integer_type p);

        static void
        apply_SR_translation(vector_type<range_elt_type> const & input,
                             vector_type<range_elt_type> & output,
                             domain_elt_type delta,
                             integer_type p);

        static void
        apply_RR_translation(vector_type<range_elt_type> const & input,
                             vector_type<range_elt_type> & output,
                             domain_elt_type delta,
                             integer_type p);
    private:
        cauchy();
    };
}

#endif // __NUFFT_CAUCHY_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
