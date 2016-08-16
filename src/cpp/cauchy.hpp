#ifndef __NUFFT_CAUCHY_HPP__
#define __NUFFT_CAUCHY_HPP__

#include <cinttypes>
#include <vector>

namespace nufft {
    template <class domain_t = double,
              class range_t = double,
              class int_t = int64_t>
    struct cauchy {
        template <class T> using vector_t = std::vector<T>;
        using index_t = std::size_t;

        static inline range_t phi(domain_t y, domain_t x) {
            return 1.0/(y - x);
        }

        static range_t phi(domain_t y,
                           domain_t const * sources,
                           range_t const * weights,
                           vector_t<int_t> const & indices);
                
        static range_t R(int_t p, domain_t x, range_t const * coefs);

        static domain_t b(int_t m, domain_t x);

        static void
        apply_SS_translation(vector_t<range_t> const & input,
                             vector_t<range_t> & output,
                             domain_t delta,
                             int_t p);

        static void
        apply_SR_translation(vector_t<range_t> const & input,
                             vector_t<range_t> & output,
                             domain_t delta,
                             int_t p);

        static void
        apply_RR_translation(vector_t<range_t> const & input,
                             vector_t<range_t> & output,
                             domain_t delta,
                             int_t p);
    private:
        cauchy();
    };
}

#include "cauchy.impl.hpp"

#endif // __NUFFT_CAUCHY_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
