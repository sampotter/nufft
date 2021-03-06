#ifndef __NUFFT_UTIL_HPP__
#define __NUFFT_UTIL_HPP__

#include <cinttypes>
#include <unordered_map>
#include <vector>

namespace nufft { namespace util {
    template <typename elt_type>
    std::vector<elt_type> linspace(elt_type min,
                                   elt_type max,
                                   int64_t num_elts);

    template <typename elt_type>
    std::vector<elt_type> zeros(int64_t num_elts);
} }

#include "util.impl.hpp"

#endif // __NUFFT_UTIL_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
