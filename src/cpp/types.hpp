#ifndef __NUFFT_TYPES_HPP__
#define __NUFFT_TYPES_HPP__

#include <cinttypes>
#include <cstddef>
#include <utility>
#include <vector>

namespace nufft {
    using domain_elt_type = double;
    using range_elt_type = double;
    using index_type = int64_t;
    using integer_type = int64_t;
    using size_type = std::size_t;
    using bookmark_type = std::pair<index_type, index_type>;
    template <typename T> using vector_type = std::vector<T>;
}

#endif // __NUFFT_TYPES_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
