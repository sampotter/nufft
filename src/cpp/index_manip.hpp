#ifndef __NUFFT_INDEX_MANIP_HPP__
#define __NUFFT_INDEX_MANIP_HPP__

#include <utility>
#include <vector>

namespace nufft {
    template <class domain_t = double,
              class range_t = double,
              class int_t = int64_t>
    struct index_manip {
        template <class T> using vector_t = std::vector<T>;
        using index_pair_type = std::pair<int_t, int_t>;

        static int_t get_parent(int_t index);
    
        static index_pair_type get_children(int_t index);
    
        static int_t get_sibling(int_t index);
    
        static vector_t<int_t> get_E2_neighbors(int_t level, int_t index);
    
        static vector_t<int_t> get_E4_neighbors(int_t level, int_t index);
    
        static int_t get_box_index(domain_t elt, int_t level);
    
        static domain_t get_box_center(int_t level, int_t index);
    
        static domain_t get_box_size(int_t level);
    };
}

#include "index_manip.impl.hpp"

#endif // __NUFFT_INDEX_MANIP_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
