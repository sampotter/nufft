#ifndef __NUFFT_INDEX_MANIP_HPP__
#define __NUFFT_INDEX_MANIP_HPP__

namespace nufft {
    template <class int_t = int64_t>
    struct index_pair {
        index_pair(int_t first, int_t second);
        int_t first;
        int_t second;
    };

    template <class domain_t = double,
              class range_t = double,
              class int_t = int64_t>
    struct index_manip {
        static int_t get_parent(int_t index);
    
        static index_pair<int_t> get_children(int_t index);
    
        static int_t get_sibling(int_t index);
    
        static void get_E4_neighbors(int_t index, int_t * neighbors);
    
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
