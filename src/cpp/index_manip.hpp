#ifndef __NUFFT_INDEX_MANIP_HPP__
#define __NUFFT_INDEX_MANIP_HPP__

#include "types.hpp"

namespace nufft {
	using index_pair_type = std::pair<index_type, index_type>;

	index_type get_parent(index_type index);
	
	index_pair_type get_children(index_type index);
	
	index_type get_sibling(index_type index);
	
	vector_type<index_type> get_E2_neighbors(size_type level, index_type index);
	
	vector_type<index_type> get_E4_neighbors(size_type level, index_type index);
	
	index_type get_box_index(domain_elt_type elt, size_type level);
	
	domain_elt_type get_box_center(size_type level, index_type index);
	
	domain_elt_type get_box_size(size_type level);
}

#endif // __NUFFT_INDEX_MANIP_HPP__
