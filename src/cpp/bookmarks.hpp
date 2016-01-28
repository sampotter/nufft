#ifndef __NUFFT_BOOKMARKS_HPP__
#define __NUFFT_BOOKMARKS_HPP__

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nufft {
	struct bookmarks {
		using domain_elt_type = double;
		using vector_type = std::vector<domain_elt_type>;
		using size_type = std::size_t;
		using index_type = int64_t;
		using bookmark_type = std::pair<index_type, index_type>;
		using bookmark_map_type =
			std::unordered_map<size_type, std::vector<index_type>>;

		bookmarks()(vector_type const & sources, size_type max_level);
		bookmark_type operator()(size_type level, size_type index) const;
	private:
		vector_type make_bookmark_map(vector_type const & sources,
									  size_type max_level) const;
		vector_type get_empty_bookmarks(size_type max_level) const;
		
		bookmark_map_type bookmark_map_;
	};
}

#endif // __NUFFT_BOOKMARKS_HPP__
