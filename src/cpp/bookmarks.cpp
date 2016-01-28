#include "bookmarks.hpp"
#include "util.hpp"

int64_t const NO_BOOKMARK = -1;

nufft::bookmarks::bookmarks(vector_type const & sources, size_type max_level):
	bookmarks_(make_bookmarks(sources, max_level))
{}

nufft::bookmarks::bookmark_map_type
nufft::bookmarks::make_bookmark_map(vector_type const & sources,
									size_type max_level) const
{
	auto const num_sources {std::size(sources)};
	auto const num_boxes {std::pow(2, max_level)};
	auto const bounds = util::linspace(0, 1, num_boxes);
	auto const bookmarks = get_empty_bookmarks(max_level);

	int64_t box_index {0};
	int64_t scan_index {-1};
	int64_t prev_scan_index {scan_index};

	while (box_index < num_boxes && scan_index < num_sources) {
		auto const right_bound = bounds[box_index + 1];
		
		while (scan_index + 1 < num_sources &&
			   sources[scan_index + 1] < right_bound) {
			++scan_index;
		}

		if (prev_scan_index != scan_index) {
			bookmarks[2 * box_index - 1] = prev_scan_index + 1;
			bookmarks[2 * box_index] = scan_index;
		}

		prev_scan_index = scan_index;
		++box_index;
	}

	auto const get_bookmark_at_level = [] (int64_t level, int64_t index) {
#if DEBUG
		assert(0 <= level <= max_level);
		assert(0 <= index < std::pow(2, level));
#endif
		
		auto const level_diff {max_level - level};
		auto const left_extent {level_diff * (index - 1) + 1};
		auto const right_extent {level_diff * index};
		auto left_index {left_extent};
		auto right_index {right_extent};

		bookmark_type bookmark;
		{
			auto first = bookmarks[2 * left_index - 1];
			while (left_index++ < right_extent && first == 0) {
				first = bookmarks[2 * left_index - 1];
			}
			bookmark.first = left_index > right_extent ? NO_BOOKMARK : first;
		}
		{
			auto second = bookmarks[2 * right_index];
			while (right_index-- >= left_extent && second == 0) {
				second = bookmarks[2 * right_index];
			}
			bookmark.second = right_index < left_index ? NO_BOOKMARK : second;
		}

#if DEBUG
		assert((bookmark.first == NO_BOOKMARK &&
				bookmark.second == NO_BOOKMARK) ||
			   (bookmark.first != NO_BOOKMARK &&
				bookmark.second != NO_BOOKMARK));
		assert(0 <= bookmark.first && bookmark.second < num_sources);
#endif

		return bookmark;
	}

	bookmark_map_type bookmark_map;
	
	for (decltype(max_level) level {0}; level <= max_level; ++level) {
		bookmark_map[level] = get_empty_bookmarks(level);
		index_type max_index {std::pow(2, level)};
		for (index_type index {0}; index < max_index; ++index) {
			auto const bookmark = get_bookmark_at_level(level, index);
			bookmark_map[level][2 * index_type] = bookmark.first;
			bookmark_map[level][2 * index_type + 1] = bookmark.second;
		}
	}

	return bookmark_map;
}

nufft::bookmarks::vector_type
nufft::bookmarks::get_empty_bookmarks(size_type max_level) const
{
	auto const size = std::pow(2, max_level + 1);
	vector_type bookmarks(size);
	bookmarks.reserve(size);
	std::fill(std::begin(bookmarks), std::end(bookmarks), NO_BOOKMARK);
	return bookmarks;
}

nufft::bookmarks::bookmark_type
nufft::bookmarks::operator()(size_type level, size_type index) const
{
	return bookmark_map_[level][index];
}
