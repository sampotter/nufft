#include "bookmarks.hpp"
#include "util.hpp"

nufft::bookmarks::bookmarks(vector_type<domain_elt_type> const & sources,
                            size_type max_level):
    bookmark_hash_(make_bookmark_hash(sources, max_level))
{}

nufft::bookmarks::bookmark_hash_type
nufft::bookmarks::make_bookmark_hash(
    vector_type<domain_elt_type> const & sources,
    size_type max_level) const
{
#ifdef NUFFT_DEBUG
    assert(std::size(sources) <= std::numeric_limits<index_type>::max());
#endif
    index_type const num_sources = std::size(sources);
    auto const num_boxes = std::pow(2, max_level);
    auto const bounds = util::linspace(0.0, 1.0, num_boxes + 1);
    vector_type<boost::optional<bookmark_type>> bookmarks = get_empty_bookmarks(max_level);

    index_type box_index {0};
    index_type scan_index {-1};
    index_type prev_scan_index {scan_index};

    while (box_index < num_boxes && scan_index < num_sources) {
        auto const right_bound = bounds[box_index + 1];
        
        while (scan_index + 1 < num_sources &&
               sources[scan_index + 1] < right_bound) {
            ++scan_index;
        }

        if (prev_scan_index != scan_index) {
            bookmarks[box_index] = std::make_pair(prev_scan_index + 1,
                                                  scan_index);
        }

        prev_scan_index = scan_index;
        ++box_index;
    }

    auto const get_bookmark_at_level = [&] (size_type level, index_type index) {
#ifdef NUFFT_DEBUG
        assert(level >= 0);
        assert(level <= max_level);
        assert(index >= 0);
        {
            index_type const max_index =
                static_cast<index_type>(std::pow(2, level)); 
            assert(index < max_index);;
        }
#endif
        
        auto const level_diff = max_level - level;
        auto const scale = std::pow(2, level_diff);
        auto const left_extent = scale * index;
#ifdef NUFFT_DEBUG
        assert(scale * (index + 1) > 0);
#endif
        auto const right_extent = scale * (index + 1) - 1;

        auto left_index = left_extent;
        auto right_index = right_extent;

        while (left_index <= right_extent && !bookmarks[left_index]) {
            ++left_index;
        }

        while (right_index >= left_extent && !bookmarks[right_index]) {
            --right_index;
        }
        
        boost::optional<bookmark_type> opt_bookmark;

        auto const past_left = right_index < left_extent;
        auto const past_right = left_index > right_extent;
#ifdef NUFFT_DEBUG
        assert(past_left && past_right || !(past_left || past_right));
#endif

        if (!(past_left || past_right)) {
            auto const first = bookmarks[left_index]->first;
            auto const second = bookmarks[right_index]->second;
            opt_bookmark = std::make_pair(first, second);
        }

#ifdef NUFFT_DEBUG
        if (opt_bookmark) {
            auto const first = opt_bookmark->first;
            auto const second = opt_bookmark->second;
            assert(0 <= first);
            assert(first <= second);
            assert(second < num_sources);
        }
#endif
        return opt_bookmark;
    };

    bookmark_hash_type bookmark_hash;
    
    for (decltype(max_level) level {0}; level <= max_level; ++level) {
        bookmark_hash[level] = get_empty_bookmarks(level);
        auto const max_index = static_cast<index_type>(std::pow(2, level));
        for (index_type index {0}; index < max_index; ++index) {
            auto opt_bookmark = get_bookmark_at_level(level, index);
            if (opt_bookmark) {
                bookmark_hash[level][index] = opt_bookmark;
            }
        }
    }

    return bookmark_hash;
}

nufft::bookmarks::vector_type<boost::optional<nufft::bookmarks::bookmark_type>>
nufft::bookmarks::get_empty_bookmarks(size_type level) const
{
    auto const size = std::pow(2, level);
    return vector_type<boost::optional<bookmark_type>>(size, boost::none);
}

boost::optional<nufft::bookmarks::bookmark_type>
nufft::bookmarks::operator()(size_type level, size_type index) const
{
    return bookmark_hash_.at(level)[index];
}

// Local Variables:
// indent-tabs-mode: nil
// End:
