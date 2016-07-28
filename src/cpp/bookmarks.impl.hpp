#include "bookmarks.hpp"
#include "util.hpp"

template <class domain_t, class int_t>
nufft::bookmarks<domain_t, int_t>::bookmarks(vector_t<domain_t> const & sources,
                                             int_t max_level):
    bookmark_hash_(make_bookmark_hash(sources, max_level))
{}

template <class domain_t, class int_t>
typename nufft::bookmarks<domain_t, int_t>::bookmark_hash_t
nufft::bookmarks<domain_t, int_t>::make_bookmark_hash(
    vector_t<domain_t> const & sources,
    int_t max_level) const
{
#ifdef NUFFT_DEBUG
    assert(std::size(sources) <= std::numeric_limits<int_t>::max());
#endif
    int_t const num_sources = std::size(sources);
    auto const num_boxes = std::pow(2, max_level);
    auto const bounds = util::linspace(0.0, 1.0, num_boxes + 1);
    vector_t<opt_t<bookmark_t>> bookmarks = get_empty_bookmarks(max_level);

    int_t box_index {0};
    int_t scan_index {-1};
    int_t prev_scan_index {scan_index};

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

    auto const get_bookmark_at_level = [&] (int_t level, int_t index) {
#ifdef NUFFT_DEBUG
        assert(level >= 0);
        assert(level <= max_level);
        assert(index >= 0);
        {
            int_t const max_index =
                static_cast<int_t>(std::pow(2, level)); 
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
        
        opt_t<bookmark_t> opt_bookmark;

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

    bookmark_hash_t bookmark_hash;
    
    for (decltype(max_level) level {0}; level <= max_level; ++level) {
        bookmark_hash[level] = get_empty_bookmarks(level);
        auto const max_index = static_cast<int_t>(std::pow(2, level));
        for (int_t index {0}; index < max_index; ++index) {
            auto opt_bookmark = get_bookmark_at_level(level, index);
            if (opt_bookmark) {
                bookmark_hash[level][index] = opt_bookmark;
            }
        }
    }

    return bookmark_hash;
}

template <class domain_t, class int_t>
nufft::bookmarks<domain_t, int_t>::vector_t<
    typename nufft::bookmarks<domain_t, int_t>::template opt_t<
        typename nufft::bookmarks<domain_t, int_t>::bookmark_t>>
nufft::bookmarks<domain_t, int_t>::get_empty_bookmarks(int_t level) const
{
    auto const size = std::pow(2, level);
    return vector_t<opt_t<bookmark_t>>(size, boost::none);
}

template <class domain_t, class int_t>
nufft::bookmarks<domain_t, int_t>::template opt_t<
    typename nufft::bookmarks<domain_t, int_t>::bookmark_t>
nufft::bookmarks<domain_t, int_t>::operator()(int_t level, int_t index) const
{
    return bookmark_hash_.at(level)[index];
}

// Local Variables:
// indent-tabs-mode: nil
// End:
