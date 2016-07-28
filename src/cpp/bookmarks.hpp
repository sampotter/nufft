#ifndef __NUFFT_BOOKMARKS_HPP__
#define __NUFFT_BOOKMARKS_HPP__

#include <boost/optional.hpp>
#include <cinttypes>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nufft {
    template <class domain_t = double, class int_t = int64_t>
    struct bookmarks {
        template <class T> using opt_t = boost::optional<T>;
        template <class T> using vector_t = std::vector<T>;
        using bookmark_t = std::pair<int_t, int_t>;

        bookmarks(vector_t<domain_t> const & sources, int_t max_level);
        
        opt_t<bookmark_t>
        operator()(int_t level, int_t index) const;
        
    private:
        template <class K, class V> using hash_t = std::unordered_map<K, V>;
        using bookmark_hash_t = hash_t<int_t, vector_t<opt_t<bookmark_t>>>;
        
        bookmark_hash_t make_bookmark_hash(vector_t<domain_t> const & sources,
                                           int_t max_level) const;
        
        vector_t<opt_t<bookmark_t>> get_empty_bookmarks(int_t level) const;
        
        bookmark_hash_t bookmark_hash_;
    };
}

#include "bookmarks.impl.hpp"

#endif // __NUFFT_BOOKMARKS_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
