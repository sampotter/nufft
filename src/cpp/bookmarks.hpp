#ifndef __NUFFT_BOOKMARKS_HPP__
#define __NUFFT_BOOKMARKS_HPP__

#include <boost/optional.hpp>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nufft {
    struct bookmarks {
        using domain_elt_type = double;
        
        template <typename T>
        using vector_type = std::vector<T>;
        
        using size_type = std::size_t;
        using index_type = int64_t;
        using bookmark_type = std::pair<index_type, index_type>;

        bookmarks(
            vector_type<domain_elt_type> const & sources,
            size_type max_level);
        
        boost::optional<bookmark_type>
        operator()(size_type level, size_type index) const;
        
    private:
        template <typename K, typename V>
        using hash_type = std::unordered_map<K, V>;
        
        using bookmark_hash_type = hash_type<
            size_type,
            vector_type<boost::optional<bookmark_type>>>;
        
        bookmark_hash_type make_bookmark_hash(
            vector_type<domain_elt_type> const & sources,
            size_type max_level) const;
        
        vector_type<boost::optional<bookmark_type>>
        get_empty_bookmarks(size_type level) const;
        
        bookmark_hash_type bookmark_hash_;
    };
}

#endif // __NUFFT_BOOKMARKS_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
