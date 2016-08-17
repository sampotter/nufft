#ifndef __NUFFT_BOOKMARKS_HPP__
#define __NUFFT_BOOKMARKS_HPP__

#include <cinttypes>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "preprocessor.hpp"

namespace nufft {
    template <class int_t = int64_t>
    struct bookmark {
        static_assert(std::is_signed<int_t>::value);

        bookmark();
        bookmark(int_t left, int_t right);
        bool empty() const;
        bool valid() const;
        void set(int_t left, int_t right);
        int_t left() const;
        int_t right() const;
    private:
        int_t left_ {-1};
        int_t right_ {-1};
    };

    template <class domain_t = double, class int_t = int64_t>
    struct bookmarks {
        static_assert(std::is_signed<int_t>::value);

        template <class T> using vector_t = std::vector<T>;
        using bookmark_t = bookmark<int_t>;

        bookmarks(vector_t<domain_t> const & sources, int_t max_level);
        bookmark_t operator()(int_t level, int_t index) const;
    private:
        template <class K, class V> using hash_t = std::unordered_map<K, V>;
        using bookmark_hash_t = hash_t<int_t, vector_t<bookmark_t>>;
        
        bookmark_hash_t make_bookmark_hash(vector_t<domain_t> const & sources,
                                           int_t max_level) const;
        vector_t<bookmark_t> get_empty_bookmarks(int_t level) const;
        
        bookmark_hash_t bookmark_hash_;
    };
}

#include "bookmarks.impl.hpp"

#endif // __NUFFT_BOOKMARKS_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
