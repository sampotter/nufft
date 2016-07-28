#define BOOST_TEST_MODULE bookmarks

#include <boost/test/included/unit_test.hpp>
#include <cinttypes>

#define private public
#include "bookmarks.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmacro-redefined"
#define private private
#pragma clang diagnostic pop

using bookmarks_t = nufft::bookmarks<double, int>;
using int_t = int;

BOOST_AUTO_TEST_CASE (empty_bookmarks_are_correct) {
    bookmarks_t bookmarks {{}, 3};
    int const level = 3;
    auto const empty_bookmarks = bookmarks.get_empty_bookmarks(level);

    BOOST_TEST(
        std::all_of(
            std::cbegin(empty_bookmarks),
            std::cend(empty_bookmarks),
            [] (boost::optional<bookmarks_t::bookmark_t> const index) {
                return index == boost::none;
            }));
}

BOOST_AUTO_TEST_CASE (bookmarks_all_full) {
    int_t const max_level {2};
    bookmarks_t bookmarks {{0.125, 0.375, 0.625, 0.875}, max_level};

    for (int_t level {0}; level <= max_level; ++level) {
        auto const level_size = std::pow(2, level);
        for (int_t index {0}; index < level_size; ++index) {
            auto const opt_bookmark = bookmarks(level, index);
            BOOST_TEST(bool {opt_bookmark});
        }
    }
}

BOOST_AUTO_TEST_CASE (bookmarks_edges) {
    int_t const max_level {2};
    auto const epsilon = std::numeric_limits<decltype(0.0)>::epsilon();
    bookmarks_t bookmarks {{0.0, 1.0 - epsilon}, max_level};

    for (int_t level {0}; level <= max_level; ++level) {
        auto const level_size = std::pow(2, level);
        BOOST_TEST(bool {bookmarks(level, 0)});
        for (int_t index {1}; index < level_size - 1; ++index) {
            BOOST_TEST(!bool {bookmarks(level, index)});
        }
        BOOST_TEST(bool {bookmarks(level, level_size - 1)});
    }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
