#define BOOST_TEST_MODULE bookmarks

#include <boost/test/included/unit_test.hpp>
#include <cinttypes>

#define private public
#include "bookmarks.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmacro-redefined"
#define private private
#pragma clang diagnostic pop

BOOST_AUTO_TEST_CASE (empty_bookmarks_are_correct) {
	using nufft::bookmarks;

	bookmarks bookmarks {{}, 3};
	bookmarks::size_type const level = 3;
	auto const empty_bookmarks = bookmarks.get_empty_bookmarks(level);

	BOOST_TEST(
		std::all_of(
			std::cbegin(empty_bookmarks),
			std::cend(empty_bookmarks),
			[] (boost::optional<bookmarks::bookmark_type> const index) {
				return index == boost::none;
			}));
}

BOOST_AUTO_TEST_CASE (bookmarks_all_full) {
	using nufft::bookmarks;

	bookmarks::size_type const max_level {2};
	bookmarks bookmarks {{0.125, 0.375, 0.625, 0.875}, max_level};

	for (bookmarks::size_type level {0}; level <= max_level; ++level) {
		auto const level_size = std::pow(2, level);
		for (bookmarks::index_type index {0}; index < level_size; ++index) {
			auto const opt_bookmark = bookmarks(level, index);
			BOOST_TEST(bool {opt_bookmark});
		}
	}
}

BOOST_AUTO_TEST_CASE (bookmarks_edges) {
	using nufft::bookmarks;
	bookmarks::size_type const max_level {2};
	auto const epsilon = std::numeric_limits<decltype(0.0)>::epsilon();
	bookmarks bookmarks {{0.0, 1.0 - epsilon}, max_level};

	for (bookmarks::size_type level {0}; level <= max_level; ++level) {
		auto const level_size = std::pow(2, level);
		BOOST_TEST(bool {bookmarks(level, 0)});
		for (bookmarks::index_type index {1}; index < level_size - 1; ++index) {
			BOOST_TEST(!bool {bookmarks(level, index)});
		}
		BOOST_TEST(bool {bookmarks(level, level_size - 1)});
	}
}
