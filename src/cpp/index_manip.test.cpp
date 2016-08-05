#define BOOST_TEST_MODULE index_manip

#include <boost/test/included/unit_test.hpp>
#include <cinttypes>
#include <vector>

#include "index_manip.hpp"

using int_t = int64_t;
template <class T> using vector_t = std::vector<T>;
using index_manip_t = nufft::index_manip<>;
auto constexpr get_parent = index_manip_t::get_parent;
auto constexpr get_children = index_manip_t::get_children;
auto constexpr get_sibling = index_manip_t::get_sibling;
auto constexpr get_E2_neighbors = index_manip_t::get_E2_neighbors;
auto constexpr get_E4_neighbors = index_manip_t::get_E4_neighbors;
auto constexpr get_box_index = index_manip_t::get_box_index;
auto constexpr get_box_center = index_manip_t::get_box_center;
auto constexpr get_box_size = index_manip_t::get_box_size;

BOOST_AUTO_TEST_CASE (get_parent_works) {
    BOOST_TEST(get_parent(0) == 0);
    BOOST_TEST(get_parent(1) == 0);
    BOOST_TEST(get_parent(2) == 1);
    BOOST_TEST(get_parent(3) == 1);
    BOOST_TEST(get_parent(4) == 2);
    BOOST_TEST(get_parent(5) == 2);
    BOOST_TEST(get_parent(6) == 3);
    BOOST_TEST(get_parent(7) == 3);
    BOOST_TEST(get_parent(8) == 4);
    BOOST_TEST(get_parent(9) == 4);
    BOOST_TEST(get_parent(10) == 5);
    BOOST_TEST(get_parent(11) == 5);
    BOOST_TEST(get_parent(12) == 6);
    BOOST_TEST(get_parent(13) == 6);
}

BOOST_AUTO_TEST_CASE (get_children_works) {
    auto const test_children = [] (int_t index,
                                   int_t first,
                                   int_t second) {
        auto const children = get_children(index);
        BOOST_TEST(children.first == first);
        BOOST_TEST(children.second == second);
    };

    test_children(0, 0, 1);
    test_children(1, 2, 3);
    test_children(2, 4, 5);
    test_children(3, 6, 7);
    test_children(4, 8, 9);
    test_children(5, 10, 11);
    test_children(6, 12, 13);
}

BOOST_AUTO_TEST_CASE (get_sibling_works) {
    BOOST_TEST(get_sibling(0) == 1);
    BOOST_TEST(get_sibling(1) == 0);
    BOOST_TEST(get_sibling(2) == 3);
    BOOST_TEST(get_sibling(3) == 2);
    BOOST_TEST(get_sibling(4) == 5);
    BOOST_TEST(get_sibling(5) == 4);
    BOOST_TEST(get_sibling(6) == 7);
    BOOST_TEST(get_sibling(7) == 6);
}

BOOST_AUTO_TEST_CASE (get_E2_neighbors_works) {
    auto const compare = [] (int_t level,
                             int_t index,
                             vector_t<int_t> other) {
        auto const N = get_E2_neighbors(level, index);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::cbegin(N), std::cend(N),
                                      std::cbegin(other), std::cend(other));
    };

    compare(0, 0, {0});
    compare(1, 0, {0, 1});
    compare(1, 1, {0, 1});
    compare(2, 0, {0, 1});
    compare(2, 1, {0, 1, 2});
    compare(2, 2, {1, 2, 3});
    compare(2, 3, {2, 3});
    compare(3, 0, {0, 1});
    compare(3, 1, {0, 1, 2});
    compare(3, 2, {1, 2, 3});
    compare(3, 3, {2, 3, 4});
    compare(3, 4, {3, 4, 5});
    compare(3, 5, {4, 5, 6});
    compare(3, 6, {5, 6, 7});
    compare(3, 7, {6, 7});
}

BOOST_AUTO_TEST_CASE (get_E4_neighbors_works) {
    auto const compare = [] (int_t index, vector_t<int_t> other) {
        int_t N[3];
        get_E4_neighbors(index, N);
        BOOST_CHECK_EQUAL_COLLECTIONS(std::cbegin(N), std::cbegin(N) + 3,
                                      std::cbegin(other), std::cend(other));
    };

    // level = 2
    compare(0, {-2, 2, 3});
    compare(1, {-2, -1, 3});
    compare(2, {0, 4, 5});
    compare(3, {0, 1, 5});

    // level = 3
    compare(0, {-2, 2, 3});
    compare(1, {-2, -1, 3});
    compare(2, {0, 4, 5});
    compare(3, {0, 1, 5});
    compare(4, {2, 6, 7});
    compare(5, {2, 3, 7});
    compare(6, {4, 8, 9});
    compare(7, {4, 5, 9});
}

BOOST_AUTO_TEST_CASE (get_box_index_works) {
    auto const compare = [] (double elt, int_t level, int_t index) {
        BOOST_CHECK_EQUAL(get_box_index(elt, level), index);
    };

    compare(0.0, 0, 0);
    compare(0.5, 0, 0);
    compare(0.0, 1, 0);
    compare(0.25, 1, 0);
    compare(0.5, 1, 1);
    compare(0.75, 1, 1);
}

BOOST_AUTO_TEST_CASE (get_box_center_works) {
    auto const compare = [] (int_t level, int_t index, double box_center) {
        BOOST_CHECK_EQUAL(get_box_center(level, index), box_center);
    };

    compare(0, 0, 0.5);
    compare(1, 0, 0.25);
    compare(1, 1, 0.75);
    compare(2, 0, 0.125);
    compare(2, 1, 0.375);
    compare(2, 2, 0.625);
    compare(2, 3, 0.875);
}

BOOST_AUTO_TEST_CASE (get_box_size_works) {
    auto const compare = [] (int_t level, double box_size) {
        BOOST_CHECK_EQUAL(get_box_size(level), box_size);
    };

    compare(0, 1.0);
    compare(1, 0.5);
    compare(2, 0.25);
    compare(3, 0.125);
    compare(4, 0.0625);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
