#define BOOST_TEST_MODULE index_manip

#include <boost/test/included/unit_test.hpp>

#include "index_manip.hpp"

BOOST_AUTO_TEST_CASE (get_parent_works) {
	using nufft::get_parent;
	
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
	using nufft::get_children;
	using nufft::index_type;

	auto const test_children = [] (index_type index,
								   index_type first,
								   index_type second) {
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
	using nufft::get_sibling;

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
	using nufft::get_E2_neighbors;
	using nufft::index_type;
	using nufft::size_type;
	using nufft::vector_type;

	auto const compare = [] (size_type level,
							 index_type index,
							 vector_type<index_type> other) {
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

BOOST_AUTO_TEST_CASE (get_E4_neighbors) {
	using nufft::get_E4_neighbors;
	using nufft::index_type;
	using nufft::size_type;
	using nufft::vector_type;

	auto const compare = [] (size_type level,
							 index_type index,
							 vector_type<index_type> other) {
		auto const N = get_E4_neighbors(level, index);
		BOOST_CHECK_EQUAL_COLLECTIONS(std::cbegin(N), std::cend(N),
									  std::cbegin(other), std::cend(other));
	};

	compare(2, 0, {2, 3});
	compare(2, 1, {3});
	compare(2, 2, {0});
	compare(2, 3, {0, 1});
	compare(3, 0, {2, 3});
	compare(3, 1, {3});
	compare(3, 2, {0, 4, 5});
	compare(3, 3, {0, 1, 5});
	compare(3, 4, {2, 6, 7});
	compare(3, 5, {2, 3, 7});
	compare(3, 6, {4});
	compare(3, 7, {4, 5});
}

BOOST_AUTO_TEST_CASE (get_box_index_works) {
	using nufft::domain_elt_type;
	using nufft::get_box_index;
	using nufft::index_type;
	using nufft::size_type;

	auto const compare = [] (domain_elt_type elt, size_type level,
							 index_type index) {
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
	using nufft::domain_elt_type;
	using nufft::get_box_center;
	using nufft::index_type;
	using nufft::size_type;

	auto const compare = [] (size_type level, index_type index,
							 domain_elt_type box_center) {
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
	using nufft::domain_elt_type;
	using nufft::get_box_size;
	using nufft::size_type;

	auto const compare = [] (size_type level, domain_elt_type box_size) {
		BOOST_CHECK_EQUAL(get_box_size(level), box_size);
	};

	compare(0, 1.0);
	compare(1, 0.5);
	compare(2, 0.25);
	compare(3, 0.125);
	compare(4, 0.0625);
}
