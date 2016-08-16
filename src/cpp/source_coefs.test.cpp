#define BOOST_TEST_MODULE source_coefs

#include <boost/test/included/unit_test.hpp>

#include "source_coefs.hpp"

using namespace nufft;

struct source_coefs_fixture {
	int max_level {4};
	int p {3};
	source_coefs<double, int> source_coefs {max_level, p};
};

BOOST_FIXTURE_TEST_CASE (set_works, source_coefs_fixture) {
	source_coefs.set(2, 0);
	BOOST_CHECK(source_coefs.mask_.test(0));
	for (int i {1}; i < source_coefs.get_num_coefs(); ++i) {
		BOOST_CHECK(!source_coefs.mask_.test(i));
	}
}

BOOST_FIXTURE_TEST_CASE (test_works, source_coefs_fixture) {
	source_coefs.mask_.set(0);
	source_coefs.mask_.set(4);
	source_coefs.mask_.set(6);
	BOOST_CHECK(source_coefs.test(2, 0));
	BOOST_CHECK(source_coefs.test(3, 0));
	BOOST_CHECK(source_coefs.test(3, 2));
}

BOOST_FIXTURE_TEST_CASE (get_coefs_works, source_coefs_fixture) {
	for (int m {0}; m < p; ++m) {
		source_coefs.coefs_[m] = p - m;
	}
	auto coefs = source_coefs.get_coefs(2, 0);
	BOOST_CHECK_EQUAL_COLLECTIONS(
		source_coefs.coefs_,
		source_coefs.coefs_ + p,
		coefs,
		coefs + p);
	for (int m {0}; m < p; ++m) {
		source_coefs.coefs_[5*p + m] = p - m;
	}
	coefs = source_coefs.get_coefs(3, 1);
	BOOST_CHECK_EQUAL_COLLECTIONS(
		source_coefs.coefs_ + 5*p,
		source_coefs.coefs_ + 5*p + p,
		coefs,
		coefs + p);
}

BOOST_FIXTURE_TEST_CASE (get_level_works, source_coefs_fixture) {
	for (int m {0}; m < p; ++m) {
		source_coefs.coefs_[(1 << 2)*p + m] = p - m;
	}
	for (int m {0}; m < p; ++m) {
		source_coefs.coefs_[((1 << 2) + (1 << 3) - 1)*p + m] = p - m;
	}
	std::vector<double> expected = {
		3, 2, 1,
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		3, 2, 1,
	};
	auto const level = source_coefs.get_level(3);
	BOOST_CHECK_EQUAL_COLLECTIONS(
		level,
		level + (1 << 3)*p,
		expected.begin(),
		expected.end());
}

BOOST_FIXTURE_TEST_CASE (get_level_index_works, source_coefs_fixture) {
	BOOST_CHECK_EQUAL(source_coefs.get_level_index(2), 0);
	BOOST_CHECK_EQUAL(source_coefs.get_level_index(3), 4);
	BOOST_CHECK_EQUAL(source_coefs.get_level_index(4), 12);
}

BOOST_FIXTURE_TEST_CASE (get_num_coefs, source_coefs_fixture) {
	BOOST_CHECK_EQUAL(source_coefs.get_num_coefs(), 28);
}
