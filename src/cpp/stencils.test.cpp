#define BOOST_TEST_MODULE stencils

#include <boost/test/included/unit_test.hpp>

#include "stencils.hpp"

using namespace nufft;

struct SS_stencil_fixture {
	int const L {4};
	SS_stencil<int> stencil {L};
};

BOOST_FIXTURE_TEST_CASE (SS_set_works, SS_stencil_fixture) {
	stencil.set(3, 0, 1);
	stencil.set(4, 0, 1);
	BOOST_CHECK(stencil.mask_.test(7));
	BOOST_CHECK(stencil.mask_.test(15));
}

BOOST_FIXTURE_TEST_CASE (SS_test_works, SS_stencil_fixture) {
	stencil.mask_.set(7);
	stencil.mask_.set(15);
	stencil.test(3, 0);
	stencil.test(4, 0);
}

BOOST_FIXTURE_TEST_CASE (SS_get_level_index_works, SS_stencil_fixture) {
	BOOST_CHECK_EQUAL(stencil.get_level_index(3), 7);
	BOOST_CHECK_EQUAL(stencil.get_level_index(4), 15);
}

BOOST_FIXTURE_TEST_CASE (SS_get_num_entries_works, SS_stencil_fixture) {
	BOOST_CHECK_EQUAL(stencil.get_num_entries(), 31);
}

struct RR_stencil_fixture {
	int const L {4};
	RR_stencil<int> stencil {L};
};

BOOST_FIXTURE_TEST_CASE (RR_set_works, RR_stencil_fixture) {
	stencil.set(3, 0, 1);
	stencil.set(4, 0, 1);
	BOOST_CHECK(stencil.mask_.test(7));
	BOOST_CHECK(stencil.mask_.test(15));
}

BOOST_FIXTURE_TEST_CASE (RR_test_works, RR_stencil_fixture) {
	stencil.mask_.set(7);
	stencil.mask_.set(15);
	stencil.test(3, 0);
	stencil.test(4, 0);
}

BOOST_FIXTURE_TEST_CASE (RR_get_level_index_works, RR_stencil_fixture) {
	BOOST_CHECK_EQUAL(stencil.get_level_index(3), 7);
	BOOST_CHECK_EQUAL(stencil.get_level_index(4), 15);
}

BOOST_FIXTURE_TEST_CASE (RR_get_num_entries_works, RR_stencil_fixture) {
	BOOST_CHECK_EQUAL(stencil.get_num_entries(), 31);
}

struct SR_stencil_fixture {
	int const L {4};
	SR_stencil<int> stencil {L};
};

BOOST_FIXTURE_TEST_CASE (SR_set_works, SR_stencil_fixture) {
	stencil.set(2, 0, 0, 1);
	stencil.set(2, 1, 1, 1);
	stencil.set(3, 0, 2, 1);
	stencil.set(4, 0, 1, 1);
	BOOST_CHECK(stencil.mask_.test(9));
	BOOST_CHECK(stencil.mask_.test(13));
	BOOST_CHECK(stencil.mask_.test(23));
	BOOST_CHECK(stencil.mask_.test(46));
}

BOOST_FIXTURE_TEST_CASE (SR_test_works, SR_stencil_fixture) {
	stencil.mask_.set(9);
	stencil.mask_.set(13);
	stencil.mask_.set(23);
	stencil.mask_.set(46);
	stencil.test(2, 0, 0);
	stencil.test(2, 1, 1);
	stencil.test(3, 0, 2);
	stencil.test(4, 0, 1);
}

BOOST_FIXTURE_TEST_CASE (SR_get_level_index_works, SR_stencil_fixture) {
	BOOST_CHECK_EQUAL(stencil.get_level_index(2), 9);
	BOOST_CHECK_EQUAL(stencil.get_level_index(3), 21);
	BOOST_CHECK_EQUAL(stencil.get_level_index(4), 45);
}

BOOST_FIXTURE_TEST_CASE (SR_get_num_entries_works, SR_stencil_fixture) {
	BOOST_CHECK_EQUAL(stencil.get_num_entries(), 93);
}
