#define BOOST_TEST_MODULE util

#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <iostream>

#include "util.hpp"

BOOST_AUTO_TEST_CASE (linspace_works_with_positive_delta) {
    std::vector<double> const gt {0.0, 0.5, 1.0, 1.5, 2.0};
    decltype(gt) const elts = nufft::util::linspace<double>(0.0, 2.0, 5);
    BOOST_TEST(std::equal(std::cbegin(gt), std::cend(gt), std::cbegin(elts)));
}

BOOST_AUTO_TEST_CASE (linspace_works_with_negative_delta) {
    std::vector<double> const gt {1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0};
    decltype(gt) const elts = nufft::util::linspace<double>(1.0, -2.0, 7);
    BOOST_TEST(std::equal(std::cbegin(gt), std::cend(gt), std::cbegin(elts)));
}

BOOST_AUTO_TEST_CASE (linspace_kahan_summation_works) {
	auto const elts = nufft::util::linspace<double>(0.0, 1.0, 11);
	BOOST_TEST(elts[10] == 1.0);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
