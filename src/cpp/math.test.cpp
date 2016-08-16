#define BOOST_TEST_MODULE math

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <random>

#include "math.hpp"

BOOST_AUTO_TEST_CASE (pow_works) {
	BOOST_CHECK_EQUAL(std::pow(-0.0230986, 0), nufft::pow(-0.0230986, 0));
	BOOST_CHECK_EQUAL(std::pow(2, 2), nufft::pow(2.0, 2));
	BOOST_CHECK_EQUAL(std::pow(2.5, 2), nufft::pow(2.5, 2));
	BOOST_CHECK_EQUAL(std::pow(-2, -2), nufft::pow(-2.0, -2));
	BOOST_CHECK_EQUAL(std::pow(-2.5, -2), nufft::pow(-2.5, -2));
	BOOST_CHECK_EQUAL(std::pow(1e-5, -2), nufft::pow(1e-5, -2));
}
