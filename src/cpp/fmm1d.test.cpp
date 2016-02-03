#define BOOST_TEST_MODULE fmm1d

#include <boost/test/included/unit_test.hpp>

#include "cauchy.hpp"
#include "fmm1d.hpp"

BOOST_AUTO_TEST_CASE (get_multipole_coefs_works) {
	using namespace nufft;
	
	vector_type<domain_elt_type> const sources = {
		0.29485291464409813,
		0.5666788017655526,
		0.5654834502137871,
		0.2657706439349883,
		0.49204494085537664,
		0.9084122210901464,
		0.41148775741412535,
		0.1644475290278926,
		0.6513973042753114,
		0.27746230058138965
	};
	vector_type<range_elt_type> const weights = {
		1.0445860347168028,
		0.5869693012415074,
		-0.3081203031469292,
		0.5737051132055659,
		1.6566494091220096,
		-1.2482587142631936,
		-0.25163013862111666,
		0.766953789456392,
		-1.1087134957611795,
		-0.522790996978465
	};
	domain_elt_type const x_star = 0.01411425398533006;
	integer_type const p = 8;
	vector_type<range_elt_type> const expected = {
		1.1893499989713934,
		-0.561401797039591,
		-0.9246407244775339,
		-0.9419912324932764,
		-0.8682254042631719,
		-0.7754211982470006,
		-0.6855300683488881,
		-0.6047525248078208
	};
	auto const actual = fmm1d<cauchy>::get_multipole_coefs(
		sources, weights, x_star, p);
	
	BOOST_CHECK_EQUAL_COLLECTIONS(
		std::cbegin(actual),
		std::cend(actual),
		std::cbegin(expected),
		std::cend(expected));
}

BOOST_AUTO_TEST_CASE (evaluate_regular_works) {
	using namespace nufft;

	vector_type<domain_elt_type> targets = {
		0.9482536724346731,
		0.7499818943527383,
		0.8298628321249875,
		0.8494297688865196,
		0.26006551164815495,
		0.9766967094351526,
		0.8485999713559158,
		0.35385393749732574,
		0.8889124953814553,
		0.494916793867477
	};
	vector_type<range_elt_type> coefs = {
		-0.44172701759315597,
		0.8759135544966249,
		1.2966258801174102,
		-1.1023501528620987,
		0.19028118911661535,
		0.12346783959953532,
		0.2758108098673406,
		-1.5661982045056975,
		0.05118043173330569,
		0.7551316717373412
	};
	domain_elt_type const x_star = 0.5327985806084505;
	integer_type const p = 10;
	vector_type<range_elt_type> expected = {
		0.07252219300692292,
		-0.20114859676536195,
		-0.09434507604911568,
		-0.06727178008279644,
		-0.5606560532840934,
		0.11305285489875792,
		-0.06842567959605964,
		-0.5504404492095917,
		-0.01187614807116009,
		-0.4729871827285726
	};
	auto const actual = fmm1d<cauchy>::evaluate_regular(
		targets, coefs, x_star, p);
	BOOST_CHECK_EQUAL_COLLECTIONS(
		std::cbegin(actual),
		std::cend(actual),
		std::cbegin(expected),
		std::cend(expected));
}
