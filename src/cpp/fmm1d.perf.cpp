#include <algorithm>
#include <iostream>
#include <random>

#include "cauchy.hpp"
#include "fmm1d.hpp"

constexpr int NUM_TRIALS = 10;

using namespace nufft;

vector_type<domain_elt_type> random_domain_elts(size_type N) {
	static std::random_device device;
	static std::default_random_engine engine(device());
	static std::uniform_real_distribution<domain_elt_type> uniform(0, 1);
	vector_type<domain_elt_type> elts(N, 0);
	std::generate(
		std::begin(elts), std::end(elts),
		[] () {
			return uniform(engine);
		});
	std::sort(std::begin(elts), std::end(elts));
	return elts;
}

vector_type<range_elt_type> random_range_elts(size_type N) {
	static std::random_device device;
	static std::default_random_engine engine(device());
	static std::normal_distribution<> normal(0, 1);
	vector_type<range_elt_type> elts(N, 0);
	std::generate(
		std::begin(elts), std::end(elts),
		[] () {
			return normal(engine);
		});
	return elts;
}

int main() {
	size_type N = 10000;
	auto const sources = random_domain_elts(N);
	auto const targets = random_domain_elts(N);
	auto const weights = random_range_elts(N);
	size_type L = 10;
	integer_type p = 10;

	for (int trial = 0; trial < NUM_TRIALS; ++trial) {
		std::cout << trial << std::endl;
		fmm1d<cauchy>::fmm(sources, targets, weights, L, p);
	}
}
