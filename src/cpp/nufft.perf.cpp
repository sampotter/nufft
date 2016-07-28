#include "nufft.hpp"

#include <algorithm>
#include <complex>
#include <random>

auto const NUM_VALUES = 1024;
auto const NUM_NODES = 1024;
auto const NUM_TRIALS = 10;
auto const FMM_DEPTH = 4;
auto const TRUNC_NUM = 4;
auto const NEIGHB_RAD = 3;

int main() {
	std::complex<double> values[NUM_VALUES];
	double nodes[NUM_NODES];
	std::complex<double> output[NUM_NODES];

	auto const randomize = [&] () {
		static std::random_device dev {};
		static std::mt19937 gen {dev()};
		static std::uniform_real_distribution<double> dist {0, 1};
		std::generate(std::begin(values),
					  std::begin(values) + NUM_VALUES,
					  [&] () { return std::complex<double>(dist(gen), dist(gen)); });
		std::generate(std::begin(nodes),
					  std::begin(nodes) + NUM_NODES,
					  [&] () { return dist(gen); });
		std::sort(std::begin(nodes), std::begin(nodes) + NUM_NODES);
	};

	for (int i = 0; i < NUM_TRIALS; ++i) {
		randomize();
		nufft::compute_P(values, nodes, NUM_VALUES, NUM_NODES, FMM_DEPTH,
						 TRUNC_NUM, NEIGHB_RAD, output);
	}
}
