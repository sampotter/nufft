#ifndef __NUFFT_IMPL_HPP__
#define __NUFFT_IMPL_HPP__

#include <algorithm>
#include <cmath>
#include <random>
#include <type_traits>

#include "cauchy.hpp"
#include "fmm1d.hpp"

template <class domain_t, class range_t, class int_t>
nufft::nufft_error
nufft::compute_P(
	std::complex<range_t> const * const values,
	domain_t const * const nodes,
	int_t const num_values,
	int_t const num_nodes,
	int_t const fmm_depth,
	int_t const truncation_number,
	int_t const neighborhood_radius,
	std::complex<range_t> * const output)
{
	// Some preliminaries:

	auto const F = values;
	auto const Y = nodes;
	auto const J = num_nodes;
	auto const K = num_values/2; // used with least-squares collocation
	auto const L = fmm_depth;
	auto const N = num_values;
	auto const p = truncation_number;
	auto const n = neighborhood_radius;

	range_t const twopi {6.283185307179586};

	// Initialize X, the uniform time domain grid:

	std::vector<domain_t> X(N);
	auto const twopi_over_N = twopi/N;
	for (int_t i {0}; i < N; ++i) {
		X[i] = i*twopi_over_N;
	}

	// Initialize X_per, the periodic extension of X to the periodic
	// summation neighborhood (X_per is computed with the correct
	// scale immediately to prevent floating point
	// errors---specifically, catastrophic cancellation):

	int_t const X_per_size {(2*n + 1)*N};
	std::vector<domain_t> X_per(X_per_size);
	for (int_t i {0}; i < X_per_size; ++i) {
		X_per[i] = static_cast<domain_t>(i)/static_cast<domain_t>(X_per_size);
	}

	// Compute F with alternating signs, and extended periodically to
	// X_per:

	std::vector<std::complex<range_t>> Fas_per(X_per_size);
	for (int_t i {0}; i < X_per_size; ++i) {
		auto const sign = std::complex<range_t>(i % 2 == 0 ? 1 : -1, 0);
		Fas_per[i] = F[i % N]*sign;
	}

	// Sum up the values of a single period of Fas_per for later use:

	std::complex<range_t> Fas_sum {0, 0};
	for (int_t i {0}; i < N; ++i) {
		Fas_sum += Fas_per[i];
	}

	// Compute N checkpoints and their periodic offsets in the
	// interval [0, 2pi):

	std::vector<domain_t> Yc(N);
	std::generate(std::begin(Yc), std::end(Yc), [twopi] () {
			static std::random_device dev {};
			static std::mt19937 gen {dev()};
			static std::uniform_real_distribution<domain_t> dist {twopi, 2*twopi};
			return dist(gen);
		});
	
	std::vector<domain_t> Yc_tilde(N);
	for (int_t i {0}; i < N; ++i) {
		Yc_tilde[i] = Yc[i] - twopi;
	}

	// Concatenate together Y, Yc, and Yc_tilde to prepare for use
	// with the FMM (we also keep the initial index of our
	// concatenated vector since we will sort it and need to be able
	// to unsort the output of the FMM later):

	std::vector<std::pair<domain_t, int_t>> Ycat_with_indices(J + 2*N);
	int_t i {0};
	for (; i < J; ++i) {
		Ycat_with_indices[i] = {Y[i], i};
	}
	for (int_t j {0}; j < N; ++i, ++j) {
		Ycat_with_indices[i] = {Yc[j], i};
	}
	for (int_t j {0}; j < N; ++i, ++j) {
		Ycat_with_indices[i] = {Yc_tilde[j], i};
	}

	// Sort Ycat by the actual values it contains (and not the
	// indices):

	std::sort(
		std::begin(Ycat_with_indices),
		std::end(Ycat_with_indices),
		[] (auto const & a, auto const & b) {
			return a.first < b.first;
		});

	// Extract Ycat and Ycat_indices (TODO: this is awful):

	std::vector<domain_t> Ycat(J + 2*N);
	std::transform(
		std::cbegin(Ycat_with_indices),
		std::cend(Ycat_with_indices),
		std::begin(Ycat),
		[] (auto const & pair) { return pair.first; });

	// Evaluate phinear using the FMM:

	auto const left = -twopi*n;
	auto const right = twopi*(n + 1);
	auto const scale_factor = 1/(right - left);
#ifdef NUFFT_DEBUG
	assert(*std::min_element(std::cbegin(Ycat), std::cend(Ycat)) >= left);
	assert(*std::max_element(std::cbegin(Ycat), std::cend(Ycat)) < right);
#endif
	for (int_t i {0}; i < J + 2*N; ++i) {
		Ycat[i] = scale_factor*(Ycat[i] - left);
	}
	// for (int_t i {0}; i < X_per_size; ++i) {
	// 	X_per[i] = scale_factor*(X_per[i] - left);
	// }
#ifdef NUFFT_DEBUG
	assert(*std::min_element(std::cbegin(Ycat), std::cend(Ycat)) >= 0);
	assert(*std::max_element(std::cbegin(Ycat), std::cend(Ycat)) < 1.0);
	assert(*std::min_element(std::cbegin(X_per), std::cend(X_per)) >= 0);
	assert(*std::max_element(std::cbegin(X_per), std::cend(X_per)) < 1.0);
#endif
	auto fmm_output = nufft::fmm1d<
		nufft::cauchy<domain_t, std::complex<range_t>, int_t>,
		domain_t,
		std::complex<range_t>,
		int_t>::fmm(X_per, Ycat, Fas_per, L, p);
	for (int_t i {0}; i < J + 2*N; ++i) {
		fmm_output[i] *= scale_factor;
	}

	// Unsort FMM output to get output associated with Y, Yc, and Yc_tilde:

	std::vector<std::complex<range_t>> V(J);
	std::vector<std::complex<range_t>> Vc(N);
	std::vector<std::complex<range_t>> Vc_tilde(N);
	for (int_t i {0}; i < J + 2*N; ++i) {
		auto const j = Ycat_with_indices[i].second;
		if (0 <= j && j < J) {
			V[j] = fmm_output[i];
		} else if (J <= j && j < J + N) {
			Vc[j - J] = fmm_output[i];
		} else if (J + N <= j && j < J + 2*N) {
			Vc_tilde[j - J - N] = fmm_output[i];
		}
	}

	// TODO: compute phifar.

	// TODO: fix output.

	// TODO: finish interpolation.

	for (int_t i {0}; i < J; ++i) {
		auto const theta = K*nodes[i];
		output[i] = (1.0/N)*(-Fas_sum*std::cos(theta) + 2*std::sin(theta)*V[i]);
	}

	return nufft_error::success;
}

#endif // __NUFFT_IMPL_HPP__
