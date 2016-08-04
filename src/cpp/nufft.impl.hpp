#ifndef __NUFFT_IMPL_HPP__
#define __NUFFT_IMPL_HPP__

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "cauchy.hpp"
#include "fmm1d.hpp"
#include "traits.hpp"

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
	static_assert(!is_complex<domain_t> {});

	// Some preliminaries:

	auto const F = values;
	// auto const Y = nodes; // TODO: replace this with a scaled vector version
	auto const J = num_nodes;
	domain_t const K = num_values/2;
	int_t const L = fmm_depth;
	auto const N = num_values;
	auto const p = truncation_number;
	auto const n = neighborhood_radius;

	range_t const twopi {6.283185307179586};
	int_t const num_cells {2*n + 1};
	auto const num_cells_recip = 1.0/static_cast<domain_t>(num_cells);
	auto const scale_factor = num_cells_recip*(1.0/twopi);

	// Initialize X_per, the periodic extension of X to the periodic
	// summation neighborhood (X_per is computed with the correct
	// scale immediately to prevent floating point
	// errors---specifically, catastrophic cancellation):

	int_t const X_per_size {num_cells*N};
	std::vector<domain_t> X_per(X_per_size);
	for (int_t i {0}; i < X_per_size; ++i) {
		X_per[i] = static_cast<domain_t>(i)/static_cast<domain_t>(X_per_size);
	}

	// Compute F with alternating signs, and extended periodically to
	// X_per:

	std::vector<std::complex<range_t>> Fas_per(X_per_size);
	for (int_t i {0}; i < X_per_size; ++i) {
		auto const f = F[i % N];
		Fas_per[i].real(f.real()*(2*static_cast<range_t>(i % 2 == 0) - 1));
		Fas_per[i].imag(f.imag());
	}

	// Sum up the values of a single period of Fas_per for later use:

	std::complex<range_t> Fas_sum {0, 0};
	for (int_t i {0}; i < N; ++i) {
		Fas_sum += Fas_per[i];
	}

	// Scale input nodes for use with FMM.

	std::vector<domain_t> Y(J);
	std::transform(
		nodes,
		nodes + J,
		std::begin(Y),
		[&] (domain_t const & y) {
			return (y + twopi*n)*scale_factor;
		});
#ifdef NUFFT_DEBUG
	assert(std::is_sorted(std::cbegin(Y), std::cend(Y)));
#endif

	// Create array of checkpoints and their periodic offsets inside
	// the scaled target domain.

	auto const num_cps = N;

	std::vector<domain_t> Yc(2*num_cps);
	for (int_t i = 0; i < 2*num_cps; ++i) {
		Yc[i] = num_cells_recip*(n + (i + 0.5)/num_cps);
	}
#ifdef NUFFT_DEBUG
	assert(std::is_sorted(std::cbegin(Y), std::cend(Y)));
#endif

	// Run the FMM on the nodes and the checkpoints separately. In
	// the future, we may be able to tune each FMM independently.

	auto V = nufft::fmm1d<
		nufft::cauchy<domain_t, std::complex<range_t>, int_t>,
		domain_t,
		std::complex<range_t>,
		int_t>::fmm(X_per, Y, Fas_per, L, p);
	for (int_t i {0}; i < J; ++i) {
		V[i] *= scale_factor;
	}

	auto Vc = nufft::fmm1d<
		nufft::cauchy<domain_t, std::complex<range_t>, int_t>,
		domain_t,
		std::complex<range_t>,
		int_t>::fmm(X_per, Yc, Fas_per, L, p);
	for (int_t i {0}; i < 2*num_cps; ++i) {
		Vc[i] *= scale_factor;
	}

	// TODO: compute phifar.

	// TODO: fix output.

	// TODO: finish interpolation.

	range_t const N_recip {1.0/static_cast<range_t>(N)};
	for (int_t i {0}; i < J; ++i) {
		auto const theta = K*nodes[i];
		output[i] = N_recip*(-Fas_sum*std::cos(theta) + 2*std::sin(theta)*V[i]);
	}

	return nufft_error::success;
}

#endif // __NUFFT_IMPL_HPP__
