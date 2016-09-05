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
	auto const J = num_nodes;
	auto const K = 0.5*domain_t(num_values);
	auto const L = fmm_depth;
	auto const N = num_values;
	auto const p = truncation_number;
	auto const n = neighborhood_radius;

	auto const twopi = range_t(6.283185307179586);
	auto const num_cells = 2*n + 1;
	auto const num_cells_recip = 1.0/domain_t(num_cells);
	auto const scale_factor = num_cells_recip*(1.0/twopi);

	// Initialize X_per, the periodic extension of X to the periodic
	// summation neighborhood:

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

	// Estimate zeroth "fitting coefficient".

	std::vector<std::complex<range_t>> C(p, 0);
	{
		domain_t const pi {3.141592653589793};
		std::complex<range_t> c0 {0, 0};
		for (int_t i {0}; i < N; ++i) {
			std::complex<range_t> tmp {0, 0};
			domain_t const x {twopi*i/domain_t(N)};
			tmp += std::atanh((x - pi)/(twopi*n));
			tmp += std::atanh((x - pi)/(twopi*(n + 1)));
			tmp *= Fas_per[i];
			c0 += tmp;
		}
		C[0] /= -twopi;
	}

	// TODO: compute first through (m - 1)th "fitting coefficients"
	// approximately.

	// This lambda evaluates the phifar polynomial using the
	// precomputed coefficients C[m].
	//
	// TODO: Modify this to use Horner's method, if that ends up being
	// the fastest way to do this sort of thing.
	//
	// TODO: it may be good to just templatize a polynomial evaluation
	// algorithm so that we can select the best one at compile time
	// further on.

	auto const phifar = [&] (domain_t const y) {
		std::complex<range_t> tmp {0};
		for (int_t m = 0; m < p; ++m) {
			tmp += C[m]*std::pow(y, m);
		}
		return tmp;
	};

	// Finally, we apply the formula that finishes and computes the
	// actual interpolated values.

	for (int_t i {0}; i < J; ++i) {
		auto const node = nodes[i];
		output[i] = std::sin(K*node)*(phifar(node) + V[i])/K;
	}

	return nufft_error::success;
}

#endif // __NUFFT_IMPL_HPP__
