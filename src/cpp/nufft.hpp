#ifndef __NUFFT_HPP__
#define __NUFFT_HPP__

#include <complex>

namespace nufft {
	enum class nufft_error: int {
		success
	};

	template <class domain_t, class range_t, class int_t>
	nufft_error compute_P(
		std::complex<range_t> const * const values,
		domain_t const * const nodes,
		int_t const num_values,
		int_t const num_nodes,
		int_t const fmm_depth,
		int_t const truncation_number,
		int_t const neighborhood_radius,
		std::complex<range_t> * const output);
}

#include "nufft.impl.hpp"

extern "C" {
	int compute_P_ddi(
		double const * const values,
		double const * const nodes,
		int const num_values,
		int const num_nodes,
		int const fmm_depth,
		int const truncation_number,
		int const neighborhood_radius,
		double * const output);
}

#endif // __NUFFT_HPP__
