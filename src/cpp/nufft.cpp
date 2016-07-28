#include "nufft.hpp"

int compute_P_ddi(
	double const * const values,
	double const * const nodes,
	int const num_values,
	int const num_nodes,
	int const fmm_depth,
	int const truncation_number,
	int const neighborhood_radius,
	double * const output)
{
	return static_cast<int>(
		nufft::compute_P<double, double, int>(
			reinterpret_cast<std::complex<double> const * const>(values),
			nodes,
			num_values,
			num_nodes,
			fmm_depth,
			truncation_number,
			neighborhood_radius,
			reinterpret_cast<std::complex<double> * const>(output)));
}
