#include "fmm.hpp"

#include "cauchy.hpp"
#include "fmm1d.hpp"

void fmm1d_cauchy_double(
	double * const output,
	double const * const sources,
	uintmax_t num_sources,
	double const * const targets,
	uintmax_t num_targets,
	double const * const weights,
	uintmax_t num_weights,
	uintmax_t max_level,
	intmax_t p)
{
    using fmm1d_t = nufft::fmm1d<nufft::cauchy<>>;

	std::vector<double> const X(sources, sources + num_sources);
	std::vector<double> const Y(targets, targets + num_targets);
	std::vector<double> const U(weights, weights + num_weights);
	auto const L = max_level;
	auto const output_vector = fmm1d_t::fmm(X, Y, U, L, p);
	std::copy(std::cbegin(output_vector), std::cend(output_vector), output);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
