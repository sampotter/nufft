#ifndef __FMM_HPP__
#define __FMM_HPP__

#include <cstdint>

extern "C" {
	void fmm1d_cauchy_double(
		double * const output,
		double const * const sources,
		uintmax_t num_sources,
		double const * const targets,
		uintmax_t num_targets,
		double const * const weights,
		uintmax_t num_weights,
		uintmax_t max_level,
		intmax_t p);
}

#endif // __FMM_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
