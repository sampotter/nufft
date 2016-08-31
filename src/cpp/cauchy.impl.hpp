#ifndef __NUFFT_CAUCHY_IMPL_HPP__
#define __NUFFT_CAUCHY_IMPL_HPP__

#include <cassert>

#include "math.hpp"

template <class domain_t, class range_t, class int_t>
range_t
nufft::cauchy<domain_t, range_t, int_t>::phi(
    domain_t y,
    domain_t const * sources,
    range_t const * weights,
    vector_t<int_t> const & indices)
{
    range_t tmp {0};
    for (auto const i: indices) {
        add(tmp, mul(weights[i], domain_t {1}/(y - sources[i])));
    }
    return tmp;
}

template <class domain_t, class range_t, class int_t>
range_t
nufft::cauchy<domain_t, range_t, int_t>::R(
    int_t p,
    domain_t x,
    range_t const * coefs)
{
    range_t tmp {coefs[p - 1]};
    for (int_t j = p - 2; j >= 0; --j) {
        tmp *= x;
        tmp += coefs[j];
    }
    return tmp;
}

template <class domain_t, class range_t, class int_t>
domain_t
nufft::cauchy<domain_t, range_t, int_t>::b(int_t m, domain_t x)
{
    return pow(x, m);
}

template <class domain_t, class range_t, class int_t>
void
nufft::cauchy<domain_t, range_t, int_t>::apply_SS_translation(
    range_t const * const input,
    vector_t<range_t> & output,
    domain_t delta,
    int_t p)
{
    // TODO: note: when I finally try to combine the coefficients and
    // deltas into a single array, I could use an inline struct to
    // make things a bit simpler to understand.
    
    static vector_t<domain_t> deltas(p);
    deltas[0] = 1;
    for (int_t i {1}; i < p; ++i) {
        deltas[i] = 0;
    }

    static vector_t<domain_t> coefs(p);
    coefs[0] = 1;
    for (int_t i {1}; i < p; ++i) {
        coefs[i] = 0;
    }

    int_t update_deltas_iter {1};
    int_t update_coefs_iter {1};
    for (int_t i {0}; i < p; ++i) {
        output[i] = 0;
        for (int_t j {0}; j <= i; ++j) {
            output[i] += coefs[j] * deltas[j] * input[j];
        }

        if (update_deltas_iter >= p) {
            return;
        }
        deltas[update_deltas_iter] = 1;
        for (int_t i {update_deltas_iter - 1}; i >= 0; --i) {
            deltas[i] *= -delta;
        }
        ++update_deltas_iter;

        if (update_coefs_iter >= p) {
            return;
        }
        for (int_t i {update_coefs_iter}; i > 0; --i) {
            coefs[i] += coefs[i - 1];
        }
        ++update_coefs_iter;
    }
}

template <class domain_t, class range_t, class int_t>
void
nufft::cauchy<domain_t, range_t, int_t>::apply_SR_translation(
    range_t const * const input,
    vector_t<range_t> & output,
    domain_t delta,
    int_t p)
{
#ifdef NUFFT_DEBUG
	assert(p > 0);
#endif

    domain_t const delta_recip = 1.0/delta;
    
    static vector_t<domain_t> deltas(p);
    for (int_t i {0}; i < p; ++i) {
        deltas[i] = delta_recip;
        if (i > 0) {
            deltas[i] *= deltas[i - 1];
        }
    }

	static_assert(std::is_signed<decltype(p)>::value);
	static decltype(p) current_p {-1};
    static vector_t<domain_t> coefs;
	if (p != current_p) {
		coefs.resize(p*p);
		coefs.shrink_to_fit();
		current_p = p;
		for (int_t i {0}; i < p; ++i) {
			coefs[i] = 1;
		}
		for (int_t i {1}; i < p; ++i) {
			for (int_t j {0}; j < p; ++j) {
				coefs[p*i + j] = coefs[p*(i - 1) + j];
			}
			for (int_t j {1}; j < p; ++j) {
				coefs[p*i + j] += coefs[p*i + j - 1];
			}
		}
	}

    for (int_t i {0}; i < p; ++i) {
        for (int_t j {0}; j < p; ++j) {
            output[i] += coefs[p*i + j] * deltas[j] * input[j];
        }
        for (auto & delta: deltas) {
            delta *= -delta_recip;
        }
    }
}

template <class domain_t, class range_t, class int_t>
void
nufft::cauchy<domain_t, range_t, int_t>::apply_RR_translation(
	vector_t<range_t> const & input,
	vector_t<range_t> & output,
	domain_t delta,
	int_t p)
{
	static vector_t<domain_t> coefs(p);
    for (int_t i {0}; i < p; ++i) {
        coefs[i] = 1;
    }

    int_t end {p - 1};
	for (int_t i {0}; i < p; ++i) {
		output[i] = 0;
		domain_t this_delta {1};
		int_t k {0};
		for (int_t j {i}; j < p; ++j) {
			output[i] += this_delta * coefs[k] * input[j];
			this_delta *= delta;
			++k;
		}
		for (int_t i {1}; i < end; ++i) {
			coefs[i] += coefs[i - 1];
		}
		--end;
	}
}

#endif // __NUFFT_CAUCHY_IMPL_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
