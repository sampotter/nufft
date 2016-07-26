#ifndef __NFFT_TIMER_HPP__
#define __NFFT_TIMER_HPP__

extern "C" {
	/**
	 * This function is a wrapper around nfft_wrapper which
	 * accomplishes two goals:
	 * 
	 * 1. It's simpler to pass separate arrays of doubles from Python
	 * than it is to spend the extra effort trying to pass a complex
	 * number directly.
	 *
	 * 2. Since C++ has more portable timing facilities (using
	 * std::chrono) that C does, we make a detour through C++ to
	 * actually time the call to nfft_wrapper (which corresponds to
	 * the smallest unit of code that we are interested in timing).
	 *
	 * The arguments are as follows:
	 *
	 * N:          number of coefficients
	 * M:          number of interpolation points
	 * f_hat_real: pointer to N doubles with real values of coefs
	 * f_hat_imag: ditto, but imaginary values
	 * x:          pointer to M double-valued interpolation nodes
	 * f_real:     pointer to space for M double values, will be filled with
	 *             real part of result
	 * f_imag:     ditto, but for imaginary values
	 * time:       pointer to a single double, will be filled with runtime of
	 *             call to nfft_wrapper
	 */
	int nfft_timer(int N, int M, double const * f_hat_real,
				   double const * f_hat_imag, double const * x, double * f_real,
				   double * f_imag, double * time);
}

#endif // __NFFT_TIMER_HPP__
