#include "nfft_timer.hpp"

#include <chrono>
#include <complex>
#ifdef DEBUG
#    include <iostream>
#endif
#include <vector>

extern "C" {
#include "nfft_wrapper.h"
}

extern "C" int nfft_timer(int N, int M, double const * f_hat_real,
						  double const * f_hat_imag, double const * x,
						  double * f_real, double * f_imag, double * time) {
	using complex = std::complex<double>;
	using clock = std::chrono::high_resolution_clock;
	using duration = std::chrono::duration<double>;

	std::vector<complex> f_hat;
	std::vector<complex> f(M, {0, 0});
	char * f_hat_bytes = nullptr;
	char * f_bytes = nullptr;

	for (int i = 0; i < N; ++i) {
		f_hat.push_back({f_hat_real[i], f_hat_imag[i]});
	}
	f_hat_bytes = (char *) &f_hat[0];
	
	f_bytes = (char *) &f[0];

	auto const start_time = clock::now();
	auto const code = nfft_wrapper(N, M, f_hat_bytes, x, f_bytes);
	*time = duration(clock::now() - start_time).count();
	if (code != NFFT_SUCCESS) {
#ifdef DEBUG
		std::cerr << "nfft_timer: nfft_wrapper failed" << std::endl;
#endif
		return code;
	}

	for (int i = 0; i < M; ++i) {
		f_real[i] = f[i].real();
		f_imag[i] = f[i].imag();
	}

	return NFFT_SUCCESS;
}
