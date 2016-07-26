#include "nfft_wrapper.h"

#include <assert.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>

#ifdef DEBUG
#    include <stdio.h>
#endif

#define NFFT_PRECISION_DOUBLE
#include "nfft3mp.h"

int nfft_wrapper(int N, int M, char const * f_hat_bytes, double const * x,
				 char * f_bytes) {
	int code = NFFT_SUCCESS;
	
	if (f_hat_bytes == NULL) {
#ifdef DEBUG
		fprintf(stderr, "nfft_wrapper: f_hat_bytes is null\n");
#endif
		code = NFFT_ERROR;
		goto cleanup;
	}
	if (x == NULL) {
#ifdef DEBUG
		fprintf(stderr, "nfft_wrapper: x is null\n");
#endif
		code = NFFT_ERROR;
		goto cleanup;
	}
	if (f_bytes == NULL) {
#ifdef DEBUG
		fprintf(stderr, "nfft_wrapper: f_bytes is null\n");
#endif
		code = NFFT_ERROR;
		goto cleanup;
	}

	_Complex double * f_hat = (_Complex double *) f_hat_bytes;
	_Complex double * f = (_Complex double *) f_bytes;
	
	char const * error = NULL;
	
	NFFT(plan) plan;
	nfft_init_1d(&plan, N, M);
#ifdef DEBUG
	if ((error = nfft_check(&plan)) != NULL) {
		fprintf(stderr, "NFFT error: %s\n", error);
	}
#endif
	memcpy(plan.f_hat, f_hat, sizeof(double complex)*N);
	memcpy(plan.x, x, sizeof(double)*M);
	
	nfft_precompute_one_psi(&plan);
#ifdef DEBUG
	if ((error = nfft_check(&plan)) != NULL) {
		fprintf(stderr, "NFFT error: %s\n", error);
	}
#endif
	nfft_trafo(&plan);
#ifdef DEBUG
	if ((error = nfft_check(&plan)) != NULL) {
		fprintf(stderr, "NFFT error: %s\n", error);
	}
#endif
	memcpy(f, plan.f, M*sizeof(double complex));
	nfft_finalize(&plan);
	
  cleanup:
	return code;
}
