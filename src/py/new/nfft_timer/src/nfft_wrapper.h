#ifndef __NFFT_TEST_H__
#define __NFFT_TEST_H__

#define NFFT_SUCCESS 0
#define NFFT_ERROR 1

/**
 * This function computes an NFFT using Potts's NFFT3. Because of how
 * C99 complex number types are implemented, it doesn't seem possible
 * to expose them in the interface here. Instead, the arguments that
 * would have been _Complex double pointers are just generic byte pointers.
 *
 * N:           number of coefficients
 * M:           number of interpolation points
 * f_hat_bytes: pointer to N*sizeof(_Complex double) bytes containing
 *              coefficients
 * x:           interpolation nodes
 * f_bytes:     should have space for M*sizeof(_Complex double)
 *              bytes---will be filled with interpolated values
 * 
 * returns: NFFT_ERROR if there was some problem, NFFT_SUCCESS otherwise
 */
int nfft_wrapper(int N, int M, char const * f_hat_bytes, double const * x,
				 char * f_bytes);

#endif // __NFFT_TEST_H__
