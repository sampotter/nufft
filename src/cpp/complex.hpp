#ifndef __NUFFT_COMPLEX_HPP__
#define __NUFFT_COMPLEX_HPP__

#include <complex>

namespace nufft {
	template <class T>
	constexpr std::complex<T> multiply(
		std::complex<T> const & lhs,
		std::complex<T> const & rhs) const;
}

#include "complex.impl.hpp"

#endif // __NUFFT_COMPLEX_HPP__
