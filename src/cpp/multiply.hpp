#ifndef __NUFFT_MULTIPLY_HPP__
#define __NUFFT_MULTIPLY_HPP__

#include <complex>
#include <type_traits>

namespace nufft {
	template <class T> struct is_complex: std::false_type {};
	template <class T> struct is_complex<std::complex<T>>: std::true_type {};

	template <class T, std::enable_if_t<is_complex<T> {}> * = nullptr>
	constexpr T multiply(T const & lhs, T const & rhs)
	{
		auto const k1 = lhs.real()*(rhs.real() + rhs.imag());
		auto const k2 = rhs.imag()*(lhs.real() + lhs.imag());
		auto const k3 = rhs.real()*(lhs.imag() - lhs.real());
		return T(k1 - k2, k1 + k3);
	}

	template <class T, std::enable_if_t<!is_complex<T> {}> * = nullptr>
	constexpr T multiply(T const & lhs, T const & rhs)
	{
		return lhs*rhs;
	}
}

#endif // __NUFFT_MULTIPLY_HPP__
