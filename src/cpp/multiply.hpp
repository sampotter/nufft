#ifndef __NUFFT_MULTIPLY_HPP__
#define __NUFFT_MULTIPLY_HPP__

#include <complex>
#include <type_traits>

namespace nufft {
	template <class T> struct get_underlying_field {};
	template <class T>
	struct get_underlying_field<std::complex<T>> {
		using type = T;
	};

	template <class T> struct is_complex: std::false_type {};
	template <class T> struct is_complex<std::complex<T>>: std::true_type {};

	// template <class T, std::enable_if_t<is_complex<T> {}> * = nullptr>
	// constexpr T multiply(T const & lhs, T const & rhs)
	// {
	// 	auto const a = lhs.real();
	// 	auto const b = lhs.imag();
	// 	auto const c = rhs.real();
	// 	auto const d = rhs.imag();
	// 	auto const tmp = a*(c + d);
	// 	return T(tmp - d*(a + b), tmp + c*(b - a));
	// }

	template <
		class S,
		class T,
		std::enable_if_t<is_complex<S> {} && !is_complex<T> {}> * = nullptr>
	constexpr S multiply(S const & lhs, T const & rhs)
	{
		using field_t = typename get_underlying_field<S>::type;
		S tmp = lhs;
		reinterpret_cast<field_t *>(&tmp)[0] *= rhs;
		reinterpret_cast<field_t *>(&tmp)[1] *= rhs;
		return tmp;
	}

	template <class T, std::enable_if_t<!is_complex<T> {}> * = nullptr>
	constexpr T multiply(T const & lhs, T const & rhs)
	{
		return lhs*rhs;
	}
}

#endif // __NUFFT_MULTIPLY_HPP__
