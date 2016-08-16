#ifndef __NUFFT_MATH_HPP__
#define __NUFFT_MATH_HPP__

#include <cinttypes>
#include <complex>
#include <type_traits>

#include "traits.hpp"

namespace nufft {
	template <
		class S,
		class T,
		std::enable_if_t<is_complex<S> {} && !is_complex<T> {}> * = nullptr>
    constexpr S mul(S const & lhs, T const & rhs)
	{
		using field_t = typename get_underlying_field<S>::type;
		return std::complex<field_t>(
			reinterpret_cast<field_t const *>(&lhs)[0] * rhs,
			reinterpret_cast<field_t const *>(&lhs)[1] * rhs);
	}

	template <
		class T,
		std::enable_if_t<!is_complex<T> {}> * = nullptr>
	constexpr T mul(T const & lhs, T const & rhs)
	{
		return lhs*rhs;
	}

	template <
		class S,
		class T,
		std::enable_if_t<is_complex<S> {} && is_complex<T> {}> * = nullptr>
    constexpr void add(S & lhs, T const & rhs)
	{
		using S_field_t = typename get_underlying_field<S>::type;
		using T_field_t = typename get_underlying_field<T>::type;
		reinterpret_cast<S_field_t *>(&lhs)[0] += reinterpret_cast<T_field_t const *>(&rhs)[0];
		reinterpret_cast<S_field_t *>(&lhs)[1] += reinterpret_cast<T_field_t const *>(&rhs)[1];
	}

	template <
		class T,
		std::enable_if_t<!is_complex<T> {}> * = nullptr>
	constexpr void add(T & lhs, T const & rhs)
	{
		lhs += rhs;
	}

	// TODO: include a version which returns void and directly
	// computes the result in the first argument (which would then be
	// passed by reference).

	namespace private_ {
		template <class T>
		constexpr T pow(T base, uint32_t expt)
		{
			T tmp = expt & 1 ? base : 1;
			while (expt >>= 1) {
				base *= base;
				if (expt & 1) {
					tmp *= base;
				}
			}
			return tmp;
		}
	}

	template <
		class S,
		class T,
		std::enable_if_t<
			!std::is_integral<S> {} && std::is_integral<T> {}> * = nullptr>
	constexpr S pow(S base, T expt)
	{
		if (expt < 0) {
			return S(1)/private_::pow(base, -expt);
		} else {
			return private_::pow(base, expt);
		}
	}
}

#endif // __NUFFT_MATH_HPP__

