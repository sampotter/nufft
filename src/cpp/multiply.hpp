#ifndef __NUFFT_MULTIPLY_HPP__
#define __NUFFT_MULTIPLY_HPP__

#include <complex>
#include <type_traits>

#include "traits.hpp"

namespace nufft {
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
