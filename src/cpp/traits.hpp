#ifndef __NUFFT_TRAITS_HPP__
#define __NUFFT_TRAITS_HPP__

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
}

#endif // __NUFFT_TRAITS_HPP__
