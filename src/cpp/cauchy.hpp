#ifndef __NUFFT_CAUCHY_HPP__
#define __NUFFT_CAUCHY_HPP__

#include "types.hpp"

namespace nufft {
	struct cauchy {
		static range_elt_type R(integer_type m, domain_elt_type x);
		static range_elt_type S(integer_type m, domain_elt_type x);
		static range_elt_type a(integer_type m, domain_elt_type x);
		static range_elt_type b(integer_type m, domain_elt_type x);
	private:
		cauchy();
	};
}

#endif // __NUFFT_CAUCHY_HPP__
