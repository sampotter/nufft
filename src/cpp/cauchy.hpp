#ifndef __NUFFT_CAUCHY_HPP__
#define __NUFFT_CAUCHY_HPP__

#include "types.hpp"

namespace nufft {
	struct cauchy {
		static range_elt_type R(integer_type m, domain_elt_type x);
		static range_elt_type S(integer_type m, domain_elt_type x);
		static range_elt_type a(integer_type m, domain_elt_type x);
		static range_elt_type b(integer_type m, domain_elt_type x);

		static matrix_type<domain_elt_type>
		get_SS_matrix(domain_elt_type delta, integer_type p);

		static matrix_type<domain_elt_type>
		get_SR_matrix(domain_elt_type delta, integer_type p);

		static matrix_type<domain_elt_type>
		get_RR_matrix(domain_elt_type delta, integer_type p);
	private:
		cauchy();
	};
}

#endif // __NUFFT_CAUCHY_HPP__
