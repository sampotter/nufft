#include "fmm1d.hpp"

#include <cassert>
#include <cmath>

namespace nufft { namespace kernel { 
	struct cauchy {
		static constexpr range_elt_type R(integer_type m, domain_elt_type x) {
			return std::pow(x, m);
		}
		
		static constexpr range_elt_type S(integer_type m, domain_elt_type x) {
			return std::pow(x, -(m + 1));
		}
		
		static constexpr range_elt_type a(integer_type m, domain_elt_type x) {
			return std::pow(-x, -(m + 1));
		}
		
		static constexpr range_elt_type b(integer_type m, domain_elt_type x) {
			return std::pow(x, m);
		}
	private:
		cauchy();
	};
} }

