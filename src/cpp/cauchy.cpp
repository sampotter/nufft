#include <cassert>
#include <cmath>

nufft::range_elt_type
nufft::cauchy::R(integer_type m, domain_elt_type x)
{
	return std::pow(x, m);
}

nufft::range_elt_type
nufft::cauchy::S(integer_type m, domain_elt_type x)
{
	return std::pow(x, -(m + 1));
}

nufft::range_elt_type
nufft::cauchy::a(integer_type m, domain_elt_type x)
{
	return std::pow(-x, -(m + 1));
}

nufft::range_elt_type
nufft::cauchy::b(integer_type m, domain_elt_type x)
{
	return std::pow(x, m);
}
