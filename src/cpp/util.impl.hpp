#ifndef __NUFFT_UTIL_IMPL_HPP__
#define __NUFFT_UTIL_IMPL_HPP__

template <typename elt_type>
std::vector<elt_type>
fmm1d::util::linspace(elt_type min, elt_type max, int64_t num_elts)
{
	std::vector<elt_type> elts(num_elts);
	elts.reserve(num_elts);
	auto current_elt = min;
	auto const delta = (max - min) / (num_elts - 1);
	for (decltype(num_elts) i {0}; i < num_elts; ++i) {
		elts[i] = current_elt;
		current_elt += delta;
	}
	return elts;
}

template <typename elt_type>
std::vector<elt_type>
fmm1d::util::zeros(int64_t num_elts)
{
	return std::vector<elt_type>(num_elts, 0);
}

#endif // __NUFFT_UTIL_IMPL_HPP__
