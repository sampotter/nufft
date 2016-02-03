#ifndef __NUFFT_UTIL_IMPL_HPP__
#define __NUFFT_UTIL_IMPL_HPP__

template <typename elt_type>
std::vector<elt_type>
nufft::util::linspace(elt_type min, elt_type max, int64_t num_elts)
{
    // NB: this implementation uses Kahan summation to avoid numerical
    // errors

    std::vector<elt_type> elts(num_elts);
    elts.reserve(num_elts);
    
    auto sum = min;
    decltype(sum) compensation {0};
    auto const delta = (max - min) / (num_elts - 1);
    
    for (decltype(num_elts) i {0}; i < num_elts; ++i) {
        elts[i] = sum;
        auto current_delta = delta - compensation;
        auto tmp = sum + current_delta;
        compensation = (tmp - sum) - current_delta;
        sum = tmp;
    }

    return elts;
}

template <typename elt_type>
std::vector<elt_type>
nufft::util::zeros(int64_t num_elts)
{
    return std::vector<elt_type>(num_elts, 0);
}

#endif // __NUFFT_UTIL_IMPL_HPP__

// Local Variables:
// indent-tabs-mode: p
// End:
