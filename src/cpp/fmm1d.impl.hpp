#ifndef __NUFFT_FMM1D_IMPL_HPP__
#define __NUFFT_FMM1D_IMPL_HPP__

template <typename kernel_type>
nufft::vector_type<nufft::domain_elt_type>
nufft::fmm1d<kernel_type>::get_multipole_coefs(
    vector_type<domain_elt_type> const & sources,
    vector_type<domain_elt_type> const & weights,
    domain_elt_type x_star,
    integer_type p)
{
#ifdef NUFFT_DEBUG
    assert(std::size(sources) <= std::numeric_limits<index_type>::max());
#endif
    auto const num_sources = static_cast<index_type>(std::size(sources));
#ifdef NUFFT_DEBUG
    {
        auto const num_weights = std::size(weights);
        assert(num_weights <= std::numeric_limits<index_type>::max());
        assert(num_sources == static_cast<index_type>(num_weights));
    }
    assert(0 <= x_star);
    assert(x_star < 1);
    assert(p > 0);
#endif

    vector_type<domain_elt_type> offset_sources(sources);
    for (auto & offset_source: offset_sources) {
        offset_source -= x_star;
    }
    
    vector_type<domain_elt_type> coefs(p, 0);
    for (index_type i {0}; i < p; ++i) {
        for (index_type j {0}; j < num_sources; ++j) {
            coefs[i] += weights[j] * kernel_type::b(i, offset_sources[j]);
        }
    }

    return coefs;
}

template <typename kernel_type>
nufft::vector_type<nufft::range_elt_type>
nufft::fmm1d<kernel_type>::evaluate_regular(
    vector_type<domain_elt_type> const & targets,
    vector_type<range_elt_type> const & coefs,
    domain_elt_type x_star,
    integer_type p)
{
#ifdef NUFFT_DEBUG
    assert(0 <= x_star);
    assert(x_star < 1);
    assert(p > 0);
#endif
    
    vector_type<domain_elt_type> offset_targets(targets);
    for (auto & offset_target: offset_targets) {
        offset_target -= x_star;
    }

#ifdef NUFFT_DEBUG
    assert(std::size(targets) <= std::numeric_limits<index_type>::max());
#endif
    auto const num_targets = static_cast<index_type>(std::size(targets));
    vector_type<range_elt_type> sums(num_targets, 0);
    for (index_type i {0}; i < num_targets; ++i) {
        auto const offset_target = offset_targets[i];
        for (index_type j {0}; j < p; ++j) {
            sums[i] += coefs[j] * kernel_type::R(j, offset_target);
        }
    }

    return sums;
}

#endif // __NUFFT_FMM1D_IMPL_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
