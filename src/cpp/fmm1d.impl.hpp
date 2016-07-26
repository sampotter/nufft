#ifndef __NUFFT_FMM1D_IMPL_HPP__
#define __NUFFT_FMM1D_IMPL_HPP__

#ifdef NUFFT_DEBUG
#    include <boost/range/adaptor/map.hpp>
#endif

#include "index_manip.hpp"
#include "util.hpp"

template <class kernel_type>
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

template <class kernel_type>
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

template <class kernel_type>
nufft::coefs_type
nufft::fmm1d<kernel_type>::get_finest_farfield_coefs(
    bookmarks const & source_bookmarks,
    vector_type<domain_elt_type> const & sources,
    vector_type<range_elt_type> const & weights,
    size_type max_level,
    integer_type p)
{
    coefs_type coefs;

#ifdef NUFFT_DEBUG
    assert(p > 0);
    assert(std::pow(2, max_level) <= std::numeric_limits<index_type>::max());
#endif
    auto const max_index = static_cast<index_type>(std::pow(2, max_level));
    
    for (index_type index {0}; index < max_index; ++index) {
        auto const opt_bookmark = source_bookmarks(max_level, index);
        if (!opt_bookmark) {
            continue;
        }

        auto const left_index = opt_bookmark->first;
        auto const right_index = opt_bookmark->second;

        auto const sources_cbegin = std::cbegin(sources);
        vector_type<domain_elt_type> const box_sources(
            sources_cbegin + left_index,
            sources_cbegin + right_index + 1); // TODO: ugly

        auto const weights_cbegin = std::cbegin(weights);
        vector_type<domain_elt_type> const box_weights(
            weights_cbegin + left_index,
            weights_cbegin + right_index + 1); // TODO: ugly

        auto const x_star = get_box_center(max_level, index);

        auto const box_coefs = get_multipole_coefs(
            box_sources, box_weights, x_star, p);

        coefs[index] = box_coefs;
    }

    return coefs;
}

template <class kernel_type>
nufft::coefs_type
nufft::fmm1d<kernel_type>::get_parent_farfield_coefs(
    coefs_type const & coefs,
    size_type level,
    integer_type p)
{
#ifdef NUFFT_DEBUG
    assert(p > 0);
    assert(level > 0);
    assert(std::pow(2, level) <= std::numeric_limits<index_type>::max());
#endif
    
    auto const max_index = static_cast<index_type>(std::pow(2, level));
    
#ifdef NUFFT_DEBUG
    auto const keys = coefs | boost::adaptors::map_keys;
    auto const min = *std::min_element(std::cbegin(keys), std::cend(keys));
    auto const max = *std::max_element(std::cbegin(keys), std::cend(keys));
    assert(0 <= min);
    assert(max < max_index);
#endif

    coefs_type parent_coefs;
    decltype(level) parent_level {level - 1};

    auto const max_parent_index = max_index / 2;
    
    for (index_type parent_index {0}; parent_index < max_parent_index; ++parent_index) {
        auto workspace = util::zeros<range_elt_type>(p); // TODO: move out of this loop
        auto parent_box_coefs = util::zeros<range_elt_type>(p);
        auto const parent_center = get_box_center(parent_level, parent_index);
        
        auto const translate_child_coefs = [&] (index_type index) {
            auto const child_center = get_box_center(level, index);
            // TODO: no need to keep recomputing delta!!! fix this.
            auto const delta = parent_center - child_center;
            kernel_type::apply_SS_translation(coefs.at(index), workspace, delta, p);
        };

        auto const children = get_children(parent_index);

        bool found_coefs = false;
        auto const add_coefs = [&] (index_type const index) {
            if (coefs.find(index) != std::cend(coefs)) {
                found_coefs = true;
                translate_child_coefs(index);
                for (index_type i {0}; i < p; ++i) {
                    parent_box_coefs[i] += workspace[i];
                }
            }
        };
        
        add_coefs(children.first);
        add_coefs(children.second);

        if (found_coefs) {
            parent_coefs[parent_index] = parent_box_coefs;
        }
    }

    return parent_coefs;
}

template <class kernel_type>
void
nufft::fmm1d<kernel_type>::do_E4_SR_translations(
    coefs_type const & input_coefs,
    coefs_type & output_coefs,
    size_type level,
    integer_type p)
{
    auto const max_key = std::pow(2, level);
#ifdef NUFFT_DEBUG
    auto const validate_coefs = [max_key] (coefs_type const & coefs) {
        for (auto const & entry: coefs) {
            auto const key = entry.first;
            assert(0 <= key);
            assert(key < max_key);
        }
    };
    validate_coefs(input_coefs);
    validate_coefs(output_coefs); // TODO: maybe unnecessary
#endif

    auto workspace = util::zeros<range_elt_type>(p);
    for (index_type i {0}; i < max_key; ++i) {
        auto const center = get_box_center(level, i);
        auto const E4_neighbors = get_E4_neighbors(level, i);
        for (auto const n: E4_neighbors) {
            if (input_coefs.find(n) == std::cend(input_coefs)) {
                continue;
            }
            auto const neighbor_center = get_box_center(level, n);
            auto const delta = center - neighbor_center;
            kernel_type::apply_SR_translation(input_coefs.at(n), workspace, delta, p);
            if (output_coefs.find(i) == std::cend(output_coefs)) {
                output_coefs[i] = util::zeros<range_elt_type>(p);
            }
            for (index_type j {0}; j < p; ++j) {
                output_coefs[i][j] += workspace[j];
            }
        }
    }
}

template <class kernel_type>
void
nufft::fmm1d<kernel_type>::do_RR_translations(
    coefs_type const & parent_coefs,
    coefs_type & child_coefs,
    size_type level,
    integer_type p)
{
    auto const parent_level = level;
    auto const child_level = level + 1;
    auto const max_parent_index = std::pow(2, parent_level);
#ifdef NUFFT_DEBUG
    auto const verify_keys = [] (coefs_type const & coefs, index_type max_index) {
        auto const keys = coefs | boost::adaptors::map_keys;
        auto const cbegin = std::cbegin(keys);
        auto const cend = std::cend(keys);
        if (cbegin != cend) {
            auto const min = *std::min_element(cbegin, cend);
            auto const max = *std::max_element(cbegin, cend);
            assert(0 <= min);
            assert(max < max_index);
        }
    };
    verify_keys(parent_coefs, max_parent_index);
    verify_keys(child_coefs, std::pow(2, child_level));
#endif
    auto workspace = util::zeros<range_elt_type>(p);
    for (index_type i {0}; i < max_parent_index; ++i) {
        auto const parent_center = get_box_center(parent_level, i);
        auto const translate = [&parent_coefs,
                                &child_coefs,
                                &workspace,
                                child_level,
                                parent_center,
                                p,
                                i] (index_type j) {
            auto const child_center = get_box_center(child_level, j);
            auto const delta = child_center - parent_center;
            kernel_type::apply_RR_translation(parent_coefs.at(i), workspace, delta, p);
            if (child_coefs.find(j) == std::cend(child_coefs)) {
                child_coefs[j] = util::zeros<range_elt_type>(p);
            }
            for (index_type k {0}; k < p; ++k) {
                child_coefs[j][k] += workspace[k];
            }
        };
        auto const children = get_children(i);
        translate(children.first);
        translate(children.second);
    }
}

template <class kernel_type>
void
nufft::fmm1d<kernel_type>::evaluate(
    bookmarks const & source_bookmarks,
    bookmarks const & target_bookmarks,
    coefs_type const & coefs,
    vector_type<range_elt_type> & output,
    vector_type<domain_elt_type> const & sources,
    vector_type<domain_elt_type> const & targets,
    vector_type<range_elt_type> const & weights,
    size_type max_level,
    integer_type p)
{
    auto const max_index = std::pow(2, max_level);
#ifdef NUFFT_DEBUG
    assert(std::size(output) == std::size(targets));
    auto const keys = coefs | boost::adaptors::map_keys;
    auto const min = *std::min_element(std::cbegin(keys), std::cend(keys));
    auto const max = *std::max_element(std::cbegin(keys), std::cend(keys));
    assert(0 <= min);
    assert(max < max_index);
#endif
    for (index_type i {0}; i < max_index; ++i) {
        auto const opt = target_bookmarks(max_level, i);
        if (!opt) {
            continue;
        }
        auto const left = opt->first;
        auto const right = opt->second;

        // TODO: this can very likely be greatly simplified
        vector_type<index_type> direct_indices;
        for (auto const j: get_E2_neighbors(max_level, i)) {
            auto const opt = source_bookmarks(max_level, j);
            if (!opt) {
                continue;
            }
            auto const left = opt->first;
            auto const right = opt->second;
            for (index_type k {left}; k <= right; ++k) {
                direct_indices.push_back(k);
            }
        }
        if (!direct_indices.empty()) {
            for (index_type j {left}; j <= right; ++j) {
                auto const y = targets[j];
                auto & tmp = output[j];
                for (auto const k: direct_indices) {
                    tmp += weights[k] * kernel_type::phi(y, sources[k]);
                }
            }
        }

        auto const center = get_box_center(max_level, i);
        auto const coef_vec = coefs.at(i);
        for (index_type j {left}; j <= right; ++j) {
            auto const y_off = targets[j] - center;
            for (index_type k {0}; k < p; ++k) {
                output[j] += coef_vec[k] * kernel_type::R(k, y_off);
            }
        }
    }
}

template <class kernel_type>
nufft::vector_type<nufft::range_elt_type>
nufft::fmm1d<kernel_type>::fmm(
    vector_type<domain_elt_type> const & sources,
    vector_type<domain_elt_type> const & targets,
    vector_type<range_elt_type> const & weights,
    size_type max_level,
    integer_type p)
{
#ifdef NUFFT_DEBUG
    {
        auto const src_begin = std::cbegin(sources);
        auto const src_end = std::cend(sources);
        auto const trg_begin = std::cbegin(targets);
        auto const trg_end = std::cend(targets);
        {
            auto const min_src = *std::min_element(src_begin, src_end);
            auto const max_src = *std::max_element(src_begin, src_end);
            assert(0 <= min_src);
            assert(max_src < 1);
        }
        {
            auto const min_trg = *std::min_element(trg_begin, trg_end);
            auto const max_trg = *std::max_element(trg_begin, trg_end);
            assert(0 <= min_trg);
            assert(max_trg < 1);
        }
        assert(std::is_sorted(src_begin, src_end));
        assert(std::is_sorted(trg_begin, trg_end));
        assert(max_level >= 2);
        assert(p >= 0);
    }
#endif
    
    bookmarks const src_bookmarks {sources, max_level};
    bookmarks const trg_bookmarks {targets, max_level};

    std::unordered_map<size_type, coefs_type> source_coefs;
    source_coefs[max_level] = get_finest_farfield_coefs(
        src_bookmarks, sources, weights, max_level, p);
    if (max_level > 2) { // TODO: this line is prob unnecessary, remove later
        for (size_type level {max_level}; level > 2; --level) {
            source_coefs[level - 1] = get_parent_farfield_coefs(
                source_coefs[level], level, p);
        }
    }

    std::unordered_map<size_type, coefs_type> target_coefs;
    for (size_type level {2}; level < max_level; ++level) {
        do_E4_SR_translations(source_coefs[level], target_coefs[level], level, p);
        do_RR_translations(target_coefs[level], target_coefs[level + 1], level, p);
    }
    do_E4_SR_translations(source_coefs[max_level], target_coefs[max_level], max_level, p);

    vector_type<range_elt_type> output(std::size(targets), 0);
    evaluate(src_bookmarks, trg_bookmarks, target_coefs[max_level],
             output, sources, targets, weights, max_level, p);

    return output;
}

#endif // __NUFFT_FMM1D_IMPL_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
