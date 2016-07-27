#include "index_manip.hpp"

#include <cassert>
#include <cmath>
#include <iterator>

template <class domain_t, class range_t, class int_t>
typename nufft::index_manip<domain_t, range_t, int_t>::index_t
nufft::index_manip<domain_t, range_t, int_t>::get_parent(index_t index)
{
#ifdef NUFFT_DEBUG
    assert(index >= 0);
#endif
    return ((index + 2) >> 1) - 1;
}

typename nufft::index_manip<domain_t, range_t, int_t>::index_pair_type
nufft::index_manip<domain_t, range_t, int_t>::get_children(index_t index)
{
#if DEBUG
    assert(index >= 0);
#endif
    auto const left_index = index << 1;
    return {left_index, left_index + 1};
}

typename nufft::index_manip<domain_t, range_t, int_t>::index_t
nufft::index_manip<domain_t, range_t, int_t>::get_sibling(index_t index)
{
#if DEBUG
    assert(index >= 0);
#endif
    return index + (index % 2 == 0 ? 1 : -1);
}

typename nufft::index_manip<domain_t, range_t, int_t>::vector_type<
    typename nufft::index_manip<domain_t, range_t, int_t>::index_t
    >
nufft::get_E2_neighbors(int_t level, index_t index)
{
    auto const max_index = std::pow(2, level);
#if DEBUG
    assert(index >= 0);
    assert(index < max_index);
#endif
    nufft::vector_type<nufft::index_t> neighbors {};
    {
        auto const left_index = index - 1;
        if (left_index >= 0) {
            neighbors.push_back(left_index);
        }
    }
    neighbors.push_back(index);
    {
        auto const right_index = index + 1;
        if (right_index < max_index) {
            neighbors.push_back(right_index);
        }
    }
    return neighbors;
}

nufft::index_manip<domain_t, range_t, int_t>::vector_type<
    typename nufft::index_manip<domain_t, range_t, int_t>::index_t
    >
nufft::get_E4_neighbors(int_t level, index_t index)
{
#if DEBUG
    assert(level >= 2);
    assert(index >= 0);
    assert(index < std::pow(2, level));
#endif

    nufft::vector_type<nufft::index_t> neighbors;
    auto const parent_neighbors = get_E2_neighbors(level - 1, get_parent(index));
    for (auto const neighbor: parent_neighbors) {
        auto const children = get_children(neighbor);
        neighbors.push_back(children.first);
        neighbors.push_back(children.second);
    }

    auto const E2_neighbors = get_E2_neighbors(level, index);
    nufft::vector_type<nufft::index_t> E4_neighbors;
    std::set_difference(
        std::cbegin(neighbors),
        std::cend(neighbors),
        std::cbegin(E2_neighbors),
        std::cend(E2_neighbors),
        std::back_inserter(E4_neighbors));
    return E4_neighbors;
}

nufft::index_t
nufft::get_box_index(domain_t elt, int_t level)
{
#ifdef NUFFT_DEBUG
    assert(elt >= 0);
    assert(elt < 1);
#endif
    return std::floor((1 << level) * elt);
}

nufft::domain_t
nufft::get_box_center(int_t level, index_t index)
{
#if DEBUG
    assert(index >= 0);
    assert(index < std::pow(2, level));
#endif
    return (index + 0.5) * get_box_size(level);
}

nufft::domain_t
nufft::get_box_size(int_t level)
{
    return 1.0 / (1 << level);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
