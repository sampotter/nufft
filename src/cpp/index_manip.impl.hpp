#ifndef __NUFFT_INDEX_MANIP_IMPL_HPP__
#define __NUFFT_INDEX_MANIP_IMPL_HPP__

#include <cassert>
#include <cmath>

template <class int_t>
nufft::index_pair<int_t>::index_pair(int_t first, int_t second):
    first {first},
    second {second}
{}

template <class domain_t, class range_t, class int_t>
int_t
nufft::index_manip<domain_t, range_t, int_t>::get_parent(int_t index)
{
#ifdef NUFFT_DEBUG
    assert(index >= 0);
#endif
    return ((index + 2) >> 1) - 1;
}

template <class domain_t, class range_t, class int_t>
nufft::index_pair<int_t>
nufft::index_manip<domain_t, range_t, int_t>::get_children(int_t index)
{
#if DEBUG
    assert(index >= 0);
#endif
    auto const left_index = index << 1;
    return {left_index, left_index + 1};
}

template <class domain_t, class range_t, class int_t>
int_t
nufft::index_manip<domain_t, range_t, int_t>::get_sibling(int_t index)
{
#if DEBUG
    assert(index >= 0);
#endif
    return index + (index % 2 == 0 ? 1 : -1);
}

template <class domain_t, class range_t, class int_t>
void
nufft::index_manip<domain_t, range_t, int_t>::get_E4_neighbors(
    int_t index,
    int_t * neighbors)
{
#if DEBUG
    assert(index >= 0);
#endif
    if (index % 2 == 0) {
        neighbors[0] = index - 2;
        neighbors[1] = index + 2;
        neighbors[2] = index + 3;
    } else {
        neighbors[0] = index - 3;
        neighbors[1] = index - 2;
        neighbors[2] = index + 2;
    }
}

template <class domain_t, class range_t, class int_t>
int_t
nufft::index_manip<domain_t, range_t, int_t>::get_box_index(
    domain_t elt,
    int_t level)
{
#ifdef NUFFT_DEBUG
    assert(elt >= 0);
    assert(elt < 1);
#endif
    return std::floor((1 << level) * elt);
}

template <class domain_t, class range_t, class int_t>
domain_t
nufft::index_manip<domain_t, range_t, int_t>::get_box_center(
    int_t level,
    int_t index)
{
#if DEBUG
    assert(index >= 0);
    assert(index < std::pow(2, level));
#endif
    return (index + 0.5) * get_box_size(level);
}

template <class domain_t, class range_t, class int_t>
domain_t
nufft::index_manip<domain_t, range_t, int_t>::get_box_size(int_t level)
{
    return 1.0 / (1 << level);
}

#endif // __NUFFT_INDEX_MANIP_IMPL_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
