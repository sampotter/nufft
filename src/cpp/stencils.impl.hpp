#ifndef __NUFFT_STENCILS_IMPL_HPP_HPP__
#define __NUFFT_STENCILS_IMPL_HPP_HPP__

#include <cassert>

template <class int_t>
nufft::SS_stencil<int_t>::SS_stencil(int_t max_level):
	max_level_ {max_level},
	mask_(get_num_entries())
{}

template <class int_t>
void
nufft::SS_stencil<int_t>::set(int_t level, int_t index, bool value)
{
#ifdef NUFFT_DEBUG
	assert(level > 2);
	assert(level <= max_level_);
	assert(index >= 0);
	assert(index < 1 << level);
#endif // NUFFT_DEBUG
	mask_.set(get_level_index(level) + index, value);
}

template <class int_t>
bool
nufft::SS_stencil<int_t>::test(int_t level, int_t index) const
{
#ifdef NUFFT_DEBUG
	assert(level > 2);
	assert(level <= max_level_);
	assert(index >= 0);
	assert(index < 1 << level);
#endif // NUFFT_DEBUG
	return mask_.test(get_level_index(level) + index);
}

template <class int_t>
constexpr
int_t
nufft::SS_stencil<int_t>::get_level_index(int_t level) const
{
#ifdef NUFFT_DEBUG
	assert(level > 2);
	assert(level <= max_level_);
#endif
	return (1 << level) - 1;
}

template <class int_t>
constexpr
int_t
nufft::SS_stencil<int_t>::get_num_entries() const
{
	return (2 << max_level_) - 1;
}

template <class int_t>
nufft::SR_stencil<int_t>::SR_stencil(int_t max_level):
	max_level_ {max_level},
	mask_(get_num_entries())
{}

template <class int_t>
void
nufft::SR_stencil<int_t>::set(int_t level, int_t index, int_t neighbor,
							  bool value)
{
#ifdef NUFFT_DEBUG
	assert(level >= 2);
	assert(level <= max_level_);
	assert(index >= 0);
	assert(index < 1 << level);
	assert(neighbor >= 0);
	assert(neighbor < 3);
#endif // NUFFT_DEBUG
	mask_.set(get_level_index(level) + 3*index + neighbor, value);
}

template <class int_t>
bool
nufft::SR_stencil<int_t>::test(int_t level, int_t index, int_t neighbor) const
{
#ifdef NUFFT_DEBUG
	assert(level >= 2);
	assert(level <= max_level_);
	assert(index >= 0);
	assert(index < 1 << level);
	assert(neighbor >= 0);
	assert(neighbor < 3);
#endif // NUFFT_DEBUG
	return mask_.test(get_level_index(level) + 3*index + neighbor);
}

template <class int_t>
constexpr
int_t
nufft::SR_stencil<int_t>::get_level_index(int_t level) const
{
#ifdef NUFFT_DEBUG
	assert(level >= 2);
	assert(level <= max_level_);
#endif
	return 3*((1 << level) - 1);
}

template <class int_t>
constexpr
int_t
nufft::SR_stencil<int_t>::get_num_entries() const
{
	return 3*((2 << max_level_) - 1);
}

#endif // __NUFFT_STENCILS_IMPL_HPP_HPP__
