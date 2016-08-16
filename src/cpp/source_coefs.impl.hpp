#ifndef __NUFFT_SOURCE_COEFS_IMPL_HPP__
#define __NUFFT_SOURCE_COEFS_IMPL_HPP__

#include <cassert>

template <class range_t, class int_t>
nufft::source_coefs<range_t, int_t>::source_coefs(int_t max_level, int_t p):
	max_level_ {max_level},
	p_ {p},
	mask_(get_num_coefs()),
	coefs_ {new range_t[get_num_coefs()*p]}
{
	memset(coefs_, 0x0, sizeof(range_t)*get_num_coefs()*p);
}

template <class range_t, class int_t>
nufft::source_coefs<range_t, int_t>::~source_coefs()
{
	delete[] coefs_;
}

template <class range_t, class int_t>
void
nufft::source_coefs<range_t, int_t>::set(int_t level, int_t index, bool value)
{
#ifdef NUFFT_DEBUG
	assert(level >= 2);
	assert(level <= max_level_);
	assert(index >= 0);
	assert(index < 1 << level);
#endif
	mask_.set(get_level_index(level) + index, value);
}

template <class range_t, class int_t>
bool
nufft::source_coefs<range_t, int_t>::test(int_t level, int_t index) const
{
#ifdef NUFFT_DEBUG
	assert(level >= 2);
	assert(level <= max_level_);
	assert(index >= 0);
	assert(index < 1 << level);
#endif
	return mask_.test(get_level_index(level) + index);
}

template <class range_t, class int_t>
constexpr
range_t *
nufft::source_coefs<range_t, int_t>::get_coefs(int_t level, int_t index) const
{
#ifdef NUFFT_DEBUG
	assert(level >= 2);
	assert(level <= max_level_);
	assert(index >= 0);
	assert(index < 1 << level);
#endif
	return &coefs_[(get_level_index(level) + index)*p_];
}

template <class range_t, class int_t>
constexpr
range_t *
nufft::source_coefs<range_t, int_t>::get_level(int_t level) const
{
#ifdef NUFFT_DEBUG
	assert(level >= 2);
	assert(level <= max_level_);
#endif
	return &coefs_[get_level_index(level)*p_];
}

template <class range_t, class int_t>
constexpr
int_t
nufft::source_coefs<range_t, int_t>::get_level_index(int_t level) const
{
#ifdef NUFFT_DEBUG
	assert(level >= 2);
	assert(level <= max_level_);
#endif
	return (1 << level) - 4;
}

template <class range_t, class int_t>
constexpr
int_t
nufft::source_coefs<range_t, int_t>::get_num_coefs() const
{
	return (2 << max_level_) - 4;
}

#endif // __NUFFT_SOURCE_COEFS_IMPL_HPP__
