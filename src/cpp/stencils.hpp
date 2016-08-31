#ifndef __NUFFT_STENCILS_HPP__
#define __NUFFT_STENCILS_HPP__

#include <boost/dynamic_bitset.hpp>
#include <cinttypes>
#include <type_traits>

#include "preprocessor.hpp"

namespace nufft {
	template <class int_t = int64_t>
	struct SS_stencil {
		SS_stencil(int_t max_level);
		void set(int_t level, int_t index, bool value = true);
		bool test(int_t level, int_t index) const;
	NUFFT_PRIVATE:
		constexpr int_t get_level_index(int_t level) const;
		constexpr int_t get_num_entries() const;

		int_t max_level_;
		boost::dynamic_bitset<> mask_;
	};

	template <class int_t = int64_t>
	using RR_stencil = SS_stencil<int_t>;

	template <class int_t = int64_t>
	struct SR_stencil {
		SR_stencil(int_t max_level);
		void set(int_t level, int_t index, int_t neighbor, bool value = true);
		bool test(int_t level, int_t index, int_t neighbor) const;
	NUFFT_PRIVATE:
		constexpr int_t get_level_index(int_t level) const;
		constexpr int_t get_num_entries() const;

		int_t max_level_;
		boost::dynamic_bitset<> mask_;
	};
}

#include "stencils.impl.hpp"

#endif // __NUFFT_STENCILS_HPP__
