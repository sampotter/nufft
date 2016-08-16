#ifndef __NUFFT_SOURCE_COEFS_HPP__
#define __NUFFT_SOURCE_COEFS_HPP__

#include <boost/dynamic_bitset.hpp>
#include <cinttypes>

#include "preprocessor.hpp"

namespace nufft {
	template <class range_t = double, class int_t = int64_t>
	struct source_coefs {
		source_coefs(int_t max_level, int_t p);
		~source_coefs();
		void set(int_t level, int_t index, bool value = true);
		bool test(int_t level, int_t index) const;
		constexpr range_t * get_coefs(int_t level, int_t index) const;
		constexpr range_t * get_level(int_t level) const;
	NUFFT_PRIVATE:
		constexpr int_t get_level_index(int_t level) const;
		constexpr int_t get_num_coefs() const;
		
		int_t max_level_;
		int_t p_;
		boost::dynamic_bitset<> mask_;
		range_t * coefs_;
	};
}

#include "source_coefs.impl.hpp"

#endif // __NUFFT_SOURCE_COEFS_HPP__
