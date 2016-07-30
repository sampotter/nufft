#ifndef __NUFFT_FMM1D_HPP__
#define __NUFFT_FMM1D_HPP__

#include <cinttypes>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "bookmarks.hpp"
#include "index_manip.hpp"

namespace nufft {
    template <class kernel_t,
              class domain_t = double,
              class range_t = double,
              class int_t = int64_t>
    struct fmm1d {
        using index_manip_t = index_manip<domain_t, range_t, int_t>;
        static auto constexpr get_box_center = index_manip_t::get_box_center;
        static auto constexpr get_children = index_manip_t::get_children;
        static auto constexpr get_E4_neighbors = index_manip_t::get_E4_neighbors;
        static auto constexpr get_E2_neighbors = index_manip_t::get_E2_neighbors;

        template <class T> using vector_t = std::vector<T>;
        using coefs_type = std::unordered_map<int_t, vector_t<range_t>>;

        static vector_t<range_t>
        get_multipole_coefs(domain_t const * sources,
                            range_t const * weights,
                            int_t num_sources,
                            domain_t x_star,
                            int_t p);

        static coefs_type
        get_finest_farfield_coefs(
            bookmarks<domain_t, int_t> const & source_bookmarks,
            vector_t<domain_t> const & sources,
            vector_t<range_t> const & weights,
            int_t max_level,
            int_t p);

        static coefs_type
        get_parent_farfield_coefs(coefs_type const & coefs,
                                  int_t level,
                                  int_t p);

        static void
        do_E4_SR_translations(coefs_type const & input_coefs,
                              coefs_type & output_coefs,
                              int_t level,
                              int_t p);

        static void
        do_RR_translations(coefs_type const & parent_coefs,
                           coefs_type & child_coefs,
                           int_t level,
                           int_t p);

        static void
        evaluate(bookmarks<domain_t, int_t> const & source_bookmarks,
                 bookmarks<domain_t, int_t> const & target_bookmarks,
                 coefs_type const & coefs,
                 vector_t<range_t> & output,
                 vector_t<domain_t> const & sources,
                 vector_t<domain_t> const & targets,
                 vector_t<range_t> const & weights,
                 int_t max_level,
                 int_t p);

        static vector_t<range_t>
        fmm(vector_t<domain_t> const & sources,
            vector_t<domain_t> const & targets,
            vector_t<range_t> const & weights,
            int_t max_level,
            int_t p);
    };
}

#include "fmm1d.impl.hpp"

#endif // __NUFFT_FMM1D_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
