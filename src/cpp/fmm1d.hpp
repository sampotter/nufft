#ifndef __NUFFT_FMM1D_HPP__
#define __NUFFT_FMM1D_HPP__

#include "bookmarks.hpp"
#include "types.hpp"

#include <unordered_map>

namespace nufft {
    using coefs_type = std::unordered_map<index_type, vector_type<range_elt_type>>;

    template <class kernel_type>
    struct fmm1d {
        static vector_type<domain_elt_type>
        get_multipole_coefs(vector_type<domain_elt_type> const & sources,
                            vector_type<domain_elt_type> const & weights,
                            domain_elt_type x_star,
                            integer_type p);

        static vector_type<range_elt_type>
        evaluate_regular(vector_type<domain_elt_type> const & targets,
                         vector_type<range_elt_type> const & coefs,
                         domain_elt_type x_star,
                         integer_type p);

        static coefs_type
        get_finest_farfield_coefs(bookmarks const & source_bookmarks,
                                  vector_type<domain_elt_type> const & sources,
                                  vector_type<range_elt_type> const & weights,
                                  size_type max_level,
                                  integer_type p);

        static coefs_type
        get_parent_farfield_coefs(coefs_type const & coefs,
                                  size_type level,
                                  integer_type p);

        static void
        do_E4_SR_translations(coefs_type const & input_coefs,
                              coefs_type & output_coefs,
                              size_type level,
                              integer_type p);

        static void
        do_RR_translations(coefs_type const & parent_coefs,
                           coefs_type & child_coefs,
                           size_type level,
                           integer_type p);

        static void
        evaluate(bookmarks const & source_bookmarks,
                 bookmarks const & target_bookmarks,
                 coefs_type const & coefs,
                 vector_type<range_elt_type> & output,
                 vector_type<domain_elt_type> const & sources,
                 vector_type<domain_elt_type> const & targets,
                 vector_type<range_elt_type> const & weights,
                 size_type max_level,
                 integer_type p);

        static vector_type<range_elt_type>
        fmm(vector_type<domain_elt_type> const & sources,
            vector_type<domain_elt_type> const & targets,
            vector_type<range_elt_type> const & weights,
            size_type max_level,
            integer_type p);
    };
}

#include "fmm1d.impl.hpp"

#endif // __NUFFT_FMM1D_HPP__

// Local Variables:
// indent-tabs-mode: nil
// End:
