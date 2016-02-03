#include "cauchy.hpp"

#include <cassert>
#include <cmath>

nufft::range_elt_type
nufft::cauchy::R(integer_type m, domain_elt_type x)
{
    return std::pow(x, m);
}

nufft::range_elt_type
nufft::cauchy::S(integer_type m, domain_elt_type x)
{
    return std::pow(x, -(m + 1));
}

nufft::range_elt_type
nufft::cauchy::a(integer_type m, domain_elt_type x)
{
    return std::pow(-x, -(m + 1));
}

nufft::range_elt_type
nufft::cauchy::b(integer_type m, domain_elt_type x)
{
    return std::pow(x, m);
}

nufft::matrix_type<nufft::domain_elt_type>
nufft::cauchy::get_SS_matrix(domain_elt_type delta, integer_type p)
{
    matrix_type<domain_elt_type> SS(p, p);

    for (index_type row {0}; row < p; ++row) {
        SS(row, 0) = 1.0;
    }

    for (index_type row {1}; row < p; ++row) {
        for (index_type col {1}; col < p; ++col) {
            SS(row, col) = SS(row - 1, col) + SS(row - 1, col - 1);
        }
    }

    for (index_type row {0}; row < p; ++row) {
        for (index_type col {0}; col < p; ++col) {
            SS(row, col) *= std::pow(-delta, row - col);
        }
    }

    return SS;
}

nufft::matrix_type<nufft::domain_elt_type>
nufft::cauchy::get_SR_matrix(domain_elt_type delta, integer_type p)
{
    matrix_type<domain_elt_type> SR(p, p);

    for (index_type row {0}; row < p; ++row) {
        SR(row, 0) = 1.0;
    }

    for (index_type col {0}; col < p; ++col) {
        SR(0, col) = 1.0;
    }

    for (index_type row {1}; row < p; ++row) {
        for (index_type col {1}; col < p; ++col) {
            SR(row, col) = SR(row - 1, col) + SR(row, col - 1);
        }
    }

    for (index_type row {0}; row < p; ++row) {
        for (index_type col {0}; col < p; ++col) {
            SR(row, col) *= std::pow(delta, -(row + col + 1));
        }
    }

    for (index_type row {1}; row < p; row += 2) {
        for (index_type col {0}; col < p; ++col) {
            SR(row, col) *= -1.0;
        }
    }

    return SR;
}

nufft::matrix_type<nufft::domain_elt_type>
nufft::cauchy::get_RR_matrix(domain_elt_type delta, integer_type p)
{
    matrix_type<domain_elt_type> RR(p, p);

    for (index_type col {0}; col < p; ++col) {
        RR(0, col) = 1.0;
    }
    
    for (index_type row {1}; row < p; ++row) {
        for (index_type col {0}; col < p; ++col) {
            RR(row, col) = 0.0;
        }
    }

    for (index_type col {1}; col < p; ++col) {
        for (index_type row {1}; row < p; ++row) {
            RR(row, col) = RR(row - 1, col - 1) + RR(row, col - 1);
        }
    }

    for (index_type col {0}; col < p; ++col) {
        for (index_type row {0}; row < col; ++row) {
            RR(row, col) *= std::pow(delta, col - row);
        }
    }

    return RR;
}

// Local Variables:
// indent-tabs-mode: nil
// End:
