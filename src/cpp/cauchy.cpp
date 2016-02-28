#include "cauchy.hpp"

#include <cassert>
#include <cmath>

// nufft::range_elt_type
// nufft::cauchy::phi(domain_elt_type y, domain_elt_type x)
// {
//     return 1.0/(y - x);
// }

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
		for (index_type col {0}; col < p; ++col) {
			SS(row, col) = 0.0;
		}
	}

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

void
nufft::cauchy::apply_SS_translation(
    vector_type<range_elt_type> const & input,
    vector_type<range_elt_type> & output,
    domain_elt_type delta,
    integer_type p)
{
    // TODO: note: when I finally try to combine the coefficients and
    // deltas into a single array, I could use an inline struct to
    // make things a bit simpler to understand.
    
    vector_type<domain_elt_type> deltas(p, 0);
    deltas[0] = 1;

    index_type update_deltas_iter {1};
    auto const update_deltas = [delta, &deltas, &update_deltas_iter, p] () {
        if (update_deltas_iter >= p) {
            return;
        }
        
        deltas[update_deltas_iter] = 1;
        for (index_type i {update_deltas_iter - 1}; i >= 0; --i) {
            deltas[i] *= -delta;
        }
        
        ++update_deltas_iter;
    };

    // TODO: we could try using compensated summation for the
    // accumulation of these coefficients, but it would end up being
    // pretty complicated and inefficient. It might be necessary to
    // maintain an array of compensation terms in parallel with the
    // coefficients themselves, doubling the memory requirements for
    // this part of the algorithm.

    vector_type<domain_elt_type> coefs(p, 0);
    coefs[0] = 1;

    index_type update_coefs_iter {1};
    auto const update_coefs = [&coefs, &update_coefs_iter, p] () {
        if (update_coefs_iter >= p) {
            return;
        }
        for (index_type i {update_coefs_iter}; i > 0; --i) {
            coefs[i] += coefs[i - 1];
        }
        ++update_coefs_iter;
    };

    for (index_type i {0}; i < p; ++i) {
        output[i] = 0;
        for (index_type j {0}; j <= i; ++j) {
            output[i] += coefs[j] * deltas[j] * input[j];
        }

        // TODO: unsure if using compensated summation here helps, but
        // just in case we need it -- what's commented below works.

        // output[i] = 0;
        // range_elt_type comp {0};
        // for (index_type j {0}; j <= i; ++j) {
        //     range_elt_type next_term {coefs[j] * deltas[j] * input[j]};
        //     next_term -= comp;
        //     range_elt_type tmp_sum = output[i] + next_term;
        //     comp = (tmp_sum - output[i]) - next_term;
        //     output[i] = tmp_sum;
        // }

        update_deltas();
        update_coefs();
    }
}

void
nufft::cauchy::apply_SR_translation(
    vector_type<range_elt_type> const & input,
    vector_type<range_elt_type> & output,
    domain_elt_type delta,
    integer_type p)
{
    domain_elt_type const delta_recip = 1.0/delta;
    
    vector_type<domain_elt_type> deltas(p, delta_recip);
    for (index_type i {1}; i < p; ++i) {
        deltas[i] *= deltas[i - 1];
    }

    auto const update_deltas = [delta_recip, &deltas] () {
        for (auto & delta: deltas) {
            delta *= -delta_recip;
        }
    };

    vector_type<domain_elt_type> coefs(p, 1);

    auto const update_coefs = [&coefs, p] () {
        for (index_type i {1}; i < p; ++i) {
            coefs[i] += coefs[i - 1];
        }
    };

    for (index_type i {0}; i < p; ++i) {
		// std::cout << std::endl << "row " << i << std::endl << std::endl << "deltas:";
		// for (auto delta: deltas) std::cout << ' ' << delta;
		// std::cout << std::endl << "coefs:";
		// for (auto coef: coefs) std::cout << ' ' << coef;
		// std::cout << std::endl;

        output[i] = 0;
        for (index_type j {0}; j < p; ++j) {
            output[i] += coefs[j] * deltas[j] * input[j];
        }
		update_deltas();
		update_coefs();
    }
}

void
nufft::cauchy::apply_RR_translation(
	vector_type<range_elt_type> const & input,
	vector_type<range_elt_type> & output,
	domain_elt_type delta,
	integer_type p)
{
	vector_type<domain_elt_type> coefs(p, 1);

    index_type end {p - 1};
	auto const update_coefs = [&coefs, &end] () {
		for (index_type i {1}; i < end; ++i) {
			coefs[i] += coefs[i - 1];
		}
		--end;
	};

	for (index_type i {0}; i < p; ++i) {
		output[i] = 0;
		domain_elt_type this_delta {1};
		index_type k = 0;
		for (index_type j {i}; j < p; ++j) {
			output[i] += this_delta * coefs[k] * input[j];
			this_delta *= delta;
			++k;
		}
		update_coefs();
	}
}

// Local Variables:
// indent-tabs-mode: nil
// End:
