#define BOOST_TEST_MODULE cauchy

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/test/included/unit_test.hpp>

#include "cauchy.hpp"

BOOST_AUTO_TEST_CASE (get_SS_matrix_works) {
    nufft::domain_elt_type const delta = 0.11;
    nufft::integer_type const p = 5;
    auto const SS = nufft::cauchy::get_SS_matrix(delta, p);

    BOOST_CHECK_EQUAL(SS(0, 0), 1.0);
    BOOST_CHECK_EQUAL(SS(0, 1), 0.0);
    BOOST_CHECK_EQUAL(SS(0, 2), 0.0);
    BOOST_CHECK_EQUAL(SS(0, 3), 0.0);
    BOOST_CHECK_EQUAL(SS(0, 4), 0.0);
    BOOST_CHECK_EQUAL(SS(1, 0), -0.11);
    BOOST_CHECK_EQUAL(SS(1, 1), 1.0);
    BOOST_CHECK_EQUAL(SS(1, 2), 0.0);
    BOOST_CHECK_EQUAL(SS(1, 3), 0.0);
    BOOST_CHECK_EQUAL(SS(1, 4), 0.0);
    BOOST_CHECK_EQUAL(SS(2, 0), 0.0121);
    BOOST_CHECK_EQUAL(SS(2, 1), -0.22);
    BOOST_CHECK_EQUAL(SS(2, 2), 1.0);
    BOOST_CHECK_EQUAL(SS(2, 3), 0.0);
    BOOST_CHECK_EQUAL(SS(2, 4), 0.0);
    BOOST_CHECK_EQUAL(SS(3, 0), -0.001331);
    BOOST_CHECK_EQUAL(SS(3, 1), 0.0363);
    BOOST_CHECK_EQUAL(SS(3, 2), -0.33);
    BOOST_CHECK_EQUAL(SS(3, 3), 1.0);
    BOOST_CHECK_EQUAL(SS(3, 4), 0.0);
    BOOST_CHECK_EQUAL(SS(4, 0), 0.00014641);
    BOOST_CHECK_EQUAL(SS(4, 1), -0.005324);
    BOOST_CHECK_EQUAL(SS(4, 2), 0.0726);
    BOOST_CHECK_EQUAL(SS(4, 3), -0.44);
    BOOST_CHECK_EQUAL(SS(4, 4), 1.0);
}

BOOST_AUTO_TEST_CASE (get_SR_matrix_works) {
    nufft::domain_elt_type const delta = 0.5;
    nufft::integer_type const p = 5;
    auto const SR = nufft::cauchy::get_SR_matrix(delta, p);
       
    BOOST_CHECK_EQUAL(SR(0, 0), 2.0);
    BOOST_CHECK_EQUAL(SR(0, 1), 4.0);
    BOOST_CHECK_EQUAL(SR(0, 2), 8.0);
    BOOST_CHECK_EQUAL(SR(0, 3), 16.0);
    BOOST_CHECK_EQUAL(SR(0, 4), 32.0);
    BOOST_CHECK_EQUAL(SR(1, 0), -4.0);
    BOOST_CHECK_EQUAL(SR(1, 1), -16.0);
    BOOST_CHECK_EQUAL(SR(1, 2), -48.0);
    BOOST_CHECK_EQUAL(SR(1, 3), -128.0);
    BOOST_CHECK_EQUAL(SR(1, 4), -320.0);
    BOOST_CHECK_EQUAL(SR(2, 0), 8.0);
    BOOST_CHECK_EQUAL(SR(2, 1), 48.0);
    BOOST_CHECK_EQUAL(SR(2, 2), 192.0);
    BOOST_CHECK_EQUAL(SR(2, 3), 640.0);
    BOOST_CHECK_EQUAL(SR(2, 4), 1920.0);
    BOOST_CHECK_EQUAL(SR(3, 0), -16.0);
    BOOST_CHECK_EQUAL(SR(3, 1), -128.0);
    BOOST_CHECK_EQUAL(SR(3, 2), -640.0);
    BOOST_CHECK_EQUAL(SR(3, 3), -2560.0);
    BOOST_CHECK_EQUAL(SR(3, 4), -8960.0);
    BOOST_CHECK_EQUAL(SR(4, 0), 32.0);
    BOOST_CHECK_EQUAL(SR(4, 1), 320.0);
    BOOST_CHECK_EQUAL(SR(4, 2), 1920.0);
    BOOST_CHECK_EQUAL(SR(4, 3), 8960.0);
    BOOST_CHECK_EQUAL(SR(4, 4), 35840.0);
}

BOOST_AUTO_TEST_CASE (get_RR_matrix_works) {
    nufft::domain_elt_type const delta = 0.1;
    nufft::integer_type const p = 5;
    auto const RR = nufft::cauchy::get_RR_matrix(delta, p);
    auto const tol = 1e-13;
    
    BOOST_CHECK_CLOSE(RR(0, 0), 1.0, tol);
    BOOST_CHECK_CLOSE(RR(0, 1), 0.1, tol);
    BOOST_CHECK_CLOSE(RR(0, 2), 0.01, tol);
    BOOST_CHECK_CLOSE(RR(0, 3), 0.001, tol);
    BOOST_CHECK_CLOSE(RR(0, 4), 0.0001, tol);
    BOOST_CHECK_CLOSE(RR(1, 0), 0.0, tol);
    BOOST_CHECK_CLOSE(RR(1, 1), 1.0, tol);
    BOOST_CHECK_CLOSE(RR(1, 2), 0.2, tol);
    BOOST_CHECK_CLOSE(RR(1, 3), 0.03, tol);
    BOOST_CHECK_CLOSE(RR(1, 4), 0.004, tol);
    BOOST_CHECK_CLOSE(RR(2, 0), 0.0, tol);
    BOOST_CHECK_CLOSE(RR(2, 1), 0.0, tol);
    BOOST_CHECK_CLOSE(RR(2, 2), 1.0, tol);
    BOOST_CHECK_CLOSE(RR(2, 3), 0.3, tol);
    BOOST_CHECK_CLOSE(RR(2, 4), 0.06, tol);
    BOOST_CHECK_CLOSE(RR(3, 0), 0.0, tol);
    BOOST_CHECK_CLOSE(RR(3, 1), 0.0, tol);
    BOOST_CHECK_CLOSE(RR(3, 2), 0.0, tol);
    BOOST_CHECK_CLOSE(RR(3, 3), 1.0, tol);
    BOOST_CHECK_CLOSE(RR(3, 4), 0.4, tol);
    BOOST_CHECK_CLOSE(RR(4, 0), 0.0, tol);
    BOOST_CHECK_CLOSE(RR(4, 1), 0.0, tol);
    BOOST_CHECK_CLOSE(RR(4, 2), 0.0, tol);
    BOOST_CHECK_CLOSE(RR(4, 3), 0.0, tol);
    BOOST_CHECK_CLOSE(RR(4, 4), 1.0, tol);
}

// Local Variables:
// indent-tabs-mode: nil
// End:
