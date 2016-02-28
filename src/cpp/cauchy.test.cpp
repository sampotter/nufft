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

BOOST_AUTO_TEST_CASE (apply_SS_translation_works) {
    using namespace nufft;

    auto const test = [] (integer_type p,
                          domain_elt_type delta,
                          vector_type<range_elt_type> input,
                          vector_type<range_elt_type> expected) {
        vector_type<range_elt_type> actual(p, 0);
        cauchy::apply_SS_translation(input, actual, delta, p);
        BOOST_CHECK_EQUAL_COLLECTIONS(
            std::cbegin(expected), std::cend(expected),
            std::cbegin(actual), std::cend(actual));
    };

    // jl: 1
    
    test(5,
         0.03125,
         {
             2.4861518709052617,
             0.020569004905276696,
             0.0007928404106781417,
             1.630627105589163e-5,
             5.371148724791965e-7,
         },
         {
             2.4861518709052617,
             -0.05712324106051273,
             0.0019351602905292677,
             -7.363309483772256e-5,
             3.004494601176515e-6,
         });

    // jl: 2
    
    test(5,
         -0.03125,
         {
             -3.012747815605832,
             -0.04650204578472563,
             -0.0007321334059643607,
             -1.432903300758953e-5,
             -3.207116545684505e-7,
         },
         {
             -3.012747815605832,
             -0.14065041502240788,
             -0.006580647806187282,
             -0.0003111447689103608,
             -1.4951384430472649e-5,
         });

    // jl: 3

    test(5,
         0.03125,
         {-0.677907963761178, -0.010669730365930353, -0.00016102851253456228, -2.636621895753053e-6, -4.309949888478858e-8},
         {-0.677907963761178, 0.01051489350160646, -0.0001561898605244406, 1.8889347185669985e-6, -1.094296414294652e-9});

    // jl: 4

    test(5,
         -0.03125,
         {2.0485153711070385, 0.0807072914574851, 0.0029544505683835372, 7.188967662504316e-5, 2.664288800980216e-6},
         {2.0485153711070385, 0.14472339680458007, 0.009999159576573075, 0.0006478322882308731, 4.076731296366488e-5});

    // jl: 5

    test(5,
         0.03125,
         {-3.2571880674423395, -0.04634763745062983, -0.0015537954032302653, -2.322815826153293e-5, -1.1950284777945614e-6},
         {-3.2571880674423395, 0.05543948965694328, -0.0018379157846775602, 8.605755801365173e-5, -4.844404652294862e-6});

    // jl: 6

    test(5,
         -0.03125,
         {1.0799899351977786, -0.04287100926847032, -0.0001411436811340207, -1.7678817651878102e-5, -3.088856639769239e-7},
         {1.0799899351977786, -0.009121323793539735, -0.0017659040893218349, -0.00012355102050280227, -7.549070462523535e-6});

    // jl: 7

    test(5,
         0.03125,
         {2.7946493161744024, -0.0005367675895145245, 0.001545495684069563, -5.091590329332566e-6, 1.0051820128414121e-6},
         {2.7946493161744024, -0.0878695587199646, 0.0043081933812407856, -0.0002368403008465901, 1.3427978241442805e-5});

    // jl: 8

    test(5,
         -0.03125,
         {-1.2159664996435338, 0.026812480550795402, -0.001231921685557535, 2.5576366733776963e-5, -8.958555252600458e-7},
         {-1.2159664996435338, -0.011186472563065028, -0.0007436089359409608, -4.847245482383787e-5, -2.803728950436288e-6});

    // jl: 9

    test(5,
         0.03125,
         {-2.4529782752529505, 0.053470787923557264, -0.0013281285434251682, 3.496740463254517e-5, -9.412339369350094e-7},
         {-2.4529782752529505, 0.13012635902521197, -0.007065539385574207, 0.000390991110727411, -2.1960700871396103e-5});

    // jl: 10

    test(5,
         -0.03125,
         {-1.0214320951534155, 0.06287557601103148, -0.0006114894121579241, 2.8789866476002085e-5, -2.3184455337913535e-7},
         {-1.0214320951534155, 0.03095582303748725, 0.0023207418081082863, 0.0001244968894177887, 6.485070638941568e-6});

    // jl: 11

    test(5,
         0.03125,
         {2.054880736142086, -0.0006315933683593428, 0.0008403098049539753, 1.489521054286529e-5, 7.403795054423231e-7},
         {2.054880736142086, -0.06484661637279954, 0.0028865038593651905, -0.0001284441882707088, 5.8389542321715906e-6});

    // jl: 12

    test(5,
         -0.03125,
         {0.2923633157755118, 0.006464584596702668, 8.752448755266928e-6, 1.3374243427377594e-6, -8.980645928071882e-9},
         {0.2923633157755118, 0.015600938214687412, 0.0006983000366112069, 3.0019399429259608e-5, 1.2774345235561972e-6});

    // jl: 13

    test(5,
         0.03125,
         {0.3856001651325943, 0.019237328880656015, 0.00011544582509588966, 2.265901077490516e-5, 6.307772959602123e-8},
         {0.3856001651325943, 0.007187323720262442, -0.0007103245686828121, 5.6427743462765574e-5, -4.073428010083886e-6});

    // jl: 14

    test(5,
         -0.03125,
         {0.02784909055806617, -0.00028272100302225225, -5.586057995007658e-5, -4.835299658999943e-7, -2.0098981938649292e-8},
         {0.02784909055806617, 0.0005875630769173155, -4.633426514085585e-5, -5.698856727945444e-6, -4.158011921098688e-7});

    // jl: 15

    test(5,
         0.03125,
         {2.6616992530831087, -0.024447447562202837, 0.0009878420377224, -2.5400387140970277e-5, 7.478195016045957e-7},
         {2.6616992530831087, -0.10762554922104998, 0.005115123187199051, -0.0002708625746085543, 1.523370661341409e-5});

    // jl: 16

    test(5,
         -0.03125,
         {-1.7721947719892979, 0.018082346378502652, -0.000350129603507875, 1.3390387291094516e-5, -6.675044600429963e-8},
         {-1.7721947719892979, -0.037298740246162906, -0.0009506419118722582, -2.0541731288899247e-5, 7.272835526999819e-8});
}

BOOST_AUTO_TEST_CASE (apply_SR_translation_works) {
    using namespace nufft;

    vector_type<range_elt_type> const expected_translation = {
        -3.385028620601427e8,
        2.6097095714834618e10,
        -1.1121491687723486e12,
        3.4610120705280227e13,
        -8.780581586249925e14,
        1.92454725767926e16,
        -3.775129404161203e17,
        6.783873501077648e18,
        -1.1352616818989236e20,
        1.7906817290346446e21,
    };

    vector_type<range_elt_type> const input = {
        -1.7083192093163444,
        -0.09075075558871018,
        -0.5493915134960792,
        0.40217713640945074,
        0.7505576381427779,
        0.25589485008199525,
        -0.0627352669310831,
        0.6638888318346806,
        -1.0816187763468461,
        -0.19039037675098727,
    };

    domain_elt_type const delta {0.125};
    integer_type const p {10};
    vector_type<range_elt_type> actual_translation(p, 0);
    cauchy::apply_SR_translation(input, actual_translation, delta, p);

    BOOST_CHECK_EQUAL_COLLECTIONS(
        std::cbegin(expected_translation),
        std::cend(expected_translation),
        std::cbegin(actual_translation),
        std::cend(actual_translation));
}

BOOST_AUTO_TEST_CASE (apply_RR_translation_works) {
    using namespace nufft;

    {
        vector_type<range_elt_type> const expected_translation = {
            -1.4812594873128075,
            0.06294450407242579,
            -0.6726878191271809,
            -1.8886946673495988,
            -1.1373560213821825,
            0.8691599774079222,
            0.955193186679812,
            2.3288361796679604,
            2.0815631773842287,
            1.0199736784144198,
        };

        vector_type<range_elt_type> const input = {
            -1.496250990422328,
            0.15241036870181457,
            -0.08581985807615482,
            -1.2048534449466162,
            -1.5842538183247659,
            0.7206196297530001,
            -0.3391940120389192,
            0.821008196391843,
            0.9340927891680064,
            1.0199736784144198,
        };

        domain_elt_type const delta {0.125};
        integer_type const p {10};
        vector_type<range_elt_type> actual_translation(p, 0);
        cauchy::apply_RR_translation(input, actual_translation, delta, p);

        BOOST_CHECK_EQUAL_COLLECTIONS(
            std::cbegin(expected_translation),
            std::cend(expected_translation),
            std::cbegin(actual_translation),
            std::cend(actual_translation));
    }

    {
        vector_type<range_elt_type> const expected_translation = {
            5.731654136683816,
            39.95857554366885,
            242.50121090735004,
            1224.8701007550403,
            3732.2818319838607,
        };

        vector_type<range_elt_type> const input = {
            4.685945552052858,
            27.935135675696912,
            149.53847782084546,
            758.3348717570577,
            3732.2818319838607,
        };

        domain_elt_type const delta {0.03125};
        integer_type const p {5};
        vector_type<range_elt_type> actual_translation(p, 0);
        cauchy::apply_RR_translation(input, actual_translation, delta, p);

        BOOST_CHECK_EQUAL_COLLECTIONS(
            std::cbegin(expected_translation),
            std::cend(expected_translation),
            std::cbegin(actual_translation),
            std::cend(actual_translation));
    }
}

// Local Variables:
// indent-tabs-mode: nil
// End:
