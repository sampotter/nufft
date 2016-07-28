#define BOOST_TEST_MODULE nufft

#include <boost/test/included/unit_test.hpp>
#include <iostream>

#include "nufft.hpp"

BOOST_AUTO_TEST_CASE (basic_test) {
	double values_double[] = {
		0.00000000,
		+0.00000000e+00,
		0.07401407,
		+2.77555756e-17,
		0.05573387,
		+4.16333634e-17,
		0.06778126,
		+9.71445147e-17,
		0.05759893,
		+0.00000000e+00,
		0.06778126,
		+3.46944695e-17,
		0.05573387,
		+4.16333634e-17,
		0.07401407,
		+9.71445147e-17,
		0.00000000,
		+0.00000000e+00,
		-0.07401407,
		-2.77555756e-17,
		-0.05573387,
		-4.16333634e-17,
		-0.06778126,
		-9.71445147e-17,
		-0.05759893,
		+0.00000000e+00,
		-0.06778126,
		-3.46944695e-17,
		-0.05573387,
		-4.16333634e-17,
		-0.07401407,
		-9.71445147e-17,
	};
	std::complex<double> values[16];
	for (int i = 0; i < 16; ++i) {
		values[i] = ((std::complex<double> *) values_double)[i];
	}
	for (int i = 0; i < 32; ++i) {
		if (i % 2 == 0) {
			BOOST_CHECK_EQUAL(values_double[i], values[i/2].real());
		} else {
			BOOST_CHECK_EQUAL(values_double[i], values[i/2].imag());
		}
	}
	double nodes[] = {
		0.06144733,
		0.12742735,
		0.19382462,
		0.76838064,
		0.81637296,
		0.97971646,
		1.04936551,
		1.21592974,
		1.25491235,
		1.45162833,
		1.53560262,
		1.59485713,
		1.75060987,
		1.79657517,
		1.8218675,
		2.54129315,
		2.6901106,
		2.87683413,
		3.01196423,
		3.10353108,
		3.23054149,
		3.59794781,
		3.83246419,
		4.30807601,
		4.61761236,
		5.31786545,
		5.62484061,
		5.66583099,
		5.67175258,
		6.07424511,
		6.11556149,
		6.19334216
	};
	std::complex<double> output[32];
	int L = 4;
	int p = 4;
	int n = 3;
	compute_P_ddi(values_double, nodes, 16, 32, L, p, n, reinterpret_cast<double *>(output));

	for (auto z: output) {
		std::cout << z << std::endl;
	}
}

BOOST_AUTO_TEST_CASE (square_K380) {
	double values[] = {
		0.00000000e+00,
		-6.29978525e-19,
		3.10260257e-03,
		+8.08015935e-19,
		2.37580227e-03,
		+1.44712458e-18,
		2.80582680e-03,
		+3.89856276e-18,
		2.49974312e-03,
		+1.83515484e-18,
		2.73752772e-03,
		+2.61147439e-18,
		2.54303940e-03,
		+3.23250147e-18,
		2.70762297e-03,
		+5.23088164e-18,
		2.56493271e-03,
		+5.82045377e-19,
		2.69090427e-03,
		+1.27601606e-18,
		2.57811503e-03,
		+2.87806331e-18,
		2.68024591e-03,
		+4.95663978e-18,
		2.58690892e-03,
		+2.20948990e-18,
		2.67286842e-03,
		+3.00883325e-18,
		2.59318510e-03,
		+3.82893288e-18,
		2.66746550e-03,
		+6.21400634e-18,
		2.59788390e-03,
		+4.90287635e-18,
		2.66334275e-03,
		+5.52977446e-18,
		2.60152940e-03,
		+6.50385487e-18,
		2.66009710e-03,
		+8.30002153e-18,
		2.60443670e-03,
		-6.23020124e-19,
		2.65747863e-03,
		+6.33903131e-19,
		2.60680662e-03,
		+1.86825138e-18,
		2.65532415e-03,
		+3.35045857e-18,
		2.60877313e-03,
		+7.74066432e-19,
		2.65352261e-03,
		+1.73646670e-18,
		2.61042907e-03,
		+2.31649527e-18,
		2.65199583e-03,
		+3.99475054e-18,
		2.61184076e-03,
		+1.44655201e-18,
		2.65068715e-03,
		+2.07150286e-18,
		2.61305686e-03,
		+3.45992662e-18,
		2.64955456e-03,
		+5.39567844e-18,
		2.61411385e-03,
		+2.43895872e-18,
		2.64856623e-03,
		+3.15830702e-18,
		2.61503964e-03,
		+4.02905376e-18,
		2.64769759e-03,
		+6.05351185e-18,
		2.61585592e-03,
		+4.51833392e-18,
		2.64692941e-03,
		+5.02517801e-18,
		2.61657980e-03,
		+6.28494432e-18,
		2.64624642e-03,
		+7.94941810e-18,
		2.61722496e-03,
		-3.98334662e-19,
		2.64563632e-03,
		+6.28776014e-19,
		2.61780249e-03,
		+1.84165265e-18,
		2.64508912e-03,
		+3.32482327e-18,
		2.61832141e-03,
		+5.52961331e-19,
		2.64459662e-03,
		+1.75862628e-18,
		2.61878919e-03,
		+2.23410291e-18,
		2.64415202e-03,
		+3.84277382e-18,
		2.61921202e-03,
		+1.75567706e-18,
		2.64374965e-03,
		+2.48856709e-18,
		2.61959512e-03,
		+3.72184682e-18,
		2.64338474e-03,
		+5.33317828e-18,
		2.61994285e-03,
		+2.65059274e-18,
		2.64305325e-03,
		+3.30162319e-18,
		2.62025895e-03,
		+4.31278260e-18,
		2.64275175e-03,
		+6.19704648e-18,
		2.62054659e-03,
		+4.69541886e-18,
		2.64247729e-03,
		+5.05741185e-18,
		2.62080849e-03,
		+6.25827640e-18,
		2.64222736e-03,
		+7.60378758e-18,
		2.62104700e-03,
		-2.73755024e-19,
		2.64199977e-03,
		+7.22222481e-19,
		2.62126414e-03,
		+1.93374190e-18,
		2.64179264e-03,
		+3.14442868e-18,
		2.62146167e-03,
		+3.21434036e-19,
		2.64160433e-03,
		+1.61608805e-18,
		2.62164111e-03,
		+2.34047043e-18,
		2.64143342e-03,
		+3.78822273e-18,
		2.62180381e-03,
		+2.10077268e-18,
		2.64127866e-03,
		+3.11993229e-18,
		2.62195090e-03,
		+4.17043905e-18,
		2.64113898e-03,
		+5.45173088e-18,
		2.62208340e-03,
		+2.62989370e-18,
		2.64101343e-03,
		+3.45343546e-18,
		2.62220220e-03,
		+4.52527816e-18,
		2.64090120e-03,
		+6.32475387e-18,
		2.62230804e-03,
		+4.86536307e-18,
		2.64080157e-03,
		+5.47241542e-18,
		2.62240161e-03,
		+6.73100255e-18,
		2.64071393e-03,
		+8.05705706e-18,
		2.62248345e-03,
		-8.61459030e-20,
		2.64063775e-03,
		+9.40285825e-19,
		2.62255408e-03,
		+2.23476491e-18,
		2.64057259e-03,
		+3.49496968e-18,
		2.62261388e-03,
		+2.97719809e-19,
		2.64051806e-03,
		+1.44055837e-18,
		2.62266322e-03,
		+2.30858346e-18,
		2.64047385e-03,
		+3.77288673e-18,
		2.62270236e-03,
		+2.29742531e-18,
		2.64043972e-03,
		+3.30436045e-18,
		2.62273153e-03,
		+4.35382377e-18,
		2.64041547e-03,
		+5.32419489e-18,
		2.62275089e-03,
		+2.64711933e-18,
		2.64040097e-03,
		+3.49362743e-18,
		2.62276054e-03,
		+4.86402876e-18,
		2.64039615e-03,
		+6.33113332e-18,
		2.62276054e-03,
		+4.70038439e-18,
		2.64040097e-03,
		+5.66053339e-18,
		2.62275089e-03,
		+6.81046055e-18,
		2.64041547e-03,
		+7.88446601e-18,
		2.62273153e-03,
		+4.03951894e-20,
		2.64043972e-03,
		+1.07649459e-18,
		2.62270236e-03,
		+2.19148187e-18,
		2.64047385e-03,
		+3.30044247e-18,
		2.62266322e-03,
		-2.13530542e-19,
		2.64051806e-03,
		+1.22078773e-18,
		2.62261388e-03,
		+2.12186237e-18,
		2.64057259e-03,
		+3.54792042e-18,
		2.62255408e-03,
		+2.24574669e-18,
		2.64063775e-03,
		+3.21494103e-18,
		2.62248345e-03,
		+4.39760405e-18,
		2.64071393e-03,
		+5.39274471e-18,
		2.62240161e-03,
		+2.83315458e-18,
		2.64080157e-03,
		+3.78860197e-18,
		2.62230804e-03,
		+4.76751052e-18,
		2.64090120e-03,
		+6.09471587e-18,
		2.62220220e-03,
		+4.38985709e-18,
		2.64101343e-03,
		+5.69414609e-18,
		2.62208340e-03,
		+6.70405971e-18,
		2.64113898e-03,
		+7.52286288e-18,
		2.62195090e-03,
		+1.50409678e-19,
		2.64127866e-03,
		+1.21354035e-18,
		2.62180381e-03,
		+2.03530405e-18,
		2.64143342e-03,
		+2.99966882e-18,
		2.62164111e-03,
		-1.70303103e-19,
		2.64160433e-03,
		+7.69508889e-19,
		2.62146167e-03,
		+1.89127735e-18,
		2.64179264e-03,
		+2.76860071e-18,
		2.62126414e-03,
		+2.25589600e-18,
		2.64199977e-03,
		+3.45991402e-18,
		2.62104700e-03,
		+4.21621111e-18,
		2.64222736e-03,
		+4.69747585e-18,
		2.62080849e-03,
		+2.95876898e-18,
		2.64247729e-03,
		+4.09402820e-18,
		2.62054659e-03,
		+4.79978463e-18,
		2.64275175e-03,
		+5.44838802e-18,
		2.62025895e-03,
		+4.16787006e-18,
		2.64305325e-03,
		+5.47625683e-18,
		2.61994285e-03,
		+6.34698390e-18,
		2.64338474e-03,
		+7.13023552e-18,
		2.61959512e-03,
		+3.45809407e-19,
		2.64374965e-03,
		+1.22024965e-18,
		2.61921202e-03,
		+2.04836965e-18,
		2.64415202e-03,
		+2.79651946e-18,
		2.61878919e-03,
		-4.21965565e-19,
		2.64459662e-03,
		+7.05990727e-19,
		2.61832141e-03,
		+1.84277451e-18,
		2.64508912e-03,
		+2.79476621e-18,
		2.61780249e-03,
		+2.19667533e-18,
		2.64563632e-03,
		+3.36665997e-18,
		2.61722496e-03,
		+4.28126689e-18,
		2.64624642e-03,
		+4.39462722e-18,
		2.61657980e-03,
		+3.15760180e-18,
		2.64692941e-03,
		+4.32294123e-18,
		2.61585592e-03,
		+5.04239741e-18,
		2.64769759e-03,
		+5.70681650e-18,
		2.61503964e-03,
		+3.94459569e-18,
		2.64856623e-03,
		+5.48989219e-18,
		2.61411385e-03,
		+6.12421233e-18,
		2.64955456e-03,
		+6.87895714e-18,
		2.61305686e-03,
		+5.21053631e-19,
		2.65068715e-03,
		+1.45251726e-18,
		2.61184076e-03,
		+2.19239680e-18,
		2.65199583e-03,
		+2.86420364e-18,
		2.61042907e-03,
		-6.53434807e-19,
		2.65352261e-03,
		+5.73492439e-19,
		2.60877313e-03,
		+1.86881795e-18,
		2.65532415e-03,
		+2.46724892e-18,
		2.60680662e-03,
		+2.44283276e-18,
		2.65747863e-03,
		+3.78917385e-18,
		2.60443670e-03,
		+4.56604902e-18,
		2.66009710e-03,
		+4.96903544e-18,
		2.60152940e-03,
		+3.07962691e-18,
		2.66334275e-03,
		+4.57837676e-18,
		2.59788390e-03,
		+5.30442003e-18,
		2.66746550e-03,
		+6.07680463e-18,
		2.59318510e-03,
		+3.78116139e-18,
		2.67286842e-03,
		+5.41248285e-18,
		2.58690892e-03,
		+6.22228023e-18,
		2.68024591e-03,
		+7.42243057e-18,
		2.57811503e-03,
		+6.38577071e-19,
		2.69090427e-03,
		+1.37750853e-18,
		2.56493271e-03,
		+2.35743502e-18,
		2.70762297e-03,
		+3.27514868e-18,
		2.54303940e-03,
		-1.34612411e-18,
		2.73752772e-03,
		+9.40114984e-20,
		2.49974312e-03,
		+1.60214604e-18,
		2.80582680e-03,
		+3.46518490e-18,
		2.37580227e-03,
		+2.49703507e-18,
		3.10260257e-03,
		+4.31905712e-18,
		2.19122965e-18,
		+1.11968550e-18,
		-3.10260257e-03,
		-5.41683035e-18,
		-2.37580227e-03,
		-1.33667494e-18,
		-2.80582680e-03,
		-2.66143884e-18,
		-2.49974312e-03,
		-2.67285344e-18,
		-2.73752772e-03,
		-4.57641853e-18,
		-2.54303940e-03,
		-4.84839010e-18,
		-2.70762297e-03,
		-5.52509431e-18,
		-2.56493271e-03,
		-6.46619427e-18,
		-2.69090427e-03,
		-8.40055630e-18,
		-2.57811503e-03,
		+6.38577071e-19,
		-2.68024591e-03,
		-7.29855089e-19,
		-2.58690892e-03,
		-1.78080046e-18,
		-2.67286842e-03,
		-3.40499293e-18,
		-2.59318510e-03,
		-8.52663081e-19,
		-2.66746550e-03,
		-1.64719793e-18,
		-2.59788390e-03,
		-2.46198318e-18,
		-2.66334275e-03,
		-4.24494000e-18,
		-2.60152940e-03,
		-1.52589997e-18,
		-2.66009710e-03,
		-2.27616939e-18,
		-2.60443670e-03,
		-3.58849116e-18,
		-2.65747863e-03,
		-5.21346333e-18,
		-2.60680662e-03,
		-2.06443533e-18,
		-2.65532415e-03,
		-2.70446576e-18,
		-2.60877313e-03,
		-3.59593909e-18,
		-2.65352261e-03,
		-5.50373721e-18,
		-2.61042907e-03,
		-4.45974825e-18,
		-2.65199583e-03,
		-5.09980802e-18,
		-2.61184076e-03,
		-6.03823194e-18,
		-2.65068715e-03,
		-7.70387342e-18,
		-2.61305686e-03,
		+5.21053631e-19,
		-2.64955456e-03,
		-5.50983265e-19,
		-2.61411385e-03,
		-1.86174021e-18,
		-2.64856623e-03,
		-3.05917115e-18,
		-2.61503964e-03,
		-5.23421569e-19,
		-2.64769759e-03,
		-1.57778083e-18,
		-2.61585592e-03,
		-2.21961754e-18,
		-2.64692941e-03,
		-3.89856264e-18,
		-2.61657980e-03,
		-1.73126592e-18,
		-2.64624642e-03,
		-2.61306485e-18,
		-2.61722496e-03,
		-3.62773244e-18,
		-2.64563632e-03,
		-5.38444669e-18,
		-2.61780249e-03,
		-2.18693216e-18,
		-2.64508912e-03,
		-2.93542689e-18,
		-2.61832141e-03,
		-3.72965828e-18,
		-2.64459662e-03,
		-5.69892693e-18,
		-2.61878919e-03,
		-4.40009646e-18,
		-2.64415202e-03,
		-4.85257319e-18,
		-2.61921202e-03,
		-6.13359115e-18,
		-2.64374965e-03,
		-7.03673026e-18,
		-2.61959512e-03,
		+3.45809407e-19,
		-2.64338474e-03,
		-6.97698923e-19,
		-2.61994285e-03,
		-1.86547349e-18,
		-2.64305325e-03,
		-3.15446608e-18,
		-2.62025895e-03,
		-4.73428297e-19,
		-2.64275175e-03,
		-1.86890881e-18,
		-2.62054659e-03,
		-2.28594708e-18,
		-2.64247729e-03,
		-3.84235732e-18,
		-2.62080849e-03,
		-2.10245752e-18,
		-2.64222736e-03,
		-2.74182044e-18,
		-2.62104700e-03,
		-3.92396197e-18,
		-2.64199977e-03,
		-5.33307711e-18,
		-2.62126414e-03,
		-2.37071672e-18,
		-2.64179264e-03,
		-2.97965872e-18,
		-2.62146167e-03,
		-4.18953162e-18,
		-2.64160433e-03,
		-5.88889514e-18,
		-2.62164111e-03,
		-4.51942495e-18,
		-2.64143342e-03,
		-5.01487339e-18,
		-2.62180381e-03,
		-6.48323323e-18,
		-2.64127866e-03,
		-7.50480877e-18,
		-2.62195090e-03,
		+1.50409678e-19,
		-2.64113898e-03,
		-1.03009506e-18,
		-2.62208340e-03,
		-2.20274125e-18,
		-2.64101343e-03,
		-3.33851872e-18,
		-2.62220220e-03,
		-4.20685261e-19,
		-2.64090120e-03,
		-1.60598609e-18,
		-2.62230804e-03,
		-2.39428533e-18,
		-2.64080157e-03,
		-4.03221368e-18,
		-2.62240161e-03,
		-2.38707106e-18,
		-2.64071393e-03,
		-3.29060727e-18,
		-2.62248345e-03,
		-4.34010549e-18,
		-2.64063775e-03,
		-5.30651910e-18,
		-2.62255408e-03,
		-2.35832015e-18,
		-2.64057259e-03,
		-3.15368075e-18,
		-2.62261388e-03,
		-4.42244728e-18,
		-2.64051806e-03,
		-5.96779119e-18,
		-2.62266322e-03,
		-4.46137213e-18,
		-2.64047385e-03,
		-5.34607888e-18,
		-2.62270236e-03,
		-6.46401824e-18,
		-2.64043972e-03,
		-7.46786137e-18,
		-2.62273153e-03,
		+4.03951894e-20,
		-2.64041547e-03,
		-1.00843906e-18,
		-2.62275089e-03,
		-2.25128042e-18,
		-2.64040097e-03,
		-3.33277486e-18,
		-2.62276054e-03,
		-4.47534401e-20,
		-2.64039615e-03,
		-1.39277570e-18,
		-2.62276054e-03,
		-2.30736857e-18,
		-2.64040097e-03,
		-3.76840094e-18,
		-2.62275089e-03,
		-2.42410234e-18,
		-2.64041547e-03,
		-3.38966650e-18,
		-2.62273153e-03,
		-4.37131235e-18,
		-2.64043972e-03,
		-5.11781210e-18,
		-2.62270236e-03,
		-2.48202362e-18,
		-2.64047385e-03,
		-3.25034902e-18,
		-2.62266322e-03,
		-4.51625541e-18,
		-2.64051806e-03,
		-5.81695948e-18,
		-2.62261388e-03,
		-4.20623205e-18,
		-2.64057259e-03,
		-5.20655371e-18,
		-2.62255408e-03,
		-6.41417098e-18,
		-2.64063775e-03,
		-7.45494003e-18,
		-2.62248345e-03,
		-8.61459030e-20,
		-2.64071393e-03,
		-1.16345915e-18,
		-2.62240161e-03,
		-2.06064293e-18,
		-2.64080157e-03,
		-3.22515945e-18,
		-2.62230804e-03,
		+1.79027840e-19,
		-2.64090120e-03,
		-1.20692024e-18,
		-2.62220220e-03,
		-2.23723773e-18,
		-2.64101343e-03,
		-3.27153194e-18,
		-2.62208340e-03,
		-2.28009741e-18,
		-2.64113898e-03,
		-3.60769032e-18,
		-2.62195090e-03,
		-4.38277180e-18,
		-2.64127866e-03,
		-4.76216457e-18,
		-2.62180381e-03,
		-2.69973347e-18,
		-2.64143342e-03,
		-3.66094560e-18,
		-2.62164111e-03,
		-4.55231594e-18,
		-2.64160433e-03,
		-5.58155578e-18,
		-2.62146167e-03,
		-4.09871883e-18,
		-2.64179264e-03,
		-5.36562205e-18,
		-2.62126414e-03,
		-6.05224198e-18,
		-2.64199977e-03,
		-6.87957272e-18,
		-2.62104700e-03,
		-2.73755024e-19,
		-2.64222736e-03,
		-1.26125190e-18,
		-2.62080849e-03,
		-1.93551820e-18,
		-2.64247729e-03,
		-2.70327480e-18,
		-2.62054659e-03,
		+1.94523315e-19,
		-2.64275175e-03,
		-7.44607881e-19,
		-2.62025895e-03,
		-1.91113286e-18,
		-2.64305325e-03,
		-2.73111763e-18,
		-2.61994285e-03,
		-2.29920135e-18,
		-2.64338474e-03,
		-3.58256352e-18,
		-2.61959512e-03,
		-4.20104566e-18,
		-2.64374965e-03,
		-3.82098959e-18,
		-2.61921202e-03,
		-2.84146191e-18,
		-2.64415202e-03,
		-3.93130810e-18,
		-2.61878919e-03,
		-4.73592663e-18,
		-2.64459662e-03,
		-5.38723845e-18,
		-2.61832141e-03,
		-3.71831903e-18,
		-2.64508912e-03,
		-5.09516645e-18,
		-2.61780249e-03,
		-5.74771904e-18,
		-2.64563632e-03,
		-6.46087166e-18,
		-2.61722496e-03,
		-3.98334662e-19,
		-2.64624642e-03,
		-1.47965922e-18,
		-2.61657980e-03,
		-2.14483147e-18,
		-2.64692941e-03,
		-2.84854713e-18,
		-2.61585592e-03,
		+3.72066054e-19,
		-2.64769759e-03,
		-7.73585682e-19,
		-2.61503964e-03,
		-2.08719887e-18,
		-2.64856623e-03,
		-2.78946476e-18,
		-2.61411385e-03,
		-2.35475266e-18,
		-2.64955456e-03,
		-3.72998837e-18,
		-2.61305686e-03,
		-4.33768763e-18,
		-2.65068715e-03,
		-4.16500381e-18,
		-2.61184076e-03,
		-2.90517966e-18,
		-2.65199583e-03,
		-4.31641182e-18,
		-2.61042907e-03,
		-5.17157949e-18,
		-2.65352261e-03,
		-5.66488683e-18,
		-2.60877313e-03,
		-3.34073511e-18,
		-2.65532415e-03,
		-4.65714705e-18,
		-2.60680662e-03,
		-5.72954268e-18,
		-2.65747863e-03,
		-6.15352001e-18,
		-2.60443670e-03,
		-6.23020124e-19,
		-2.66009710e-03,
		-1.37144653e-18,
		-2.60152940e-03,
		-2.22603472e-18,
		-2.66334275e-03,
		-3.04551221e-18,
		-2.59788390e-03,
		+5.93356117e-19,
		-2.66746550e-03,
		-8.36958208e-19,
		-2.59318510e-03,
		-2.02448402e-18,
		-2.67286842e-03,
		-2.85747922e-18,
		-2.58690892e-03,
		-2.71581007e-18,
		-2.68024591e-03,
		-4.17954357e-18,
		-2.57811503e-03,
		-4.52586664e-18,
		-2.69090427e-03,
		-4.53683012e-18,
		-2.56493271e-03,
		-3.35972869e-18,
		-2.70762297e-03,
		-4.89843830e-18,
		-2.54303940e-03,
		-5.73106631e-18,
		-2.73752772e-03,
		-6.68445608e-18,
		-2.49974312e-03,
		-1.91282392e-18,
		-2.80582680e-03,
		-4.01474451e-18,
		-2.37580227e-03,
		-4.13092387e-18,
		-3.10260257e-03,
		-6.79495902e-18,		
	};
	double nodes[] = {
		0.01635386,
		0.01648199,
		0.02688253,
		0.0531834,
		0.08785123,
		0.13826541,
		0.15444706,
		0.1584932,
		0.16333774,
		0.18395088,
		0.19079139,
		0.21554478,
		0.23684837,
		0.28126913,
		0.29063021,
		0.37259512,
		0.3730457,
		0.37819027,
		0.38021724,
		0.40752094,
		0.4302922,
		0.43850044,
		0.46564138,
		0.49666382,
		0.50036821,
		0.51070172,
		0.51954359,
		0.53371768,
		0.53893182,
		0.5454686,
		0.57386102,
		0.57888838,
		0.59337926,
		0.61983973,
		0.65035086,
		0.66341019,
		0.67670209,
		0.71451814,
		0.72675074,
		0.72719017,
		0.73818586,
		0.74363857,
		0.75203503,
		0.7615933,
		0.76212421,
		0.79561992,
		0.82695357,
		0.84932972,
		0.89053724,
		0.91463046,
		0.91577682,
		0.94542806,
		0.98261586,
		0.99724441,
		1.00135786,
		1.02464769,
		1.04468607,
		1.05166558,
		1.07068059,
		1.07512694,
		1.08661401,
		1.08967515,
		1.10138753,
		1.10480311,
		1.13541177,
		1.13702892,
		1.14367582,
		1.15343072,
		1.15977802,
		1.15978391,
		1.1639762,
		1.17293736,
		1.17391943,
		1.19127818,
		1.19946401,
		1.23605574,
		1.24070066,
		1.27325241,
		1.31017864,
		1.3554046,
		1.36192315,
		1.36634333,
		1.39021814,
		1.40028786,
		1.4115707,
		1.4118071,
		1.42807018,
		1.46813317,
		1.46969072,
		1.49914654,
		1.53906605,
		1.54448356,
		1.55201682,
		1.55636229,
		1.55678291,
		1.56016193,
		1.5846043,
		1.59719723,
		1.63086435,
		1.63486903,
		1.64453129,
		1.6608502,
		1.67490028,
		1.69591825,
		1.73258806,
		1.77904546,
		1.81062594,
		1.83828316,
		1.84204134,
		1.84261823,
		1.84975576,
		1.86747,
		1.88316031,
		1.92363686,
		1.9348082,
		1.94370182,
		1.99130415,
		1.99827093,
		2.01163139,
		2.0199503,
		2.02450017,
		2.03549099,
		2.04209955,
		2.0657729,
		2.08314199,
		2.10897236,
		2.13836024,
		2.14184381,
		2.14370619,
		2.15472485,
		2.1678258,
		2.21195143,
		2.21581633,
		2.22296909,
		2.30246148,
		2.30492903,
		2.31452522,
		2.32999145,
		2.34491528,
		2.3480099,
		2.36782839,
		2.36932876,
		2.39288047,
		2.40484024,
		2.40781139,
		2.4695463,
		2.49888748,
		2.5079553,
		2.51425736,
		2.51491255,
		2.5343665,
		2.53977564,
		2.5594871,
		2.56112761,
		2.5907462,
		2.60010786,
		2.61439758,
		2.61595528,
		2.62424338,
		2.63354402,
		2.64240689,
		2.65332224,
		2.65415128,
		2.71419251,
		2.75319435,
		2.79197801,
		2.79928498,
		2.82848949,
		2.83128963,
		2.84778522,
		2.8591734,
		2.86433542,
		2.86464447,
		2.87031636,
		2.88143409,
		2.891364,
		2.89769197,
		2.8991116,
		2.9072677,
		2.90789434,
		2.93226384,
		2.93701839,
		2.95837726,
		2.97101939,
		2.9728294,
		2.98200699,
		3.02376262,
		3.02923448,
		3.04501317,
		3.06365518,
		3.06910546,
		3.09199723,
		3.11878641,
		3.13724885,
		3.21037003,
		3.21788795,
		3.22428924,
		3.23770678,
		3.27258571,
		3.28423663,
		3.32332206,
		3.32565645,
		3.33632666,
		3.35160003,
		3.3672491,
		3.38332439,
		3.39047524,
		3.39747986,
		3.41148846,
		3.43063584,
		3.44240514,
		3.48293435,
		3.5158438,
		3.52611477,
		3.5349924,
		3.53555939,
		3.53630225,
		3.57003157,
		3.58403061,
		3.58416932,
		3.58461892,
		3.58699807,
		3.6361829,
		3.65650599,
		3.65676837,
		3.66832459,
		3.69019328,
		3.71102108,
		3.7117796,
		3.74540537,
		3.7574264,
		3.77218449,
		3.77607563,
		3.79094813,
		3.85164485,
		3.86799907,
		3.8857311,
		3.90158232,
		3.95311051,
		3.95322132,
		3.98447648,
		4.01961406,
		4.03092654,
		4.04929957,
		4.05093993,
		4.05920176,
		4.07706376,
		4.08437966,
		4.08786399,
		4.1256478,
		4.13805497,
		4.15499599,
		4.15816263,
		4.15882713,
		4.15885268,
		4.1691038,
		4.17272847,
		4.17655808,
		4.22156712,
		4.25555366,
		4.33719349,
		4.34503486,
		4.36299445,
		4.38885796,
		4.39673058,
		4.4500306,
		4.47016977,
		4.49295334,
		4.51060654,
		4.51787236,
		4.52737812,
		4.54582357,
		4.56273082,
		4.5653396,
		4.5707031,
		4.59002912,
		4.59611204,
		4.64562346,
		4.6468254,
		4.66522633,
		4.68901652,
		4.70118552,
		4.70195301,
		4.70509489,
		4.71842221,
		4.72453255,
		4.72599778,
		4.7285024,
		4.74624988,
		4.74836893,
		4.77839985,
		4.78760437,
		4.79551428,
		4.81538073,
		4.85182945,
		4.85763206,
		4.88915131,
		4.90929618,
		4.92379918,
		4.93735211,
		4.94937129,
		4.95694798,
		4.96805122,
		4.98801657,
		5.01564169,
		5.02888032,
		5.08010416,
		5.11206086,
		5.11532452,
		5.12147498,
		5.13787444,
		5.14135546,
		5.15902269,
		5.18435696,
		5.20138532,
		5.22770212,
		5.27242799,
		5.27365948,
		5.27470331,
		5.28057801,
		5.31091221,
		5.31390317,
		5.32117882,
		5.3316226,
		5.3689122,
		5.37053527,
		5.38772459,
		5.41082631,
		5.47334317,
		5.47874857,
		5.50909506,
		5.53471769,
		5.53544686,
		5.54185011,
		5.57277439,
		5.58168418,
		5.5925987,
		5.6022795,
		5.61393788,
		5.65573248,
		5.66350756,
		5.66813737,
		5.67594689,
		5.69276315,
		5.69874552,
		5.73695069,
		5.74984315,
		5.7507408,
		5.78203939,
		5.79960248,
		5.80385911,
		5.8465879,
		5.8608906,
		5.87954402,
		5.88171896,
		5.88470367,
		5.89249195,
		5.93048391,
		5.93342113,
		5.94382067,
		5.9499357,
		5.99181368,
		6.00929607,
		6.02735823,
		6.06219776,
		6.07197917,
		6.09484905,
		6.10405199,
		6.10589133,
		6.10676296,
		6.10886552,
		6.12697903,
		6.14388245,
		6.14531944,
		6.15658562,
		6.18665356,
		6.21849376,
		6.22216383,
		6.27383424,
		6.27981524		
	};
	std::complex<double> output[380];
	int L = 4;
	int p = 4;
	int n = 3;
	int num_values = 380;
	int num_nodes = 380;
	compute_P_ddi(values, nodes, num_values, num_nodes, L, p, n,
				  reinterpret_cast<double *>(output));
}
