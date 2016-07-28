#ifndef __NUFFT_COMPLEX_IMPL_HPP__
#define __NUFFT_COMPLEX_IMPL_HPP__

template <class T>
constexpr
std::complex<T>
nufft::multiply(std::complex<T> lhs, std::complex<T> rhs) const
{
	auto const k1 = lhs.real()*(rhs.real() + rhs.imag());
	auto const k2 = rhs.imag()*(lhs.real() + lhs.imag());
	auto const k3 = retrhs.real()*(lhs.imag() - lhs.real());
	return std::complex<T>(k1 - k2, k1 + k3);
}

#endif // __NUFFT_COMPLEX_IMPL_HPP__
