# TODO: gabor wavelet for complex?

# TODO: reduce boilerplate in
# methods that actually compute values (they're basically all the
# same)

# TODO: documentation

module TestSeries

function sawtooth_coefs(N::Integer)
    c(n) = n == 0 ? 1/2 : im/(2*pi*n)
    Dict{Integer, Complex128}([n => c(n) for n = -(N-1):(N-1)])
end

function sawtooth(x::Real, N::Integer)
    C = sawtooth_coefs(N)
    sum(map(n -> C[n]*exp(im*n*x), -(N-1):(N-1)))
end

function sawtooth(X::AbstractArray, N::Integer)
    C = sawtooth_coefs(N)
    map(x -> sum(map(n -> C[n]*exp(im*n*x), -(N-1):(N-1))), X)
end

function square_coefs(N::Integer)
    c(n) = n => iseven(n) ? 0 : -2im/(n*pi)
    Dict{Integer, Complex128}([c(n) for n = -(N-1):(N-1)])
end

function square(x::Real, N::Integer)
    C = square_coefs(N)
    sum(map(n -> C[n]*exp(im*n*x), -(N-1):(N-1)))
end

function square(X::AbstractArray, N::Integer)
    C = square_coefs(N)
    map(x -> sum(map(n -> C[n]*exp(im*n*x), -(N-1):(N-1))), X)
end

function triangle_coefs(N::Integer)
    c(n) = -(4*im*(-1)^((n - 1)/2))/(4*pi^2*n^2)
    Dict{Integer, Complex128}([n => iseven(n) ? 0 : c(n) for n = -(N-1):(N-1)])
end

function triangle(x::Real, N::Integer)
    C = triangle_coefs(N)
    sum(map(n -> C[n]*exp(im*n*x), -(N-1):(N-1)))
end

function triangle(X::AbstractArray, N::Integer)
    C = triangle_coefs(N)
    map(x -> sum(map(n -> C[n]*exp(im*n*x), -(N-1):(N-1))), X)
end

function semicircle_coefs(N::Integer)
    c(n) = n == 0 ? pi^2/4 : pi*(-1.0)^n*besselj1(pi*n)/(2n)
    Dict{Integer, Complex128}([n => c(n) for n = -(N-1):(N-1)])
end

function semicircle(x::Real, N::Integer)
    C = semicircle_coefs(N)
    sum(map(n -> C[n]*exp(im*n*x), -(N-1):(N-1)))
end

function semicircle(X::AbstractArray, N::Integer)
    C = semicircle_coefs(N)
    map(x -> sum(map(n -> C[n]*exp(im*n*x), -(N-1):(N-1))), X)
end

end
