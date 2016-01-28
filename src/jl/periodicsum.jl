# TODO: DON'T sort before returning check points—this doesn't work!
# instead, we need to figure out a way to pass unsorted elements to
# the FMM and have the sorting be taken care of transparently

# TODO: write a macro or a function to take care of some of the
# boilerplate here in the check point generators

module PeriodicSum

import FMM1D
   
module CheckPoints

using Distributions

function beta(r::Real, L::Integer, β::Integer)
    assert(r > π)
    let b = Beta(1, β), L1::Integer = floor(L/2), L2::Integer = ceil(L/2)
        Z_left = (π - r)*rand(b, L1)
        Z_left_per = (Z_left % 2π) + 2π
        Z_right = (r - π)*rand(b, L2) + 2π
        Z_right_per = Z_right % 2π
        reg = [Z_left; Z_right]
        assert(all(z -> z < 0 || 2π < z, reg))
        per = [Z_left_per; Z_right_per]
        assert(all(z -> 0 < z < 2π, per))
        reg, per
    end
end

function uniform(r::Real, L::Integer)
    assert(r > π)
    let u = Uniform(), L1::Integer = floor(L/2), L2::Integer = ceil(L/2)
        Z_left = (π - r)*rand(u, L1)
        Z_left_per = (Z_left % 2π) + 2π
        Z_right = (r - π)*rand(u, L2) + 2π
        Z_right_per = Z_right % 2π
        reg = [Z_left; Z_right]
        assert(all(z -> z < 0 || 2π < z, reg))
        per = [Z_left_per; Z_right_per]
        assert(all(z -> 0 < z < 2π, per))
        reg, per
    end
end

function linspace(r::Real, L::Integer)
    assert(r > π)
    let L1::Integer = floor(L/2), L2::Integer = ceil(L/2)
        Z_left = linspace(π - r, 0)
        Z_left_per = (Z_left % 2π) + 2π
        Z_right = linspace(2π, π + r)
        Z_right_per = Z_right % 2π
        reg = [Z_left; Z_right]
        assert(all(z -> z < 0 || 2π < z, reg))
        per = [Z_left_per; Z_right_per]
        assert(all(z -> 0 < z < 2π, per))
        reg, per
    end
end

function logspace(r::Real, L::Integer, range::Integer)
    assert(r > π)
    assert(range > 0)
    scaled_logspace = (logspace(0, range) - 1)/(10^range - 1)
    let L1::Integer = floor(L/2), L2::Integer = ceil(L/2)
        Z_left = (π - r)*scaled_logspace
        Z_left_per = (Z_left % 2π) + 2π
        Z_right = (r - π)*scaled_logspace + 2π
        Z_right_per = Z_right % 2π
        reg = [Z_left; Z_right]
        assert(all(z -> z < 0 || 2π < z, reg))
        per = [Z_left_per; Z_right_per]
        assert(all(z -> 0 < z < 2π, per))
        reg, per
    end
end

# TODO: this function should generate checkpoints by generating points
# that are halfway between the time grid points, and then periodically
# offsetting them into the central box
#
# TODO: could also look into
# adding a small random perturbation to the pairs of points?
function halfway(r::Real, L::Integer)
    assert(r > π)
end

end # module CheckPoints

function periodic_sum{T<:Number}(X::Array{Float64, 1}, # time grid (sources)
                                 U::Array{T, 1},       # coefs
                                 Y::Array{Float64, 1}, # target points
                                 p::Integer,       # truncation number
                                 depth::Integer,   # depth of FMM
                                 x_star::Real = π, # expansion center
                                 n::Integer = 1,   # neighborhood size
                                 r::Real = 2π - 0.1, # inner radius
                                 R::Real = 2π + 0.1, # outer radius
                                 L::Integer = 100;   # number of check points
                                 cp_method::AbstractString = "beta",
                                 cp_param::Integer = 3)
    Z, Z_per = CheckPoints.beta(r, L, cp_param) # TODO: make use of cp_method
    reg = (m::Integer, y) -> (y - x_star).^m

    A = zeros(T, L, p)
    for l = 1:L
        for m = 1:p
            A[l, m] = reg(m - 1, Z_per[l]) - reg(m - 1, Z[l])
        end
    end

    X_per = reduce((S, T) -> [S; T], map(k -> X + 2π*k, -n:n))
    assert(length(unique(X_per)) == length(X_per))

    U_per = real(repmat(U, 2n + 1, 1)[:, 1])
    
    fmm = S -> FMM1D.Cauchy.scaled_fmm(S, X_per, U_per, depth, p)

    f = fmm(Z) - fmm(Z_per)
    c = pinv(A)*f

    M = length(Y)
    R = zeros(M, p)
    for j = 1:M
        for m = 1:p
            R[j, m] = reg(m - 1, Y[j]) # TODO: the paper says this
                                       # should be Y[j] - x
                                       # something...
        end
    end

    φ_near_Y = fmm(Y)
    φ_far_Y = -R*c
    φ_near_Y + φ_far_Y
end

end # module PeriodicSum
