import FMM1D
import PeriodicSum
import TestSeries
import Util

reload("PeriodicSum")

#
# fix parameters that don't depend on the iteration
#

x₀ = π                      # expansion center for per. sum.
L = 50                      # number of check points
p = 10                       # truncation number
depth = 5                   # fmm depth
δ = 1/10                    # 2δ ≡ |R - r|
N = 100                     # number of time domain points
X = Float64[2π*(0:N-1)/N;]  # time domain grid
ψ = TestSeries.semicircle   # bandlimited test function
B = round(Integer, N/10)    # the "bandlimit"
F = ψ(X, B)                 # weights (i.e. ψ evaluated at X)
Bs = (1-B):(B-1)            # support of ψ's fourier series
M = N                       # number of evaluation points
Y = sort(2π*rand(M))        # evaluation points

results = Dict()

n_fmm = 25

# compute periodic extension of time domain grid
X_per = reduce((S, T) -> [S; T], map(k -> X + 2π*k, -n_fmm:n_fmm))
assert(length(unique(X_per)) == length(X_per))
assert(issorted(X_per))

# get per. extension of coefficients
F_per = real(repmat(F, 2*n_fmm + 1, 1)[:, 1])

# compute approximate answer using FMM
φ_approx = FMM1D.Cauchy.scaled_fmm(Y, X_per, F_per, depth, p)

# iterate over different sizes of neighborhood
for n = 1:100
    #
    # pick varying parameters
    #
    
    nwidth = (n + 1)π           # neighborhood width
    r = nwidth - δ              # inner radius
    R = nwidth + δ              # outer radius

    #
    # compute ground truth answer over neighborhood
    #

    # compute periodic extension of time domain grid
    X_per = reduce((S, T) -> [S; T], map(k -> X + 2π*k, -n:n))
    assert(length(unique(X_per)) == length(X_per))
    assert(issorted(X_per))

    # get per. extension of coefficients
    F_per = real(repmat(F, 2n + 1, 1)[:, 1])

    # compute using periodic summation method
    φ_per_sum = PeriodicSum.periodic_sum(X, F, Y, p, depth, x₀, n, r, R, L)

    @printf("n = %d: mse (φ_approx ↔ φ_per_sum) = %g\n",
            n,
            Util.mse(φ_approx, φ_per_sum))

    # aggregate results
    results[n] = φ_approx, φ_per_sum
end
