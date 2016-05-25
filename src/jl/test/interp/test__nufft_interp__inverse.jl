import FMM1D
import GroundTruth
import TestSeries

#
# problem parameters
#

N = 10                          # bandlimit
f = TestSeries.semicircle       # test function
K = 100                         # number of time domain grid points
X = Float64[2π*(0:K-1)/K;]      # time domain grid
F = f(X, N)                     # known function values
J = 150                         # number of interpolation points
Y = 2π*rand(J)                  # points to interpolate at

#
# fmm parameters
#

p = 10                    # truncation number
depth = 5                 # what depth MLFMM to use
n = 50                    # size of periodically extended neighborhoo

#
# use mlfmm to compute φ_Y
#

X_per = reduce((S, T) -> [S; T], map(k -> X + 2π*k, -n:n))
F_per = real(repmat(F, 2n + 1, 1)[:, 1])
φ_Y_fmm = FMM1D.Cauchy.scaled_fmm(Y, X_per, F_per, depth, p)
# normalizeinput(lst) = (lst + 2pi*n)/(2pi*(2n + 1))
# X_per_normalized = normalizeinput(X_per)
# Y_normalized = normalizeinput(Y)
# φ_Y_fmm = FMM1D.Cauchy.fmm(Y_normalized, X_per_normalized, F_per, depth, p)

#
# interpolation
#

α = exp(-im*(N - 1)*Y) - exp(im*N*Y)
β = sum(F)
G_fmm = α.*(β/2 + im*φ_Y_fmm)/K

#
# ground truth interpolation
#

KK = GroundTruth.compute_K(X, Y, -(N-1):N-1)
G_gt = KK*F
