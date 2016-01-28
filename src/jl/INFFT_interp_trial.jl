#
# this file contains a sketch of the project's INFFT interpolation
# method -- TODO: not currently functional
#

import FMM1D
import GroundTruth
import PeriodicSum
import TestSeries
import Util

using Gadfly

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
# periodic summation parameters
#

p = 10                    # truncation number
depth = 5                 # what depth MLFMM to use
n = 50                    # size of periodically extended neighborhood
nwidth = (n + 1)π         # one-sided neighborhood width
δ = 0.2*nwidth            # 2δ ≡ |r - R|
r = nwidth - δ            # inner radius
R = nwidth + δ            # outer radius
x₀ = π                    # expansion center
L = 100                   # number of check points

#
# first, we use our direct method that we know works to compute the
# result for comparison
#

KK = GroundTruth.compute_K(X, Y, -(N-1):N-1)
G = KK*F
G_real = f(Y, N)

#
# now we compute the same thing with periodic sum method
#

φ_Y = PeriodicSum.periodic_sum(X, F, Y, p, depth, x₀, n, r, R, L)

# interpolation step using results from per. sum. method
α = exp(-im*(N - 1)*Y) - exp(im*N*Y)
β = sum(F)
G_per = α.*(β/2 + im*φ_Y)/K

#
# report results
#

@printf("mse (G ↔ G_real): %g\n", Util.mse(G, G_real))
@printf("mse (G_real ↔ G_per): %g\n", Util.mse(G_real, G_per))
@printf("mse (G_per ↔ G): %g\n", Util.mse(G_per, G))

plot(layer(x = Y, y = real(G), Geom.point,
           Theme(default_color = colorant"blue")),
     layer(x = Y, y = real(G_per), Geom.point,
           Theme(default_color = colorant"green")),
     layer(x = Y, y = real(G_real), Geom.point,
           Theme(default_color = colorant"red")))
