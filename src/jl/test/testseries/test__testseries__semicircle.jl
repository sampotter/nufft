import GroundTruth
import TestSeries

using Gadfly

M = 101 # the number of samples at which to interpolate
N = 10 # the "order" of the func
L = 2(2N - 2) # this is the number of samples -- must be at least 2N - 2

X = (2π/L)*(0:L-1)
Y = 2π*rand(M)
# Y = linspace(0, 2π, M)
Ns = -(N-1):(N-1)
func = TestSeries.semicircle
F = func(X, N)
K = GroundTruth.compute_K(X, Y, Ns)
G = K*F

plot(layer(x = X,
           y = real(F),
           Geom.point,
           Theme(default_color = colorant"red")),
     layer(x = Y,
           y = real(G),
           Geom.line,
           Theme(default_color = colorant"blue")),
     layer(x = Y,
           y = imag(G),
           Geom.line,
           Theme(default_color = colorant"blue")),
     layer(x = Y,
           y = real(func(Y, N)),
           Geom.line,
           Theme(default_color = colorant"green")))
