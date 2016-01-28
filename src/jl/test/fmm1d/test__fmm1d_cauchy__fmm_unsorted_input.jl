import FMM1D
import GroundTruth
import Util

using Gadfly

M = N = 10

X = rand(N)
Y = rand(M)

U = randn(N)
L = 5
p = 10

Φ = GroundTruth.get_Φ_direct(X, U)
V = Φ(Y)

V_fmm = FMM1D.Cauchy.fmm(Y, X, U, L, p)

plot(layer(x = Y, y = V, Geom.point,
           Theme(default_color = colorant"red")),
     layer(x = Y, y = V_fmm, Geom.point,
           Theme(default_color = colorant"blue")))

@printf("mse: %g\n", Util.mse(V, V_fmm))
