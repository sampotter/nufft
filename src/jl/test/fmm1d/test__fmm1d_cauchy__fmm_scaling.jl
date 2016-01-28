import FMM1D

import FMM1D
import GroundTruth
import Util

using Gadfly

M = N = 1000

# X = Array{Float64, 1}(linspace(0, 1, M))
# Y = Array{Float64, 1}(linspace(0, 1, M))

x_scale = 100*randn()
y_scale = 100*randn()
x_offset, y_offset = 100*sort(randn(2))
X = sort(x_scale*rand(N)) + x_offset
Y = sort(y_scale*rand(M)) + y_offset

a = min(minimum(X), minimum(Y))
# a = floor(min(minimum(X), minimum(Y)))
b = max(maximum(X), maximum(Y))
# b = ceil(max(maximum(X), maximum(Y)))
fmm_offset = -a
fmm_scale = b - a
Y_fmm = (Y + fmm_offset)/fmm_scale
X_fmm = (X + fmm_offset)/fmm_scale

U = randn(N)
L = 5
p = 5

Φ = GroundTruth.get_Φ_direct(X, U)
V = Φ(Y)

V_fmm = FMM1D.Cauchy.fmm(Y_fmm, X_fmm, U, L, p)/fmm_scale

V_scaled_fmm = FMM1D.Cauchy.scaled_fmm(Y, X, U, L, p)

plot(layer(x = Y, y = V, Geom.point,
           Theme(default_color = colorant"red")),
     layer(x = Y, y = V_fmm, Geom.point,
           Theme(default_color = colorant"blue")))

@printf("mse (fmm <-> direct): %g\n", Util.mse(V, V_fmm))
@printf("mse (direct <-> fmm_scaled): %g\n", Util.mse(V, V_scaled_fmm))
@printf("mse (fmm_scaled <-> fmm): %g\n", Util.mse(V_scaled_fmm, V_fmm))
