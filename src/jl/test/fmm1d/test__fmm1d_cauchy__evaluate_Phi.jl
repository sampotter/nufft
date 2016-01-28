import FMM1D
import GroundTruth

using Gadfly

X = sort(rand(20))              # sources
U = randn(20)                   # weights
l = 2                           # level
i = 3                           # box index
x_star = FMM1D.get_box_center(l, i) # expansion center
p = 100                             # truncation number

finder = FMM1D.get_box_finder(X, l)
left, right = finder(l, i)
X_box = X[left:right]
U_box = U[left:right]
          
Φ_multipole_direct = GroundTruth.get_Φ_direct(X_box, U_box)
Y = FMM1D.get_E3_linspace(l, i, 100)
V_direct = Φ_multipole_direct(Y)

B = FMM1D.Cauchy.get_multipole_coefs(X_box, U_box, x_star, p)
V_approx = FMM1D.Cauchy.evaluate_Φ_singular(Y, B, x_star, p)

println((1/length(Y))*norm(V_approx - V_direct, 2)^2)

# plot(layer(x = Y, y = V_approx, Geom.point,
#            Theme(default_color = colorant"blue")),
#      layer(x = Y, y = V_direct, Geom.point,
#            Theme(default_color = colorant"green")))

