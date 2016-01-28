import FMM1D
import Gadfly
import GroundTruth
import Util

grid_pts = 100
M = N = 100
X = sort(rand(N))
U = randn(N)
l = 5
p = 50

source_finder = FMM1D.get_box_finder(X, l)
source_coefs = FMM1D.Cauchy.get_finest_farfield_coefs(source_finder, X, U, l, p)

for i = 1:2^l
    y_grid = FMM1D.get_E3_linspace(l, i, grid_pts)
    
    left, right = source_finder(l, i)
    I = left:right
    if I == 0:0
        continue
    end
    Φ = GroundTruth.get_Φ_direct(X[I], U[I])
    V_direct = Φ(y_grid)

    B = source_coefs[i]
    x_star = FMM1D.get_box_center(l, i)
    V_singular = FMM1D.Cauchy.evaluate_Φ_singular(y_grid, B, x_star, p)

    @printf("i = %d, mse = %g\n", i, Util.mse(V_direct, V_singular))
end
