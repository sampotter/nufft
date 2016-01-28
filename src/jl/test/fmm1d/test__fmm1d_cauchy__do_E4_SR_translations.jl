import FMM1D
import GroundTruth
import Util

grid_pts = 100
M = N = 100
X = sort(rand(N))
Y = sort(rand(M))
U = randn(N)
L = 5
p = 10

source_finder = FMM1D.get_box_finder(X, L)
source_coefs = FMM1D.Cauchy.get_finest_farfield_coefs(source_finder, X, U, L, p)

target_finder = FMM1D.get_box_finder(Y, L)
target_coefs = FMM1D.initialize_target_coefs(Float64, target_finder, L, p)

FMM1D.Cauchy.do_E4_SR_translations!(source_coefs, target_coefs, L, p)

data = Dict()

for i = keys(target_coefs)
    y_grid = FMM1D.get_E1_linspace(L, i, grid_pts)
    
    # direct Φ

    E4 = FMM1D.get_E4_neighbors(L, i)
    I = []
    for j = E4
        left, right = source_finder(L, j)
        J = left:right
        if J ≠ 0:0
            I = union(I, J)
        end
    end
    assert(I ≠ 0:0)

    Φ = GroundTruth.get_Φ_direct(X[I], U[I])
    V_direct = Φ(y_grid)

    # translated regular expansion

    x_star = FMM1D.get_box_center(L, i)
    A = target_coefs[i]
    V_regular = FMM1D.Cauchy.evaluate_Φ_regular(y_grid, A, x_star, p)

    @printf("mse = %g\n", Util.mse(V_direct, V_regular))

    data[i] = y_grid, V_direct, V_regular
end
