import FMM1D
import GroundTruth
import Util

grid_pts = 100
M = N = 100
X = sort(rand(N))
Y = sort(rand(M))
U = randn(N)
L = 5
p = 20

parent_l = L - 1
child_l = parent_l + 1

assert(child_l ≤ L)

source_finder = FMM1D.get_box_finder(X, L)
source_coefs =
    FMM1D.Cauchy.get_finest_farfield_coefs(source_finder, X, U, parent_l, p)

target_finder = FMM1D.get_box_finder(Y, L)
target_coefs = FMM1D.initialize_target_coefs(Float64, target_finder, parent_l, p)
RR_target_coefs = FMM1D.initialize_target_coefs(Float64, target_finder, child_l, p)

FMM1D.Cauchy.do_E4_SR_translations!(source_coefs, target_coefs, parent_l, p)
FMM1D.Cauchy.do_RR_translations!(target_coefs, RR_target_coefs, parent_l, p)

data = Dict()

for parent = keys(target_coefs)
    @printf("parent: %d\n", parent)
    
    # get parent's reg. coefs for approx evaluation

    A_orig = target_coefs[parent]
    x_star_orig = FMM1D.get_box_center(parent_l, parent)

    # get parent's sources in E4 neighborhood for direct eval

    I = []
    for i = FMM1D.get_E4_neighbors(parent_l, parent)
        left, right = source_finder(parent_l, i)
        if left:right ≠ 0:0
            I = union(I, left:right)
        end
    end
    if isempty(I)
        continue
    end

    Φ = GroundTruth.get_Φ_direct(X[I], U[I])
    
    function check_child!(child)
        @printf("- child: %d\n", child)
        y_grid = FMM1D.get_E1_linspace(child_l, child, grid_pts)
        A = RR_target_coefs[child]
        x_star = FMM1D.get_box_center(child_l, child)

        V_orig = FMM1D.Cauchy.evaluate_Φ_regular(y_grid, A_orig, x_star_orig, p)
        V = FMM1D.Cauchy.evaluate_Φ_regular(y_grid, A, x_star, p)
        V_direct = Φ(y_grid)
        @printf("- child <-> parent mse: %g\n", Util.mse(V, V_orig))
        @printf("- parent <-> direct mse: %g\n", Util.mse(V_orig, V_direct))
        @printf("- direct <-> child mse: %g\n", Util.mse(V_direct, V))
    end

    for child = FMM1D.get_children(parent)
        if child ∈ keys(RR_target_coefs)
            check_child!(child)
        end
    end
end
