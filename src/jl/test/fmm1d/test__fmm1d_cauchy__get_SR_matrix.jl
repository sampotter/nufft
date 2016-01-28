import FMM1D
import Gadfly
import GroundTruth
import Util

grid_pts = 100
M = N = 100
X = sort(rand(N))
U = randn(N)
l = 5
p = 10

source_finder = FMM1D.get_box_finder(X, l)
source_coefs = FMM1D.Cauchy.get_finest_farfield_coefs(source_finder, X, U, l, p)

# selecting a box and a random E4 neighbor

i = 10
y_grid = FMM1D.get_E1_linspace(l, i, grid_pts)

E4 = FMM1D.get_E4_neighbors(l, i)
j = rand(E4)

i_center = FMM1D.get_box_center(l, i)
j_center = FMM1D.get_box_center(l, j)

# trying to evaluate the original singular function

B = source_coefs[j]
V_singular = FMM1D.Cauchy.evaluate_Φ_singular(y_grid, B, j_center, p)

# trying to evaluate the SR translation

δ = i_center - j_center
SR = FMM1D.Cauchy.get_SR_matrix(δ, p)

A = SR*B
V_regular = FMM1D.Cauchy.evaluate_Φ_regular(y_grid, A, i_center, p)

# evaluate directly

left, right = source_finder(l, j)
J = left:right
Φ = GroundTruth.get_Φ_direct(X[J], U[J])
V_direct = Φ(y_grid)

@printf("mse (singular <-> regular): %g\n", Util.mse(V_singular, V_regular));
@printf("mse (regular <-> direct): %g\n", Util.mse(V_regular, V_direct));
@printf("mse (direct <-> singular): %g\n", Util.mse(V_direct, V_singular));
