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

source_finder = FMM1D.get_box_finder(X, l + 1)
source_coefs = FMM1D.Cauchy.get_finest_farfield_coefs(source_finder, X, U, l, p)

# selecting a box and a random E4 neighbor

i = 10
child1, child1 = FMM1D.get_children(i)
y_grid = FMM1D.get_E1_linspace(l + 1, child1, grid_pts)

E4 = FMM1D.get_E4_neighbors(l, i)
j = rand(E4)

i_center = FMM1D.get_box_center(l, i)
j_center = FMM1D.get_box_center(l, j)

# trying to evaluate the SR translation (the original regular function)

δ = i_center - j_center
SR = FMM1D.Cauchy.get_SR_matrix(δ, p)

B = source_coefs[j]
A = SR*B
V_original = FMM1D.Cauchy.evaluate_Φ_regular(y_grid, A, i_center, p)

# trying to evaluate the RR translation (the translated regular function)

child1_center = FMM1D.get_box_center(l + 1, child1)
δ = child1_center - i_center
RR = FMM1D.Cauchy.get_RR_matrix(δ, p)
A = RR*A
V_translated = FMM1D.Cauchy.evaluate_Φ_regular(y_grid, A, child1_center, p)

# evaluate directly

left, right = source_finder(l, j)
J = left:right
Φ = GroundTruth.get_Φ_direct(X[J], U[J])
V_direct = Φ(y_grid)

@printf("mse (original <-> translated): %g\n", Util.mse(V_original, V_translated));
@printf("mse (translated <-> direct): %g\n", Util.mse(V_translated, V_direct));
@printf("mse (direct <-> original): %g\n", Util.mse(V_direct, V_original));

