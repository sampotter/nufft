import FMM1D
import GroundTruth
import Util

using Gadfly

N = 10000
X = sort(rand(N))
U = randn(N)
L = 4 # TODO: bump this up
p = 10

finder = FMM1D.get_box_finder(X, L)
coefs = FMM1D.Cauchy.get_finest_farfield_coefs(finder, X, U, L, p)

data = Dict()

for l = L:-1:3
    @printf("l = %d\n", l)

    for i = 1:2^l
        if i ∈ keys(coefs)
            @printf("- i = %d, ", i)

            left, right = finder(l, i)
            
            I = left:right
            if I ≠ 0:0
                Y = FMM1D.get_E3_linspace(l, i, 1000)
                Φ = GroundTruth.get_Φ_direct(X[I], U[I])
                B = coefs[i]
                x_star = FMM1D.get_box_center(l, i)
                V = Φ(Y)
                V_approx = FMM1D.Cauchy.evaluate_Φ_singular(Y, B, x_star, p)
                MSE = Util.mse(V, V_approx)
                @printf("mse: %g\n", MSE)

                data[l, i] = Y, V, V_approx, MSE
            end
        end
    end

    coefs = FMM1D.Cauchy.get_parent_farfield_coefs(coefs, l, p)
end
