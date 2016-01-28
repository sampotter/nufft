import FMM1D
import GroundTruth
import Util

using Gadfly

N = 100
L = 4
p = 300

X = sort(rand(N))
U = randn(N)

finder = FMM1D.get_box_finder(X, L)
coefs = FMM1D.Cauchy.get_finest_farfield_coefs(finder, X, U, L, p)

results = Dict()

l = L - 1
for i = 1:2^l
    @printf("i = %d\n", i)
    
    left, right = FMM1D.get_children(i)
    parent_center = FMM1D.get_box_center(l, i)

    B_parent = zeros(p)
    
    if left ∈ keys(coefs)
        child_center = FMM1D.get_box_center(l + 1, left)
        δ = parent_center - child_center
        SS = FMM1D.Cauchy.get_SS_matrix(δ, p)
        B = coefs[left]
        B_parent += SS*B
    end
    
    if right ∈ keys(coefs)
        child_center = FMM1D.get_box_center(l + 1, right)
        δ = parent_center - child_center
        SS = FMM1D.Cauchy.get_SS_matrix(δ, p)
        B = coefs[right]
        B_parent += SS*B
    end

    # First, let's make sure the children are evaluating correctly:

    begin # left child
        Y = FMM1D.get_E3_linspace(l + 1, left, 100)
        B = coefs[left]
        x_star = FMM1D.get_box_center(l + 1, left)
        V_approx = FMM1D.Cauchy.evaluate_Φ_singular(Y, B, x_star, p)
        cl, cr = finder(l + 1, left)
        I = cl:cr
        if I ≠ 0:0
            Φ = GroundTruth.get_Φ_direct(X[I], U[I])
            @printf("- left mse: %g\n", Util.mse(Φ(Y), V_approx))
        end
    end

    begin # right child
        Y = FMM1D.get_E3_linspace(l + 1, right, 100)
        B = coefs[right]
        x_star = FMM1D.get_box_center(l + 1, right)
        V_approx = FMM1D.Cauchy.evaluate_Φ_singular(Y, B, x_star, p)
        cl, cr = finder(l + 1, right)
        I = cl:cr
        if I ≠ 0:0
            Φ = GroundTruth.get_Φ_direct(X[I], U[I])
            @printf("- right mse: %g\n", Util.mse(Φ(Y), V_approx))
        end
    end

    # Let's try evaluating the translated Φ versus the direct Φ...

    begin
        Y = FMM1D.get_E3_linspace(l, i, 100) # grid of test targets

        # Set up everything for the parent...

        parent_left, parent_right = finder(l, i)
        
        I = parent_left:parent_right
        if I == 0:0
            continue
        end
        X_parent = X[I]
        U_parent = U[I]
        Φ_direct = GroundTruth.get_Φ_direct(X_parent, U_parent)

        # ... and the evaluation:

        V_direct = Φ_direct(Y)
        V_trans = FMM1D.Cauchy.evaluate_Φ_singular(Y, B_parent, parent_center, p)

        # Collect the results and functions for plotting the results
        # in a dictionary

        results[i] = Y, V_direct, V_trans, function ()
            plot(layer(x = results[i][1],
                       y = results[i][2],
                       Geom.point,
                       Theme(default_color = colorant"red")),
                 layer(x = results[i][1],
                       y = results[i][3],
                       Geom.point,
                       Theme(default_color = colorant"blue")))
        end
        
        @printf("- parent mse: %g\n", Util.mse(V_direct, V_trans))
    end
end
