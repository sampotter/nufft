module GroundTruth

# TODO: document me!
function compute_K(X::AbstractArray, Y::AbstractArray, Ns::AbstractArray)
    M = length(Y)
    N = length(X)
    K = zeros(Complex128, M, N)
    for j = 1:M
        for k = 1:N
            K[j, k] = (1/N)*sum(n -> exp(im*n*(Y[j] - X[k])), Ns)
        end
    end
    K
end

function get_Φ_direct(X, U)
    N = length(X)
    assert(N == length(U))
    Φ(y, x) = (y - x).^(-1)
    y -> sum(n -> U[n]*Φ(y, X[n]), 1:N)
end

end
