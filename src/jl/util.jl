module Util

function mse(X::AbstractArray, Y::AbstractArray)
    N = length(X)
    assert(N == length(Y))
    (1/N)*norm(X - Y, 2)^2
end

end
