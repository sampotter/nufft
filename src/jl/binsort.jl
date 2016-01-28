#
# a crappy binsort... the existence of sortperm might render this
# pretty useless unless we figure out a way to really optimize this
# function. should be possible in principle, but it's low priority for
# now.
# 

module BinSort

function binsort(X::AbstractArray, k::Integer, m, M)
    T = eltype(X)
    binsort(X, k, T(m), T(M))
end

function binsort{T<:Real}(X::Array{T, 1}, k::Integer, m::T, M::T)
    bins = Dict{Integer, Array{T, 1}}()
    for i = 1:k
        bins[i] = typeof(X)()
    end
    
    X_bin_indices = ceil(Integer, k*(X - m)/(M - m))
    for i = 1:length(X)
        bin_index = X_bin_indices[i]
        bins[bin_index] = [bins[bin_index]; X[i]]
    end

    sorted = typeof(X)()
    for i = 1:k
        sorted = [sorted; bins[i]]
    end
    sorted
end

end # module BinSort
