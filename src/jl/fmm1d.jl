"""
The FMM1D module provides an implementation of the full fast multipole
method (FMM) in one-dimension.

- TODO: investigate performance of Unsigned
- TODO: compare bit twiddling
implementation with straightforward implementation, see if there is
actually any performance gain
- TODO: decide how I want to deal with boundary cases
"""
module FMM1D

module Cauchy

import FMM1D
import GroundTruth

typealias Coefs{T} Dict{Integer, Array{T, 1}}

R(m::Integer, x) = x.^m

S(m::Integer, x) = x.^(-m-1)

a(m::Integer, x) = -x.^(-m-1)

b(m::Integer, x) = x.^m

function get_multipole_coefs{T<:Number}(X::Array{Float64, 1},
                                        U::Array{T, 1},
                                        x_star::Float64,
                                        p::Integer)
    N = length(X)
    assert(N == length(U))
    assert(p ≥ 0)
    B = zeros(p)
    for m = 0:p-1
        B[m + 1] = sum(k -> U[k]*b(m, X[k] - x_star), 1:N)
    end
    B
end

function evaluate_Φ_regular{T<:Number}(Y::Array{Float64, 1},
                                       A::Array{T, 1},
                                       x_star::Float64,
                                       p::Integer)
    P = 0:p-1
    Φ(y::Float64) = sum(m -> A[m + 1]*R(m, y - x_star), P)
    map(Φ, Y)
end

function evaluate_Φ_singular{T<:Number}(Y::Array{Float64, 1},
                                        B::Array{T, 1},
                                        x_star::Float64,
                                        p::Integer)
    P = 0:p-1
    sum(m -> B[m+1]*S(m, Y - x_star), P)
end

function isnormalized(A::Array{Float64, 1})
    minimum(A) ≥ 0 && maximum(A) ≤ 1
end

function get_finest_farfield_coefs{T<:Number}(get_source_box::Function,
                                              X::Array{Float64, 1},
                                              U::Array{T, 1},
                                              L::Integer,
                                              p::Integer)
    coefs = Coefs{T}()
    for i = 1:2^L
        l, r = get_source_box(L, i)
        J = l:r
        if J == 0:0
            continue
        end
        x_star = FMM1D.get_box_center(L, i)
        coefs[i] = get_multipole_coefs(X[J], U[J], x_star, p)
    end
    coefs
end

function get_parent_farfield_coefs{T<:Number}(coefs::Coefs{T},
                                              l::Integer,
                                              p::Integer)
    assert(1 ≤ maximum(keys(coefs)) ≤ 2^l)
    
    parent_coefs = Coefs{T}()

    l_child = l
    l_parent = l - 1
    for i = 1:2^l_parent
        B_parent = zeros(p)
        parent_center = FMM1D.get_box_center(l_parent, i)
        
        function translate_child_coefs(child_index)
            child_center = FMM1D.get_box_center(l_child, child_index)
            δ = parent_center - child_center
            SS = FMM1D.Cauchy.get_SS_matrix(δ, p)
            SS*coefs[child_index]
        end

        left, right = FMM1D.get_children(i)
        if left ∈ keys(coefs)
            B_parent += translate_child_coefs(left)
        end
        if right ∈ keys(coefs)
            B_parent += translate_child_coefs(right)
        end

        parent_coefs[i] = B_parent
    end
    parent_coefs
end

function do_E4_SR_translations!{T<:Number}(source_coefs::Coefs{T},
                                           target_coefs::Coefs{T},
                                           l::Integer,
                                           p::Integer)
    source_keys = keys(source_coefs)
    target_keys = keys(target_coefs)

    function validate_coefs(K)
        assert(1 ≤ minimum(K))
        assert(maximum(K) ≤ 2^l)
    end
    validate_coefs(source_keys)
    validate_coefs(target_keys)
    
    for i = intersect(target_keys, 1:2^l)
        i_center = FMM1D.get_box_center(l, i)
        E4_neighbors = FMM1D.get_E4_neighbors(l, i)
        for j = intersect(source_keys, E4_neighbors)
            j_center = FMM1D.get_box_center(l, j)
            δ = i_center - j_center
            SR = get_SR_matrix(δ, p)
            target_coefs[i] += SR*source_coefs[j]
        end
    end
end

function do_RR_translations!{T<:Number}(par_coefs::Coefs{T},
                                        children_coefs::Coefs{T},
                                        l::Integer,
                                        p::Integer)
    par_l = l
    children_l = l + 1
    
    par_keys = keys(par_coefs)
    assert(1 ≤ minimum(par_keys))
    assert(maximum(par_keys) ≤ 2^par_l)
           
    children_keys = keys(children_coefs)
    assert(1 ≤ minimum(children_keys))
    assert(maximum(children_keys) ≤ 2^children_l)

    for par = 1:2^par_l
        par_center = FMM1D.get_box_center(par_l, par)

        function translate!(child)
            if child ∈ children_keys
                child_center = FMM1D.get_box_center(children_l, child)
                δ = child_center - par_center
                RR = get_RR_matrix(δ, p)
                A = par_coefs[par]
                children_coefs[child] += RR*A
            end
        end
        
        map(translate!, FMM1D.get_children(par))
    end
end

function evaluate{T<:Number}(source_finder::Function,
                             target_finder::Function,
                             target_coefs::Coefs{T},
                             X::Array{Float64, 1},
                             Y::Array{Float64, 1},
                             U::Array{T, 1},
                             L::Integer,
                             p::Integer)
    # initialize the evaluation vector
    V::Array{T, 1} = zeros(length(Y))

    # for each box at the finest level...
    for i = 1:2^L
        # get the indices of the target points in the current box
        left, right = target_finder(L, i)
        I_target = left:right
        if I_target == 0:0
            continue
        end
        
        # get the indices of the sources points in the E2 neighborhood
        # of this box
        I_direct = []
        for j = FMM1D.get_E2_neighbors(L, i)
            left, right = source_finder(L, j)
            J = left:right
            I_direct = union(I_direct, J == 0:0 ? [] : J)
        end
        if !isempty(I_direct)
            # compute the contribution to this box's targets directly for
            # the source points in E2
            Φ = GroundTruth.get_Φ_direct(X[I_direct], U[I_direct])
            V[I_target] += Φ(Y[I_target])
        end

        # compute the contribution due to the regular expansion in the
        # current box
        A = target_coefs[i]
        i_center = FMM1D.get_box_center(L, i)
        V[I_target] += evaluate_Φ_regular(Y[I_target], A, i_center, p)
    end
    V
end

# TODO: look into LowerTriangular?
function get_SS_matrix(δ::Float64, p::Integer)
    SS = zeros(Float64, p, p)
    SS[:, 1] = 1
    for m = 2:p
        for n = 2:m
            SS[m, n] = SS[m - 1, n] + SS[m - 1, n - 1]
        end
    end
    for m = 1:p
        for n = 1:m
            expt = m - n
            SS[m, n] *= float(-1)^expt * float(δ)^expt
        end
    end
    SS
end

function get_SR_matrix(δ::Float64, p::Integer)
    SR = zeros(Float64, p, p)
    SR[:, 1] = 1
    SR[1, :] = 1
    for m = 2:p
        for n = 2:p
            SR[m, n] = SR[m - 1, n] + SR[m, n - 1]
        end
    end
    for m = 1:p
        for n = 1:p
            SR[m, n] *= float(δ)^(1-m-n)
        end
    end
    for m = 1:p
        SR[m, :] *= iseven(m) ? -1 : 1
    end
    SR
end

# TODO: make sure this is correct!!
# TODO: consider UpperTriangular?
function get_RR_matrix(δ::Float64, p::Integer)
    RR = zeros(Float64, p, p)
    RR[1, :] = 1
    for n = 2:p
        for m = 2:n
            RR[m, n] = RR[m - 1, n - 1] + RR[m, n - 1]
        end
    end
    for n = 1:p
        for m = 1:n
            RR[m, n] *= δ^(n - m)
        end
    end
    RR
end

function scaled_fmm{T<:Number}(Y::Array{Float64, 1},
                               X::Array{Float64, 1},
                               U::Array{T, 1},
                               L::Integer,
                               p::Integer,
                               verbose::Bool = false)
    a = min(minimum(X), minimum(Y))
    b = max(maximum(X), maximum(Y))
    offset = -a
    α = b - a
    assert(α > 0)
    adjust = S -> (S + offset)/α
    fmm(adjust(Y), adjust(X), U, L, p, verbose)/α
end

function fmm{T<:Number}(Y::Array{Float64, 1}, # normalized and sorted targets
                        X::Array{Float64, 1}, # normalized and sorted sources
                        U::Array{T, 1},       # weights for each X
                        L::Integer, # depth, TODO: remove by making adaptive
                        p::Integer, # truncation number
                        verbose::Bool = false)
    assert(isnormalized(Y))
    assert(isnormalized(X))
    assert(L ≥ 2)
    assert(p ≥ 0)

    # Compute the sorting permutation of X. We use this to sort X and
    # maintain the order of U with respect to X.

    X_sortperm = sortperm(X)
    X_sorted = X[X_sortperm]
    U_sorted = U[X_sortperm]

    # Compute the sorting permutation of Y and sort Y—the rest of the
    # algorithm will operate on the sorted copy. At the end of the
    # algorithm, we will use the permutation to return the computed
    # values in the original, unsorted order.

    Y_sortperm = sortperm(Y)
    Y_sorted = Y[Y_sortperm]

    # Create functions that will return the boxes at different levels
    # of the sorted X and Y. This is the algorithm's main data
    # structure.

    get_source_box = FMM1D.get_box_finder(X_sorted, L)
    get_target_box = FMM1D.get_box_finder(Y_sorted, L)

    if verbose @printf("upward pass\n") end
    
    if verbose @printf("- getting finest singular coefs\n") end
    source_coefs = Dict{Integer, Coefs{T}}()
    source_coefs[L] =
        get_finest_farfield_coefs(get_source_box, X_sorted, U_sorted, L, p)
    if L > 2
        for l = L:-1:3
            if verbose @printf("- SS translations (l = %d)\n", l) end
            source_coefs[l - 1] =
                get_parent_farfield_coefs(source_coefs[l], l, p)
        end
    end

    if verbose @printf("downward pass\n") end
    
    target_coefs = Dict{Integer, Coefs{T}}()
    for l = 2:L
        if verbose @printf("- initializing regular coefs (l = %d)\n", l) end
        target_coefs[l] = FMM1D.initialize_target_coefs(T, get_target_box, l, p)
    end
    for l = 2:L-1
        if verbose @printf("- E4 SR translations (l = %d)\n", l) end
        do_E4_SR_translations!(source_coefs[l], target_coefs[l], l, p)
        if verbose @printf("- RR translations (l = %d -> %d)\n", l, l + 1) end
        do_RR_translations!(target_coefs[l], target_coefs[l + 1], l, p)
    end
    do_E4_SR_translations!(source_coefs[L], target_coefs[L], L, p)

    if verbose @printf("- final evaluation\n") end
    V_sorted = evaluate(get_source_box,
                        get_target_box,
                        target_coefs[L],
                        X_sorted,
                        Y_sorted,
                        U_sorted,
                        L,
                        p)

    M = length(Y)
    V = zeros(eltype(V_sorted), M)
    for i = 1:M
        V[Y_sortperm[i]] = V_sorted[i]
    end
    V
end

end # module Cauchy

typealias Coefs{T} Dict{Integer, Array{T, 1}}

get_parent(n::Integer) = (n + 1) >> 1

get_children(n::Integer) = let m = n << 1; [m - 1; m] end

get_siblings(n::Integer) = (n >> 1 << 1) + 1 - (n & 1)

function get_E1_linspace(l::Integer, i::Integer, N::Integer)
    assert(0 ≤ 1)
    assert(1 ≤ i ≤ 2^l)
    assert(0 < N)
    c = get_box_center(l, i)
    Δ = get_box_size(l)/2
    Float64[linspace(c - Δ, c + Δ, N);]
end

function get_E2_neighbors(l::Integer, i::Integer)
    assert(0 ≤ l)
    max_i = 2^l
    assert(1 ≤ i ≤ max_i)
    filter(j -> 1 ≤ j ≤ max_i, [i - 1; i; i + 1])
end

function get_E2_interval(l::Integer, i::Integer)
    assert(0 ≤ l)
    assert(1 ≤ i ≤ 2^l)

    c = get_box_center(l, i)
    s = get_box_size(l)
    Δ = 3s/2

    max(0, c - Δ), min(1, c + Δ)
end

function get_E2_linspace(l::Integer, i::Integer, N::Integer)
    l, r = get_E2_interval(l, i)
    filter(x -> l <= x <= r, linspace(0, 1, N))
end

function get_E3_intervals(l::Integer, i::Integer)
    E2_left, E2_right = get_E2_interval(l, i)
    0, E2_left, E2_right, 1
end

function get_E3_linspace(l::Integer, i::Integer, N::Integer)
    l1, r1, l2, r2 = get_E3_intervals(l, i)
    L = linspace(0, 1, N)

    if l1 == 0 == r1 && l2 == 1 == r2
        return L
    elseif l1 == 0 == r1
        return filter(x -> l2 <= x, L)
    elseif l2 == 1 == r2
        return filter(x -> x <= r1, L)
    else
        return filter(x -> x <= r1 || l2 <= x, L)
    end
end

function get_E4_neighbors(l::Integer, i::Integer)
    assert(l > 0)
    assert(1 ≤ i ≤ 2^l)
    neighbors = []
    for j = get_E2_neighbors(l - 1, get_parent(i))
        neighbors = [neighbors; get_children(j)]
    end
    setdiff(neighbors, get_E2_neighbors(l, i))
end

get_box_index(x::Real, l::Integer) = floor(Integer, (1 << l)x)

function get_box_center(l::Integer, i::Integer)
    assert(0 ≤ l)
    assert(1 ≤ i ≤ 2^l)
    (i - 1/2)*get_box_size(l)
end

get_box_size(l::Integer) = 1/(1 << l)

"""
This function computes and returns a closure that returns bookmark pairs
that effectively serve as the algorithm's main data structure.
"""
function get_box_finder(X::AbstractArray, # normalized and sorted reals
                        L::Integer)       # maximum level
    assert(issorted(X))
    N = length(X)
    K = 2^L                     # the number of boxes

    # We start by performing a linear scan of X and create an array of
    # "bookmarks" -- pairs of integers which indicate the first and
    # last index of elements in X that are in one of the boxes at
    # level L (where L is thought of as the deepest level).
    
    bounds = linspace(0, 1, K + 1) # the boundaries of the boxes
    B = zeros(Integer, 2K)         # the bookmark data structure
    s = 0                          # the index of the linear scan
    t = s                          # the previous s
    k = 1                          # the box index
    
    while k ≤ K && s ≤ N
        R = bounds[k + 1]       # the right bound of the current box
        while s + 1 ≤ N && X[s + 1] ≤ R
            s += 1
        end
        if t ≠ s
            B[2k-1:2k] = [t + 1; s]
        end
        t = s
        k += 1
    end
    
    # This function is what is used to calculate the adjusted
    # bookmarks at each level.
    
    get_bookmark_at_level = function(l::Integer, # level
                                     i::Integer) # box index
        assert(0 ≤ l ≤ L)
        assert(1 ≤ i ≤ 2^l)

        # Convert the box index into the leftmost and rightmost
        # endpoints of the corresponding bookmarks at the finest level.
        
        α = 2^(L - l)
        l, r = α*(i - 1) + 1, α*i
        i, j = l, r

        # Scan inward from the leftmost endpoint to find the first
        # nonzero bookmark in the corresponding bookmark range.

        while i ≤ r && B[2i - 1] == 0
            i += 1
        end
        B_left = i > r ? 0 : B[2i - 1]

        # As before, but scanning from the right.

        while j ≥ l && B[2j] == 0
            j -= 1
        end
        B_right = j < l ? 0 : B[2j]

        # Make sure that the adjusted bookmark pair is valid.

        assert((B_left == 0 && B_right == 0) || (B_left ≠ 0 && B_right ≠ 0))
        assert(0 ≤ B_left ≤ B_right ≤ N)

        B_left, B_right
    end

    # Construct a dictionary of all of the (adjusted) bookmarks at each level.
        
    D = Dict{Integer, Array{Integer, 1}}()
    for l = 0:L
        D[l] = zeros(Integer, 2^(l + 1))
        for i = 1:2^l
            D[l][2i - 1], D[l][2i] = get_bookmark_at_level(l, i)
        end
    end

    # Return a closure which just indexes the bookmark dictionary to
    # desired bookmark at a given level and for a given box.

    function(l::Integer,        # level
             i::Integer)        # box index
        assert(0 ≤ l ≤ L)
        assert(1 ≤ i ≤ 2^l)
        D[l][2i - 1], D[l][2i]
    end
end

function initialize_target_coefs{T<:Number}(::Type{T},
                                            get_target_box::Function,
                                            l::Integer,
                                            p::Integer)
    coefs = Coefs{T}()
    for i = 1:2^l
        left, right = get_target_box(l, i)
        if left == 0 && right == 0
            continue
        else
            coefs[i] = zeros(T, p)
        end
    end
    coefs
end

end
