################################################################################
# get_parent_farfield_coefs_works

include("/Volumes/Molly/Dropbox/Research/nufft/src/jl/fmm1d.jl")
X = sort(rand(100))
U = randn(100)
L = 4
p = 5
finder = FMM1D.get_box_finder(X, L)
finest = FMM1D.Cauchy.get_finest_farfield_coefs(finder, X, U, L, p)
println("finest coefs"); println()
map(key -> begin println(key - 1); map(println, finest[key]); println() end,
    keys(finest))
parent = get_parent_farfield_coefs(finest, L, p)
println("parent coefs"); println()
map(key -> begin println(key - 1); map(println, parent[key]); println() end,
    keys(parent))
