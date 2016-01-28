import TestSeries

using MAT

N = 5
M = 100
X = (2π/M)*(0:M-1)

semicircle = zeros(Float64, M + 1, N)
triangle = zeros(Float64, M + 1, N)
square = zeros(Float64, M + 1, N)
sawtooth = zeros(Float64, M + 1, N)

function fill!(ψ, data)
    for n = 1:N
        func = z -> real(ψ(z, n))
        for m = 1:M
            data[m, n] = func(X[m])
        end
        data[M + 1, n] = data[1, n]
    end
end

fill!(TestSeries.semicircle, semicircle)
fill!(TestSeries.triangle, triangle)
fill!(TestSeries.square, square)
fill!(TestSeries.sawtooth, sawtooth)

matwrite("numexp__test_series.mat",
         Dict{Any, Any}("X" => Float64[X; 2π],
                        "semicircle" => semicircle,
                        "triangle" => triangle,
                        "square" => square,
                        "sawtooth" => sawtooth))
