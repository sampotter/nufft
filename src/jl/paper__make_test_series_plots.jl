using Gadfly
using TestSeries

import TestSeries

N = 5

semicircle_plot = let ψ = TestSeries.semicircle
    plot(map(n -> (z -> real(ψ(z, n))), 1:N), 0, 2π)
end

triangle_plot = let ψ = TestSeries.triangle
    plot(map(n -> (z -> real(ψ(z, n))), 1:N), 0, 2π)
end

square_plot = let ψ = TestSeries.square
    plot(map(n -> (z -> real(ψ(z, n))), 1:N), 0, 2π)
end

sawtooth_plot = let ψ = TestSeries.sawtooth
    plot(map(n -> (z -> real(ψ(z, n))), 1:N), 0, 2π)
end

draw(PS("semicircle.ps", 3inch, 2inch), semicircle_plot)
draw(PS("triangle.ps", 3inch, 2inch), triangle_plot)
draw(PS("square.ps", 3inch, 2inch), square_plot)
draw(PS("sawtooth.ps", 3inch, 2inch), sawtooth_plot)
