f1(t, N) = sum(n -> exp(im*n*t), -(N-1):N-1)

function f2(t, N)
    if abs(t) ≤ sqrt(eps())
        f1(t, N)
    else
        (exp(-im*(N-1)*t) - exp(im*N*t))/(1 - exp(im*t))
    end
end

N = 10

I = 100
X = linspace(0, 2π, I)

F1 = zeros(Complex128, I)
F2 = zeros(Complex128, I)

for i = 1:I
    display(i)
    F1[i] = f1(X[i], N)
    F2[i] = f2(X[i], N)
end

plot(layer(x = X, y = real(F1), Geom.line, Theme(default_color = colorant"red")),
     layer(x = X, y = real(F2), Geom.line, Theme(default_color = colorant"green")))
