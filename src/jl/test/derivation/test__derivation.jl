import GroundTruth
import PeriodicSum
import TestSeries
import Util

using Gadfly

n₀ = 10                         # our bandlimit
K = 100                         # our bandlimit bound/guess
# f = TestSeries.semicircle       # our bandlimited function
f = TestSeries.square
X = Float64[2π*(0:2K-1)/2K;]    # the time domain grid
F = f(X, n₀)                  # b.l. function evaluated at t.d. grid
J = 2K                        # the number of points to interpolate at
Y = 2π*rand(J)                # the points to interpolate at

assert(K > n₀)

DFT_range = -(K - 1):(K - 1)
KK = GroundTruth.compute_K(X, Y, DFT_range)
G = KK*F
G_real = f(Y, n₀)

get_fresh_KK = () -> zeros(Complex128, J, 2K)
function print_mse(step::Integer, current_G)
    @printf("step %d: mse = %g\n", step, Util.mse(G, current_G))
end

#
# first step
#

KK1 = get_fresh_KK()
for j = 1:J
    for k = 1:2K
        t = Y[j] - X[k]
        KK1[j, k] = -1
        KK1[j, k] += sum(n -> exp(im*n*t), 0:K-1)
        KK1[j, k] += sum(n -> exp(-im*n*t), 0:K-1)
    end
end
KK1 /= 2K
G1 = KK1*F
print_mse(1, G1)

#
# second step
#

KK2 = get_fresh_KK()
for j = 1:J
    for k = 1:2K
        t = Y[j] - X[k]
        neg = (-1)^(k - 1)
        KK2[j, k] = -1
        KK2[j, k] += (1 - neg*exp(im*K*Y[j]))/(1 - exp(im*t))
        KK2[j, k] += (1 - neg*exp(-im*K*Y[j]))/(1 - exp(-im*t))
    end
end
KK2 /= 2K
G2 = KK2*F
print_mse(2, G2)

#
# third step
#

KK3 = get_fresh_KK()
for j = 1:J
    for k = 1:2K
        t = Y[j] - X[k]
        neg = (-1)^(k - 1)
        KK3[j, k] = -1
        KK3[j, k] += (1 - neg*exp(im*K*Y[j]))*((1 + im*cot(t/2))/2)
        KK3[j, k] += (1 - neg*exp(-im*K*Y[j]))*((1 + im*cot(-t/2))/2)
    end
end
KK3 /= 2K
G3 = KK3*F
print_mse(3, G3)

#
# fourth step
#

KK4 = get_fresh_KK()
for j = 1:J
    for k = 1:2K
        t = Y[j] - X[k]
        neg = (-1)^(k - 1)
        KK4[j, k] = -1
        KK4[j, k] += (1/2)*(2 - neg*(exp(im*K*Y[j]) + exp(-im*K*Y[j])))
        KK4[j, k] += (im/2)*neg*(exp(-im*K*Y[j]) - exp(im*K*Y[j]))*cot(t/2)
    end
end
KK4 /= 2K
G4 = KK4*F
print_mse(4, G4)

#
# fifth step
#

KK5 = get_fresh_KK()
for j = 1:J
    for k = 1:2K
        t = Y[j] - X[k]
        neg = (-1)^(k - 1)
        KK5[j, k] = -neg*cos(K*Y[j])
        KK5[j, k] += neg*sin(K*Y[j])*cot(t/2)
    end
end
KK5 /= 2K
G5 = KK5*F
print_mse(5, G5)

#
# sixth step
#

G6 = zeros(Complex128, 2K)
negs = (-1).^(0:2K-1)
ΣF = sum(negs.*F)
for j = 1:2K
    Σcot = sum(negs.*F.*cot((Y[j] - X)/2))
    G6[j] = -ΣF*cos(K*Y[j])
    G6[j] += sin(K*Y[j])*Σcot
end
G6 /= 2K
print_mse(6, G6)

#
# seventh step
#

G7 = zeros(Complex128, 2K)
negs = (-1).^(0:2K-1)
ΣF = sum(negs.*F)
for j = 1:2K
    per_sum_approx = 0
    for l = -3:3
        per_sum_approx += sum(k -> ((-1)^(k-1))*F[k]/(Y[j] - X[k] - 2π*l), 1:2K)
    end
    
    G7[j] = -ΣF*cos(K*Y[j])
    G7[j] += 2*sin(K*Y[j])*per_sum_approx
end
G7 /= 2K
print_mse(7, G7)

#
# eighth step
#

p = 10
depth = 5
x₀ = π
n = 5
nwidth = (n + 1)π           # neighborhood width
r = 0.9*nwidth              # inner radius
R = 1.1*nwidth              # outer radius
L = 50                      # num check points

G8 = zeros(Complex128, 2K)

negs = (-1).^(0:2K-1)
F_alt_sign = negs.*F

ΣFas = sum(F_alt_sign)
G8 += -ΣFas*cos(K*Y)

per_sum = PeriodicSum.periodic_sum(X, F_alt_sign, Y, p, depth, x₀, n, r, R, L)
G8 += 2*sin(K*Y).*per_sum

G8 /= 2K
print_mse(8, G8)
