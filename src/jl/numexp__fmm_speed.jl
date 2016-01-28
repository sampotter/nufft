import FMM1D
import GroundTruth

using DataFrames
using Gadfly
using MAT

timing_samples = 50

Ns = unique(round(Integer, logspace(1, log10(5000))), 100)

direct_times_avg = zeros(length(Ns))
fmm_times_avg = zeros(length(Ns))

for i = 1:length(Ns)
    N = Ns[i]
    @printf("i = %d, N = %d\n", i, N)
    X = Float64[(2π/N)*(0:N-1);]
    U = randn(N)
    Y = 2π*rand(N)

    Φ = GroundTruth.get_Φ_direct(X, U)

    @printf("- direct:")
    direct_times = zeros(timing_samples)
    tmp = Φ(Y) # warm the function up
    for j = 1:timing_samples
        @printf(" %d", j)
        tic()
        Φ(Y)
        direct_times[j] = toq();
    end
    direct_times_avg[i] = mean(direct_times)
    @printf("\n")

    p = 5
    L = 4

    @printf("- fmm:")
    fmm_times = zeros(timing_samples)
    tmp = FMM1D.Cauchy.scaled_fmm(Y, X, U, L, p) # warm the function up
    for j = 1:timing_samples
        @printf(" %d", j)
        tic()
        FMM1D.Cauchy.scaled_fmm(Y, X, U, L, p)
        fmm_times[j] = toq();
    end
    fmm_times_avg[i] = mean(fmm_times)
    @printf("\n")
end

matwrite("numexp__fmm_speed.mat",
         Dict{Any, Any}("Ns" => Ns,
                        "directtimes" => direct_times_avg,
                        "fmmtimes" => fmm_times_avg))
