import GroundTruth
import PeriodicSum
import TestSeries
import Util

using MAT

timing_samples = 10

# Ks = unique(round(Integer, logspace(1, log10(500), 100)))
Ks = 10:100
groundtruth_time_avg = zeros(length(Ks))
persum_time_avg = zeros(length(Ks))
persum_mse_avg = zeros(length(Ks))

for i = 1:length(Ks)
    K = Ks[i]
    @printf("i = %d, K = %d\n", i, K)
    
    n₀ = 10                         # our bandlimit
    f = TestSeries.square
    X = Float64[2π*(0:2K-1)/2K;]    # the time domain grid
    F = f(X, n₀)                  # b.l. function evaluated at t.d. grid
    J = 2K                        # the number of points to interpolate at
    Y = 2π*rand(J)                # the points to interpolate at

    DFT_range = -(K - 1):(K - 1)
    
    gt_runs = Dict()

    @printf("- groundtruth:")
    groundtruth_times = zeros(timing_samples)
    KK = GroundTruth.compute_K(X, Y, DFT_range)
    KK*F # warm up
    for j = 1:timing_samples
        @printf(" %d", j)
        tic()
        KK = GroundTruth.compute_K(X, Y, DFT_range)
        G = KK*F
        groundtruth_times[j] = toq();
        gt_runs[j] = G
    end
    groundtruth_time_avg[i] = mean(groundtruth_times)
    @printf("\n")

    p = 10
    depth = 5
    x₀ = π
    n = 5
    nwidth = (n + 1)π           # neighborhood width
    r = 0.9*nwidth              # inner radius
    R = 1.1*nwidth              # outer radius
    L = 50                      # num check points

    persum_runs = Dict()

    function persum()
        G = zeros(Complex128, 2K)
        negs = (-1).^(0:2K-1)
        F_alt_sign = negs.*F
        ΣFas = sum(F_alt_sign)
        G += -ΣFas*cos(K*Y)
        per_sum = PeriodicSum.periodic_sum(X, F_alt_sign, Y, p, depth, x₀, n, r, R, L)
        G += 2*sin(K*Y).*per_sum
        G
    end

    @printf("- periodic sum:")
    persum_times = zeros(timing_samples)
    persum() # warm up
    for j = 1:timing_samples
        @printf(" %d", j)
        tic()
        G = persum()
        persum_times[j] = toq();
        persum_runs[j] = G
    end
    persum_time_avg[i] = mean(persum_times)
    @printf("\n")

    @printf("- computing mses\n");
    mses = Dict()
    for j = 1:timing_samples
        mses[j] = Util.mse(persum_runs[j], gt_runs[j])
    end
    persum_mse_avg[i] = mean(values(mses))
end

matwrite("numexp__interp_speed.mat",
         Dict{Any, Any}("Ks" => Float64[Ks;],
                        "groundtruth_time_avg" => groundtruth_time_avg,
                        "persum_time_avg" => persum_time_avg,
                        "persum_mse_avg" => persum_mse_avg))
