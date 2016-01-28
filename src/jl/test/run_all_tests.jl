reload("FMM1D")

tests = ["test__fmm1d_cauchy__do_E4_SR_translations.jl",
         "test__fmm1d_cauchy__do_RR_translations.jl",
         "test__fmm1d_cauchy__evaluate_Phi.jl",
         "test__fmm1d_cauchy__fmm.jl",
         "test__fmm1d_cauchy__fmm_unsorted_input.jl",
         "test__fmm1d_cauchy__fmm_scaling.jl",
         "test__fmm1d_cauchy__get_RR_matrix.jl",
         "test__fmm1d_cauchy__get_SR_matrix.jl",
         "test__fmm1d_cauchy__get_finest_farfield_coefs.jl",
         "test__fmm1d_cauchy__get_parent_farfield_coefs.jl",
         "test__fmm1d_cauchy__merge_child_ffes.jl"]

for test in tests
    @printf("\nrunning test: \"%s\"\n", test)
    include(test)
    @printf("\n")
end
