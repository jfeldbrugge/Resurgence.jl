module Resurgence

    using DynamicPolynomials, MultivariatePolynomials, LinearAlgebra, GenericSchur, HomotopyContinuation, SpecialFunctions, QuadGK, CairoMakie
	import Singular, Polynomials

    include("Saddles.jl")
    include("GaussManinSystem.jl")
    include("Asymptotic.jl")
    include("Pade.jl")
    include("Borel-Pade.jl")
    include("Plots.jl")
    include("Hyperlogarithm.jl")
    include("Mould.jl")
    include("SouthEast.jl")
    include("Superasymptotics.jl")
    include("Hyperterminants.jl")

    export findAllSaddles, refineSaddle
    export GaussManinSystem, GaussManin
    export Expansion, asymptotic, findPoles, findStokesConstants, BorelPadeResum
    export plotPoles, plotAdjacencyPoles

    export Series, one_s, letter, mul, sinv, sexp, base_value, transport,
        local_singular, loop_path, monodromy_factor, continue_along,
        evaluate_series, evaluate_series_by_weight, monodromy_pieces, monodromy_matrices,
        keyhole_paths, stokes_constant, stokes_constants,
        direct_loop_factor, trace_contributions,
        matrix_transport, frobenius_solution, frobenius_gauge, local_monodromy,
        path_strength, truncation_bound, sufficient_N, stokes_constant_increment
    export plotAdjacencyGM, plotStokesGraphGM
    export morse_orientation, initial_intersection_numbers, southeast_edges, southeast, relevance
    export saddle_series, singulants, divergence_rate, superasymptotic
    export hyperterminant, hyperasymptotic, path_lengths
end
