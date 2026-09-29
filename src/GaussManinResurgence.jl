module GaussManinResurgence

    using DynamicPolynomials, MultivariatePolynomials, LinearAlgebra, HomotopyContinuation, SpecialFunctions, QuadGK, CairoMakie
	import Singular, Polynomials

    include("Saddles.jl")
    include("GaussManinSystem.jl")
    include("Asymptotic.jl")
    include("Pade.jl")
    include("Borel-Pade.jl")
    include("Plots.jl")
    include("Hyperlogarithm.jl")

    export findAllSaddles, refineSaddle
    export GaussManinSystem, GaussManin
    export Expansion, asymptotic, findPoles, findStokesConstants, BorelPadeResum
    export plotPoles, plotAdjacencyPoles

    export Series, one_s, letter, mul, sinv, sexp, base_value, transport,
        local_singular, loop_path, monodromy_factor, continue_along
end
