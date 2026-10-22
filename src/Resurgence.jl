module Resurgence

    using DynamicPolynomials, MultivariatePolynomials, LinearAlgebra, GenericSchur, HomotopyContinuation,
        SpecialFunctions, QuadGK, CairoMakie
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

    # Saddle points and the Gauss-Manin system
    export findAllSaddles, refineSaddle, GaussManin

    # Adjacency: Stokes constants from the Gauss-Manin connection
    export stokes_constant, stokes_constants

    # Adjacency from Borel-Padé (numerical alternative)
    export Expansion, asymptotic, findPoles, findStokesConstants, BorelPadeResum

    # Relevance: South-East rule
    export southeast, relevance

    # Asymptotics of the integral
    export transseries, singulants, divergence_rate, superasymptotic
    export hyperterminant, hyperasymptotic, path_lengths

    # Plots
    export plotStokesGraphGM, plotAdjacencyGM, plotAdjacencyPoles

end
