cd(@__DIR__)
using Pkg
Pkg.activate(".")

using GaussManinResurgence
using DynamicPolynomials, MultivariatePolynomials, LinearAlgebra, Printf

setprecision(BigFloat, 2 * 512)

######################################
# Single-loop formula (method = :loop) versus the two-path keyhole difference (method = :matrix)
# for the Stokes constants of the Pearcey integral, the Lorentzian lens (polynomial lift) and the
# parabolic umbilic of the manuscript.
######################################

function compare(name, f)
    println("\n== $name ==")
    vars = variables(f)
    hessian = differentiate(differentiate(f, vars), vars)
    saddles = [refineSaddle(f, sd)[1] for sd in findAllSaddles(f)]
    fs = f.(saddles)
    T0s = [1 / sqrt(det([h(p) for h in hessian])) for p in saddles]
    t = @elapsed GM = GaussManin(f)
    @printf("Gauss-Manin system: %.1f s, n = %d\n", t, GM.n)
    Σ  = ComplexF64.(-im .* GM.λs)
    Rs = [ComplexF64.(R) for R in GM.Rs]
    T0 = [ComplexF64(T0s[argmin(abs.(fs .- λ))]) for λ in GM.λs]
    n = length(Σ)

    results = Dict{Symbol, Any}()
    for method in (:matrix, :loop)
        stokes_constant(Σ, Rs, 1, 2, 6; T0 = T0, method = method)       # compile
        t = @elapsed out = [i == j ? (0.0im, nothing, 0.0) : stokes_constant(Σ, Rs, i, j, 6; T0 = T0, method = method)
                            for j in 1:n, i in 1:n]
        K = first.(out); res = maximum(last.(out))
        results[method] = K
        @printf("%-7s: %.2f s, max |K - round(K)| = %.1e, max residual = %.1e\n",
                method, t, maximum(abs.(K .- round.(real.(K)))), res)
    end
    Δ = maximum(abs.(results[:loop] .- results[:matrix]))
    @printf("max |K_loop - K_matrix| = %.1e\n", Δ)
    display(round.(Int, real.(results[:loop])))
    round.(Int, real.(results[:loop])) == round.(Int, real.(results[:matrix])) ||
        @warn "$name: the integer Stokes matrices differ"
end

@polyvar q
compare("Pearcey, f = q^4 + q^2/2 + q", q^4 + (1 // 2) * q^2 + q)

@polyvar x1 x2 x3 monomial_order = Graded{MultivariatePolynomials.Reverse{InverseLexOrder}}
PP = 2 + (1 + x1^2) * (x1 - 1)^2
QQ = 2 * (1 + x1^2)
compare("Lorentzian lens (lift), y = 1", PP * x2 + x3 * (x2 * QQ - 1))

@polyvar u v
compare("parabolic umbilic, (x, y, z, w) = (-5/4, 1/4, 1, 0)", u^4 + u * v^2 + v^2 + (1 // 4) * v - (5 // 4) * u)
