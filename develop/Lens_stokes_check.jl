cd(@__DIR__)
using Pkg
Pkg.activate(".")
using GaussManinResurgence, DynamicPolynomials, MultivariatePolynomials, LinearAlgebra
setprecision(BigFloat, 2 * 512)

# Stokes constants of the one-dimensional Lorentzian lens (y = 1) from its polynomial lift, with
# stokes_constants (K[l, j] = K_{l←j}, minus the convention of the paper), to compare with the
# matrix printed in the manuscript.
y, β = 1, 1
@polyvar x1 x2 x3 monomial_order = Graded{MultivariatePolynomials.Reverse{InverseLexOrder}}
PP = 2β + (1 + x1^2) * (x1 - y)^2
QQ = 2 * (1 + x1^2)
f = PP * x2 + x3 * (x2 * QQ - 1)

vars = variables(f)
hessian = differentiate(differentiate(f, vars), vars)
saddles = [refineSaddle(f, sd)[1] for sd in findAllSaddles(f)]
T0s = [1 / sqrt(det([hess(p) for hess in hessian])) for p in saddles]
fs = f.(saddles)
GM = GaussManin(f)
Σ  = ComplexF64.(-im .* GM.λs)
Rs = [ComplexF64.(R) for R in GM.Rs]
T0 = [ComplexF64(T0s[argmin(abs.(fs .- λ))]) for λ in GM.λs]
K = stokes_constants(Σ, Rs, 6; T0 = T0, method = :matrix)
println("max deviation from integers: ", maximum(abs.(K .- round.(K))))
println("Σ = ", round.(Σ, digits = 4))
display(round.(Int, real.(K)))
