cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Resurgence
using DynamicPolynomials, MultivariatePolynomials, LinearAlgebra, Serialization

setprecision(BigFloat, 2 * 512)

######################################
# Regulated parabolic umbilic: f + ε v^4 is bounded from below, so that the South-East rule for
# bounded exponents applies. The regulator adds four auxiliary critical points, which move to
# t = +i∞ as ε → 0. Usage: julia D5_regulated.jl 16   (ε = 1/16)
######################################

εinv = parse(Int, get(ARGS, 1, "16"))
ε = 1 // εinv
x, y, z, w = -5 // 4, 1 // 4, 1 // 1, 0 // 1
@polyvar u v
f = u^4 + u * v^2 + w * u^2 + z * v^2 + y * v + x * u + ε * v^4

vars = variables(f);
hessian = differentiate(differentiate(f, vars), vars);

saddles = [refineSaddle(f, sd)[1] for sd in findAllSaddles(f)];
T0s = [1 / sqrt(det([hess(p) for hess in hessian])) for p in saddles];
fs = f.(saddles);

GM = GaussManin(f);
Σ  = ComplexF64.(-im .* GM.λs);
Rs = [ComplexF64.(R) for R in GM.Rs];
T0 = [ComplexF64(T0s[argmin(abs.(fs .- λ))]) for λ in GM.λs];
xs = [saddles[argmin(abs.(fs .- λ))] for λ in GM.λs];

Kmat = stokes_constants(Σ, Rs, 6; T0 = T0, method = :matrix)
Kmat_int = round.(Int, real.(Kmat))
println("ε = 1/$εinv, max deviation from integers: ", maximum(abs.(Kmat .- Kmat_int)))
println("Σ = ", round.(Σ, digits = 4))
display(Kmat_int)

serialize("D5_regulated_$(εinv).jls", (; ε, Σ, T0, xs = [ComplexF64.(p) for p in xs], Kmat, Kmat_int))
println("saved D5_regulated_$(εinv).jls")
