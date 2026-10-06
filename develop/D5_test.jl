cd(@__DIR__)
using Pkg
Pkg.activate(".")

using GaussManinResurgence
using DynamicPolynomials, MultivariatePolynomials, LinearAlgebra, Serialization

setprecision(BigFloat, 2 * 512)

######################################
# Parabolic umbilic (D5) diffraction integral
#   Ψ = (k/2πi) ∫ exp(ik f(u, v)) du dv,  f = u^4 + u v^2 + w u^2 + z v^2 + y v + x u
######################################

x, y, z, w = -5 // 4, 1 // 4, 1 // 1, 0 // 1
@polyvar u v
f = u^4 + u * v^2 + w * u^2 + z * v^2 + y * v + x * u

vars = variables(f);
jacobi = differentiate(f, vars);
hessian = differentiate(jacobi, vars);

######################################
# Saddle points and the Gauss-Manin system
######################################

saddles = sort(findAllSaddles(f), by = p -> abs.(f(p)))
saddles = [refineSaddle(f, sd)[1] for sd in saddles];
T0s = [1 / sqrt(det([hess(p) for hess in hessian])) for p in saddles];
fs = f.(saddles);

GM = GaussManin(f);
@show GM.ϕ
@show GM.M
@show GM.B

######################################
# Stokes constants from the Fuchsian system
######################################

Σ  = ComplexF64.(-im .* GM.λs);
Rs = [ComplexF64.(R) for R in GM.Rs];
T0 = [ComplexF64(T0s[argmin(abs.(fs .- λ))]) for λ in GM.λs];

# K[j, i] = K_{j←i}: Laplace ray of saddle i crossing t_j
Kmat = stokes_constants(Σ, Rs, 6; T0 = T0, method = :matrix)
Kmat_int = round.(Int, real.(Kmat))
println("max deviation from integers: ", maximum(abs.(Kmat .- Kmat_int)))
display(Kmat_int)

######################################
# Asymptotic series of every saddle point (first component = Ψ^(j)), normalised by 1/√det H
######################################

M = 100
series = map(1:GM.n) do j
    λⱼ, μⱼ, Ts = asymptotic(j, GM, M)
    index = argmin(abs.(fs .- λⱼ))
    (λ = λⱼ, μ = μⱼ, x = saddles[index], T = T0s[index] .* Ts[1, :])
end

serialize("D5_results.jls", (; params = (x, y, z, w), Σ, Kmat, Kmat_int, T0, series,
                               M = GM.M, B = GM.B, ϕ = GM.ϕ))
println("saved D5_results.jls")
