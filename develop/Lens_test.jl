cd(@__DIR__)
using Pkg
Pkg.activate(".")

using GaussManinResurgence
using DynamicPolynomials, MultivariatePolynomials, LinearAlgebra, CairoMakie, Alert

setprecision(BigFloat, 2 * 512)

######################################
# Define integral
######################################

# y, β = 1, 1;
# @polyvar x1 x2 x3 monomial_order = Graded{MultivariatePolynomials.Reverse{InverseLexOrder}}

# PP = 2β + (1 + x1^2) * (x1 - y)^2
# QQ = 2 * (1 + x1^2)

# f = PP * x2 + x3 * (x2 * QQ - 1)

# @polyvar x 
# f = x^3 + x

@polyvar x 
f = x^4 + 2x^2 + x

# @polyvar x y 
# f = 1//3 * x^3 + 1//3 * y^3 + 1//2 * x * y - x - 2 * y

vars = variables(f)

jacobi = differentiate(f, vars)
hessian = differentiate(jacobi, vars)

######################################
# Find saddle points
######################################

saddles = findAllSaddles(f)
saddles = sort(saddles, by=p->abs.(f(p)))
saddles = [refineSaddle(f, sd)[1] for sd in saddles]
T0s = [1 / sqrt(det([hess(p) for hess in hessian])) for p in saddles]
u = f.(saddles)

######################################
# Find the Gauss-Manin system
######################################

GM = GaussManin(f);

GM.ϕ
GM.M
GM.B
GM.λs

######################################
# Find asymptotic series
######################################

λⱼ, μⱼ, Ts = asymptotic(1, GM, 10)
Ts = T0s[argmin(abs.(u .- λⱼ))] .* Ts

transpose(Ts)

begin 
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="x", ylabel="y", yscale=log10)

	for i in 1:GM.n
		scatter!(ax, collect(0:size(Ts, 2) - 1), abs.(Ts[i,:]), markersize=10, label = string(i))
	end
	
	axislegend(ax)
	fig
end

M = 100
transseries = Vector{Expansion}(undef, GM.n);
for j in 1:GM.n
    λⱼ, μⱼ, Ts = asymptotic(j, GM, M)
    index = argmin(abs.(u .- λⱼ))
    Ts = T0s[index] .* Ts
    transseries[j] = Expansion(saddles[index], -im * λⱼ, μⱼ, im * Ts[1,:] ./ sqrt(big(im)))
end;

######################################
# Borel-Pade
######################################

begin  
    fig = Figure()
    ax = Axis(fig[1,1], xlabel = "m", ylabel = "|Tₘ⁽ʲ⁾|")
    for (i, trans) in pairs(transseries)
        scatter!(ax, 0:length(trans.Ts) - 1, log10.(abs.(trans.Ts)), label = string(i))
    end
    
    axislegend(ax)
    fig
end

K = findStokesConstants(transseries)
K_int = round.(Int, real.(K))

begin
    poles = [findPoles(trans) for trans in transseries];
    fig, ax = plotAdjacencyPoles(transseries, K_int, poles)
    fig
end

# BorelPadeResum(transseries[1], 1)[1]

######################################
# Ecalle's Mould expansion
######################################

ComplexF64.([4π^2 * tr(Ri * Rj) for Ri in GM.Rs, Rj in GM.Rs])

N = 4
Σ = [trans.f₀ for trans in transseries]

T = monodromy_matrices(Σ,  GM.Rs, ComplexF64(0), N; K = 40);
K_estimated = [tr((T[i] - I) * (T[j] - I)) for i in 1:GM.n, j in 1:GM.n];

K_int
sum(abs.(abs.(K_int) .- abs.(K_estimated)))

K_estimated

######################################
# Stokes constants via the keyhole contour
######################################

# Borel singularities t_a = -iλ_a and residues R_a of A(t) = Σ R_a/(t - t_a)
Σ  = ComplexF64.(-im .* GM.λs)
Rs = [ComplexF64.(R) for R in GM.Rs]

# Normalise each saddle solution like its asymptotic series: T₀ = 1/√det H of the saddle of λ_a
T0 = [ComplexF64(T0s[argmin(abs.(u .- λ))]) for λ in GM.λs]

# Stokes constant K_{j←i}: Laplace ray of saddle i crossing t_j (res ≈ 0 when converged)
K21, S, res = stokes_constant(Σ, Rs, 1, 2, 6; T0 = T0)
K21, res

Kmat = stokes_constants(Σ, Rs, 6; T0 = T0)   # Kmat[j, i] = K_{j←i}
Kmat_int = round.(Int, real.(Kmat))

# Convergence check in the hyperlogarithm weight N
# for N in 4:2:8
#     println("N = $N: ", round.(stokes_constants(Σ, Rs, N; T0 = T0), digits = 6))
# end

alert("Done")