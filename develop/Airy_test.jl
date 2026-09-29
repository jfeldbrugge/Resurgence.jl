cd(@__DIR__)
using Pkg
Pkg.activate(".")

using GaussManinResurgence
using DynamicPolynomials, MultivariatePolynomials, LinearAlgebra, HomotopyContinuation, CairoMakie, Alert

######################################
# Define integral
######################################

@polyvar x
f = x^3 + x
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

######################################
# Borel-Pade
######################################

transseries = Vector{Expansion}(undef, GM.n);
for j in 1:GM.n
    λⱼ, μⱼ, Ts = asymptotic(j, GM, 50)
    index = argmin(abs.(u .- λⱼ))
    Ts = T0s[index] .* Ts
    transseries[j] = Expansion(saddles[index],-im * λⱼ, μⱼ, Ts[1,:] ./ sqrt(im))
end;

begin  
    fig = Figure()
    ax = Axis(fig[1,1], xlabel = "m", ylabel = "|Tₘ⁽ʲ⁾|")
    for (i, trans) in pairs(transseries)
        scatter!(ax, 0:length(trans.Ts) - 1, log10.(abs.(trans.Ts)), label = string(i))
    end
    
    axislegend(ax)
    fig
end

begin
    poles = [findPoles(trans) for trans in transseries];
    K = findStokesConstants(transseries)
    K_int = round.(Int, real.(K))
    
    fig, ax = plotAdjacencyPoles(transseries, K_int, poles)
    fig
end

# BorelPadeResum(transseries[1], 1)[1]

alert("Done")