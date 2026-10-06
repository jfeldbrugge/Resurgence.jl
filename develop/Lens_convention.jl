cd(@__DIR__)
using Pkg
Pkg.activate(".")
using GaussManinResurgence, DynamicPolynomials, MultivariatePolynomials, LinearAlgebra
setprecision(BigFloat, 2 * 512)

# Convert the Stokes constants of the lens lift (stokes_constants, K[l,j] = K_{l←j}, package
# normalisation T0 = 1/sqrt(det H_lift)) to the convention of the manuscript: S⁺ - S⁻ = K_lj I^(l),
# thimbles of the one-dimensional integral normalised by 1/sqrt(f''(x_j)) (principal branch; for the
# real saddle this is the stationary-phase contribution, n = +1). J^lift_j = s_j J^1D_j with
# s_j = i Q(x_j) sqrt(f''(x_j)) T0_j, such that K^paper_lj = -K^code_lj s_l / s_j.
y, β = 1, 1
@polyvar x1 x2 x3 monomial_order = Graded{MultivariatePolynomials.Reverse{InverseLexOrder}}
PP = 2β + (1 + x1^2) * (x1 - y)^2
QQ = 2 * (1 + x1^2)
F = PP * x2 + x3 * (x2 * QQ - 1)
hessian = differentiate(differentiate(F, variables(F)), variables(F))
saddles = [refineSaddle(F, sd)[1] for sd in findAllSaddles(F)]
T0s = [ComplexF64(1 / sqrt(det([hess(p) for hess in hessian]))) for p in saddles]
ts = [ComplexF64(-im * F(p)) for p in saddles]
lab(t) = real(t) > 1 ? 1 : real(t) < -1 ? 2 : abs(real(t)) < 1e-6 ? 3 : real(t) > 0 ? 4 : 5
order = sortperm(lab.(ts))
xs = [ComplexF64(saddles[i][1]) for i in order]; T0 = T0s[order]
d2f(x) = 1 - (2 - 6x^2) / (1 + x^2)^3
s = [im * 2(1 + x^2) * sqrt(d2f(x)) * T0[j] for (j, x) in enumerate(xs)]
println("s = ", round.(s, digits = 8))

Kcode = [0 0 -1 -2 -1; 0 0 -1 1 2; 1 1 0 -1 -1; 2 -1 1 0 1; 1 -2 1 -1 0]   # Lens_stokes_check.jl
S = round.(Int, real.(s))
Kpaper = [-Kcode[l, j] * S[l] ÷ S[j] for l in 1:5, j in 1:5]
println("K (manuscript convention) = "); display(Kpaper)
println("K_43 = ", Kpaper[4, 3], "  (direct measurement of the Stokes jump: +1)")
