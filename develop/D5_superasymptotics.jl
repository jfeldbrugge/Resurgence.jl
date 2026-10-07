cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Resurgence, DynamicPolynomials
using LinearAlgebra, Serialization, Printf

setprecision(BigFloat, 2 * 512)

######################################
# Parabolic umbilic: relevance and superasymptotics, using the output of D5_test.jl.
#   Ψ = (k/2πi) ∫ exp(ik f(u, v)) du dv,  f = u^4 + u v^2 + w u^2 + z v^2 + y v + x u.
# The Gaussian v-integral reduces Ψ to a one-dimensional integral, which we evaluate along a
# deformed contour (the "exact" value) and along the thimbles of the reduced exponent
#   g(u) = u^4 + w u^2 + x u - y^2/(4(u+z)),
# whose critical points and values coincide with those of f. Writes D5Superasymptotics.dat.
######################################

R = deserialize("D5_results.jls")
x, y, z, w = Float64.(R.params)
n = length(R.series)
λs = [ComplexF64(s.λ) for s in R.series]
ts = -im .* λs
us = [ComplexF64(s.x[1]) for s in R.series]
Ts = [ComplexF64.(s.T) for s in R.series]
K  = R.Kmat_int
adjacent(j) = [l for l in 1:n if l != j && K[l, j] != 0]

g(u)   = u^4 + w*u^2 + x*u - y^2 / (4(u + z))
gp(u)  = 4u^3 + 2w*u + x + y^2 / (4(u + z)^2)
gpp(u) = 12u^2 + 2w - y^2 / (2(u + z)^3)

# exact integral along the real line, indented by a small bump above the essential singularity at
# u = -z (where the integrand decays) and with tails tilted into the valleys of exp(ik u^4), such
# that Im g ≥ 0 along the whole contour (no exponentially large intermediate values)
softplus(t; c = 4) = log1p(exp(c * t)) / c
dsoftplus(t; c = 4) = 1 / (1 + exp(-c * t))
function exact(k; h = 2e-5, L = 5.0, δ = 0.03, σ = 0.1, α = 0.3, a = 1.5, b = -z - 1.5)
    acc = zero(ComplexF64)
    for s in -L:h:L
        H  = δ * exp(-((s + z) / σ)^2) + α * softplus(s - a) - α * softplus(b - s)
        dH = -2(s + z) / σ^2 * δ * exp(-((s + z) / σ)^2) + α * dsoftplus(s - a) + α * dsoftplus(b - s)
        u = s + im * H
        acc += sqrt(π / (-im * k * (u + z))) * exp(im * k * g(u)) * (1 + im * dH)
    end
    k / (2π * im) * acc * h
end

# thimble integral of saddle j, normalised like the asymptotic series (leading term e^{ikλ} T_0)
function thimble(j, k; h = 1e-3)
    uj, λ = us[j], λs[j]
    v = sqrt(2im / gpp(uj))
    S = sqrt(60 / k)
    pref = k / (2π * im) * sqrt(π / (-im * k))
    total = zero(ComplexF64)
    for sgn in (1, -1)
        u = uj; uprev = uj; r = sqrt(uj + z); acc = 0.0im; s = 0.0
        while s < S
            s += h
            ug = s ≈ h ? uj + sgn * v * s : 2u - uprev
            uprev = u; u = ug
            for _ in 1:40
                du = (g(u) - λ - im * s^2) / gp(u); u -= du
                abs(du) < 1e-14 && break
            end
            rn = sqrt(u + z); r = abs(rn - r) < abs(rn + r) ? rn : -rn
            acc += exp(-k * s^2) * 2im * (sgn * s) / gp(u) / r
        end
        total += acc * h
    end
    total += h * v / sqrt(uj + z)
    lead = pref * v / sqrt(uj + z) * sqrt(π / k)
    σ = lead / Ts[j][1]
    abs(abs(σ) - 1) < 1e-8 || @warn "saddle $j: normalisation |σ| = $(abs(σ))"
    exp(im * k * λ) * pref * total / round(real(σ))
end

######################################
# Relevance: fit Ψ = Σ n_j Ψ^(j)
######################################

ks = 3.0:1.0:8.0
A = [thimble(j, k) for k in ks, j in 1:n]
b = [exact(k) for k in ks]
nfit = [real(A); imag(A)] \ [real(b); imag(b)]
println("intersection numbers n = ", round.(nfit, digits = 6))
println("relative residual = ", norm(A * round.(nfit) - b) / norm(b))
nj = round.(Int, nfit)

######################################
# Divergence of the series: the nearest adjacent singularity, not the nearest singularity
######################################

for j in 1:n
    M = length(Ts[j])
    est = [exp((sum(log, 1:m-1; init = 0.0) - log(abs(Ts[j][m+1]))) / m) for m in M-11:M-1]
    Fadj = minimum(abs(ts[l] - ts[j]) for l in adjacent(j))
    Fall = minimum(abs(ts[l] - ts[j]) for l in 1:n if l != j)
    @printf("saddle %d: (Γ(m)/|T_m|)^(1/m) ≈ %.3f,  nearest adjacent |F| = %.3f,  nearest |F| = %.3f\n",
            j, est[end], Fadj, Fall)
end

######################################
# Superasymptotics
######################################

F = [minimum(abs(ts[l] - ts[j]) for l in adjacent(j)) for j in 1:n]
relevant = [j for j in 1:n if nj[j] != 0]
real_relevant = [j for j in relevant if abs(imag(us[j])) < 1e-8]
SA(j, k) = nj[j] * exp(im * k * λs[j]) * sum(Ts[j][m+1] / k^m for m in 0:ceil(Int, F[j] * k)-1)

open("D5Superasymptotics.dat", "w") do io
    println(io, "k ReExact ImExact ReSA ImSA errSA errSAreal")
    for k in 1.0:0.25:30.0
        Ψ = exact(k)
        Ψsa = sum(SA(j, k) for j in relevant)
        Ψreal = sum(SA(j, k) for j in real_relevant)
        @printf(io, "%.2f %.12e %.12e %.12e %.12e %.6e %.6e\n", k, real(Ψ), imag(Ψ),
                real(Ψsa), imag(Ψsa), abs(Ψ - Ψsa), abs(Ψ - Ψreal))
    end
end
println("relevant saddles: ", relevant, ", F = ", round.(F, digits = 4))
println("wrote D5Superasymptotics.dat")
