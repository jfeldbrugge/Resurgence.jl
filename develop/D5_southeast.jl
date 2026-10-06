cd(@__DIR__)
using Pkg
Pkg.activate(".")

using GaussManinResurgence, DynamicPolynomials
using LinearAlgebra, Serialization, Printf

######################################
# South-East rule for the parabolic umbilic, using the regulated exponent f + ε v^4 (bounded from
# below) computed by D5_regulated.jl. stokes_constants returns K[l, j] = K_{l←j} with the
# keyhole orientation γ₊ = before the transition, i.e. minus the Stokes constant of the paper
# (S⁺ - S⁻ = K_lj I^(l)). The paper's update rule n_l ↦ n_l + n_j K_lj thus reads
# n_l ↦ n_l - n_j K[l, j] in terms of the package output.
######################################

x, y, z, w = -5 / 4, 1 / 4, 1.0, 0.0

# orientation of a real critical point relative to the normalisation T0 = 1/sqrt(det H):
# the real Gaussian integral (k/2πi) ∫ e^{ik Δuᵀ H Δu/2} = (1/i) Π_i (-i h_i)^{-1/2}
function orientation(p, T0, ε)
    u, v = real.(p)
    H = [12u^2 + 2w  2v; 2v  2u + 2z + 12ε * v^2]
    G = (1 / im) * prod((-im * h)^(-1 / 2) for h in eigvals(Symmetric(H)))
    σ = G / T0
    abs(abs(σ) - 1) < 1e-8 || @warn "orientation: |σ| = $(abs(σ))"
    round(Int, real(σ))
end

function southeast(Σ, K, xs, T0, ε; verbose = false)
    n = [abs(imag(p[1])) < 1e-9 && abs(imag(p[2])) < 1e-9 ? orientation(p, T0[a], ε) : 0
         for (a, p) in enumerate(xs)]
    # South-East edges, excluding the vertical edges between real critical points (θ = π/2 itself)
    edges = [(j, l) for j in eachindex(Σ), l in eachindex(Σ)
             if j != l && K[l, j] != 0 && -π/2 + 1e-9 < angle(Σ[l] - Σ[j]) <= 1e-12]
    for (j, l) in sort(edges, by = e -> angle(Σ[e[2]] - Σ[e[1]]))
        if n[j] != 0
            verbose && @printf("  edge %d → %d (%.1f°): n_%d = %d - (%d)(%d)\n", j, l,
                               rad2deg(angle(Σ[l] - Σ[j])), l, n[l], n[j], K[l, j])
            n[l] -= n[j] * K[l, j]
        end
    end
    n
end

# unregulated system: labels of D5_results.jls
R = deserialize("D5_results.jls")
Σ0 = R.Σ
xs0 = [ComplexF64.(s.x) for s in R.series]
T00 = [ComplexF64(s.T[1]) for s in R.series]

for εinv in (128, 256)
    Rr = deserialize("D5_regulated_$(εinv).jls")
    println("ε = 1/$εinv")
    n = southeast(Rr.Σ, Rr.Kmat_int, Rr.xs, Rr.T0, 1 / εinv; verbose = true)
    # map the regulated critical points onto the unregulated labels
    for j in eachindex(Σ0)
        a = argmin(abs.(Rr.Σ .- Σ0[j]))
        @printf("  saddle %d: t = %s  ->  n = %d\n", j, round(Σ0[j], digits = 3), n[a])
    end
    aux = [a for a in eachindex(Rr.Σ) if minimum(abs.(Σ0 .- Rr.Σ[a])) > 1]
    println("  auxiliary: t = ", round.(Rr.Σ[aux], digits = 1), ", n = ", n[aux])
end

println("unregulated, no auxiliary points:")
n0 = southeast(Σ0, R.Kmat_int, xs0, T00, 0.0; verbose = true)
println("  n = ", n0)
