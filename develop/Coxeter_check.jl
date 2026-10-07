cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Resurgence
using DynamicPolynomials, LinearAlgebra, Printf

setprecision(BigFloat, 2 * 512)

######################################
# Coxeter-Dynkin check for the Pearcey integral (A3), f = q^4 + y q^2 + x q.
# For d = 1 the fibre f^{-1}(s) consists of four points, a vanishing cycle is the difference of the
# two points that collide at a critical value, and the intersection form is the dot product of
# point vectors. Monodromy around a critical value swaps the colliding points (a reflection).
#  1. Distinguished basis from a base point s0: Gram matrix and Coxeter element.
#  2. Topological Stokes constants along the straight segments t_j -> t_l, compared with the
#     Stokes constants of the Fuchsian system (stokes_constants), up to orientation conventions.
######################################

roots_of(x, y, s) = eigvals([0 0 0 (s - 0); 1 0 0 -x; 0 1 0 -y; 0 0 1 0] .+ 0im)   # q^4 + y q^2 + x q - s

# track the four roots of f = s along a polyline in the s-plane, labelled by the roots at its start
function track(x, y, path; steps = 2000)
    r = roots_of(x, y, path[1])
    for (a, b) in zip(path[1:end-1], path[2:end]), k in 1:steps
        new = roots_of(x, y, a + (k / steps) * (b - a))
        r = [new[argmin(abs.(new .- ri))] for ri in r]           # continuity
        allunique(r) || error("track: roots collided along the path")
    end
    r
end
track(x, y, s1, s2; kwargs...) = track(x, y, [s1, s2]; kwargs...)

# straight path from λ[j] + ε e to λ[l] - ε e, avoiding critical values on the segment by
# half-circles on the left of the directed line (as keyhole_paths with side = +1)
function segment_path(λ, j, l, ε; ρ = 0.1, npts = 60)
    e = (λ[l] - λ[j]) / abs(λ[l] - λ[j]); len = abs(λ[l] - λ[j])
    pos(k) = real(conj(e) * (λ[k] - λ[j]))
    hit = [k for k in eachindex(λ) if k ∉ (j, l) && 0 < pos(k) < len && abs(imag(conj(e) * (λ[k] - λ[j]))) < ρ * len]
    path = [λ[j] + ε * e]
    for k in sort(hit; by = pos)
        r = ρ * len
        append!(path, [λ[k] + r * e * cis(π * (1 - m / npts)) for m in 0:npts])   # left half-circle
    end
    push!(path, λ[l] - ε * e)
    path
end

# vanishing cycle of the critical point xc, oriented consistently with the normalisation of the
# Fuchsian system: φ(t) = Σ_p ±(dx/dt)(p) = C (t - t_c)^{-1/2} with the principal branch and the
# constant C = 2i/(f'' v), v = sqrt(2i/f''), of the θ = 0 thimble x = x_c + v s, t - t_c = s^2.
function vanishing(r, xc, tc, t, fpp, fp)
    i, j = sortperm(abs.(r .- xc))[1:2]
    φ = im / fp(r[i]) - im / fp(r[j])                             # dx/dt = 1/(-i f') = i/f'
    C = 2im / (fpp * sqrt(2im / fpp))
    σ = real(φ / (C * (t - tc)^(-1 / 2))) > 0 ? 1 : -1
    abs(abs(φ / (C * (t - tc)^(-1 / 2))) - 1) < 1e-2 || @warn "vanishing: |φ/leading| = $(abs(φ / (C * (t - tc)^(-1/2))))"
    δ = zeros(Int, 4); δ[i] = σ; δ[j] = -σ
    δ
end

function check(xp, yp)
    println("\n== f = q^4 + ($yp) q^2 + ($xp) q ==")
    x, y = Float64(xp), Float64(yp)
    fp(q) = 4q^3 + 2y * q + x
    fpp(q) = 12q^2 + 2y
    xs = filter(z -> true, eigvals([0 0 -x/4; 1 0 -y/2; 0 1 0] .+ 0im))      # f' = 4q^3 + 2y q + x
    λ = [z^4 + y*z^2 + x*z for z in xs]
    ε = 1e-3 * minimum(abs(λ[a] - λ[b]) for a in 1:3 for b in a+1:3)

    # 1. distinguished basis from a far base point s0, paths ordered by angle
    s0 = 20 * maximum(abs.(λ)) * cis(0.37)
    order = sortperm([angle((λ[j] - s0) * cis(-angle(-s0))) for j in 1:3])
    δ = Vector{Vector{Int}}(undef, 3)
    for j in 1:3
        e = (λ[j] - s0) / abs(λ[j] - s0)
        r_end = track(x, y, s0, λ[j] - ε * e)                     # roots at the end, labelled at s0
        δ[j] = vanishing(r_end, xs[j], -im * λ[j], -im * (λ[j] - ε * e), fpp(xs[j]), fp)
    end
    G = [dot(δ[a], δ[b]) for a in order, b in order]
    println("Gram matrix of the distinguished basis:"); display(G)
    @printf("det G = %.1f (A3 Cartan matrix: 4), eigenvalues of G: %s\n", det(G), string(round.(eigvals(G), digits = 3)))
    refl(d) = I - d * d' // 1                                      # x ↦ x - <x, δ> δ on Z^4
    C = prod(refl(δ[j]) for j in order)
    println("eigenvalues of the Coxeter element on Z^4: ", round.(eigvals(Float64.(C)), digits = 6))

    # 2. Stokes constants along straight segments t_j -> t_l (t = -i f), topologically
    Ktop = zeros(Int, 3, 3)
    for j in 1:3, l in 1:3
        j == l && continue
        e = (λ[l] - λ[j]) / abs(λ[l] - λ[j])
        path = segment_path(λ, j, l, ε)
        r_start = roots_of(x, y, path[1])
        r_end = track(x, y, path)
        δj = vanishing(r_start, xs[j], -im * λ[j], -im * path[1], fpp(xs[j]), fp)
        δl = vanishing(r_end, xs[l], -im * λ[l], -im * path[end], fpp(xs[l]), fp)   # labelled at the start
        Ktop[l, j] = -dot(δj, δl)                                  # jump of δ_j across t_l: -<δ_j, δ_l> δ_l
    end

    # Stokes constants of the Fuchsian system from the package
    @polyvar q
    f = q^4 + yp * q^2 + xp * q
    GM = GaussManin(f)
    Σ  = ComplexF64.(-im .* GM.λs)
    Rs = [ComplexF64.(R) for R in GM.Rs]
    perm = [argmin(abs.(λ .- ComplexF64(μ))) for μ in GM.λs]      # package index -> topological index
    T0 = [1 / sqrt(ComplexF64(12 * xs[p]^2 + 2y)) for p in perm]
    Kp = round.(Int, real.(stokes_constants(Σ, Rs, 6; T0 = T0, method = :loop)))
    Kpkg = zeros(Int, 3, 3)
    for a in 1:3, b in 1:3; Kpkg[perm[a], perm[b]] = Kp[a, b]; end
    println("topological Stokes constants (straight segments):"); display(Ktop)
    println("Stokes constants of the Fuchsian system:"); display(Kpkg)
    println("K_package == -K_topological (package convention = minus the paper's): ", Kpkg == -Ktop)
    println("adjacency (|K|) agrees: ", abs.(Ktop) == abs.(Kpkg))
end

check(1 // 1, 1 // 2)       # one real and two complex saddle points
check(1 // 4, -1 // 1)      # three real saddle points
