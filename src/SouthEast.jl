# South-East rule: from adjacency (Stokes constants) to relevance (intersection numbers).
#
# For an exponent f bounded from below on ℝᵈ, rotate k ↦ k e^{iθ}. At θ = π/2 the integral is
# Euclidean and ℝᵈ is a union of steepest-descent manifolds of the real critical points: these
# are relevant (n_j = ±1, the sign fixed by the normalisation of the series), the complex critical
# points are not (n_j = 0). Rotating back to θ = 0, the Borel singularities t_j = -iλ_j rotate
# clockwise, and a relevant saddle x_j switches on x_l when t_j and t_l line up horizontally and
# the saddles are adjacent:
#     n_l ↦ n_l + n_j K_lj         (paper convention, S⁺ - S⁻ = K_lj I^(l)).
# Only edges pointing South-East, arg(t_l - t_j) ∈ [-π/2, 0], are crossed, from steep to shallow.
#
# `stokes_constants` returns K[l, j] = -K_lj (keyhole orientation γ₊ before the transition), so in
# terms of the package output the update reads n_l ↦ n_l - n_j K[l, j]  (convention = :package).

"""
    morse_orientation(H, T0; atol = 1e-6) -> Int

Intersection number n = ±1 of a real non-degenerate critical point in the Euclidean integral,
relative to the normalisation T0 (leading coefficient of its series, `1/√det H` in the package).
Stationary phase gives the contribution e^{-iπν/2}/√|det H| (ν the Morse index), so
n = e^{-iπν/2}/(√|det H| T0).
"""
function morse_orientation(H, T0; atol = 1e-6)
    h = eigvals(Symmetric(Float64.(real.(H))))
    minimum(abs, h) > atol || error("morse_orientation: degenerate critical point (Hessian eigenvalues $h)")
    ν = count(<(0), h)
    σ = cis(-π * ν / 2) / sqrt(prod(abs, h)) / ComplexF64(T0)
    abs(σ - round(real(σ))) < atol && abs(abs(σ) - 1) < atol ||
        error("morse_orientation: T0 = $T0 is not ±1/√det H for this real critical point (ratio $σ)")
    round(Int, real(σ))
end

"""
    initial_intersection_numbers(f, xs, T0; atol = 1e-8) -> Vector{Int}

Intersection numbers at θ = π/2 for the critical points `xs` of the polynomial f (bounded from
below on ℝᵈ): `morse_orientation` for the real critical points, 0 for the complex ones.
`T0[a]` is the normalisation of the series of `xs[a]`, as passed to `stokes_constants`.
"""
function initial_intersection_numbers(f, xs, T0; atol = 1e-8)
    vars = variables(f)
    Hpoly = differentiate(differentiate(f, vars), vars)
    map(eachindex(xs)) do a
        x = xs[a]
        all(abs(imag(c)) < atol * max(1, abs(c)) for c in x) || return 0
        xr = real.(x)
        morse_orientation([h(vars => xr) for h in Hpoly], T0[a])
    end
end

"""
    southeast_edges(Σ, K; include_vertical = false, atol = 1e-9)

Edges j → l of the adjacency graph (K[l, j] ≠ 0) pointing South-East in the Borel plane,
arg(t_l - t_j) ∈ [-π/2, 0], ordered from steep to shallow. Exactly vertical edges, crossed at the
Euclidean point θ = π/2 itself, are left out unless `include_vertical = true`.
Returns a vector of named tuples `(from = j, to = l, angle = θ, K = K[l, j])`.
"""
function southeast_edges(Σ, K; include_vertical = false, atol = 1e-9)
    n = length(Σ)
    size(K) == (n, n) || error("southeast_edges: K must be $(n)×$(n)")
    lo = include_vertical ? -π / 2 - atol : -π / 2 + atol
    edges = [(from = j, to = l, angle = angle(ComplexF64(Σ[l] - Σ[j])), K = K[l, j])
             for j in 1:n, l in 1:n if j != l && abs(K[l, j]) > 0.5]
    sort!(filter!(e -> lo <= e.angle <= atol, edges), by = e -> e.angle)
end

"""
    southeast(Σ, K, n0; convention = :package, include_vertical = false, verbose = false, atol = 1e-9)

Apply the South-East rule to the Borel singularities Σ (t_j = -iλ_j), the Stokes constants K and
the intersection numbers n0 at θ = π/2 (see `initial_intersection_numbers`); returns the
intersection numbers n at θ = 0, i.e. Ψ = Σ_j n_j Ψ^(j).

- `convention = :package`: K[l, j] as returned by `stokes_constants` (update n_l -= n_j K[l, j]);
  `convention = :paper`: K[l, j] = K_lj of the paper (update n_l += n_j K[l, j]). For the output of
  `findStokesConstants` (transposed with respect to `stokes_constants`) pass `transpose(K)`.
- K is rounded to integers (with a warning when it is not close to integers).
- A warning is issued when a firing edge is horizontal (the integral lies on a Stokes line, where
  the jump is only half) or when two firing edges at the same angle share a saddle.

For f unbounded from below the Euclidean integral diverges; n0 must then be supplied by other
means (e.g. from a regulated exponent f + ε‖x‖⁴, see Aniceto, Feldbrugge & Howls).
"""
function southeast(Σ, K, n0; convention = :package, include_vertical = false, verbose = false, atol = 1e-9)
    convention in (:package, :paper) || error("southeast: `convention` must be :package or :paper")
    s = convention === :package ? -1 : 1
    Kint = round.(Int, real.(K))
    maximum(abs, K - Kint; init = 0.0) < 1e-3 ||
        @warn "southeast: K is not close to integers (max deviation $(maximum(abs, K - Kint)))"
    n = collect(Int, n0)
    fired = []
    for e in southeast_edges(Σ, Kint; include_vertical = include_vertical, atol = atol)
        n[e.from] == 0 && continue
        Δ = s * n[e.from] * e.K
        abs(e.angle) <= atol &&
            @warn "southeast: edge $(e.from) → $(e.to) is horizontal; the integral lies on a Stokes line"
        any(f -> abs(f.angle - e.angle) <= atol && !isempty(intersect((f.from, f.to), (e.from, e.to))), fired) &&
            @warn "southeast: simultaneous transitions at angle $(rad2deg(e.angle))° share a saddle; the order is ambiguous"
        verbose && println("  edge $(e.from) → $(e.to) ($(round(rad2deg(e.angle), digits = 2))°): ",
                           "n_$(e.to) = $(n[e.to]) + $Δ")
        n[e.to] += Δ
        push!(fired, e)
    end
    n
end

"""
    relevance(f, Σ, K, T0; saddles = nothing, convention = :package, kwargs...)

South-East rule from start to finish for a polynomial f bounded from below on ℝᵈ: matches the
critical points (`saddles`, by default `findAllSaddles(f)`) to the Borel singularities
Σ[a] = -i f(x_a), determines the intersection numbers at θ = π/2 and applies `southeast`.
`T0[a]` must be the normalisation used in `stokes_constants` for Σ[a].
Returns `(n = n, n0 = n0, saddles = xs)` with `xs[a]` the critical point of Σ[a].
"""
function relevance(f, Σ, K, T0; saddles = nothing, convention = :package, atol = 1e-8, kwargs...)
    cands = saddles === nothing ? findAllSaddles(f) : saddles
    vars = variables(f)
    ts = [-im * ComplexF64(f(vars => x)) for x in cands]
    idx = [argmin(abs.(ts .- t)) for t in Σ]
    allunique(idx) || error("relevance: the critical values do not match Σ one-to-one")
    tscale = max(1, maximum(abs, Σ))
    maximum(abs.(ts[idx] .- Σ)) < 1e-6 * tscale ||
        error("relevance: some Borel singularities have no critical point (max mismatch $(maximum(abs.(ts[idx] .- Σ))))")
    xs = cands[idx]
    n0 = initial_intersection_numbers(f, xs, T0; atol = atol)
    n = southeast(Σ, K, n0; convention = convention, kwargs...)
    (n = n, n0 = n0, saddles = xs)
end
