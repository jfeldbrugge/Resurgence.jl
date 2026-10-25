"""
Borel transform: Σₘ Tₘ / kᵐ ↦ Σₘ Tₘ / Γ(m + α) tᵐ
"""
function BorelTransform(Ts, α)
    return [t / gamma(big(r-1) + α) for (r, t) in pairs(Ts)]
end

"""
    findPoles(trans)

Poles of the diagonal Padé approximant of the Borel transform of `trans.Ts` (an `Expansion`),
relative to the Borel singularity `trans.f₀`.
"""
function findPoles(trans)
    order = length(trans.Ts) ÷ 2 - 1
    borelTransform = BorelTransform(trans.Ts, big"0")
    _, den = TaylorToPade(borelTransform, order, order)
    poles = Polynomials.roots(Polynomials.Polynomial(den))
    return poles
end

"""
Find the Stokes constants
"""
function findStokesConstants(transseries, ϵ = 1e-2)
    K = zeros(Complex{BigFloat}, (length(transseries), length(transseries)))
    for j in eachindex(transseries), k in eachindex(transseries)
        if j ≠ k
            trans_j, trans_k = transseries[j], transseries[k]
            Fⱼₖ = -(trans_j.f₀ - trans_k.f₀)

            borelTransform = BorelTransform(trans_j.Ts, big"0.5")
            Φt = Polynomials.Polynomial(borelTransform)
            # t=Fⱼₖ-Fⱼₖ(s-1)^2
            Φs = Φt(Polynomials.Polynomial([0, 2Fⱼₖ, -Fⱼₖ]))

            order = length(trans_j.Ts) ÷ 2 - 1
            num, den = TaylorToPade(Polynomials.coeffs(Φs), order, order)
            pole = argmin(p -> abs(p - 1), Polynomials.roots(Polynomials.Polynomial(den)))

            if abs(pole - 1) < ϵ
                p = Polynomials.Polynomial(num)
                q = Polynomials.Polynomial(den)
                dq = Polynomials.derivative(q)
                K[j, k] = 2im * sqrt(π) / trans_k.Ts[1] * p(pole) / dq(pole)
            end
        end
    end
    return K
end

"""
    evaluateAdjacency(transseries, ϵ = 0.1)

Adjacency matrix (0/1) from the Padé poles of the Borel transforms, without Stokes constants.
"""
function evaluateAdjacency(transseries, ϵ = 0.1)
    K = zeros(Int, (length(transseries), length(transseries)))

    f0s = [trans.f₀ for trans in transseries]

    poles = [findPoles(trans) for trans in transseries]
    for i in eachindex(transseries), j in eachindex(transseries)
        if i ≠ j
            for pole in poles[i]
                if abs(pole .+ f0s[i] - f0s[j]) / abs(f0s[i] - f0s[j]) < ϵ
                    K[i, j] = 1
                end
            end

        end
    end

    return K
end

"""
    cut_poles(poles, zeros; doublet = 1e-6)

Poles of a Padé approximant that lie on its cut system: poles within a relative distance `doublet`
of a zero (Froissart doublets, which nearly cancel) are dropped.
"""
cut_poles(poles, zeros; doublet = 1e-6) =
    ComplexF64[p for p in poles if minimum((abs(p - z) for z in zeros); init = Inf) > doublet * abs(p)]

"""
    pole_links(p; ρ = 0.75)

Segments joining every pole to its two nearest neighbours, when closer than `ρ` times the smaller of
their moduli: a polyline approximation of Padé's cut system, which follows curved arcs as well.
"""
function pole_links(p; ρ = 0.75)
    links = Tuple{ComplexF64,ComplexF64}[]
    for (i, a) in enumerate(p)
        d = [(abs(b - a), m) for (m, b) in enumerate(p) if m != i]
        for (dist, m) in sort(d)[1:min(2, length(d))]
            dist < ρ * min(abs(a), abs(p[m])) && push!(links, (a, p[m]))
        end
    end
    links
end

# Does the ray arg t = θ meet the cut system: a pole direction within ϵ (radial cuts) or a link (arcs)?
function ray_blocked(θ, p, links; ϵ = 0.01)
    any(q -> abs(mod(angle(q) - θ + π, 2π) - π) < ϵ, p) && return true
    e = cis(-θ)
    for (a, b) in links
        a, b = a * e, b * e
        if sign(imag(a)) != sign(imag(b))
            x = real(a) - imag(a) * (real(b) - real(a)) / (imag(b) - imag(a))
            x > 0 && return true
        end
    end
    false
end

"""
    laplace_direction(poles, zeros, θ₀; ϵ = 0.01, side = :above, maxangle = π/4, step = 0.002)

Rotation δθ of the Laplace ray, relative to the Borel direction θ₀, that clears Padé's cut system
(`cut_poles`, `pole_links`, `ray_blocked`): 0 if the ray arg t = θ₀ is free, otherwise halfway from the
edge of the blocked window to the next blocked direction on `side` (`:above` or `:below`), at most
`maxangle` beyond the edge. Returns `(δθ, width)` with the width of the blocked window around θ₀
(0 if free): a radial cut (Stokes line) blocks a window of width ≈ 2ϵ, curved arcs a wider one.
"""
function laplace_direction(poles, zeros, θ₀; ϵ = 0.01, side = :above, maxangle = π/4, step = 0.002)
    side in (:above, :below) || error("laplace_direction: side must be :above or :below")
    p = cut_poles(poles, zeros)
    links = pole_links(p)
    blocked(δ) = ray_blocked(θ₀ + δ, p, links; ϵ = ϵ)
    blocked(0.0) || return 0.0, 0.0
    edge(σ) = (δ = 0.0; while blocked(δ) && abs(δ) < π; δ += σ * step; end; δ)
    σ = side === :above ? 1 : -1
    δ₁ = edge(σ)                                        # first free direction on `side`
    δ₂ = δ₁
    while !blocked(δ₂) && abs(δ₂ - δ₁) < 2maxangle
        δ₂ += σ * step
    end
    δ₁ + σ * min(abs(δ₂ - δ₁) / 2, maxangle), abs(δ₁ - edge(-σ))
end

"""
    BorelPadeResum(trans, k; ϵ = 0.01, side = :above, rtol = sqrt(eps(real(float(k)))))

Borel-Padé sum of the series Σₘ Tₘ/kᵐ (`trans.Ts`): the Borel transform Φ(t) = Σₘ Tₘ tᵐ/m! is
replaced by its diagonal Padé approximant and Laplace transformed, k ∫₀^∞ e^{-kt} Φ(t) dt, which
reproduces Σₘ Tₘ/kᵐ term by term. The Laplace ray runs in the Borel direction arg t = -arg k,
unless it meets Padé's cut system, the rows of poles along which the approximant imitates the branch
cuts of Φ (`laplace_direction`). Beyond such a cut the approximant converges to the wrong branch of Φ and
the error stalls at e^{-Re(k t_c)}, with t_c the crossing point. The ray is then rotated to `side`
(`:above` or `:below`), halfway to the next blocked direction (at most π/4 beyond the edge):
- a radial cut along the ray (a Stokes line) gives the lateral sum from that side;
- a band of curved arcs (Chebotarev cuts between several singularities) gives the Borel sum of the
  neighbouring sector, which differs from the Borel sum in the direction -arg k by the Stokes jumps
  of the singularities in between (a warning is issued). The method
  `BorelPadeResum(transseries, j, K, k)` adds these jumps.
The Laplace integral is computed in the precision of `k` (pass `k::BigFloat` for results beyond
double precision). Returns `(integral, error)`.
"""
function BorelPadeResum(trans, k; ϵ = 0.01, side = :above, rtol = sqrt(eps(real(float(k)))))
    r = _borel_pade_ray(trans, k; ϵ = ϵ, side = side, rtol = rtol)
    if r.width > 4ϵ
        @warn "BorelPadeResum: the Laplace ray arg t = $(round(r.θ₀, digits = 3)) crosses Padé's cut " *
              "system; integrating along arg t = $(round(r.θ₀ + r.δθ, digits = 3)) instead. This is the " *
              "Borel sum of the neighbouring sector: it differs by the Stokes jumps of the singularities " *
              "between the two directions (use BorelPadeResum(transseries, j, K, k) to add them)."
    end
    return r.integral, r.error
end

# Borel-Padé sum along the Borel direction θ₀ = -arg k, rotated by δθ around Padé's cut system.
function _borel_pade_ray(trans, k; ϵ, side, rtol)
    borelTransform = BorelTransform(trans.Ts, big"1")    # Tₘ tᵐ / m!

    order = length(trans.Ts) ÷ 2 - 1
    num, den = TaylorToPade(borelTransform, order, order)

    integrand(t, k) = k * exp(-k * t) * PadeEvaluate(num, den, t)

    poles = Polynomials.roots(Polynomials.Polynomial(den))
    zeros = Polynomials.roots(Polynomials.Polynomial(num))

    R = real(float(typeof(k)))
    θ₀ = -Float64(angle(k))                             # Borel direction
    δθ, width = laplace_direction(poles, zeros, θ₀; ϵ = ϵ, side = side)
    abs(δθ) < π/2 - 0.05 || error("BorelPadeResum: no pole-free Laplace direction on side $side " *
                                  "within the half-plane Re(kt) > 0")
    θ = R(θ₀ + δθ)

    order = R == Float64 ? 7 : 40                       # high order beyond double precision
    int, err = quadgk(t -> cis(θ) * integrand(t * cis(θ), k), zero(R), R(Inf); rtol = rtol, order = order)
    return (integral = int, error = err, θ₀ = θ₀, δθ = δθ, width = width)
end

"""
    BorelPadeResum(transseries, j, K, k; ϵ = 0.01, side = :above, depth = 3,
                   rtol = sqrt(eps(real(float(k)))))

Borel sum of the series of saddle point j in the Borel direction arg t = -arg k, from the Borel-Padé
sums of all series (`transseries`, a vector of `Expansion`s) and the Stokes constants K
(`stokes_constants` convention). The series of j is summed as by `BorelPadeResum(trans, k)`; when its
Laplace ray has to bend around Padé's cut system from -arg k to θ, the Stokes jumps of the adjacent
singularities in between are added:

    S_j(-arg k) = S_j(θ) - σ Σ_l K[l, j] e^{-k F_l} S_l(k),     F_l = t_l - t_j,

with σ = +1 for `side = :above` (counter-clockwise rotation) and σ = -1 for `:below`. The sum runs over
the l with K[l, j] ≠ 0 whose directions arg F_l lie strictly between -arg k and θ (more than `ϵ` from
-arg k; a singularity on the ray itself gives the lateral sum, as for a single series). The S_l are
computed in the same way, recursively up to `depth` levels. Returns `(integral, error)`.
"""
function BorelPadeResum(transseries::AbstractVector, j::Integer, K, k; ϵ = 0.01, side = :above, depth = 3,
                        rtol = sqrt(eps(real(float(k)))))
    r = _borel_pade_ray(transseries[j], k; ϵ = ϵ, side = side, rtol = rtol)
    value, err = r.integral, r.error
    σ = side === :above ? 1 : -1
    for l in eachindex(transseries)
        (l == j || abs(K[l, j]) < 0.5) && continue
        F = transseries[l].f₀ - transseries[j].f₀
        δ = mod(Float64(angle(F)) - r.θ₀ + π, 2π) - π
        crossed = r.δθ > 0 ? ϵ < δ < r.δθ : r.δθ < δ < -ϵ
        crossed || continue
        depth > 0 || error("BorelPadeResum: Stokes jump at singularity $l needed beyond the recursion depth")
        Sl, el = BorelPadeResum(transseries, l, K, k; ϵ = ϵ, side = side, depth = depth - 1, rtol = rtol)
        w = K[l, j] * exp(-k * F)
        value -= σ * w * Sl
        err += abs(w) * el
    end
    return value, err
end
