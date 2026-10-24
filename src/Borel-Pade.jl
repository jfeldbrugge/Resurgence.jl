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
    BorelPadeResum(trans, k; ϵ = 0.01, rtol = sqrt(eps(real(float(k)))))

Borel-Padé sum of the series Σₘ Tₘ/kᵐ (`trans.Ts`): the Borel transform Φ(t) = Σₘ Tₘ tᵐ/m! is
replaced by its diagonal Padé approximant and Laplace transformed, k ∫₀^∞ e^{-kt} Φ(t) dt, which
reproduces Σₘ Tₘ/kᵐ term by term. When a Padé pole lies within an angle `ϵ` of the positive real
axis (a Stokes line), the ray is rotated upwards, halfway to the next pole direction (at most π/4):
the lateral sum from above. The Laplace integral is computed in the precision of `k` (pass
`k::BigFloat` for results beyond double precision). Returns `(integral, error)`.
"""
function BorelPadeResum(trans, k; ϵ = 0.01, rtol = sqrt(eps(real(float(k)))))
    borelTransform = BorelTransform(trans.Ts, big"1")    # Tₘ tᵐ / m!

    order = length(trans.Ts) ÷ 2 - 1
    num, den = TaylorToPade(borelTransform, order, order)

    integrand(t, k) = k * exp(-k * t) * PadeEvaluate(num, den, t)

    poles = Polynomials.roots(Polynomials.Polynomial(den))
    angles = angle.(poles)

    R = real(float(typeof(k)))
    θ = zero(R)
    if any(a -> abs(a) < ϵ, angles)
        above = filter(>(ϵ), angles)
        θ = R(min(minimum(above; init = π / 2) / 2, π / 4))
    end

    order = R == Float64 ? 7 : 40                       # high order beyond double precision
    int, err = quadgk(t -> cis(θ) * integrand(t * cis(θ), k), zero(R), R(Inf); rtol = rtol, order = order)
    return int, err
end
