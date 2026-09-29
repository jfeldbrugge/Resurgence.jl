"""
Borel transform: Σₘ Tₘ / kᵐ ↦ Σₘ Tₘ / Γ(m + α) tᵐ
"""
function BorelTransform(Ts, α)
    return [t / gamma(big(r-1) + α) for (r, t) in pairs(Ts)]
end

"""
Find poles
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
Evaluate the adjacency without Stokes constants
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
Resumation with Borel-Pade
"""
function BorelPadeResum(trans, k)
    borelTransform = BorelTransform(trans.Ts, big"0");

    order = length(trans.Ts) ÷ 2 - 1
    num, den = TaylorToPade(borelTransform, order, order)

    integrand(t, k) = exp(-k * t) * PadeEvaluate(num, den, t)

    poles = Polynomials.roots(Polynomials.Polynomial(den))
    poles_angles = sort(angle.(poles))

    ϵ = 0.01
    θ = 0
    if poles_angles[1] < ϵ
        θ = poles_angles[2] / 2
    end

    int, err = quadgk(t -> exp(im * θ) * integrand(t * exp(im * θ), k), 0.0, Inf)
    return int, err
end