"""
    Expansion(x, f₀, μ, Ts)

Asymptotic expansion of a saddle point `x`: the exponent f₀ (Borel singularity), the exponent μ of
the prefactor and the coefficients `Ts`, as used by the Borel-Padé functions (`findPoles`,
`findStokesConstants`, `BorelPadeResum`).
"""
mutable struct Expansion
    x
    f₀
    μ
    Ts
end

"""
    asymptotic(j, GM::GaussManinSystem, order)

Asymptotic series of saddle point `j` (ordered as `GM.λs`) to `order`, from the recursion of the
Gauss-Manin system. Returns `(λ, μ, Ts)`: the critical value, the exponent of the prefactor and the
(components × orders) coefficient matrix, normalised such that `Ts[1, 1] = 1`.
"""
function asymptotic(j, GM::GaussManinSystem, order)
    # Dimension of the system
    d = length(GM.vars)

    # Number of saddle points
    n = size(GM.M, 1)

    # B in the eigenbasis of M
    B̃ = d//2 * I - (inv(GM.V) * GM.B * GM.V)

    # Parameters
    μ = diag(B̃)

    # Set Y₀
    Ys = zeros(Complex{BigFloat}, n, order + 1)
    Ys[j, 1] = 1

    # Perform recursion
    for m in 1:order
        for i in 1:n
            if i ≠ j
                ΣBY = sum(B̃[i, l] * Ys[l, m] for l in 1:n if l ≠ i)
                Ys[i, m + 1] = (ΣBY + (μ[i] - μ[j] + m - 1) * Ys[i, m]) / (GM.λs[j] - GM.λs[i])
            end
        end

        Ys[j, m + 1] = - sum(B̃[j, l] * Ys[l, m + 1] for l in 1:n if l ≠ j) / m
    end

    # Transform the Yₘ coefficient to the Tₘ coefficients
    Ts = copy(Ys)
    for m in 1:order + 1
        Ts[:, m] =  GM.V * Ys[:, m] ./ im^m
    end
    Ts = Ts ./ Ts[1,1]

    return GM.λs[j], μ[j], Ts
end
