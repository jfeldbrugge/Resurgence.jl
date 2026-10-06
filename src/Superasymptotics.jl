# Superasymptotics: optimally truncated transseries of the relevant saddle points.
#
#     Ψ(k) ≈ Σ_j n_j e^{ikλ_j} (ik)^{μ_j} Σ_{m=0}^{N_j-1} T_m^(j) / k^m ,
#
# with n_j the intersection numbers (South-East rule) and T^(j) the asymptotic series of saddle j,
# normalised by its leading coefficient T_0^(j) = 1/√det H. The late terms of saddle j grow as
# Γ(m)/F_j^m, with F_j = min_{l: K_lj ≠ 0} |t_l - t_j| the distance to the nearest *adjacent*
# Borel singularity, so the terms are smallest near N_j ≈ |k| F_j and the error is O(e^{-|k| F_j}).

"""
    morse_normalisation(H; atol = 1e-8)

Leading coefficient T0 = 1/√det H of the series of a critical point with Hessian H. For a complex
critical point, the principal branch is used. For a real critical point with Morse index ν (the
number of negative eigenvalues of H), the branch is fixed by stationary phase,
√det H = e^{iπν/2} √|det H|, such that T0 = (-i)^ν / √|det H| is the contribution of the real
critical point to the integral: real critical points then enter with intersection number n = +1
(see `morse_orientation`). For det H < 0 the principal branch would be ambiguous, as it depends on
the sign of the rounding error in the imaginary part of det H.
"""
function morse_normalisation(H; atol = 1e-8)
    d = det(H)
    scale = max(1, maximum(abs, H))
    if maximum(abs ∘ imag, H) > atol * scale
        return 1 / sqrt(d)
    end
    h = eigvals(Symmetric(Float64.(real.(H))))
    minimum(abs, h) > atol * scale || error("morse_normalisation: degenerate critical point (Hessian eigenvalues $h)")
    ν = count(<(0), h)
    (-im)^ν / sqrt(abs(real(d)))
end

"""
    saddle_series(f, GM, M; saddles = nothing)

Asymptotic series of every saddle point to order M, ordered as `GM.λs` and normalised such that
the leading coefficient of the first component (Ψ itself) is T0 = 1/√det H, with the branch fixed
by `morse_normalisation` (principal branch for complex critical points, Morse convention for real
ones, so that real critical points enter with n = +1).
Returns a vector of named tuples `(λ, t, μ, x, T0, T)` with t = -iλ the Borel singularity,
x the critical point and T the (components × orders) coefficient matrix; T[1, :] is the series of Ψ.
Pass the same `T0` to `stokes_constants` and `relevance`.
"""
function saddle_series(f, GM, M; saddles = nothing)
    vars = variables(f)
    xs = saddles === nothing ? [refineSaddle(f, s)[1] for s in findAllSaddles(f)] : saddles
    Hpoly = differentiate(differentiate(f, vars), vars)
    fs = [f(vars => x) for x in xs]
    map(eachindex(GM.λs)) do j
        λ, μ, Ts = asymptotic(j, GM, M)
        a = argmin(abs.(fs .- λ))
        isreal_point = maximum(abs ∘ imag, xs[a]) < 1e-8 * max(1, maximum(abs, xs[a]))
        H = [h(vars => (isreal_point ? real.(xs[a]) : xs[a])) for h in Hpoly]
        T0 = morse_normalisation(H)
        (λ = λ, t = -im * λ, μ = μ, x = xs[a], T0 = T0, T = T0 .* Ts)
    end
end

"""
    singulants(Σ, K)

Distance F_j = min_{l: K[l, j] ≠ 0} |t_l - t_j| from every Borel singularity to the nearest adjacent
one (Inf for a saddle without adjacent saddles, whose series then converges).
"""
singulants(Σ, K) = [minimum((abs(Σ[l] - Σ[j]) for l in eachindex(Σ) if l != j && abs(K[l, j]) > 0.5);
                            init = Inf) for j in eachindex(Σ)]

"""
    divergence_rate(T; tail = 10, component = 1)

Estimate of the singulant |F| from the late terms of a series, (Γ(m)/|T_m|)^{1/m}, averaged over
the last `tail` orders (row `component` when T is a coefficient matrix). Compare with `singulants`
to check the adjacency of a saddle point; the estimate approaches F slowly, with O(log m/m) corrections.
"""
function divergence_rate(T; tail = 10, component = 1)
    T isa AbstractMatrix && (T = T[component, :])
    M = length(T) - 1
    est = [exp((loggamma(m) - log(abs(T[m+1]))) / m) for m in max(1, M - tail + 1):M if T[m+1] != 0]
    Float64(sum(est) / length(est))
end

"Truncation order: `:singulant` (N = ⌈|k| F⌉), `:smallest` (index of the smallest term) or an Int."
function truncation_order(T, F, k, truncation)
    M = length(T)
    truncation isa Integer && return min(truncation, M)
    if truncation === :singulant
        isfinite(F) || return M
        N = ceil(Int, abs(k) * F)
        N > M && @warn "superasymptotic: N = $N exceeds the $M available terms; using all of them"
        return min(N, M)
    elseif truncation === :smallest
        mags = [T[m+1] == 0 ? Inf : Float64(log(abs(T[m+1]))) - m * log(abs(k)) for m in 0:M-1]
        return argmin(mags) - 1      # sum m < N: up to and excluding the smallest term
    end
    error("superasymptotic: unknown truncation $truncation")
end

"""
    superasymptotic(series, n, Σ, K, k; component = 1, truncation = :singulant)

Superasymptotic approximation Σ_j n_j e^{ikλ_j} (ik)^{μ_j} Σ_{m<N_j} T_m^(j)/k^m of component
`component` (1 = Ψ) of the moments, with `series` from `saddle_series`, intersection numbers n
(from `relevance`), Borel singularities Σ and Stokes constants K (`stokes_constants` convention,
only adjacency K[l, j] ≠ 0 is used).

`truncation`: `:singulant` truncates at N_j = ⌈|k| F_j⌉, `:smallest` at the smallest term of each
series, an Int at a fixed order (fixed-order asymptotics, for comparison).

Returns `(value, N, error, terms)`: the approximation, the truncation orders, the size of the first
omitted terms Σ_j |n_j e^{ikλ_j} T_{N_j}/k^{N_j}| as an error estimate, and the contribution of every
saddle point.
"""
function superasymptotic(series, n, Σ, K, k; component = 1, truncation = :singulant)
    F = singulants(Σ, K)
    N = zeros(Int, length(series))
    terms = Vector{Any}(undef, length(series))
    err = 0.0
    for (j, s) in enumerate(series)
        T = s.T isa AbstractMatrix ? s.T[component, :] : s.T
        if n[j] == 0
            terms[j] = zero(complex(float(k)))
            continue
        end
        N[j] = truncation_order(T, F[j], k, truncation)
        pref = n[j] * exp(im * k * s.λ) * (im * k)^s.μ
        terms[j] = pref * sum(T[m+1] / k^m for m in 0:N[j]-1)
        N[j] < length(T) && (err += Float64(abs(pref * T[N[j]+1] / k^N[j])))
    end
    (value = sum(terms), N = N, error = err, terms = terms)
end
