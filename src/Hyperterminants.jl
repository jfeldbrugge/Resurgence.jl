# Hyperasymptotics: the remainder of each optimally truncated series is re-expanded in terms of the
# series of the adjacent saddle points, weighted by the Stokes constants and universal functions, the
# hyperterminants (Berry & Howls 1991, Howls 1997, Olde Daalhuis 1996-1998).
#
# With S_j(k) = e^{-ikλ_j} (ik)^{-μ_j} Ψ^(j)(k) ~ Σ_m T_m^(j) k^{-m} and the singulants
# F_jl = t_l - t_j (t = -iλ), the exact remainder after N terms reads
#
#     R_N^(j)(k) = Σ_l K_lj/(2πi) (k F_jl)^{-N} ∫_0^∞ dv v^{N-1} e^{-v} S_l(v/F_jl) / (1 - v/(k F_jl)),
#
# with K_lj = K[l, j] the Stokes constants in the convention of `stokes_constants` (computed with the
# same normalisation T0 of the series). Expanding 1/(1 - v/(kF)) reproduces the late-term relation
# T_N^(j) ≈ Σ_l K_lj/(2πi) Σ_s Γ(N-s) F_jl^{s-N} T_s^(l). Substituting for S_l its truncated series
# plus its own remainder, recursively, yields the hyperasymptotic levels. At level one the
# remainder is a sum of level-one hyperterminants,
#
#     R_N^(j)(k) ≈ Σ_l K_lj/(2πi) Σ_{s<N_l} T_s^(l) F_jl^s Γ(N-s) (k F_jl)^{-N} h(N - s, k F_jl),
#
# with the closed form h(M, w) = -w e^{-w} E_M(-w) in terms of the generalised exponential integral,
# equivalent to F^(1)(z; M; σ) = -e^{σz} (-z)^{M-1} Γ(M) Γ(1-M, σz) of Olde Daalhuis (Hyperterminants II,
# J. Comput. Appl. Math. 89 (1997) 87). At level L ≥ 2, the remainder of the re-expanded series is
# integrated against the level-one kernel (L - 1 nested quadratures over closed-form level-one sums).
# The optimal numbers of terms follow Olde Daalhuis (Proc. R. Soc. Lond. A 454 (1998) 1): the series of
# saddle j is truncated at N_j = α_j^(L) |k|, with α_j^(L) the length of the shortest path of L + 1
# steps from t_j in the adjacency graph, and a re-expanded series at N_child = N_parent - |k F_edge|.
# In the Laplace representation of the hyperterminants (Lemma 1 of Hyperterminants II), the Borel
# functions are hyperlogarithms with letters at the cumulative singulants for integer M; see Sec. VI C
# (Hyperasymptotics) of Crew & Feldbrugge, "Asymptotic evaluation of multidimensional oscillatory
# integrals", arXiv:2610.22767.

"""
    hyperterminant(M, w; method = :closed, rtol = 1e-10)

Normalised level-one hyperterminant

    h(M, w) = Γ(M)⁻¹ ∫_0^∞ v^{M-1} e^{-v} / (1 - v/w) dv = -w e^{-w} E_M(-w),       M > 0,

so that h → 1 for |w| → ∞, with E_M the generalised exponential integral (`method = :closed`, the
default) or by quadrature (`method = :quadrature`). The integral has a pole at v = w when w lies on
the positive real axis (a Stokes line), the branch cut of E_M(-w): off the cut h is the analytic
continuation from either side, and exactly on it the average of both sides (the median value) is
returned.
"""
function hyperterminant(M, w; method = :closed, rtol = 1e-10)
    M > 0 || error("hyperterminant: M must be positive (got M = $M)")
    w = complex(float(w))
    method in (:closed, :quadrature) || error("hyperterminant: method must be :closed or :quadrature")
    if real(w) > 0 && abs(imag(w)) <= 1e-12 * abs(w)          # on a Stokes line: median value
        ϵ = min(1e-10, sqrt(eps(real(w))))                    # offset from the cut, in the precision of w
        side(σ) = method == :closed ? _hyperterminant_closed(M, complex(real(w), σ * ϵ * abs(w))) :
                                      _hyperterminant(M, w, -0.5σ; rtol = rtol)
        return (side(1) + side(-1)) / 2
    end
    if method == :closed
        return _hyperterminant_closed(M, w)
    end
    δ = (real(w) > 0 && abs(angle(w)) < 0.5) ? -0.5 * sign(imag(w)) : 0.0
    _hyperterminant(M, w, δ; rtol = rtol)
end

# closed form, valid off the branch cut w > 0 of E_M(-w)
_hyperterminant_closed(M, w) = -w * exp(-w) * expint(M, -w)

# h(M, w) for M = Mlo, …, Mhi from one closed-form evaluation and the recurrence of E_M,
#     h(M + 1, w) = w (h(M, w) - 1)/M,
# run upwards from M₀ ≈ |w| (where it is stable, errors shrink by |w|/M) and downwards below M₀
# (h(M) = 1 + M h(M + 1)/w, errors shrink by M/|w|). On a Stokes line, the median value as in `hyperterminant`.
# This is the level-one case of the recurrences of Olde Daalhuis (Hyperterminants II, J. Comput. Appl. Math.
# 89 (1997) 87). In double precision, where E_M is cheap, the closed form is used for every M.
function _hyperterminants(Mlo, Mhi, w)
    w = complex(float(w))
    real(w) isa Float64 && return [hyperterminant(M, w) for M in Mlo:Mhi]
    if real(w) > 0 && abs(imag(w)) <= 1e-12 * abs(w)
        ϵ = min(1e-10, sqrt(eps(real(w))))
        return (_hyperterminants(Mlo, Mhi, complex(real(w), ϵ * abs(w))) .+
                _hyperterminants(Mlo, Mhi, complex(real(w), -ϵ * abs(w)))) ./ 2
    end
    h = Vector{typeof(w)}(undef, Mhi - Mlo + 1)
    M₀ = clamp(ceil(Int, abs(w)), Mlo, Mhi)
    h[M₀ - Mlo + 1] = _hyperterminant_closed(M₀, w)
    for M in M₀:Mhi-1
        h[M - Mlo + 2] = w * (h[M - Mlo + 1] - 1) / M
    end
    for M in M₀-1:-1:Mlo
        h[M - Mlo + 1] = 1 + M * h[M - Mlo + 2] / w
    end
    h
end

# Window of the Gamma weight v^{M-1} e^{-v} needed for rtol: the half-width in standard deviations,
# plus a margin for the e^{-v} tail when M is small (12 for the default rtol = 1e-10)
_window(rtol) = max(12, sqrt(2 * log(1 / Float64(rtol))) + 2)
_tail(rtol) = max(12, log(1 / Float64(rtol)) - 11)

# h(M, w) by quadrature along the ray v = ρ e^{iδ}, over the window in which the Gamma weight lives.
# Computed in the precision of w.
function _hyperterminant(M, w, δ; rtol = 1e-10)
    R = real(typeof(w))
    δ = R(δ); e = cis(δ); c = cos(δ)
    lg = loggamma(R(M))
    f(ρ) = (v = ρ * e; exp((M - 1) * (log(ρ) + im * δ) - v - lg) / (1 - v / w) * e)
    peak = max(M - 1, 0) / c
    width = (_window(rtol) * sqrt(R(M)) + _tail(rtol)) / c
    lo = max(zero(R), peak - width)
    quadgk(f, lo, max(peak, lo + R(1e-12)), peak + width; rtol = rtol)[1]
end

# Data shared by the recursion: Borel singularities, series and Stokes constants, in the precision C
struct HyperData{C<:Complex}
    t::Vector{C}
    T::Vector{Vector{C}}
    K::Matrix{Int}
end
adjacent(d::HyperData, j) = [l for l in eachindex(d.t) if l != j && d.K[l, j] != 0]

"""
    path_lengths(Σ, K, steps)

Olde Daalhuis' numbers α_j^(steps-1): for every Borel singularity t_j, the length of the shortest path
of `steps` steps t_j → t_{j_0} → … in the adjacency graph (edges j → l with K[l, j] ≠ 0, revisits
allowed). For steps = 1 this is the distance to the nearest adjacent singularity (`singulants`). The
minimal remainder at hyperasymptotic level L is of the order e^{-α^(L) |k|}, attained for
N_j = α_j^(L) |k| (Proc. R. Soc. Lond. A 454 (1998) 1, eqs. (2.14), (4.15)).
"""
function path_lengths(Σ, K, steps)
    n = length(Σ)
    α = fill(Inf, n)
    for j in 1:n
        dist = fill(Inf, n); dist[j] = 0.0
        for _ in 1:steps
            new = fill(Inf, n)
            for y in 1:n, x in 1:n
                (x != y && abs(K[x, y]) > 0.5 && isfinite(dist[y])) || continue
                new[x] = min(new[x], dist[y] + abs(Σ[x] - Σ[y]))
            end
            dist = new
        end
        α[j] = minimum(dist)
    end
    α
end

"""
Approximation of the remainder R_N^(j)(κ) with `L` levels of re-expansion. The re-expanded series
of the adjacent saddle point l is truncated at N_l = max(0, N - ⌈scale |F_jl|⌉), with scale = |k|
of the top level (Olde Daalhuis 1998, eq. (4.18)). The re-expanded series contributes a closed-form
sum of level-one hyperterminants; for L > 1 the remainder of saddle l, one level deeper, is integrated
against the level-one kernel,

    w^{-N} ∫_0^∞ v^{N-1} e^{-v}/(1 - v/w) R_{N_l}^(l)(v/F) dv,       w = κF.
"""
function _remainder(d::HyperData{C}, j, κ, N, L, scale; rtol = 1e-10) where C
    R = real(C)
    (L == 0 || N == 0) && return zero(C)
    total = zero(C)
    for l in adjacent(d, j)
        F = d.t[l] - d.t[j]
        w = κ * F
        Nl = min(max(0, N - ceil(Int, scale * abs(F))), length(d.T[l]))
        Nl == 0 && continue                            # beyond the accuracy of this level
        c = d.K[l, j] / (2 * R(π) * im)
        logw = log(w); logF = log(F)
        hs = _hyperterminants(N - Nl + 1, N, w)        # level-one hyperterminants h(N - s, w), s < Nl
        for s in 0:Nl-1
            Ts = d.T[l][s+1]
            iszero(Ts) && continue
            total += c * exp(log(Ts) + s * logF + loggamma(R(N - s)) - N * logw) * hs[Nl - s]
        end
        if L > 1                                       # next level: remainder of saddle l
            total += c * _edge_integral(κl -> _remainder(d, l, κl, Nl, L - 1, scale; rtol = rtol),
                                        F, w, N, Nl; rtol = rtol)
        end
    end
    total
end

# w^{-N} ∫_0^∞ v^{N-1} e^{-v} / (1 - v/w) R(v/F) dv for a remainder R that decays like v^{-Nl}: the
# integrand is supported around v ≈ N - Nl - 1. Near a Stokes line (w close to the positive real axis)
# the ray is rotated away from the pole at v = w, and exactly on it the average of both sides is taken.
function _edge_integral(R, F, w, N, Nl; rtol = 1e-10)
    if real(w) > 0 && abs(imag(w)) <= 1e-12 * abs(w)
        return (_edge_integral(R, F, w, N, Nl, 0.5; rtol = rtol) + _edge_integral(R, F, w, N, Nl, -0.5; rtol = rtol)) / 2
    end
    δ = (real(w) > 0 && abs(angle(w)) < 0.5) ? -0.5 * sign(imag(w)) : 0.0
    _edge_integral(R, F, w, N, Nl, δ; rtol = rtol)
end
function _edge_integral(R, F, w, N, Nl, δ; rtol = 1e-10)
    Re = real(typeof(w))
    δ = Re(δ); e = cis(δ); c = cos(δ)
    lg = loggamma(Re(N))
    f(ρ) = begin
        v = ρ * e
        val = exp((N - 1) * (log(ρ) + im * δ) - v - lg) / (1 - v / w) * R(v / F) * e
        isfinite(val) ? val : zero(val)            # negligible tail where the weight underflows
    end
    peak = max(N - Nl - 1, 0) / c
    width = (_window(rtol) * sqrt(Re(N - Nl + 1)) + _tail(rtol)) / c
    lo = max(zero(Re), peak - width)
    exp(lg - N * log(w)) * quadgk(f, lo, max(peak, lo + Re(1e-12)), peak + width; rtol = rtol)[1]
end

"""
    hyperasymptotic(transseries, n, Σ, K, k; level = 1, component = 1, N = nothing,
                    truncation = :daalhuis, rtol = 1e-10)

Hyperasymptotic approximation of the integral,

    Σ_j n_j e^{ikλ_j} (ik)^{μ_j} [ Σ_{m<N_j} T_m^(j)/k^m + R_{N_j}^(j)(k) ],

with the remainders re-expanded `level` times in terms of the series of the adjacent saddle points
(`level = 0` is the superasymptotic approximation). `transseries` is the vector of `Expansion`s from
`transseries` (a named tuple `(λ, μ, T)` per saddle point is accepted as well; `component` selects a
row when T is a matrix), `n` from
`relevance`, Σ are the Borel singularities and K the Stokes constants of `stokes_constants` (package
convention K[l, j] = K_{l←j}, computed with the normalisation of `transseries`).

Truncation (`truncation = :daalhuis`, the default): the series of saddle j is truncated at
N_j = ⌈α_j^(level) |k|⌉, with α_j^(level) the shortest path of level + 1 steps from t_j in the adjacency
graph (`path_lengths`), and every re-expanded series at N_child = max(0, N_parent - ⌈|k F_edge|⌉).
`truncation = :uniform` uses N_j = ⌈(level + 1)|k| F_j⌉ instead, with F_j the distance to the nearest
adjacent singularity. Pass a vector `N` to set the top-level truncations directly.

Level one is evaluated in closed form; level L nests L - 1 quadratures, so levels beyond 2 become
expensive. The computation runs in the precision of `k`: pass `k::BigFloat` (with the series in
BigFloat, as returned by `transseries`, and a matching `rtol`) for results beyond double precision. The relation used assumes μ_j = 0 (non-degenerate saddle points in the normalisation of the
package). Returns `(value, N, terms)` with `terms[j]` the contribution of saddle j.
"""
function hyperasymptotic(transseries, n, Σ, K, k; level = 1, component = 1, N = nothing,
                         truncation = :daalhuis, rtol = 1e-10)
    series = transseries
    level >= 0 || error("hyperasymptotic: level must be non-negative")
    all(abs(s.μ) < 1e-8 for s in series) ||
        error("hyperasymptotic: the resurgence relation is implemented for μ_j = 0 only")
    Kint = round.(Int, real.(K))
    maximum(abs.(K .- Kint)) < 1e-6 || @warn "hyperasymptotic: the Stokes constants are not integers"
    C = complex(float(typeof(k)))
    Ts = [C.(_coefficients(s, component)) for s in series]
    t = [C(-im * _critical_value(s)) for s in series]  # Borel singularities in the precision of k
    maximum(abs.(ComplexF64.(t) .- ComplexF64.(Σ))) < 1e-8 * max(1, maximum(abs, Σ)) ||
        error("hyperasymptotic: Σ must be ordered like the transseries")
    d = HyperData(t, Ts, Kint)
    α = truncation == :daalhuis ? path_lengths(Σ, Kint, level + 1) :
        truncation == :uniform  ? (level + 1) .* singulants(Σ, Kint) :
        error("hyperasymptotic: truncation must be :daalhuis or :uniform")
    Ns = zeros(Int, length(series))
    terms = Vector{C}(undef, length(series))
    for (j, s) in enumerate(series)
        if n[j] == 0
            terms[j] = 0
            continue
        end
        Nj = N === nothing ? (isfinite(α[j]) ? ceil(Int, α[j] * abs(k)) : length(Ts[j])) : N[j]
        if Nj > length(Ts[j])
            @warn "hyperasymptotic: N_$j = $Nj exceeds the $(length(Ts[j])) available terms; using all of them"
            Nj = length(Ts[j])
        end
        Ns[j] = Nj
        S = sum((Ts[j][m+1] / k^m for m in 0:Nj-1); init = zero(C))
        S += _remainder(d, j, complex(float(k)), Nj, level, abs(k); rtol = rtol)
        terms[j] = n[j] * exp(im * k * C(_critical_value(s))) * (im * k)^C(s.μ) * S
    end
    (value = sum(terms), N = Ns, terms = terms)
end
