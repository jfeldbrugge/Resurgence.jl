# Stokes constants of the Fuchsian system  Y' = A(t) Y,  A(t) = Σ_a R_a/(t - σ_a),
# from keyhole contours, with the propagators computed from hyperlogarithms (Hyperlogarithm.jl)
# or by direct matrix-valued continuation.

# Stokes constants from a keyhole contour (Sec. VI.A):  U± = P exp ∫_{γ±} A(t) dt  with
# A(t) = Σ_a R_a/(t - σ_a), where γ± run from a (near σ_i) to b (beyond σ_j) passing σ_j on the
# left (+) / right (−) of the directed line σ_i → σ_j.  The keyhole holonomy is S_j = U₋⁻¹ U₊.
# The saddle solutions are the local solutions Φ̂_a(t) = Y_a(t) v_a, with Y_a = ev(L_a) the
# local Frobenius solution at σ_a and v_a the eigenvector of R_a (R_a v_a = tr(R_a) v_a), on the
# principal branch along the ray σ_i → σ_j. For the crossing i → j,
#     Φ̂_i^+(b) - Φ̂_i^-(b) = (U₊ - U₋) Φ̂_i(a) = K_{j←i} Φ̂_j(b).

"""
Keyhole paths (γ₊, γ₋) from a = σ_i + r_i e to b = σ_j + r_j e, e = (σ_j - σ_i)/|σ_j - σ_i|.
Singular points σ_k on (or close to) the segment σ_i → σ_j are avoided by half-circles on the
left (side = +1) or right (side = -1) of the directed line; γ₊ and γ₋ differ only around σ_j.
"""
function keyhole_paths(Σ, i, j; npts = 12, side = 1)
    e  = (Σ[j] - Σ[i]) / abs(Σ[j] - Σ[i]); len = abs(Σ[j] - Σ[i])
    r(k) = 0.3 * dist(Σ[k], Σ; skip = k)
    arc(k, o) = [Σ[k] + r(k)*e*cis(o*π*(1 - m/npts)) for m in 0:npts]  # σ_k - r_k e → σ_k + r_k e
    s(k) = real(conj(e) * (Σ[k] - Σ[i]))                                # position along the line
    between = [k for k in eachindex(Σ) if k ∉ (i, j) && 0 < s(k) < len &&
               abs(imag(conj(e) * (Σ[k] - Σ[i]))) < r(k)]
    stem = [Σ[i] + r(i)*e; reduce(vcat, (arc(k, side) for k in sort(between; by = s)); init = eltype(Σ)[])]
    [stem; arc(j, 1)], [stem; arc(j, -1)]
end

"""
Stokes constant K_{j←i} for the Laplace contour of saddle i crossing the Borel singularity σ_j,
from the keyhole contour around σ_j. `vs[a]` is the leading vector of Φ̂_a; by default it is
T0[a] times the eigenvector of R_a with unit first entry. For integer Stokes constants, T0[a] should
be the normalisation of the asymptotic series of saddle a, T0[a] = 1/√det H(x_a), ordered like Σ.
Returns (K, S, rel_residual) with S = U₋⁻¹U₊ the keyhole holonomy and rel_residual the part of
the jump not proportional to Φ̂_j (should be ≈ 0).

`method = :hyperlog` evaluates the free hyperlogarithm series truncated at weight N (giving access
to the weight expansion); `method = :matrix` transports the matrices directly, with no weight
truncation (N is then ignored). Use :matrix for more than a few saddles. `method = :loop` uses the
single-loop formula U₊ - U₋ = U_arc₊ (I - L) U_stem: the two paths share their stem and differ by
one loop around σ_j, whose monodromy L = G(p) e^{2πi R_j} G(p)⁻¹ follows from the Frobenius normal
form at σ_j (`local_monodromy`). It avoids subtracting two transported propagators; the second
return value is then L instead of the keyhole holonomy.

`N` is either an integer (used as given) or `:auto`. With `:auto` (hyperlog only) the smallest N
with certified truncation error |K_N - K| < 1/2 is used (`sufficient_N`, keywords `Nmax`, `sharp`,
`a_posteriori`), so that `round(K)` is exact when the Stokes constants are integers. The bound
assumes nilpotent residues (ρ = 0, the package convention). If it certifies no N ≤ Nmax, or the
certified N would need more than `max_words` words (about n^N for n letters), it falls back to
`stokes_constant_increment` within that budget.
"""
function stokes_constant(Σ, Rs, i, j, N; K = 40, npts = 12, paths = keyhole_paths(Σ, i, j; npts = npts),
                         T0 = ones(length(Σ)), vs = [T0[a] * leading_vector(Rs[a]) for a in eachindex(Rs)],
                         method = :hyperlog, Nmax = 30, sharp = true, a_posteriori = true, max_words = 1e5)
    if N === :auto && method == :hyperlog
        Nbudget = max_weight(length(Σ), max_words)
        Nstar = all(abs(tr(Rs[s])) < 1e-8 * max(1, opnorm(Rs[s])) for s in (i, j)) ?
            sufficient_N(Σ, Rs, i, j; Nmax = Nmax, K = K, paths = paths, vs = vs, sharp = sharp,
                         a_posteriori = a_posteriori) : nothing
        if Nstar === nothing || Nstar > Nbudget
            why = Nstar === nothing ? "no N ≤ $Nmax certified (the bound needs ρ = 0)" :
                                      "certified N = $Nstar exceeds the budget N ≤ $Nbudget (max_words = $max_words)"
            @warn "stokes_constant: K[$j,$i]: $why; using the increment rule up to N = $Nbudget"
            return stokes_constant_increment(Σ, Rs, i, j; Nmax = Nbudget, K = K, paths = paths, vs = vs)[1]
        end
        @info "stokes_constant: K[$j,$i] certified at N = $Nstar (|K_N - K| < 1/2)"
        N = Nstar
    end
    N === :auto && (N = 0)                                     # :matrix ignores N
    γp, γm = paths
    if method == :hyperlog
        Up = evaluate_series(transport(Σ, one_s(), γp, N; K = K), Rs, N)
        Um = evaluate_series(transport(Σ, one_s(), γm, N; K = K), Rs, N)
        Φi = evaluate_series(local_singular(Σ, i, γp[1],   N; K = K), Rs, N) * vs[i]
        Φj = evaluate_series(local_singular(Σ, j, γp[end], N; K = K), Rs, N) * vs[j]
    elseif method == :matrix
        Up = matrix_transport(Σ, Rs, γp; K = K)
        Um = matrix_transport(Σ, Rs, γm; K = K)
        Φi = frobenius_solution(Σ, Rs, i, γp[1],   vs[i]; K = K)
        Φj = frobenius_solution(Σ, Rs, j, γp[end], vs[j]; K = K)
    elseif method == :loop
        # single-loop formula: γ₊ and γ₋ share the stem up to p = σ_j - r_j e and differ by one loop
        # around σ_j, so U₊ - U₋ = U_arc₊ (I - L) U_stem, with L = G(p) e^{2πi R_j} G(p)⁻¹ the local
        # monodromy of the counterclockwise loop at p. No difference of two transports is formed.
        narc = npts + 1                                        # points on the half-circle around σ_j
        tol = 1e-12 * max(1, maximum(abs, γp))                 # cis(π) and cis(-π) differ by rounding
        length(γp) == length(γm) && γp[1:end-narc] == γm[1:end-narc] && abs(γp[end-npts] - γm[end-npts]) < tol ||
            error("stokes_constant: method = :loop needs paths that share their stem (keyhole_paths)")
        p = γp[end-npts]
        Ustem = matrix_transport(Σ, Rs, γp[1:end-npts]; K = K)
        Uarc  = matrix_transport(Σ, Rs, γp[end-npts:end]; K = K)
        L = local_monodromy(Σ, Rs, j, p; K = K)
        Φi = frobenius_solution(Σ, Rs, i, γp[1],   vs[i]; K = K)
        Φj = frobenius_solution(Σ, Rs, j, γp[end], vs[j]; K = K)
        d  = Uarc * ((I - L) * (Ustem * Φi))
        Kji = dot(Φj, d) / dot(Φj, Φj)
        return Kji, L, norm(d - Kji * Φj) / max(norm(d), norm(Φj))
    else
        error("stokes_constant: method must be :hyperlog, :matrix or :loop")
    end
    d  = (Up - Um) * Φi
    Kji = dot(Φj, d) / dot(Φj, Φj)
    Kji, Um \ Up, norm(d - Kji * Φj) / max(norm(d), norm(Φj))
end

"""
Image of the (rank-one) residue R, normalised to unit first entry: the leading vector v with R v = tr(R) v.
Taken from the SVD rather than an eigen-decomposition, which is ill-conditioned (≈ √eps) when R is
nilpotent (tr R = 0).
"""
function leading_vector(R)
    v = svd(ComplexF64.(R)).U[:, 1]
    iszero(v[1]) ? v / norm(v) : v / v[1]
end

"Matrix of Stokes constants K[j, i] = K_{j←i} for all ordered pairs i ≠ j (zero on the diagonal); N may be `:auto`."
function stokes_constants(Σ, Rs, N; kwargs...)
    n = length(Σ)
    [i == j ? zero(ComplexF64) : stokes_constant(Σ, Rs, i, j, N; kwargs...)[1] for j in 1:n, i in 1:n]
end

"""
Weight-resolved  tr((T_i - I)(T_j - I)):  entry r-1 is the total-weight-r contribution
Σ_{p=1}^{r-1} tr(D_i[p+1] D_j[r-p+1]), r = 2..N, with D_i the weight pieces of T_i
(e.g. from `monodromy_pieces`). The leading entry is (2πi)² tr(R_i R_j).
"""
trace_contributions(Di, Dj, N) =
    ComplexF64[sum(tr(Di[p+1] * Dj[r-p+1]) for p in 1:r-1) for r in 2:N]

# Matrix-valued continuation of  Y' = A(t) Y,  A(t) = Σ_s R_s/(t - σ_s),  without weight truncation.

"Taylor coefficients A_m of A(a+u) = Σ_m A_m u^m, over the letters s ≠ skip."
taylor_A(Σ, Rs, a, K; skip = 0) =
    [sum(Rs[s] * ((-1)^m / (a - Σ[s])^(m+1)) for s in eachindex(Σ) if s != skip) for m in 0:K-1]

"Propagator U(b ← a) of Y' = A Y for one step, by Taylor expansion at a (needs |b-a| < dist(a, Σ))."
function matrix_step(Σ, Rs, a, b, K)
    A = taylor_A(Σ, Rs, a, K); u = b - a; n = size(Rs[1], 1)
    c = [Matrix{ComplexF64}(I, n, n)]                          # c[k+1] = coefficient of u^k (scaled)
    for k in 0:K-1
        push!(c, u * sum(A[m+1] * c[k-m+1] * u^m for m in 0:k) / (k + 1))
    end
    sum(c)
end

"""
    frobenius_gauge(Σ, Rs, j, z; K = 40)

Holomorphic factor G(z) of the Frobenius normal form Y(t) = G(t) (t - σ_j)^{R_j} C of the Fuchsian
system at σ_j, with G(σ_j) = I. With A(t) = R_j/(t - σ_j) + Σ_m A_m (t - σ_j)^m, the coefficients
of G(t) = Σ_m G_m (t - σ_j)^m solve  m G_m + G_m R_j - R_j G_m = Σ_{i<m} A_i G_{m-1-i},
which is uniquely solvable when no two eigenvalues of R_j differ by a positive integer (e.g. R_j
nilpotent). Needs |z - σ_j| < distance from σ_j to the other letters.
"""
function frobenius_gauge(Σ, Rs, j, z; K = 40)
    R = ComplexF64.(Rs[j]); n = size(R, 1)
    A = taylor_A(Σ, Rs, Σ[j], K; skip = j)
    G = [Matrix{ComplexF64}(I, n, n)]
    for m in 1:K
        rhs = sum(A[i+1] * G[m-i] for i in 0:m-1)
        push!(G, sylvester(m * I - R, R, -rhs))                # (mI - R) G_m + G_m R = rhs
    end
    u = z - Σ[j]
    sum(G[m+1] * u^m for m in 0:K)
end

"""
    local_monodromy(Σ, Rs, j, p; K = 40)

Monodromy L = G(p) e^{2πi R_j} G(p)⁻¹ of the counterclockwise loop around σ_j based at p (in the
Frobenius disk of σ_j): a solution Y continued once around σ_j becomes L Y. For nilpotent R_j,
e^{2πi R_j} = I + 2πi R_j and L - I has the rank of R_j.
"""
function local_monodromy(Σ, Rs, j, p; K = 40)
    G = frobenius_gauge(Σ, Rs, j, p; K = K)
    G * exp(2π * im * ComplexF64.(Rs[j])) / G
end

"Propagator U with Y(path[end]) = U Y(path[1]) along the polyline `path` (same steps as `transport`)."
function matrix_transport(Σ, Rs, path; K = 40, ρ = 0.3)
    U = Matrix{ComplexF64}(I, size(Rs[1])...)
    for i in 1:length(path)-1
        a, b = path[i], path[i+1]; pts = [a]
        while true
            p = pts[end]; d = dist(p, Σ)
            d > 1e-8 * abs(b - a) || error("matrix_transport: path $a → $b runs into a singular point")
            abs(b - p) <= ρ*d && (push!(pts, b); break)
            push!(pts, p + ρ*d*(b - p)/abs(b - p))
        end
        for k in 1:length(pts)-1
            U = matrix_step(Σ, Rs, pts[k], pts[k+1], K) * U
        end
    end
    U
end

"""
Saddle solution Φ̂_j(z) = (z - σ_j)^ρ Σ_k c_k (z - σ_j)^k  with c_0 = v, R_j v = ρ v (principal
branch), from the Frobenius recursion ((ρ + k) I - R_j) c_k = Σ_{m<k} A_m c_{k-1-m}.
Needs |z - σ_j| < distance to the other letters, and ρ not a negative integer.
"""
function frobenius_solution(Σ, Rs, j, z, v; K = 40)
    R = ComplexF64.(Rs[j]); ρ = dot(v, R * v) / dot(v, v); u = z - Σ[j]
    isinteger(real(ρ)) && real(ρ) < 0 && abs(imag(ρ)) < 1e-12 &&
        error("frobenius_solution: resonant exponent ρ = $ρ (negative integer) is not supported")
    A = taylor_A(Σ, Rs, Σ[j], K; skip = j)
    c = [ComplexF64.(v)]                                       # c[k+1] = c_k u^k
    for k in 1:K
        rhs = sum(A[m+1] * c[k-m] * u^(m+1) for m in 0:k-1)
        push!(c, ((ρ + k) * I - R) \ rhs)
    end
    exp(ρ * log(u)) * sum(c)
end

# Truncation control for the hyperlogarithm back-end (keyhole notes, Sec. 9): certified bound
# B(N) ≥ |K_N - K| for nilpotent residues (ρ = 0), and the smallest N with B(N) < 1/2.

"Midpoint quadrature nodes and weights |dt| on the polyline `path`, refined near the letters Σ."
function path_quadrature(Σ, path; per = 50)
    ts = eltype(path)[]; ws = Float64[]
    for k in 1:length(path)-1
        a, b = path[k], path[k+1]; L = abs(b - a)
        d = minimum(dist(a + q*(b - a), Σ) for q in range(0, 1, 21))
        m = max(per, ceil(Int, per * L / d))
        append!(ts, [a + (q - 0.5)/m * (b - a) for q in 1:m]); append!(ws, fill(L/m, m))
    end
    ts, ws
end

"‖A(t)‖ over the letters s ≠ skip: operator norm of the full sum (sharp) or sum of residue norms."
A_norm(Σ, Rs, t; sharp = true, skip = 0) = sharp ?
    opnorm(sum(Rs[s] / (t - Σ[s]) for s in eachindex(Σ) if s != skip)) :
    sum(opnorm(Rs[s]) / abs(t - Σ[s]) for s in eachindex(Σ) if s != skip)

"Path strength Λ_γ = ∫_γ ‖A(t)‖ |dt| (numerical quadrature)."
function path_strength(Σ, Rs, path; sharp = true)
    ts, ws = path_quadrature(Σ, path)
    sum(w * A_norm(Σ, Rs, t; sharp = sharp) for (t, w) in zip(ts, ws))
end

"Local constants (α_s, λ_s) = (‖R_s‖, r_s max‖Â_s‖) of the stem from σ_s to z, Â_s = Σ_{u≠s} R_u/(t-σ_u)."
function local_constants(Σ, Rs, s, z; sharp = true, npts = 200)
    r = abs(z - Σ[s])
    λ = sharp ? r * maximum(A_norm(Σ, Rs, Σ[s] + q*(z - Σ[s]); skip = s) for q in range(0, 1, npts)) :
                r * sum(opnorm(Rs[u]) for u in eachindex(Σ) if u != s) / (dist(Σ[s], Σ; skip = s) - r)
    opnorm(Rs[s]), λ
end

"Dyson tail Σ_{r>N} Λ^r/r! ≤ e^Λ Λ^{N+1}/(N+1)!."
dyson_tail(Λ, N) = exp(Λ + (N + 1) * log(Λ) - loggamma(N + 2))

"Tail of the local majorant exp(λ(g + αg²)) at g = 1, by the Chernoff bound inf_{G>1} e^{λ(G+αG²)}/(G^{N+1}(1-1/G))."
local_tail(α, λ, N) =
    minimum(exp(λ*(G + α*G^2) - (N + 1)*log(G)) / (1 - 1/G) for G in exp.(range(log(1.001), log(1e4), 4000)))

"""
Certified bound N ↦ B(N) ≥ |K_N - K_{j←i}| for the hyperlogarithm back-end truncated at weight N,
valid for nilpotent residues (ρ = 0). `sharp` uses ‖Σ_s R_s/(t - σ_s)‖ instead of Σ_s ‖R_s‖/|t - σ_s|;
`a_posteriori` takes ‖Φ̂_a‖, ‖Φ̂_b‖, ‖U₊ - U₋‖ and |K| from the matrix back-end instead of a-priori bounds.
"""
function truncation_bound(Σ, Rs, i, j; K = 40, npts = 12, paths = keyhole_paths(Σ, i, j; npts = npts),
                          T0 = ones(length(Σ)), vs = [T0[a] * leading_vector(Rs[a]) for a in eachindex(Rs)],
                          sharp = true, a_posteriori = true)
    γp, γm = paths
    Λ = max(path_strength(Σ, Rs, γp; sharp = sharp), path_strength(Σ, Rs, γm; sharp = sharp))
    αa, λa = local_constants(Σ, Rs, i, γp[1]; sharp = sharp)
    αb, λb = local_constants(Σ, Rs, j, γp[end]; sharp = sharp)
    va, vb = norm(vs[i]), norm(vs[j])
    if a_posteriori
        Φa = frobenius_solution(Σ, Rs, i, γp[1], vs[i]; K = K); Φb = frobenius_solution(Σ, Rs, j, γp[end], vs[j]; K = K)
        dU = matrix_transport(Σ, Rs, γp; K = K) - matrix_transport(Σ, Rs, γm; K = K)
        d = dU * Φa; Pa0 = norm(Φa); nb = norm(Φb); nU = opnorm(dU); Kabs = abs(dot(Φb, d) / dot(Φb, Φb))
    else
        Pa0 = va * exp(λa * (1 + αa)); nb = vb * (2 - exp(λb * (1 + αb))); nU = 2exp(Λ)
        Kabs = nb > 0 ? nU * Pa0 / nb : Inf
    end
    function B(N)
        EU, Ea, Eb = dyson_tail(Λ, N), local_tail(αa, λa, N), local_tail(αb, λb, N)
        Pa = a_posteriori ? Pa0 + va * Ea : Pa0              # bound on ‖Φ̂_a,N‖
        den = nb - vb * Eb                                   # lower bound on ‖Φ̂_b,N‖
        den > 0 ? (2Pa * EU + nU * va * Ea + Kabs * vb * Eb) / den : Inf
    end
    B
end

"Largest weight N whose words (Σ_{r≤N} n^r for n letters) fit in `max_words`."
max_weight(n, max_words) = (N = 0; while sum(float(n)^r for r in 0:N+1) <= max_words; N += 1; end; N)

"Smallest N ≤ Nmax with certified |K_N - K_{j←i}| < 1/2 (see `truncation_bound`), or `nothing`."
function sufficient_N(Σ, Rs, i, j; Nmax = 30, kwargs...)
    B = truncation_bound(Σ, Rs, i, j; kwargs...)
    for N in 0:Nmax
        B(N) < 1/2 && return N
    end
    nothing
end

"""
Practical stopping rule (not a certificate): increase N from `Nmin` until K_N and K_{N+1} round to the
same integer and differ by less than `tol`. Returns ((K, S, residual), N).
"""
function stokes_constant_increment(Σ, Rs, i, j; Nmin = 2, Nmax = 12, tol = 0.05, kwargs...)
    prev = stokes_constant(Σ, Rs, i, j, Nmin; kwargs...)
    for N in Nmin+1:Nmax
        cur = stokes_constant(Σ, Rs, i, j, N; kwargs...)
        abs(cur[1] - prev[1]) < tol && round(cur[1]) == round(prev[1]) && return cur, N
        prev = cur
    end
    @warn "stokes_constant_increment: no stable integer for K[$j,$i] up to N = $Nmax"
    prev, Nmax
end
