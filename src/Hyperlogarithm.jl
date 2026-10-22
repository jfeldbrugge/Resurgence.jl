# Hyperlogarithms and their analytic continuation along loops.
#
# Generating series  L(z) = Σ_w L_w(z) w,  ∂_z L = Ω L,  Ω = Σ_σ x_σ/(z-σ),
# normalised by L(z) ~ exp(x_0 log z) at z → 0  (so L_{0^n} = log^n z / n!).
# A word w = [i1,...,in] (indices into the alphabet Σ) stands for L_{σ_i1 ... σ_in}.
#
# Continuation along a loop γ around σ_i (base point z0):
#     L^γ(z0) = L(z0) · M_γ ,   M_γ = C⁻¹ exp(±2πi x_i) C ,
# where C = lim_{z→σ_i} exp(-x_i log(z-σ_i)) L(z) is the regularised value of the
# generating series at σ_i, taken along the path  0 → z0 → σ_i  followed by γ.
# Composition of loops based at the same point:  M_{γ1 then γ2} = M_{γ2} · M_{γ1}.

const Word   = Vector{Int}
const Series = Dict{Word,ComplexF64}          # truncated noncommutative series

one_s()   = Series(Int[] => 1.0 + 0im)
letter(i) = Series([i] => 1.0 + 0im)
scale(A::Series, c) = Series(w => c*a for (w, a) in A)
function add!(A::Series, B::Series, c = 1)
    for (w, b) in B; A[w] = get(A, w, 0) + c*b; end
    A
end

"""
Concatenation product, truncated at weight N.
"""
function mul(A::Series, B::Series, N::Int)
    C = Series()
    for (u, a) in A, (v, b) in B
        length(u) + length(v) <= N || continue
        w = vcat(u, v); C[w] = get(C, w, 0) + a*b
    end
    C
end

"""
Inverse of a series with invertible constant term.
"""
function sinv(A::Series, N)
    a0 = A[Int[]]; X = scale(A, 1/a0); X[Int[]] -= 1
    R = one_s(); P = one_s()
    for k in 1:N; P = mul(P, X, N); add!(R, P, (-1)^k); end
    scale(R, 1/a0)
end

"""
exp(A) for A without constant term.
"""
function sexp(A::Series, N)
    R = one_s()
    P = one_s()
    for k in 1:N
        P = scale(mul(P, A, N), 1/k)
        add!(R, P)
    end
    R
end

"""
Distance function
"""
dist(p, Σ; skip = 0) = minimum(abs(p - Σ[s]) for s in eachindex(Σ) if s != skip)

"""
Transport series G_a(b) = Σ_w L^{(a)}_w(b) w with base point a (a regular point),
by Taylor expansion in u = t-a. Needs |b-a| < dist(a, Σ).
"""
function regular_transport(Σ, a, b, N; K = 40)
    m = length(Σ); u = b - a
    g = [[(-1)^j / (a - Σ[s])^(j+1) for j in 0:K-1] for s in 1:m]  # 1/(t-σ_s) in powers of u
    coeffs = Dict{Word,Vector{ComplexF64}}(Int[] => [1.0 + 0im; zeros(ComplexF64, K)])
    res = one_s(); layer = [Int[]]
    for n in 1:N
        newlayer = Word[]
        for w in layer, s in 1:m
            c = coeffs[w]; d = zeros(ComplexF64, K + 1)
            for k in 0:K-1, j in 0:K-1-k
                d[k+j+2] += c[k+1] * g[s][j+1] / (k + j + 1)
            end
            sw = vcat(s, w); coeffs[sw] = d; push!(newlayer, sw)
            res[sw] = evalpoly(u, d)
        end
        layer = newlayer
    end
    res
end

"""
Local solution at the singular point σ_j:  L_j(z) = H(z-σ_j) exp(x_j log(z-σ_j)),
H holomorphic with H(0) = 1 (principal branch of log). Needs |z-σ_j| < dist to other letters.
Solves  k h_k - [x_j, h_k] = Σ_m A_m h_{k-1-m}  order by order.
"""
function local_singular(Σ, j, z, N; K = 40)
    m = length(Σ); σ = Σ[j]; u = z - σ; xj = letter(j)
    A = [Series() for _ in 0:K-1]
    for mm in 0:K-1, s in 1:m
        s == j && continue
        A[mm+1][[s]] = (-1)^mm / (σ - Σ[s])^(mm+1)
    end
    h = Vector{Series}(undef, K + 1); h[1] = one_s()
    for k in 1:K
        R = Series()
        for mm in 0:k-1; add!(R, mul(A[mm+1], h[k-mm], N)); end
        hk = scale(R, 1/k)
        for _ in 1:N                               # ad_{x_j} raises weight: converges in N steps
            comm = add!(mul(xj, hk, N), mul(hk, xj, N), -1)
            hk = scale(add!(copy(R), comm), 1/k)
        end
        h[k+1] = hk
    end
    H = Series(); for k in 0:K; add!(H, h[k+1], u^k); end
    mul(H, sexp(scale(xj, log(u)), N), N)
end

"""
Continue a solution L (given at path[1]) along the polyline `path`, left-multiplying transports.
"""
function transport(Σ, L, path, N; K = 40, ρ = 0.3)
    for i in 1:length(path)-1
        a, b = path[i], path[i+1]; pts = [a]
        while true
            p = pts[end]; d = dist(p, Σ)
            d > 1e-8 * abs(b - a) || error("transport: path $a → $b runs into a singular point")
            abs(b - p) <= ρ*d && (push!(pts, b); break)
            push!(pts, p + ρ*d*(b - p)/abs(b - p))
        end
        for k in 1:length(pts)-1
            L = mul(regular_transport(Σ, pts[k], pts[k+1], N; K = K), L, N)
        end
    end
    L
end

"""
L(z0) on the principal branch: straight path from the (tangential) base point 0.
"""
function base_value(Σ, z0, N; K = 40)
    j = findfirst(==(0), Σ)
    j === nothing && return transport(Σ, one_s(), [0.0im, z0], N; K = K)   # 0 regular
    r0 = 0.3 * dist(0.0im, Σ; skip = j)
    abs(z0) <= r0 && return local_singular(Σ, j, z0, N; K = K)
    z1 = r0 * z0 / abs(z0)
    transport(Σ, local_singular(Σ, j, z1, N; K = K), [z1, z0], N; K = K)
end

"""
Polyline for the loop: z0 → straight to radius r around σ_i → circle (ccw if orient=+1) → back.
"""
function loop_path(Σ, i, z0; r = 0.3*dist(Σ[i], Σ; skip = i), orient = 1, npts = 24)
    σ = Σ[i]; q = σ + r*(z0 - σ) / abs(z0 - σ)
    θ0 = angle(q - σ)
    circ = [σ + r*cis(θ0 + orient*2π*k/npts) for k in 1:npts]
    [z0; q; circ; z0]
end

"""
Monodromy factor M with L^γ(z0) = L(z0) · M for the loop around σ_i based at z0
(straight segment z0 → σ_i, one turn, back). Returns (M, C), C the regularised value at σ_i.
A different (homotopy class of) stem can be given as a polyline `stem` from z0 to a point q
close to σ_i (within the local disk, |q - σ_i| < distance to the other letters).
"""
function monodromy_factor(Σ, i, z0, N; orient = 1, K = 40, stem = nothing)
    r = 0.3 * dist(Σ[i], Σ; skip = i)
    path = stem === nothing ? [z0, Σ[i] + r*(z0 - Σ[i])/abs(z0 - Σ[i])] : stem
    path[1] == z0 || error("monodromy_factor: the stem must start at z0")
    q = path[end]
    abs(q - Σ[i]) < dist(Σ[i], Σ; skip = i) || error("monodromy_factor: the stem must end close to σ_i")
    Lq = transport(Σ, base_value(Σ, z0, N; K = K), path, N; K = K)
    C  = mul(sinv(local_singular(Σ, i, q, N; K = K), N), Lq, N)
    M  = mul(mul(sinv(C, N), sexp(scale(letter(i), orient*2π*im), N), N), C, N)
    M, C
end

"""
L^γ(z0) for a sequence of loops, e.g. [(2,+1),(3,-1)] = around σ_2 ccw, then σ_3 cw.
"""
function continue_along(Σ, z0, loops, N; K = 40)
    L = base_value(Σ, z0, N; K = K)
    Mtot = one_s()          # continuing L·M_prev along γ gives L·M_γ·M_prev
    for (i, o) in loops
        Mtot = mul(monodromy_factor(Σ, i, z0, N; orient = o, K = K)[1], Mtot, N)
    end
    mul(L, Mtot, N)
end

# Matrix representation of the Fuchsian system  Y' = Σ_s R_s/(z-σ_s) Y :
# the algebra homomorphism x_s ↦ R_s maps L to Y, i.e. [i1,...,in] ↦ R_i1 ⋯ R_in
# (no reversal: regular_transport already builds words as vcat(s, w)).

"""
Weight-resolved evaluation: D[r+1] is the contribution of the words of length r.
"""
function evaluate_series_by_weight(S::Series, Rs, N)
    n = size(Rs[1], 1)
    D = [zeros(ComplexF64, n, n) for _ in 0:N]
    for (w, c) in S
        length(w) <= N && (D[length(w)+1] .+= c .*
        prod((Rs[a] for a in w); init = Matrix{ComplexF64}(I, size(Rs[1])...)))
    end
    D
end

"""
Based monodromy matrices T[j] of  Y' = Σ_s R_s/(z-σ_s) Y  for the ccw loops around σ_j
(base point z0, straight stems), weight-resolved: T[j][r+1] is the weight-r part.
"""
function monodromy_pieces(Σ, Rs, z0, N; K = 40)
    [evaluate_series_by_weight(monodromy_factor(Σ, j, z0, N; orient = 1, K = K)[1], Rs, N)
     for j in eachindex(Σ)]
end

monodromy_matrices(Σ, Rs, z0, N; K = 40) = sum.(monodromy_pieces(Σ, Rs, z0, N; K = K))

"""
Evaluate a free series in the residue matrices: [i1,...,in] ↦ R_i1 ⋯ R_in (no reversal).
"""
evaluate_series(S::Series, Rs, N) = sum(evaluate_series_by_weight(S, Rs, N))

"""
Loop factor M = L(z0)⁻¹ L^γ(z0) from explicit transport around `loop_path` (straight stem, circle,
back). An independent check of `monodromy_factor`, which uses C⁻¹ exp(2πi x_j) C instead.
"""
function direct_loop_factor(Σ, j, z0, N; orient = 1, K = 40, npts = 96)
    L0 = base_value(Σ, z0, N; K = K)
    Lγ = transport(Σ, L0, loop_path(Σ, j, z0; orient = orient, npts = npts), N; K = K)
    mul(sinv(L0, N), Lγ, N)
end
