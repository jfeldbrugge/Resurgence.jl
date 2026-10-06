# Superasymptotic approximation of the 1D Lorentzian lens (y = 1) versus the exact integral
#   Ψ(k) = sqrt(k/2πi) ∫ exp(ik f(x)) dx,  f(x) = (x - y)^2/2 + 1/(1 + x^2).
# The relevant saddle points are x_3 (real) and x_4 (complex), n = (0, 0, 1, 1, 0) (Lens_relevance.jl).
# The asymptotic series of each saddle follows from the thimble f(x(u)) = λ + i u^2 by Lagrange
# inversion and Watson's lemma:  Ψ^(j) ~ e^{ikλ_j} Σ_m b_m k^{-m}.
# Writes a data file for the pgfplots figure of the paper. Standalone: only needs LinearAlgebra.
using LinearAlgebra, Printf

setprecision(BigFloat, 512)
const CB = Complex{BigFloat}
y = big(1)

f(x)   = (x - y)^2 / 2 + 1 / (1 + x^2)
df(x)  = (x - y) - 2x / (1 + x^2)^2
d2f(x) = 1 - (2 - 6x^2) / (1 + x^2)^3

# critical points, refined in high precision, labelled as in the paper
c = [-1.0, -1, -2, 2, -1]
C = zeros(5, 5); for i in 2:5; C[i, i-1] = 1; end; C[:, 5] = -c
xs = CB.(eigvals(C))
for _ in 1:12; global xs = xs .- df.(xs) ./ d2f.(xs); end
ts = -im .* f.(xs)
lab(t) = real(t) > 1 ? 1 : real(t) < -1 ? 2 : abs(real(t)) < 1e-6 ? 3 : real(t) > 0 ? 4 : 5
xs = xs[sortperm([lab(t) for t in ts])]

# --- power series helpers (coefficient vectors, index 1 = order 0) ---
mul(a, b, N) = [sum(a[i+1] * b[n-i+1] for i in 0:n) for n in 0:N-1]
function inv_series(a, N)
    b = zeros(CB, N); b[1] = 1 / a[1]
    for n in 1:N-1; b[n+1] = -sum(a[i+1] * b[n-i+1] for i in 1:min(n, length(a)-1)) / a[1]; end
    b
end
function sqrt_series(a, N)          # principal branch of the constant term
    b = zeros(CB, N); b[1] = sqrt(a[1])
    for n in 1:N-1; b[n+1] = (a[n+1] - sum(b[i+1] * b[n-i+1] for i in 1:n-1; init = zero(CB))) / (2b[1]); end
    b
end

# asymptotic coefficients b_m, m = 0..M-1, of the saddle point xj
function series(xj, M)
    N = 2M + 2
    # Taylor coefficients of f(xj + w) - f(xj) = Σ_{n≥2} c_n w^n
    q = zeros(CB, N); q[1] = 1 + xj^2; q[2] = 2xj; q[3] = 1
    r = inv_series(q, N + 2)
    c = copy(r[1:N+2]); c[1] += (xj - y)^2 / 2; c[2] += xj - y; c[3] += big(1) / 2
    # s := sqrt(i) u solves w * sqrt(c_2 + c_3 w + ...) = s; Lagrange: w = s ψ(w), ψ = 1/sqrt(...)
    ψ = inv_series(sqrt_series(c[3:N+2], N), N)
    w = zeros(CB, N + 1); P = [one(CB); zeros(CB, N - 1)]
    for n in 1:N
        P = mul(P, ψ, N)                # ψ^n
        w[n+1] = P[n] / n               # [s^n] w = [w^{n-1}] ψ^n / n
    end
    # orientation: dx/du(0) = sqrt(2i/f''(xj)) with the principal square root
    σ = sign(real(sqrt(CB(im)) * w[2] / sqrt(2im / d2f(xj))))
    e = [σ * w[n+1] * sqrt(CB(im))^n for n in 0:N]      # x(u) = xj + Σ e_n u^n
    # Watson, with Γ(m + 1/2) = (2m)! √π / (4^m m!): sqrt(k/2πi) ∫ e^{-k u^2} Σ (2m+1) e_{2m+1} u^{2m} du
    [(2m + 1) * e[2m+2] * (factorial(big(2m)) * sqrt(big(pi)) / (big(4)^m * factorial(big(m)))) / sqrt(2 * big(pi) * im) for m in 0:M-1]
end

M = 80
b3 = ComplexF64.(series(xs[3], M)); λ3 = ComplexF64(f(xs[3]))
b4 = ComplexF64.(series(xs[4], M)); λ4 = ComplexF64(f(xs[4]))
println("b0 check: ", b3[1], " vs ", 1 / sqrt(ComplexF64(d2f(xs[3]))))
for (name, b) in (("3", b3), ("4", b4))
    # root test (robust when equidistant singularities interfere): (Γ(m)/|b_m|)^{1/m} -> |F|
    est = [exp((sum(log, 1:m-1; init = 0.0) - log(abs(b[m+1]))) / m) for m in M-10:M-1]
    println("saddle $name: late terms (Γ(m)/|b_m|)^{1/m} -> |F| ≈ ", round(maximum(est), digits = 3))
end

# optimal truncation N_j = ⌈k min_l |F_jl|⌉ over the adjacent saddle points l (all but the pair 1-2)
tB = ComplexF64.(-im .* f.(xs))
adjacent = Dict(3 => [1, 2, 4, 5], 4 => [1, 2, 3, 5])
F3 = minimum(abs(tB[l] - tB[3]) for l in adjacent[3])
F4 = minimum(abs(tB[l] - tB[4]) for l in adjacent[4])
println("|F_3| = ", round(F3, digits = 4), ",  |F_4| = ", round(F4, digits = 4))

function truncated(b, λ, k, N)
    exp(im * k * λ) * sum(b[m+1] / k^m for m in 0:N-1), N
end

# exact integral along a contour tilted by 45° for large |x|
fd(x) = (x - 1)^2 / 2 + 1 / (1 + x^2)
function exact(k; h = 1e-4, L = 14)
    w(s)  = 0.5 * (1 + tanh((s - 1)^2 - 25))
    dw(s) = 0.5 * (1 - tanh((s - 1)^2 - 25)^2) * 2(s - 1)
    acc = zero(ComplexF64)
    for s in (1 - L):h:(1 + L)
        x  = s + im * (s - 1) * w(s)
        acc += exp(im * k * fd(x)) * (1 + im * (w(s) + (s - 1) * dw(s)))
    end
    sqrt(k / (2π * im)) * acc * h
end

out = joinpath(@__DIR__, "LensSuperasymptotics.dat")
open(out, "w") do io
    println(io, "k ReExact ImExact ReSA ImSA errSA errSA3 errFixed N3 N4")
    for k in 1.0:0.25:30.0
        Ψ = exact(k)
        s3, N3 = truncated(b3, λ3, k, ceil(Int, F3 * k))
        s4, N4 = truncated(b4, λ4, k, ceil(Int, F4 * k))
        fixed, _ = truncated(b3, λ3, k, 3)               # real image, three terms
        SA = s3 + s4
        @printf(io, "%.2f %.12e %.12e %.12e %.12e %.6e %.6e %.6e %d %d\n", k, real(Ψ), imag(Ψ),
                real(SA), imag(SA), abs(Ψ - SA), abs(Ψ - s3), abs(Ψ - fixed), N3, N4)
    end
end
println("wrote ", out)
