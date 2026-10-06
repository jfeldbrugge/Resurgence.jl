# Numerical Picard-Lefschetz check of the South-East rule for the 1D Lorentzian lens, y = 1.
# Fits Ψ = Σ_j n_j Ψ^(j) over several k, with the thimble integrals Ψ^(j) and the original
# integral Ψ evaluated numerically. Expected (South-East rule): n = (0, 0, 1, 1, 0).
# Standalone: only needs LinearAlgebra. Run with `julia Lens_relevance.jl`.
using LinearAlgebra

y = 1.0
f(x)   = (x - y)^2 / 2 + 1 / (1 + x^2)
df(x)  = (x - y) - 2x / (1 + x^2)^2
d2f(x) = 1 - (2 - 6x^2) / (1 + x^2)^3

# critical points: (x-1)(1+x^2)^2 - 2x = x^5 - x^4 + 2x^3 - 2x^2 - x - 1
c = [-1.0, -1, -2, 2, -1]
C = zeros(5, 5); for i in 2:5; C[i, i-1] = 1; end; C[:, 5] = -c
xs = eigvals(C)
for _ in 1:5; global xs = xs .- df.(xs) ./ d2f.(xs); end

# label as in the paper: t = -i f(x)
ts = -im .* f.(xs)
lab(t) = real(t) > 1 ? 1 : real(t) < -1 ? 2 : abs(real(t)) < 1e-6 ? 3 : real(t) > 0 ? 4 : 5
order = sortperm([lab(t) for t in ts]); xs = xs[order]; ts = ts[order]
println("saddles and Borel singularities:")
for j in 1:5; println("  $j  x = ", round(xs[j], digits=4), "   t = ", round(ts[j], digits=4)); end

# thimble integral, without the prefactor sqrt(k/2πi):
#   f(x(u)) = λ + i u^2,  Ψ_j = e^{ikλ} ∫ e^{-k u^2} 2i u / f'(x(u)) du,
# oriented such that dx/du(0) = sqrt(2i/f''(x_j)) with the principal square root, i.e. the
# leading term (with the prefactor) is e^{ikλ}/sqrt(f''(x_j)), as T0 = 1/sqrt(det H) in the package.
function thimble(xj, k; h = 1e-3)
    λ = f(xj)
    U = sqrt(60 / k)
    v = sqrt(2im / d2f(xj))
    total = zero(ComplexF64)
    for sgn in (1, -1)
        x = xj
        u = 0.0
        acc = 0.0im
        xprev = xj
        while u < U
            u += h
            xg = u ≈ h ? xj + sgn * v * u : 2x - xprev   # predictor
            xprev = x
            x = xg
            for _ in 1:30
                dx = (f(x) - λ - im * u^2) / df(x)
                x -= dx
                abs(dx) < 1e-14 && break
            end
            w = (u >= U - h/2) ? 0.5 : 1.0
            acc += w * exp(-k * u^2) * 2im * (sgn * u) / df(x)
        end
        total += acc * h
    end
    # contribution at u = 0: 2iu/f' -> v (finite), counted once
    total += h * v
    exp(im * k * λ) * total
end

# original integral along a contour that is real near the lens and tilted by 45° far away
function original(k; h = 2e-4, L = 14)
    w(s)  = 0.5 * (1 + tanh((s - 1)^2 - 25))
    dw(s) = 0.5 * (1 - tanh((s - 1)^2 - 25)^2) * 2(s - 1)
    acc = zero(ComplexF64)
    for s in (1 - L):h:(1 + L)
        x  = s + im * (s - 1) * w(s)
        dx = 1 + im * (w(s) + (s - 1) * dw(s))
        acc += exp(im * k * f(x)) * dx
    end
    acc * h
end

ks = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
A = [thimble(xs[j], k) for k in ks, j in 1:5]
b = [original(k) for k in ks]
# real integer unknowns: stack real and imaginary parts
n = [real(A); imag(A)] \ [real(b); imag(b)]
println("least-squares intersection numbers n_j = ", round.(n, digits=6))
println("residual = ", norm(A * round.(n) - b) / norm(b))
