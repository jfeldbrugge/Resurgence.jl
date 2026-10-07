cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Resurgence
using DynamicPolynomials, MultivariatePolynomials, LinearAlgebra, Printf

setprecision(BigFloat, 2 * 512)

######################################
# Sign convention of stokes_constants (K[l, j] = K_{l←j}) in the South-East update rule,
# tested on the Pearcey integral (bounded from below) against a numerical Picard-Lefschetz fit.
######################################

for (xp, yp) in [(1 // 1, 1 // 2), (2 // 1, 1 // 1), (1 // 2, -1 // 2), (-3 // 2, 1 // 1)]
    @polyvar q
    f = q^4 + yp * q^2 + xp * q
    GM = GaussManin(f)
    Σ  = ComplexF64.(-im .* GM.λs)
    Rs = [ComplexF64.(R) for R in GM.Rs]
    saddles = [refineSaddle(f, sd)[1] for sd in findAllSaddles(f)]
    fs = f.(saddles)
    d2(p) = differentiate(differentiate(f, q), q)(p)
    xs = [ComplexF64(saddles[argmin(abs.(fs .- λ))][1]) for λ in GM.λs]
    T0 = [1 / sqrt(ComplexF64(d2([p]))) for p in xs]
    K = round.(Int, real.(stokes_constants(Σ, Rs, 6; T0 = T0, method = :matrix)))

    # South-East rule with n_l += n_j K[l, j] (sign = +1) or n_l -= n_j K[l, j] (sign = -1)
    function southeast(sign)
        n = [abs(imag(p)) < 1e-10 ? 1 : 0 for p in xs]   # real saddles relevant at θ = π/2
        edges = [(j, l) for j in eachindex(Σ), l in eachindex(Σ)
                 if j != l && K[l, j] != 0 && -π/2 < angle(Σ[l] - Σ[j]) <= 0]
        for (j, l) in sort(edges, by = e -> angle(Σ[e[2]] - Σ[e[1]]))
            n[j] != 0 && (n[l] += sign * n[j] * K[l, j])
        end
        n
    end

    # numerical Picard-Lefschetz: thimbles normalised to leading term e^{ikλ}/sqrt(f'')
    fn(u) = u^4 + yp*u^2 + xp*u; fp(u) = 4u^3 + 2yp*u + xp; fpp(u) = 12u^2 + 2yp
    function thimble(uj, k; h = 1e-3)
        λ = fn(uj); v = sqrt(2im / fpp(uj)); S = sqrt(60 / k); tot = 0.0im
        for sg in (1, -1)
            u = uj; up = uj; acc = 0.0im; t = 0.0
            while t < S
                t += h; ug = t ≈ h ? uj + sg*v*t : 2u - up; up = u; u = ug
                for _ in 1:40; du = (fn(u) - λ - im*t^2) / fp(u); u -= du; abs(du) < 1e-14 && break; end
                acc += exp(-k*t^2) * 2im * sg * t / fp(u)
            end
            tot += acc * h
        end
        sqrt(k / (2π*im)) * exp(im*k*λ) * (tot + h*v)
    end
    function exact(k; h = 1e-4)
        acc = 0.0im
        for t in -6:h:6
            u = t + im * 0.25 * tanh(t)            # tilt into the valleys of exp(ik u^4)
            acc += exp(im * k * fn(u)) * (1 + im * 0.25 * sech(t)^2)
        end
        sqrt(k / (2π*im)) * acc * h
    end
    ks = 2.0:1.0:6.0
    A = [thimble(p, k) for k in ks, p in xs]; b = [exact(k) for k in ks]
    nfit = round.(Int, [real(A); imag(A)] \ [real(b); imag(b)])
    @printf("x=%s y=%s  Σ=%s\n  fit n=%s   SE(+)=%s   SE(-)=%s\n", xp, yp,
            string(round.(Σ, digits = 3)), nfit, southeast(1), southeast(-1))
end
