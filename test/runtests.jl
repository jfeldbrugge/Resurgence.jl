using Resurgence
using DynamicPolynomials, MultivariatePolynomials, LinearAlgebra
using Test

# The one-dimensional Lorentzian lens (y = β = 1),
#
#     Ψ(k) = √(k/2πi) ∫ exp(ik f(x)) dx,    f(x) = (x - y)²/2 + β/(1 + x²),
#
# in its polynomial lift f(x₁, x₂, x₃) = P(x₁) x₂ + x₃ (x₂ Q(x₁) - 1), with P = 2β + (1 + x₁²)(x₁ - y)²
# and Q = 2(1 + x₁²), the example of Sec. VII B (The Lorentzian lens) of Crew & Feldbrugge, "Asymptotic
# evaluation of multidimensional oscillatory integrals", arXiv:2610.22767. The test covers the saddle
# points, the Gauss-Manin system, Borel-Padé, the Stokes constants, the South-East rule and the
# Borel-Padé, superasymptotic and hyperasymptotic approximations of the integral.

setprecision(BigFloat, 512)

# progress messages with the elapsed time
const t_start = time()
progress(msg) = (println("[", lpad(round(time() - t_start, digits = 1), 6), " s] ", msg); flush(stdout))

y, β = 1, 1
@polyvar x1 x2 x3 monomial_order = Graded{MultivariatePolynomials.Reverse{InverseLexOrder}}
PP = 2β + (1 + x1^2) * (x1 - y)^2
QQ = 2 * (1 + x1^2)
f = PP * x2 + x3 * (x2 * QQ - 1)

vars = variables(f)
hessian = differentiate(differentiate(f, vars), vars)

# one-dimensional exponent, its second derivative and the exact integral along the real line,
# tilted by 45° for large |x|
fd(x)  = (x - y)^2 / 2 + β / (1 + x^2)
d2f(x) = 1 - 2β * (1 - 3x^2) / (1 + x^2)^3
function exact(k; h = 1e-4, L = 14)
    w(s)  = 0.5 * (1 + tanh((s - y)^2 - 25))
    dw(s) = 0.5 * (1 - tanh((s - y)^2 - 25)^2) * 2(s - y)
    acc = zero(ComplexF64)
    for s in (y - L):h:(y + L)
        x  = s + im * (s - y) * w(s)
        acc += exp(im * k * fd(x)) * (1 + im * (w(s) + (s - y) * dw(s)))
    end
    sqrt(k / (2π * im)) * acc * h
end

# Stokes constants of the paper (rows l, columns j, thimbles normalised by 1/√f''(x_j)), with the
# saddle points labelled by their Borel singularities t = -iλ: Re t > 1, Re t < -1, the real saddle
# point (Re t = 0), 0 < Re t < 1 and -1 < Re t < 0
paper_label(t) = real(t) > 1 ? 1 : real(t) < -1 ? 2 : abs(real(t)) < 1e-6 ? 3 : real(t) > 0 ? 4 : 5
Kpaper = [ 0  0 -1  2 -1;
           0  0  1  1 -2;
           1 -1  0 -1  1;
          -2 -1  1  0  1;
           1  2 -1 -1  0]

@testset "Resurgence.jl: Lorentzian lens" begin

    # ---------------------------------------------------------------- saddle points
    progress("finding and refining the saddle points")
    saddles = sort(findAllSaddles(f), by = p -> abs.(f(p)))
    saddles = [refineSaddle(f, sd)[1] for sd in saddles]
    T0s = [1 / sqrt(det([h(p) for h in hessian])) for p in saddles]
    u = f.(saddles)

    @testset "saddle points" begin
        @test length(saddles) == 5
        @test all(norm(ComplexF64.([g(p) for g in differentiate(f, vars)])) < 1e-30 for p in saddles)
        # the x₁ coordinates are the critical points of the one-dimensional exponent
        @test all(abs(ComplexF64(fd(p[1]) - f(p))) < 1e-12 for p in saddles)
    end

    # ---------------------------------------------------------------- Gauss-Manin system
    progress("computing the Gauss-Manin system")
    GM = GaussManin(f)

    @testset "Gauss-Manin system" begin
        @test GM.n == 5
        # the matrices of the Gauss-Manin system in the good basis of the paper (Sec. VII B),
        # ϕ = (1, x₁, -1/2 + 2x₂, 2x₃ + 4x₂ - 2x₁, 4x₁x₂)
        @test GM.M == Rational{BigInt}[ 1//2  -1   1    -1//2   0;
                                        1      0   0     1//2   1;
                                        5//4   1   1//2  3//4  -1//2;
                                       -3//2   0   1     0      1//2;
                                       -2      2   2     0      0]
        @test GM.B == Diagonal(Rational{BigInt}[1, 1, 2, 3//2, 2])     # paper, Sec. VII B
        # the eigenvalues of M are the critical values
        @test abs(ComplexF64(tr(GM.M) - sum(u))) < 1e-30
        @test all(minimum(abs.(u .- λ)) < 1e-30 for λ in GM.λs)
        # the residues of the Fuchsian system are nilpotent
        @test all(abs(tr(R)) < 1e-30 for R in GM.Rs)
    end

    # ---------------------------------------------------------------- asymptotic series
    progress("computing the asymptotic series")
    # The fourth basis element ϕ₄ = 2x₃ + 4x₂ - 2x₁ generates the series of the one-dimensional lens,
    # normalised by its leading term 1/√f''(x_j)
    M = 120
    lens = map(1:GM.n) do j
        λ, μ, Ts = asymptotic(j, GM, M)
        x  = saddles[argmin(abs.(u .- λ))][1]
        T0 = 1 / sqrt(d2f(Complex{BigFloat}(x)))
        (λ = λ, μ = μ, x = x, T = T0 .* Ts[4, :] ./ Ts[4, 1], v = ComplexF64.(T0 .* Ts[:, 1] ./ Ts[4, 1]))
    end
    Σ  = ComplexF64[-im * s.λ for s in lens]
    Rs = [ComplexF64.(R) for R in GM.Rs]
    p  = sortperm(paper_label.(Σ))                   # p[a] = index (in GM.λs) of the saddle a of the paper
    Kref = zeros(Int, 5, 5); Kref[p, p] = -Kpaper    # expected package Stokes constants, GM.λs order

    @testset "asymptotic series" begin
        @test all(abs(s.μ) < 1e-30 for s in lens)
        @test all(length(s.T) == M + 1 for s in lens)
        # the late terms diverge at the rate of the nearest adjacent Borel singularity
        @test sort(paper_label.(Σ)) == 1:5
        F = singulants(Σ, Kref)
        @test all(isapprox(divergence_rate(lens[j].T; tail = 10), F[j]; rtol = 0.1) for j in 1:5)
    end

    # ---------------------------------------------------------------- Borel-Padé adjacency
    progress("Borel-Padé poles and Stokes constants")
    @testset "Borel-Padé" begin
        transseries = map(1:GM.n) do j
            λ, μ, Ts = asymptotic(j, GM, 100)
            index = argmin(abs.(u .- λ))
            Expansion(saddles[index], -im * λ, μ, sqrt(big(im)) .* T0s[index] .* Ts[1, :])
        end
        poles = [findPoles(trans) for trans in transseries]
        @test all(length(poles[j]) == 49 for j in 1:5)
        K_bp = round.(Int, real.(findStokesConstants(transseries)))
        # the detected edges belong to the adjacency graph of the Fuchsian system
        # (findStokesConstants is transposed with respect to stokes_constants)
        @test count(!iszero, K_bp) >= 4
        @test all((K_bp .!= 0) .<= transpose(Kref .!= 0))
        @test plotAdjacencyPoles(transseries, K_bp, poles) isa Tuple
    end

    # ---------------------------------------------------------------- Stokes constants
    progress("Stokes constants from the Fuchsian system")
    # leading vectors normalised such that their fourth component is the leading lens coefficient
    vs = [s.v for s in lens]
    Kmat = stokes_constants(Σ, Rs, 6; vs = vs, method = :matrix)
    K = round.(Int, real.(Kmat))

    @testset "Stokes constants" begin
        @test maximum(abs, Kmat - K) < 1e-10
        @test K == Kref                                              # package convention K = -K_paper
        @test K == -transpose(K)
        @test plotStokesGraphGM(Σ, K) isa Tuple
        @test plotAdjacencyGM(Σ, K) isa Tuple
    end

    # ---------------------------------------------------------------- South-East rule
    progress("South-East rule")
    # the lifted exponent is unbounded from below: start from the real minimum x₃ of the 1D exponent
    n0 = [abs(imag(ComplexF64(s.x))) < 1e-10 ? 1 : 0 for s in lens]
    n  = southeast(Σ, K, n0)

    @testset "South-East rule" begin
        @test n0[p] == [0, 0, 1, 0, 0]
        @test n[p] == [0, 0, 1, 1, 0]                                # paper: Ψ = Ψ⁽³⁾ + Ψ⁽⁴⁾
    end

    # ---------------------------------------------------------------- approximations
    progress("approximations of the integral")
    expansions = [Expansion(s.x, -im * s.λ, s.μ, s.T) for s in lens]
    series = [(λ = ComplexF64(s.λ), μ = 0.0, T = ComplexF64.(s.T)) for s in lens]
    borel_pade(k) = sum(n[j] * exp(im * k * ComplexF64(lens[j].λ)) *
                        ComplexF64(BorelPadeResum(expansions[j], k)[1]) for j in 1:5 if n[j] != 0)

    @testset "approximations, k = $k" for k in (4.0, 8.0)
        progress("  k = $k: exact integral, Borel-Padé, superasymptotic and hyperasymptotic (levels 1, 2)")
        Ψ   = exact(k)
        sup = superasymptotic(series, n, Σ, K, k)
        eSA = abs(Ψ - sup.value)
        eH1 = abs(Ψ - hyperasymptotic(series, n, Σ, K, k; level = 1).value)
        eH2 = abs(Ψ - hyperasymptotic(series, n, Σ, K, k; level = 2).value)
        @test abs(Ψ - borel_pade(k)) < 1e-10
        @test eSA < 10 * sup.error
        @test eH1 < eSA / 5
        @test eH2 < eH1 / 5
    end
end
progress("done")
