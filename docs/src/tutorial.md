```@meta
CurrentModule = Resurgence
```

# Tutorial

In this tutorial, we demonstrate the main features of Resurgence.jl with two examples: the Pearcey integral, a one-dimensional integral with three saddle points, and the Lorentzian lens of wave optics, a rational exponent that we treat through a polynomial lift (Sec. VII B of the paper).

## The Pearcey integral

We consider the Pearcey integral

```math
\Psi(k) = \sqrt{\frac{k}{2\pi i}} \int_{\mathbb{R}} e^{i k f(q)}\, \mathrm{d}q, \qquad f(q) = q^4 + \frac{q^2}{2} + q.
```

### Saddle points and the Gauss–Manin system

The exponent has three saddle points, the roots of $f'(q) = 4q^3 + q + 1$: one real saddle point at $q = -1/2$ and a complex conjugate pair. They can be found numerically with [`findAllSaddles`](@ref) and refined to high precision with [`refineSaddle`](@ref); here, we supply them exactly.

```@example pearcey
using Resurgence, DynamicPolynomials, CairoMakie

@polyvar q
f = q^4 + 1//2 * q^2 + q

s7 = sqrt(big(7))
saddles = [[complex(-big(1)/2)], [(1 - s7 * im)/4], [(1 + s7 * im)/4]]
nothing # hide
```

The Gauss–Manin system follows from Singular's `gmspoly.lib`. It consists of the good basis `GM.ϕ` and the matrices `GM.M` and `GM.B`:

```@example pearcey
GM = GaussManin(f)
GM.ϕ
```

```@example pearcey
GM.M
```

```@example pearcey
GM.B
```

The eigenvalues of $M$ are the critical values $\lambda_j = f(q_j)$:

```@example pearcey
ComplexF64.(GM.λs)
```

### The transseries

[`transseries`](@ref) computes the asymptotic series of all saddle points, ordered as `GM.λs`, as a vector of [`Expansion`](@ref)s. Each expansion holds the saddle point `x`, the Borel singularity `f₀` $= t_j = -i\lambda_j$, the exponent `μ` and the coefficients `Ts`, normalised such that `Ts[1]` $= 1/\sqrt{f''(q_j)}$.

```@example pearcey
trans = transseries(f, GM, 60; saddles = saddles)
Σ  = ComplexF64[s.f₀    for s in trans]        # Borel singularities t = -iλ
T0 = ComplexF64[s.Ts[1] for s in trans]        # normalisation of the series
Rs = [ComplexF64.(R) for R in GM.Rs]           # residues of the Fuchsian system
Σ
```

The coefficients grow factorially. On a logarithmic scale:

```@example pearcey
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "m", ylabel = "log₁₀ |Tₘ|")
for (j, s) in enumerate(trans)
    scatter!(ax, 0:length(s.Ts) - 1, Float64.(log10.(abs.(s.Ts))), label = "saddle $j")
end
axislegend(ax, position = :lt)
fig
```

### Adjacency: the Stokes constants

The Stokes constants follow from the monodromy of the Fuchsian system in the Borel plane. With `method = :matrix`, the solutions are transported numerically along keyhole contours.

```@example pearcey
K = stokes_constants(Σ, Rs, 0; T0 = T0, method = :matrix)
K = round.(Int, real.(K))
```

The entry `K[l, j]` is the Stokes constant of the transition in which saddle point $j$ picks up saddle point $l$. All three pairs are adjacent. The adjacency graph in the Borel plane:

```@example pearcey
fig, ax = plotStokesGraphGM(Σ, K)
fig
```

The adjacency controls the divergence of the series: the late terms grow as $\Gamma(m)/|F_j|^m$, with $F_j$ the distance to the nearest *adjacent* Borel singularity ([`singulants`](@ref)). [`divergence_rate`](@ref) estimates $|F_j|$ from the late terms:

```@example pearcey
[singulants(Σ, K) [divergence_rate(s.Ts) for s in trans]]
```

### Relevance: the South-East rule

The exponent is bounded from below, so at $\theta = \pi/2$ only the real saddle point is relevant. Rotating back to the oscillatory integral, the South-East rule tracks which complex saddle points are switched on along the way:

```@example pearcey
rel = relevance(f, Σ, K, T0; saddles = saddles)
rel.n0
```

```@example pearcey
n = southeast(Σ, K, rel.n0; verbose = true)
```

The real saddle point switches on one of the two complex saddle points, the one whose contribution is exponentially suppressed. The other complex saddle point is irrelevant.

### Approximations of the integral

We compare the approximations with the exact integral, evaluated by quadrature along the real line, tilted into the valleys of $e^{ikq^4}$ for large $|q|$:

```@example pearcey
fn(u) = u^4 + u^2 / 2 + u
function exact(k; h = 1e-4)
    acc = 0.0im
    for t in -6:h:6
        u = t + im * 0.25 * tanh(t)
        acc += exp(im * k * fn(u)) * (1 + im * 0.25 * sech(t)^2)
    end
    sqrt(k / (2π * im)) * acc * h
end
exact(6.0)
```

The superasymptotic approximation truncates each relevant series near its smallest term; the hyperasymptotic approximation re-expands the remainder in the series of the adjacent saddle points:

```@example pearcey
k = 6.0
(superasymptotic = superasymptotic(trans, n, Σ, K, k).value,
 hyperasymptotic = hyperasymptotic(trans, n, Σ, K, k; level = 1).value)
```

The Borel–Padé sum of the relevant series provides an independent resummation:

```@example pearcey
borel_pade(k) = sum(n[j] * exp(im * k * ComplexF64(im * trans[j].f₀)) *
                    ComplexF64(BorelPadeResum(trans[j], k)[1]) for j in eachindex(trans) if n[j] != 0)
borel_pade(6.0)
```

The errors as a function of $k$:

```@example pearcey
ks = 2.0:1.0:10.0
errors = map(ks) do k
    Ψ = exact(k)
    (SA = abs(Ψ - superasymptotic(trans, n, Σ, K, k).value),
     H1 = abs(Ψ - hyperasymptotic(trans, n, Σ, K, k; level = 1).value),
     H2 = abs(Ψ - hyperasymptotic(trans, n, Σ, K, k; level = 2).value),
     BP = abs(Ψ - borel_pade(k)))
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "k", ylabel = "|Ψ - Ψₐₚₚᵣₒₓ|", yscale = log10)
for (key, label) in ((:SA, "superasymptotic"), (:H1, "hyperasymptotic, level 1"),
                     (:H2, "hyperasymptotic, level 2"), (:BP, "Borel-Padé"))
    scatterlines!(ax, collect(ks), [max(e[key], 1e-16) for e in errors], label = label)
end
axislegend(ax, position = :lb)
fig
```

The quadrature for the exact integral is accurate to about $10^{-14}$, which sets the floor of the errors.

## The Lorentzian lens

The one-dimensional Lorentzian lens (Sec. VII B of the paper),

```math
\Psi(k) = \sqrt{\frac{k}{2\pi i}} \int_{\mathbb{R}} e^{i k f(x)}\, \mathrm{d}x, \qquad f(x) = \frac{(x - y)^2}{2} + \frac{\beta}{1 + x^2},
```

has a rational exponent. Writing $f = P/Q$ with $P(x) = 2\beta + (1 + x^2)(x - y)^2$ and $Q(x) = 2(1 + x^2)$, a delta function and its Fourier representation lift the integral to the polynomial exponent

```math
f(x_1, x_2, x_3) = P(x_1)\, x_2 + x_3 \left(x_2\, Q(x_1) - 1\right).
```

### Gauss–Manin system and Stokes constants

```@example lens
using Resurgence, DynamicPolynomials, MultivariatePolynomials, LinearAlgebra, CairoMakie
setprecision(BigFloat, 512)

y, β = 1, 1
@polyvar x1 x2 x3 monomial_order = Graded{MultivariatePolynomials.Reverse{InverseLexOrder}}
PP = 2β + (1 + x1^2) * (x1 - y)^2
QQ = 2 * (1 + x1^2)
f = PP * x2 + x3 * (x2 * QQ - 1)

saddles = [refineSaddle(f, s)[1] for s in findAllSaddles(f)]
GM = GaussManin(f)
GM.ϕ
```

The lifted integral has five saddle points. In the lifted representation, the fourth basis element $\phi_4 = 2x_3 + 4x_2 - 2x_1$ generates the asymptotic series of the one-dimensional lens. We normalise each series by its leading term $1/\sqrt{f''(x_j)}$ of the one-dimensional exponent and scale the leading vectors of the saddle solutions accordingly:

```@example lens
d2f(x) = 1 - 2β * (1 - 3x^2) / (1 + x^2)^3
u = [f(p) for p in saddles]

lens = map(1:GM.n) do j
    λ, μ, Ts = asymptotic(j, GM, 120)
    x  = saddles[argmin(abs.(u .- λ))][1]          # x₁ coordinate of the saddle point
    T0 = 1 / sqrt(d2f(Complex{BigFloat}(x)))
    (expansion = Expansion(x, -im * λ, μ, T0 .* Ts[4, :] ./ Ts[4, 1]),
     v = ComplexF64.(T0 .* Ts[:, 1] ./ Ts[4, 1]))
end
trans = [l.expansion for l in lens]
vs    = [l.v for l in lens]

Σ  = ComplexF64[s.f₀ for s in trans]
Rs = [ComplexF64.(R) for R in GM.Rs]
K  = round.(Int, real.(stokes_constants(Σ, Rs, 6; vs = vs, method = :matrix)))
```

All pairs of saddle points are adjacent, except for one pair:

```@example lens
fig, ax = plotStokesGraphGM(Σ, K)
fig
```

### Relevance and approximations

The lifted exponent is unbounded from below, so we start the South-East rule from the one-dimensional integral, in which only the real minimum of $f$ is relevant:

```@example lens
n0 = [abs(imag(ComplexF64(s.x))) < 1e-10 ? 1 : 0 for s in trans]
n  = southeast(Σ, K, n0; verbose = true)
```

Besides the real saddle point (the real image of the lens), a complex saddle point contributes: an evanescent complex image. The approximations converge to the exact integral:

```@example lens
fd(x) = (x - y)^2 / 2 + β / (1 + x^2)
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

borel_pade(k) = sum(n[j] * exp(im * k * ComplexF64(im * trans[j].f₀)) *
                    ComplexF64(BorelPadeResum(trans[j], k)[1]) for j in eachindex(trans) if n[j] != 0)

ks = 2.0:2.0:16.0
errors = map(ks) do k
    Ψ = exact(k)
    (SA = abs(Ψ - superasymptotic(trans, n, Σ, K, k).value),
     H1 = abs(Ψ - hyperasymptotic(trans, n, Σ, K, k; level = 1).value),
     H2 = abs(Ψ - hyperasymptotic(trans, n, Σ, K, k; level = 2).value),
     BP = abs(Ψ - borel_pade(k)))
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "k", ylabel = "|Ψ - Ψₐₚₚᵣₒₓ|", yscale = log10)
for (key, label) in ((:SA, "superasymptotic"), (:H1, "hyperasymptotic, level 1"),
                     (:H2, "hyperasymptotic, level 2"), (:BP, "Borel-Padé"))
    scatterlines!(ax, collect(ks), [max(e[key], 1e-16) for e in errors], label = label)
end
axislegend(ax, position = :rt)
fig
```
