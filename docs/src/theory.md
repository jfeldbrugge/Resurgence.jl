```@meta
CurrentModule = Resurgence
```

# Theory

Resurgence.jl evaluates oscillatory integrals

```math
\Psi(k) = \left(\frac{k}{2\pi i}\right)^{d/2} \int_{\mathbb{R}^d} e^{i k f(\boldsymbol{x})}\, \mathrm{d}\boldsymbol{x}
```

with a polynomial exponent $f$ and a large parameter $k$. The method replaces the geometric analysis of steepest descent manifolds in $\mathbb{C}^d$ by algebra: integration by parts yields a finite-dimensional differential equation in $k$, the Gauss–Manin system, whose Borel transform is a Fuchsian system. From these, the package computes the three ingredients of an exponentially accurate approximation of $\Psi$: the asymptotic series of all saddle points, the Stokes constants that encode their adjacency, and the intersection numbers that encode their relevance. This page summarises the derivation; for the details, see [Crew & Feldbrugge (2026)](https://arxiv.org/abs/2610.22767), whose section numbers are quoted below.

## The Picard–Lefschetz formula

Let $f$ be a tame Morse polynomial with non-degenerate critical points $\boldsymbol{x}_j$, $\nabla f(\boldsymbol{x}_j) = 0$, and distinct critical values $\lambda_j = f(\boldsymbol{x}_j)$, $j = 1, \dots, n$. The critical points are in general complex; we refer to them as saddle points. The steepest descent manifold (Lefschetz thimble) $\mathcal{J}_j$ of $\boldsymbol{x}_j$ defines the thimble integral

```math
\Psi^{(j)}(k) = \left(\frac{k}{2\pi i}\right)^{d/2} \int_{\mathcal{J}_j} e^{i k f(\boldsymbol{x})}\, \mathrm{d}\boldsymbol{x},
```

and the original integral decomposes as (Sec. II)

```math
\Psi = \sum_{j=1}^n n_j\, \Psi^{(j)},
```

with integer intersection numbers $n_j$. A saddle point is *relevant* when $n_j \neq 0$. The thimbles are oriented such that

```math
\Psi^{(j)}(k) = \frac{e^{i k \lambda_j}}{\sqrt{\det \mathcal{H}f(\boldsymbol{x}_j)}}\left[1 + \mathcal{O}(k^{-1})\right],
```

with the principal branch for complex saddle points and $\sqrt{\det \mathcal{H}f} = e^{i\pi\nu_j/2}\sqrt{|\det \mathcal{H}f|}$ for a real saddle point of Morse index $\nu_j$, such that every real saddle point enters with $n_j = 1$. Expanding each thimble integral yields the transseries

```math
\Psi \sim \sum_{j=1}^n n_j\, e^{i k \lambda_j} \sum_{m=0}^\infty \frac{T_m^{(j)}}{k^m}, \qquad T_0^{(j)} = \frac{1}{\sqrt{\det \mathcal{H}f(\boldsymbol{x}_j)}}.
```

## The Gauss–Manin system

Integration by parts shows that a moment $\langle g \rangle = \int_{\mathcal{J}_j} g(\boldsymbol{x})\, e^{ikf(\boldsymbol{x})}\,\mathrm{d}\boldsymbol{x}$ is unchanged by $g \mapsto g + \partial_i h + i k\, h\, \partial_i f$ for any polynomial $h$. Modulo these relations, every polynomial reduces to a combination of $n$ basis functions $\boldsymbol{\phi} = (\phi_1, \dots, \phi_n)$, with $\phi_1 = 1$, and the standard moments

```math
I_i^{(j)}(k) = \left(\frac{k}{2\pi i}\right)^{d/2} \int_{\mathcal{J}_j} \phi_i(\boldsymbol{x})\, e^{ikf(\boldsymbol{x})}\, \mathrm{d}\boldsymbol{x}
```

satisfy, for a *good basis* in the sense of Saito, the Gauss–Manin system (Sec. III)

```math
\frac{\mathrm{d}\boldsymbol{I}^{(j)}}{\mathrm{d}k} = \left[i M + \frac{1}{k}\left(\frac{d}{2} I - B\right)\right] \boldsymbol{I}^{(j)}.
```

The first component $I_1^{(j)} = \Psi^{(j)}$ is the thimble integral itself. The matrix $M$ represents multiplication by $f$, and its eigenvalues are the critical values $\lambda_j$. [`GaussManin`](@ref) computes the good basis `GM.ϕ` and the matrices `GM.M` and `GM.B` with the `gmspoly.lib` library of Singular, via [Singular.jl](https://github.com/oscar-system/Singular.jl).

## Asymptotic series

Substituting the ansatz

```math
\boldsymbol{I}^{(j)}(k) \sim e^{ik\lambda_j} (ik)^{\mu_j} \sum_{m=0}^\infty \frac{\boldsymbol{T}_m^{(j)}}{k^m}
```

into the Gauss–Manin system yields an eigenvalue problem, $M \boldsymbol{T}_0^{(j)} = \lambda_j \boldsymbol{T}_0^{(j)}$, followed by a linear recursion for the higher coefficients (Sec. IV). The exponent $\mu_j$ vanishes for non-degenerate saddle points. [`asymptotic`](@ref) solves the recursion to any order in arbitrary precision, and [`transseries`](@ref) returns the series of all saddle points as a vector of [`Expansion`](@ref)s, normalised by $T_0^{(j)} = 1/\sqrt{\det \mathcal{H}f(\boldsymbol{x}_j)}$.

## Into the Borel plane

With the Borel variable $t = -i f(\boldsymbol{x})$, each thimble maps to a ray from the Borel singularity $t_j = -i\lambda_j$, and the Borel transform of the moments,

```math
\widehat{\boldsymbol{I}}^{(j)}(t) = \sum_{m=0}^\infty \frac{\boldsymbol{T}_m^{(j)} (t - t_j)^m}{m!},
```

solves the Fuchsian system (Sec. V)

```math
\frac{\mathrm{d}\widehat{\boldsymbol{I}}}{\mathrm{d}t} = A(t)\, \widehat{\boldsymbol{I}}, \qquad A(t) = \sum_{j=1}^n \frac{R_j}{t - t_j},
```

with simple poles at the Borel singularities. The residues $R_j$ (`GM.Rs`) are nilpotent in the normalisation of the package. Solutions are continued along any path in $\mathbb{C}\setminus\{t_1,\dots,t_n\}$, either by transporting the solution numerically or with hyperlogarithms.

### Stokes constants

When the phase of $k$ rotates, the Laplace ray of saddle point $\boldsymbol{x}_j$ can cross the singularity $t_l$. Across this Stokes transition the thimble $\mathcal{J}_j$ picks up $K_{lj}$ copies of $\mathcal{J}_l$,

```math
\mathcal{S}^+ \boldsymbol{I}^{(j)} - \mathcal{S}^- \boldsymbol{I}^{(j)} = K_{lj}\, \boldsymbol{I}^{(l)},
```

with the integer Stokes constant $K_{lj}$. Two saddle points are *adjacent* when $K_{lj} \neq 0$. As the difference of two continuations is again a solution of the Fuchsian system, $K_{lj}$ follows from continuing the Borel transform of $\boldsymbol{x}_j$ along two keyhole paths that pass $t_l$ on either side, and comparing their difference with the Borel transform of $\boldsymbol{x}_l$ at a single point. [`stokes_constants`](@ref) evaluates all $K_{lj}$ in this way; [`plotStokesGraphGM`](@ref) draws the resulting adjacency graph.

!!! note "Sign convention"
    The package returns `K[l, j]` $= -K_{lj}$ with $K_{lj}$ the Stokes constants of the paper (the keyhole orientation of the package), so that the late-term relations below hold without a minus sign. The functions [`southeast`](@ref), [`superasymptotic`](@ref) and [`hyperasymptotic`](@ref) use the package convention by default.

Alternatively, [`findStokesConstants`](@ref) estimates the Stokes constants from Borel–Padé approximants of the asymptotic series ([`findPoles`](@ref)), and [`BorelPadeResum`](@ref) resums a single series by a Laplace transform of its Padé approximant.

## Approximating the integral

### Relevance: the South-East rule

For an exponent bounded from below on $\mathbb{R}^d$, rotate $k \mapsto |k|\,e^{i\theta}$. At $\theta = \pi/2$ the integral is Euclidean, and exactly the real saddle points are relevant ($n_j = 1$), the complex ones not ($n_j = 0$). Rotating back to $\theta = 0$, the Borel singularities rotate clockwise, and a relevant saddle point $\boldsymbol{x}_j$ switches on the adjacent saddle point $\boldsymbol{x}_l$ when they line up horizontally (Sec. VI A):

```math
n_l \mapsto n_l + n_j K_{lj}.
```

Only edges $t_j \to t_l$ of the adjacency graph pointing South-East, $\arg(t_l - t_j) \in [-\pi/2, 0]$, contribute, ordered from steep to shallow. [`relevance`](@ref) applies the rule from start to finish; [`southeast`](@ref) applies it to given initial intersection numbers, for instance for exponents unbounded from below.

### Superasymptotics

The late terms of the series of $\boldsymbol{x}_j$ are governed by the adjacent saddle points (Sec. VI B),

```math
T_m^{(j)} \approx \sum_l \frac{K_{lj}}{2\pi i} \sum_{r \geq 0} \frac{\Gamma(m - r)}{F_{jl}^{\,m-r}}\, T_r^{(l)}, \qquad F_{jl} = t_l - t_j,
```

in the package convention for `K`. The terms are smallest near $N_j \approx |k|\,\min_{l:K_{lj}\neq 0} |F_{jl}|$ ([`singulants`](@ref)). Truncating each series there yields the superasymptotic approximation [`superasymptotic`](@ref),

```math
\Psi(k) \approx \sum_j n_j\, e^{ik\lambda_j} \sum_{m=0}^{N_j - 1} \frac{T_m^{(j)}}{k^m},
```

with an exponentially small error. It is not the nearest singularity in the Borel plane that controls the divergence, but the nearest *adjacent* one.

### Hyperasymptotics

The remainder after optimal truncation is re-expanded in the series of the adjacent saddle points (Sec. VI C),

```math
R_N^{(j)}(k) = \sum_l \frac{K_{lj}}{2\pi i} (kF_{jl})^{-N} \int_0^\infty \frac{v^{N-1} e^{-v}}{1 - v/(kF_{jl})}\, S_l\!\left(\frac{v}{F_{jl}}\right) \mathrm{d}v,
```

with $S_l$ the series of $\boldsymbol{x}_l$. Truncating $S_l$ yields the first hyperasymptotic level in terms of the hyperterminant

```math
h(M, w) = \frac{1}{\Gamma(M)} \int_0^\infty \frac{v^{M-1} e^{-v}}{1 - v/w}\, \mathrm{d}v = -w\, e^{-w} E_M(-w)
```

([`hyperterminant`](@ref)), with $E_M$ the generalised exponential integral. Higher levels follow paths in the adjacency graph. [`hyperasymptotic`](@ref) implements the first two levels, with the optimal truncations of Olde Daalhuis ([`path_lengths`](@ref)).
