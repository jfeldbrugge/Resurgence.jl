# Resurgence.jl
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jfeldbrugge.github.io/Resurgence.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jfeldbrugge.github.io/Resurgence.jl/dev/)
[![Build Status](https://github.com/jfeldbrugge/Resurgence.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jfeldbrugge/Resurgence.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jfeldbrugge/Resurgence.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jfeldbrugge/Resurgence.jl)

Multidimensional oscillatory integrals

$$\Psi(k) = \left(\frac{k}{2\pi i}\right)^{d/2} \int_{\mathbb{R}^d} e^{i k f(\boldsymbol{x})}\, \mathrm{d}\boldsymbol{x}$$

with a polynomial (or, by a polynomial lift, rational) exponent $f$ are evaluated asymptotically with resurgence, implemented in Julia. Using integration by parts and Brieskorn reduction, the package constructs the Gauss–Manin system of the exponent, which generates the asymptotic series of all saddle points. Its Borel transform is a Fuchsian system, whose monodromy yields the Stokes constants and hence the adjacency of the saddle points. Combined with the South-East rule, this identifies the relevant saddle points and enables superasymptotic and hyperasymptotic approximations of the integral, without computing steepest-descent manifolds. This code accompanies the publication [Asymptotic evaluation of multidimensional oscillatory integrals](https://arxiv.org/abs/2610.22767).

## Installation

The Resurgence package is not yet registered in the General registry, but can be installed directly from GitHub with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add https://github.com/jfeldbrugge/Resurgence.jl
```

Or, equivalently, via the `Pkg` API:

```julia
julia> using Pkg; Pkg.add(url = "https://github.com/jfeldbrugge/Resurgence.jl")
```

## Usage
For the Pearcey integral with exponent $f(q) = q^4 + q^2/2 + q$, we compute the asymptotic series and the Stokes constants of the saddle points, determine the relevant saddle points with the South-East rule and approximate the integral $\Psi(k)$ at $k = 6$ (exact value $\Psi(6) = -0.1148611265675526 - 0.4893380385050138i$) as follows:

```julia
using Resurgence, DynamicPolynomials

@polyvar q
f = q^4 + 1//2 * q^2 + q

# Saddle points: the roots of f'(q) = 4q³ + q + 1
s7 = sqrt(big(7))
saddles = [[complex(-big(1)/2)], [(1 - s7 * im)/4], [(1 + s7 * im)/4]]

# Gauss-Manin system and the transseries: the asymptotic series of all saddle points to order 60
GM    = GaussManin(f)
trans = transseries(f, GM, 60; saddles = saddles)
Σ  = ComplexF64[s.f₀    for s in trans]        # Borel singularities t = -iλ
T0 = ComplexF64[s.Ts[1] for s in trans]        # normalisation of the series
Rs = [ComplexF64.(R) for R in GM.Rs]           # residues of the Fuchsian system

# Adjacency: Stokes constants K[l, j] = K_{l←j}
K = stokes_constants(Σ, Rs, 0; T0 = T0, method = :matrix)
fig, ax = plotStokesGraphGM(Σ, K)

# Relevance: intersection numbers from the South-East rule
n = relevance(f, Σ, K, T0; saddles = saddles).n

# Superasymptotic and hyperasymptotic approximations of Ψ(k)
k = 6.0
superasymptotic(trans, n, Σ, K, k).value                 # -0.11331 - 0.48913im
hyperasymptotic(trans, n, Σ, K, k; level = 1).value      # -0.114860 - 0.489336im
```

The adjacency graph of the Pearcey integral, drawn with `plotStokesGraphGM`, shows the three Borel singularities $t_j = -i\lambda_j$ of the saddle points in the Borel plane. Every pair of saddle points is adjacent, and each edge is labelled by its Stokes constant.

<figure>
<a href='docs/src/assets/figures/stokes_graph.png'><img src='docs/src/assets/figures/stokes_graph.png' width=100% /></a>
</figure>

Please have a look at the docstrings (e.g. `?stokes_constants`) for more details. The documentation is under development.

## Contributors
This code was written by:
* Job Feldbrugge ([jobfeldbrugge@ntu.edu.tw](mailto:jobfeldbrugge@ntu.edu.tw))

## Citations
When using this code in work, please cite

```
@article{CrewFeldbrugge2026,
    author = {Crew, Samuel and Feldbrugge, Job},
    title = {Asymptotic evaluation of multidimensional oscillatory integrals},
    year = {2026},
    eprint = {2610.22767},
    archivePrefix = {arXiv},
    url = {https://arxiv.org/abs/2610.22767},
}
```
