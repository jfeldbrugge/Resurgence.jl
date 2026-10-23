"""
Capture all the relevant variables of the Gauss-Manin system.
"""
mutable struct GaussManinSystem
    M
    B
    vars
    ϕ
    λs
    V
    Rs
    Πs
    n
end

const GMSPOLY_LIB = joinpath(@__DIR__, "singular", "gmspoly_resonant.lib")

"""
Run a Singular script with the Singular executable and return everything it prints. Each call is a separate
process, so Singular uses its own memory management (much faster than running it inside Julia).
"""
function run_singular(script::AbstractString)
    mktempdir() do dir
        path = joinpath(dir, "script.sing")
        write(path, script)
        out = IOBuffer()
        run(pipeline(ignorestatus(`$(Singular_jll.Singular()) -q --no-rc $path`); stdout = out, stderr = out))
        return String(take!(out))
    end
end

"""
Write the polynomial f in Singular syntax, with the variables named x(1), ..., x(n).
"""
function singular_string(f, vars = variables(f))
    term(t) = begin
        c = coefficient(t)
        c = c isa Union{Integer, Rational} ? Rational{BigInt}(c) : rationalize(BigInt, c)
        m = join(("*x($i)^$(degree(monomial(t), v))" for (i, v) in enumerate(vars) if degree(monomial(t), v) > 0))
        "($(numerator(c))/$(denominator(c)))" * m
    end
    return isempty(terms(f)) ? "0" : join((term(t) for t in terms(f)), "+")
end

"""
Singular header that loads the patched gmspoly.lib and defines the polynomial f.
"""
function singular_header(f, vars = variables(f))
    return """
    LIB "$(GMSPOLY_LIB)";
    ring R=0,(x(1..$(length(vars)))),dp;
    poly f=$(singular_string(f, vars));
    """
end

"""
Collect the lines printed with the given tag, or report the Singular output if the script did not finish.
"""
function singular_lines(out, tag)
    occursin("RESURGENCE_DONE", out) || error("Singular failed:\n" * out)
    return [split(l)[2:end] for l in split(out, '\n') if startswith(l, tag * " ")]
end

"""
Parse a Singular rational number such as -25/8.
"""
function parseQ(c::AbstractString)
    p = split(c, '/')
    return length(p) == 1 ? Rational{BigInt}(parse(BigInt, p[1])) : parse(BigInt, p[1]) // parse(BigInt, p[2])
end

"""
    isTame(f)

Test whether the polynomial `f` is tame in the sense of Broughton (Singular's `gmspoly.lib`).
"""
function isTame(f)
    out = run_singular(singular_header(f) * """
    "RESURGENCE_TAME "+string(isTame(f));
    "RESURGENCE_DONE";
    quit;
    """)
    return only(singular_lines(out, "RESURGENCE_TAME"))[1] == "1"
end

"""
    goodBasis(f; check_tame = true)

Good basis of the Brieskorn lattice of the tame polynomial `f` (Singular's `gmspoly.lib`, patched to keep
resonant terms in A). Returns the basis elements `b` as lists of (coefficient, exponents [s, x...]) and the
matrices A0 and A1 with `t[b] = [b](A0 + s A1 + s^2 d/ds)`.
"""
function goodBasis(f; check_tame = true)
    tame = check_tame ? "if(isTame(f)!=1){\"RESURGENCE_NOTTAME\"; \"RESURGENCE_DONE\"; quit;}" : ""
    out = run_singular(singular_header(f) * """
    $tame
    def Rs=goodBasis(f);
    setring(Rs);
    int mu=ncols(b);
    matrix Z[mu][mu];
    if(A/var(1)^2!=Z){"RESURGENCE_NOTLINEAR"; "RESURGENCE_DONE"; quit;}
    int i,j; poly p;
    for(i=1;i<=mu;i++){p=b[i]; while(p!=0){"RESURGENCE_B "+string(i)+" "+string(leadcoef(p))+" "+string(leadexp(p)); p=p-lead(p);}}
    for(i=1;i<=mu;i++){for(j=1;j<=mu;j++){"RESURGENCE_A "+string(i)+" "+string(j)+" "+string(jet(A[i,j],0))+" "+string(jet(A[i,j]/var(1),0));}}
    "RESURGENCE_DONE";
    quit;
    """)
    isempty(singular_lines(out, "RESURGENCE_NOTTAME")) || error("Polynomial is not tame.")
    isempty(singular_lines(out, "RESURGENCE_NOTLINEAR")) || error("goodBasis: A is not of the form A0 + s*A1")

    Alines = singular_lines(out, "RESURGENCE_A")
    n = isqrt(length(Alines))
    A0 = zeros(Rational{BigInt}, n, n)
    A1 = zeros(Rational{BigInt}, n, n)
    for (i, j, a0, a1) in Alines
        A0[parse(Int, i), parse(Int, j)] = parseQ(a0)
        A1[parse(Int, i), parse(Int, j)] = parseQ(a1)
    end
    b = [Tuple{Rational{BigInt}, Vector{Int}}[] for _ in 1:n]
    for (i, c, e) in singular_lines(out, "RESURGENCE_B")
        push!(b[parse(Int, i)], (parseQ(c), parse.(Int, split(e, ','))))
    end
    return (b = b, A0 = A0, A1 = A1)
end

"""
DynamicPolynomials variables [s, x...] for the good basis, named after the variables of f.
"""
function mp_vars(vars)
    return [DynamicPolynomials.Variable(name,
            DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder},
            DynamicPolynomials.Graded{DynamicPolynomials.LexOrder}) for name in ["s"; string.(vars)]]
end

"""
Convert a basis element, given as a list of (coefficient, exponents), to a polynomial.
"""
function to_polynomial(p, vars)
    one_ = prod(vars)^0
    isempty(p) && return polynomial(zero(Rational{BigInt}) * one_)
    return polynomial(sum(c * prod(vars .^ e) for (c, e) in p))
end

"""
Evaluate the x-exponent vector of the leading monomial (degrevlex) of the s = 0 part of p
"""
function s0_leadexp(p)
    terms = [e[2:end] for (_, e) in p if e[1] == 0]
    isempty(terms) && error("basis element $p has no s = 0 part")
    return reduce((a, b) -> degrevlex_less(a, b) ? b : a, terms)
end

"""
a < b in degree reverse lexicographical order.
"""
function degrevlex_less(a::Vector{Int}, b::Vector{Int})
    sum(a) != sum(b) && return sum(a) < sum(b)
    for i in length(a):-1:1                 # reverse lex: last differing variable decides
        a[i] != b[i] && return a[i] < b[i]
    end
    return false
end

"""
Permutation that sorts the basis by the leading monomial of its s = 0 part (degrevlex).
"""
function basis_order(b; rev::Bool = false)
    keys = [s0_leadexp(p) for p in b]
    return sortperm(1:length(b); lt = (i, j) -> degrevlex_less(keys[i], keys[j]), rev = rev)
end

"""
    GaussManin(f)

Gauss-Manin system of the tame polynomial `f` (Singular's `gmspoly.lib`, see `goodBasis`): the matrices M and B
of the connection in a good basis of the Brieskorn lattice, the critical values λs (eigenvalues of M),
the eigenvectors V, the spectral projectors Πs and the residues Rs of the Borel-plane Fuchsian system.
"""
function GaussManin(f)
    vars = variables(f)
    dim = length(vars)

    gb = goodBasis(f)                    # t[b] = [b](A0 + s A1 + s^2 d/ds)
    perm = basis_order(gb.b)
    b = gb.b[perm]
    n = length(b)

    M = collect(transpose(gb.A0[perm, perm]))
    B = collect(transpose(gb.A1[perm, perm]))

    # Project B - (d/2) I onto the partial fraction matrices Rⱼ: the Borel transform of k^(d/2-1) ∫ ϕ e^{ikf},
    # for which every Rⱼ is nilpotent (tr Rⱼ = 0), the optimal choice for the hyperlogarithm expansion
    M_eigen = eigen(Complex{BigFloat}.(M))
    λs = Complex{BigFloat}.(M_eigen.values)
    V  = Complex{BigFloat}.(M_eigen.vectors)

    Πs = [prod((M - λs[i] * I) / (λs[j] - λs[i]) for i in eachindex(λs) if i ≠ j) for j in eachindex(λs)]
    Rs = [Π * (B - dim//2 * I) for Π in Πs]

    vs = mp_vars(vars)
    ϕ = [to_polynomial(p, vs) for p in b]
    GM = GaussManinSystem(M, B, vars, ϕ, λs, V, Rs, Πs, n)
    return GM
end
