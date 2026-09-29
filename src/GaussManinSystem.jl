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

function toS(c::Integer)
    return Singular.QQ(c)
end

function toS(c::Rational) 
    return Singular.QQ(BigInt(numerator(c)), BigInt(denominator(c)))
end

function toS(c)
    return Singular.QQ(BigInt(numerator(c)), BigInt(denominator(c))) 
end

function to_singular(p::AbstractPolynomialLike; ordering = :degrevlex)
    vs = variables(p)
    R, _ = Singular.polynomial_ring(Singular.QQ, string.(vs); ordering = ordering)
    B = Singular.MPolyBuildCtx(R)
    for t in terms(p)
        Singular.push_term!(B, toS(rationalize(coefficient(t))), [degree(monomial(t), v) for v in vs])
    end
    return Singular.finish(B)
end

# Singular rational number -> Julia Rational{BigInt}
function fromS(c) 
    return Rational{BigInt}(BigInt(numerator(c)), BigInt(denominator(c)))
end

# DynamicPolynomials variables named like the Singular ring's variables, e.g. [s, x, y, z]
function mp_vars(R) 
    return [DynamicPolynomials.Variable(string(v),
            DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder},
            DynamicPolynomials.Graded{DynamicPolynomials.LexOrder}) for v in Singular.gens(R)]
end

function from_singular(p::Singular.spoly, vars = mp_vars(parent(p)))
    length(vars) == Singular.nvars(parent(p)) ||
        error("need one variable per ring variable $(Singular.gens(parent(p)))")
    one_ = prod(vars)^0
    iszero(p) && return polynomial(zero(Rational{BigInt}) * one_)
    return polynomial(sum(fromS(c) * prod(vars .^ e)
                for (c, e) in zip(Singular.coefficients(p), Singular.exponent_vectors(p))))
end

# vector of Singular polynomials, or a Singular ideal -> Vector of polynomials
function from_singular(ps::AbstractVector{<:Singular.spoly}, vars = mp_vars(parent(first(ps))))
    return [from_singular(p, vars) for p in ps]
end

function from_singular(I::Singular.sideal, vars = mp_vars(Singular.base_ring(I))) 
    return from_singular(Singular.gens(I), vars)
end

# x-exponent vector of the leading monomial (degrevlex) of the s = 0 part of p
function s0_leadexp(p)
    terms = [e[2:end] for e in Singular.exponent_vectors(p) if e[1] == 0]
    isempty(terms) && error("basis element $p has no s = 0 part")
    return reduce((a, b) -> degrevlex_less(a, b) ? b : a, terms)
end

# a < b in degree reverse lexicographical order
function degrevlex_less(a::Vector{Int}, b::Vector{Int})
    sum(a) != sum(b) && return sum(a) < sum(b)
    for i in length(a):-1:1                 # reverse lex: last differing variable decides
        a[i] != b[i] && return a[i] < b[i]
    end
    return false
end

function reorder_basis(b, A; rev::Bool = false)
    bv = b isa Singular.sideal ? Singular.gens(b) : collect(b)
    keys = [s0_leadexp(p) for p in bv]

    p = sortperm(1:length(bv); lt = (i, j) -> degrevlex_less(keys[i], keys[j]), rev = rev)
    n = length(bv)
    Ap = [A[p[i], p[j]] for i in 1:n, j in 1:n]
    return bv[p], Ap, p
end

"""
Build the Gauss-Manin system.
"""
function GaussManin(f)
	vars = variables(f)
	dim = length(vars)
	
	f_singular = to_singular(f)
	
	@assert Singular.LibGmspoly.isTame(f_singular) == 1 "Polynomial is not tame." 

	Rs, objs = Singular.LibGmspoly.goodBasis(f_singular)
	b = Singular.gens(objs[:b]) # good basis of the Brieskorn lattice
	A = objs[:A] # matrix of t with respect to b:  t = A + s^2*d/ds

	b, A, perm = reorder_basis(b, A)
	
	n = Singular.nrows(A)
	s = Singular.gen(Rs, 1) 

	# Singular's jet(p,0): the constant term of p.
	jet0(p) = iszero(p) ? p : Rs(Singular.constant_coefficient(p))

	# Singular's p/var(1): divide termwise by s, dropping terms not divisible by s.
	function divs(p)
	    B = Singular.MPolyBuildCtx(Rs)
	    for (c, e) in zip(Singular.coefficients(p), Singular.exponent_vectors(p))
	        e[1] >= 1 || continue
	        e[1] -= 1
	        Singular.push_term!(B, c, e)
	    end
	    return Singular.finish(B)
	end

	A0 = [jet0(A[i, j]) for i in 1:n, j in 1:n]          # jet(A,0)
	A1 = [jet0(divs(A[i, j])) for i in 1:n, j in 1:n]    # jet(A/var(1),0)

	toQ(p) = iszero(p) ? zero(Rational{BigInt}) : Rational{BigInt}(Singular.constant_coefficient(p))
	
	M = collect(transpose(toQ.(A0)))
	B = collect(transpose(toQ.(A1)))

	# Project B - (1 + d/2) I onto the partial fraction matrices Rⱼ
	M_eigen = eigen(M)
	λs = M_eigen.values
	V = M_eigen.vectors
	Πs = [prod((M - λs[i] * I) / (λs[j] - λs[i]) for i in eachindex(λs) if i ≠ j) for j in eachindex(λs)]

	Rs = [Π * (B - (1 - dim / 2) * I) for Π in Πs]

	GM = GaussManinSystem(M, B, vars, from_singular(b), λs, V, Rs, Πs, n)
	return GM
end
