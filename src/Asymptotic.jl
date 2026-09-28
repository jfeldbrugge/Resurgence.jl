"""
Expansion consists of a f₀ and Tₘ
"""
mutable struct Expansion
	x
    f₀
    μ
    Ts
end

"""
Evaluate the asymptotic series expansion.
"""
function asymptotic(j, GM::GaussManinSystem, order)
	# Dimension of the system
	d = length(GM.vars)
	
	# Number of saddle points
	n = size(GM.M, 1)

	# Eigenvalue decomposition
	M_eigen = eigen(ComplexF64.(GM.M))
	Λ = diagm(M_eigen.values)
	V = M_eigen.vectors
	B̃ = d/2 * I - (inv(V) * GM.B * V)

	# Prameters
	λ = M_eigen.values
	μ = diag(B̃)

	# Set Y₀
	Ys = zeros(Complex{BigFloat}, n, order + 1)
	# Ys = zeros(ComplexF64, n, order + 1)
	Ys[j, 1] = 1

	# Perform recursion
	for m in 1:order
		for i in 1:n
			if i ≠ j
				ΣBY = sum(B̃[i, l] * Ys[l, m] for l in 1:n if l ≠ i)
				Ys[i, m + 1] = (ΣBY + (μ[i] - μ[j] + m - 1) * Ys[i, m]) / (λ[j] - λ[i])
			end
		end
	
		Ys[j, m + 1] = - 1 / m * sum(B̃[j, l] * Ys[l, m + 1] for l in 1:n if l ≠ j)
	end
	
	# Transform the Yₘ coefficient to the Tₘ coefficients
	Ts = copy(Ys)
	for m in 1:order + 1
		Ts[:, m] =  V * Ys[:, m] ./ im^m
	end
	Ts = Ts ./ Ts[1,1]
	
	return λ[j], μ[j], Ts
end