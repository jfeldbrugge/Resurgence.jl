"""
    findAllSaddles(f)

All critical points of the multivariate polynomial `f` (HomotopyContinuation, double precision),
sorted by the imaginary part of the first coordinate.
"""
function findAllSaddles(f::DynamicPolynomials.Polynomial)
    f_d = sum(ComplexF64(c) * m for (c, m) in zip(coefficients(f), monomials(f)))
    grad = differentiate(f_d, variables(f_d))

    # Find all saddles using HomotopyContinuation
    result = solve(System(grad); show_progress=false)
    saddles = sort(solutions(result), by=p->imag(p[1]))
    return saddles
end

"""
    refineSaddle(f, x0, tol = 10 * eps(BigFloat), max_iterations = 100)

Refine the critical point `x0` of `f` by Newton's method in `Complex{BigFloat}`.
Returns `(x, residual)`.
"""
function refineSaddle(f, x0, tol = 10 * eps(BigFloat), max_iterations = 100)
    f = sum(Complex{BigFloat}(c) * m for (c, m) in zip(coefficients(f), monomials(f)))
    vars = variables(f)
    grad = differentiate(f, vars)
    jacob = [differentiate(g, vars) for g in grad]

    function grad_eval(grad, x0)
        vars = variables(grad[1])
        return map(g -> g(vars => x0), grad)
    end

    function jacob_eval(jacob, x0)
        vars = variables(jacob[1][1])
        return reduce(hcat, map.(g -> g(vars => x0), jacob))
    end

    x = copy(x0)
    ∇f = grad_eval(grad, x)
    res_norm = norm(∇f)

    for k in 1:max_iterations
        if res_norm ≤ tol
            return x, res_norm
        else
            J = jacob_eval(jacob, x)
            Δ = -(J \ ∇f)
            x = x + Δ

            ∇f = grad_eval(grad, x)
            res_norm = norm(∇f)
        end
    end
    return x, res_norm
end
