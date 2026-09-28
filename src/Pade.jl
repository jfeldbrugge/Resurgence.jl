"""
    TaylorToPade(a, m, n)

Compute the [m/n] Padé approximant from Taylor coefficients `a`,
where `a[k+1]` is the coefficient of x^k.

Returns `(num, den)` where
- `num` are numerator coefficients in increasing order
- `den` are denominator coefficients in increasing order, with `den[1] == 1`
"""
function TaylorToPade(a::AbstractVector, m::Int, n::Int)
    length(a) >= m + n + 1 || error("Need at least m+n+1 Taylor coefficients")

    T = promote_type(eltype(a), Float64)
    a = collect(T, a)

    if n == 0
        return a[1:m+1], T[1]
    end

    A = zeros(T, n, n)
    b = zeros(T, n)

    for k in 1:n
        b[k] = -a[m + k + 1]
        for j in 1:n
            A[k, j] = a[m + k - j + 1]
        end
    end

    q = A \ b
    den = vcat(one(T), q)

    num = zeros(T, m + 1)
    for k in 0:m
        s = zero(T)
        for j in 0:min(k, n)
            s += den[j + 1] * a[k - j + 1]
        end
        num[k + 1] = s
    end

    return num, den
end

"""
    poly_eval(c, x)

Evaluate polynomial with coefficients in increasing order.
"""
function polyEvaluate(c, x)
    y = zero(x)
    for ck in reverse(c)
        y = muladd(y, x, ck)
    end
    return y
end

"""
    pade_eval(num, den, x)

Evaluate the rational approximant.
"""
function PadeEvaluate(num, den, x) 
    return polyEvaluate(num, x) / polyEvaluate(den, x)
end