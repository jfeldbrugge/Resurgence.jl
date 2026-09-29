using GaussManinResurgence
using LinearAlgebra
using DynamicPolynomials
using Test

# RK4 solution of Y' = Σ_s R_s/(z-σ_s) Y along a polyline, Y(path[1]) = I.
function rk4_transport(Σ, Rs, path; h = 1e-3)
    A(z) = sum(Rs[s] / (z - Σ[s]) for s in eachindex(Σ))
    Y = Matrix{ComplexF64}(I, size(Rs[1])...)
    for k in 1:length(path)-1
        a, b = path[k], path[k+1]
        m = max(1, ceil(Int, abs(b - a) / h)); dz = (b - a) / m
        for i in 0:m-1
            z = a + i*dz
            k1 = A(z)*Y; k2 = A(z + dz/2)*(Y + dz/2*k1)
            k3 = A(z + dz/2)*(Y + dz/2*k2); k4 = A(z + dz)*(Y + dz*k3)
            Y += dz/6*(k1 + 2k2 + 2k3 + k4)
        end
    end
    Y
end

@testset "GaussManinResurgence.jl" begin
    @testset "hyperlogarithmic monodromy vs. ODE" begin
        Σ  = ComplexF64[0.4im, -0.4im, 0.7]
        Rs = [ComplexF64[0.2 0.1; 0.0 -0.1], ComplexF64[0.1 0.0; 0.15 0.05], ComplexF64[-0.05 0.1; 0.1 0.1]]
        N  = 10
        for z0 in (0.0im, 0.1 + 0.05im)
            Yz0 = rk4_transport(Σ, Rs, [0.0im, z0])
            T = monodromy_matrices(Σ, Rs, z0, N)
            for j in eachindex(Σ)
                Yγ = rk4_transport(Σ, Rs, [0.0im; loop_path(Σ, j, z0; npts = 200)])
                @test T[j] ≈ Yz0 \ Yγ  rtol = 1e-4
            end
        end
    end

    @testset "keyhole Stokes constant (Airy)" begin
        # x^3/3 - x: the two saddles are connected with Stokes constant K = i.
        @polyvar x
        GM = GaussManin(x^3/3 - x)
        Σ  = ComplexF64.(-im .* GM.λs); Rs = [ComplexF64.(R) for R in GM.Rs]
        K21, S, res = stokes_constant(Σ, Rs, 1, 2, 12)
        @test K21 ≈ im  atol = 1e-10
        @test res < 1e-10
        γp, γm = keyhole_paths(Σ, 1, 2)
        @test S ≈ rk4_transport(Σ, Rs, γm) \ rk4_transport(Σ, Rs, γp)  rtol = 1e-6
    end

    @testset "hyperlogarithmic monodromy in BigFloat (commuting case)" begin
        # Diagonal Rs commute, so the weight-r part of the monodromy around σ_j is (2πi R_j)^r / r!.
        setprecision(BigFloat, 256) do
            CB = Complex{BigFloat}
            Σ  = CB[0, big"0.5" + big"0.3"*im, -big"0.4"*im]
            Rs = [diagm(CB[1//3, -1//5]), diagm(CB[1//7, 2//9]), diagm(CB[-1//4, 1//6])]
            N  = 4
            T  = monodromy_pieces(Σ, Rs, big"0.1" + big"0.05"*im, N)
            @test eltype(T[1][1]) == CB
            for j in eachindex(Σ), r in 0:N
                @test maximum(abs, T[j][r+1] - (2 * CB(π) * im * Rs[j])^r / factorial(r)) < big"1e-60"
            end
        end
    end
end
