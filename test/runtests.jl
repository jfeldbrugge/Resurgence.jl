using Resurgence
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

@testset "Resurgence.jl" begin
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

    @testset "monodromy: formula vs. direct loop, stems, traces" begin
        Σ  = ComplexF64[0.4im, -0.4im, 0.7]
        Rs = [ComplexF64[0.2 0.1; 0.0 -0.1], ComplexF64[0.1 0.0; 0.15 0.05], ComplexF64[-0.05 0.1; 0.1 0.1]]
        N, z0 = 6, 0.1 + 0.05im
        for j in eachindex(Σ)
            Mf, _ = monodromy_factor(Σ, j, z0, N)
            @test evaluate_series(Mf, Rs, N) ≈ evaluate_series(direct_loop_factor(Σ, j, z0, N), Rs, N)  rtol = 1e-10
        end
        # a stem homotopic to the straight one gives the same monodromy
        r = 0.3 * Resurgence.dist(Σ[3], Σ; skip = 3); q = Σ[3] + r*(z0 - Σ[3])/abs(z0 - Σ[3])
        @test evaluate_series(monodromy_factor(Σ, 3, z0, N; stem = [z0, 0.3 + 0.15im, q])[1], Rs, N) ≈
              evaluate_series(monodromy_factor(Σ, 3, z0, N)[1], Rs, N)  rtol = 1e-10
        D = monodromy_pieces(Σ, Rs, z0, N)
        @test trace_contributions(D[1], D[2], N)[1] ≈ (2π*im)^2 * tr(Rs[1] * Rs[2])
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

    @testset "keyhole Stokes constants, matrix transport" begin
        # Airy: the matrix back-end agrees with the hyperlogarithm one.
        @polyvar x
        GM = GaussManin(x^3/3 - x)
        Σ  = ComplexF64.(-im .* GM.λs); Rs = [ComplexF64.(R) for R in GM.Rs]
        Km, Sm, resm = stokes_constant(Σ, Rs, 1, 2, 0; method = :matrix)
        @test Km ≈ im  atol = 1e-10
        @test resm < 1e-10
        γp, γm = keyhole_paths(Σ, 1, 2)
        @test matrix_transport(Σ, Rs, γp) ≈ rk4_transport(Σ, Rs, γp)  rtol = 1e-6
        # four saddles: with T₀ = 1/√det H normalisation all Stokes constants are integers
        @polyvar x y
        f = 1//3*x^3 + 1//3*y^3 + 1//2*x*y - x - 2y
        hess = differentiate(differentiate(f, [x, y]), [x, y])
        saddles = [refineSaddle(f, s)[1] for s in findAllSaddles(f)]
        u = f.(saddles); T0s = [1 / sqrt(det([h(p) for h in hess])) for p in saddles]
        GM = GaussManin(f)
        Σ  = ComplexF64.(-im .* GM.λs); Rs = [ComplexF64.(R) for R in GM.Rs]
        T0 = [ComplexF64(T0s[argmin(abs.(u .- λ))]) for λ in GM.λs]
        Kmat = stokes_constants(Σ, Rs, 0; T0 = T0, method = :matrix)
        @test maximum(abs, Kmat - round.(real.(Kmat))) < 1e-8
        @test count(!iszero, round.(Int, real.(Kmat))) == 10
    end

end
