# begin 
# 	fig = Figure()
# 	ax = Axis(fig[1,1], aspect = 1, xlabel="r", ylabel="|α|", limits=(-0.2, 3M+1, -0.2, 3M+1))

# 	lines!(ax, [0, 2M], [0, 3M], color = :red, linewidth=3)
# 	lines!(ax, [0, 2M], [M, 3M], color = :red, linewidth=3)
# 	lines!(ax, [0, 0],  [0, M], color = :red, linewidth=3)
# 	for m in 0:M
# 		lines!(ax, [0, 2m], [m, 3m], color= :black)
# 	end

# 	for r in 0:2M
# 		k = map(sum, collect(keys(A[r+1])))
# 		plot!(ax, r .* ones(size(k)),k)
# 	end
# 	fig
# end

function plotPoles(poles, transseries, plotRange = 3) 
    fig = Figure()
    ax = Axis(fig[1,1], xlabel = "Re", ylabel = "Im", limits=(-plotRange, plotRange,-plotRange, plotRange), aspect = 1)
    
    for i in eachindex(transseries)
        points = poles[i] .+ transseries[i].f₀
        scatter!(ax, real.(points), imag.(points), label = string(i))
    end
    
    for i in eachindex(transseries)
        f0 = transseries[i].f₀
        scatter!(ax, [real(f0)], [imag(f0)], color = :black)
        text!(ax, string(i), position = Point2f(real(f0), imag(f0)), align = (:right, :bottom), offset = (8, 8))
    end
    axislegend(ax)
    return fig, ax
end

"""
Plot the evaluated coefficients A.
"""
function plotA(A, M)
	fig = Figure()
	ax = Axis(fig[1,1], aspect = 1, xlabel="r", ylabel="|α|", limits=(-0.2, 3M+1, -0.2, 3M+1))

	lines!(ax, [0, 2M], [0, 3M], color = :red, linewidth=3)
	lines!(ax, [0, 2M], [M, 3M], color = :red, linewidth=3)
	lines!(ax, [0, 0],  [0, M], color = :red, linewidth=3)
	for m in 0:M
		lines!(ax, [0, 2m], [m, 3m], color= :black)
	end

	for r in 0:2M
		# k = map(sum, collect(keys(A[r+1])))
		k = map(sum, collect(keys(A[r+1]))) ./ 2
		if length(k) ≠ 0
			plot!(ax, r .* ones(size(k)), k)
		end
	end
	return fig, ax
end

"""
Plot the evaluated coefficients A.
"""
function plotAdjacency(transseries, K, plotRange = 3) 
	fig = Figure()
	ax = Axis(fig[1,1], xlabel = "Re", ylabel = "Im", limits=(-plotRange, plotRange,-plotRange, plotRange), aspect = 1)
	
	for i in axes(K, 1), j in axes(K, 2)
		if K[i, j] != 0
			f0i = transseries[i].f₀
			f0j = transseries[j].f₀
			
			lines!(ax, [Point2f.(real(f0i), imag(f0i)), Point2f.(real(f0j), imag(f0j))], color = :red, linewidth = 2)
		end
	end
	
	for i in eachindex(transseries)
		f0 = transseries[i].f₀
		scatter!(ax, [real(f0)], [imag(f0)], color = :black)
		text!(ax, string(i), position = Point2f(real(f0), imag(f0)), align = (:right, :bottom), offset = (8, 8))
	end

	return fig, ax
end

"""
Plot the evaluated coefficients A.
"""
function plotAdjacencyPoles(transseries, K, poles, plotRange = 3) 
	fig = Figure()
	ax = Axis(fig[1,1], xlabel = "Re", ylabel = "Im", limits=(-plotRange, plotRange,-plotRange, plotRange), aspect = 1)
	
	for i in eachindex(transseries)
        points = poles[i] .+ transseries[i].f₀
        scatter!(ax, real.(points), imag.(points), label = string(i))
    end

	for i in axes(K, 1), j in axes(K, 2)
		if K[i, j] != 0
			f0i = transseries[i].f₀
			f0j = transseries[j].f₀
			
			lines!(ax, [Point2f.(real(f0i), imag(f0i)), Point2f.(real(f0j), imag(f0j))], color = :red, linewidth = 2)
		end
	end
	
	for i in eachindex(transseries)
		f0 = transseries[i].f₀
		scatter!(ax, [real(f0)], [imag(f0)], color = :black)
		text!(ax, string(i), position = Point2f(real(f0), imag(f0)), align = (:right, :bottom), offset = (8, 8))
	end

	axislegend(ax)
    return fig, ax
end


"""
Plot the evaluated coefficients A.
"""
function plotAdjacencyGM(Σ, K, plotRange = 3) 
	fig = Figure()
	ax = Axis(fig[1,1], xlabel = "Re", ylabel = "Im", limits=(-plotRange, plotRange,-plotRange, plotRange), aspect = 1)
	
	for i in axes(K, 1), j in axes(K, 2)
		if K[i, j] != 0
			f0i = Σ[i]
			f0j = Σ[j]
			
			lines!(ax, [Point2f.(real(f0i), imag(f0i)), Point2f.(real(f0j), imag(f0j))], color = :red, linewidth = 2)
		end
	end
	
	for i in eachindex(Σ)
		f0 = Σ[i]
		scatter!(ax, [real(f0)], [imag(f0)], color = :black)
		text!(ax, string(i), position = Point2f(real(f0), imag(f0)), align = (:right, :bottom), offset = (8, 8), label = string(i))
	end

	axislegend(ax)
    return fig, ax
end