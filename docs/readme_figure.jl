# The figure at the top of README.md: the closed-form circle model of the README's minimal
# example, its field, and the three queries of the README's table answered by that one model.
#     julia --project=docs docs/readme_figure.jl
using VariationalDiffusion, LuxCore, Random, LinearAlgebra
using CairoMakie

sched = VPSDE()
θ = range(0, 2π; length = 49)[1:48]
circle = NoisePredictor(GaussianMixtureEps(sched, vcat(cos.(θ)', sin.(θ)'); s = 0.05), sched)
ps, st = LuxCore.setup(Xoshiro(0), circle)
m = ImplicitDiffusion(circle, field_nodes(Xoshiro(1), 2; samples = 8))

query(z₀, ρ) = first(implicit_infer(m, z₀, ρ, ps, st)).z
up, dn = query([0.6, 0.5], [Inf, 0.0]), query([0.6, -0.5], [Inf, 0.0])        # x given: two answers
left, right = query([-0.5, 0.6], [0.0, Inf]), query([0.5, 0.6], [0.0, Inf])   # y given
off = query([1.05, 0.3], [Inf, 0.0])                                          # just off the circle

field(x, y) = first(prior_field(m, [x, y], ps, st))
grid = range(-1.5, 1.5; length = 151)
logmag = [log10(norm(field(x, y))) for x in grid, y in grid]
arrowgrid = range(-1.35, 1.35; length = 13)
pts = vec([Point2f(x, y) for x in arrowgrid, y in arrowgrid])
dirs = [Vec2f(-field(p...) / norm(field(p...))) for p in pts]
ring = (cos.(range(0, 2π; length = 300)), sin.(range(0, 2π; length = 300)))

set_theme!(fontsize = 17)
fig = Figure(size = (1100, 520), backgroundcolor = :white)
ax1 = Axis(fig[1, 1]; aspect = DataAspect(), title = "a relation: the stable roots of a field",
           subtitle = "the field of a diffusion model of points on a circle", xlabel = "x", ylabel = "y")
heatmap!(ax1, grid, grid, logmag; colormap = Reverse(:Blues), colorrange = (-2, maximum(logmag)), lowclip = :navy)
arrows2d!(ax1, pts, 0.11 .* dirs; color = (:black, 0.55))

ax2 = Axis(fig[1, 2]; aspect = DataAspect(), title = "one model, queried in any direction",
           subtitle = "the query decides what is input and what is output", xlabel = "x", ylabel = "y")
lines!(ax2, ring...; color = :gray80, linewidth = 2)
vlines!(ax2, [0.6]; color = (:steelblue, 0.6), linestyle = :dash)
hlines!(ax2, [0.6]; color = (:orangered, 0.6), linestyle = :dash)
vlines!(ax2, [1.05]; color = (:black, 0.4), linestyle = :dot)
scatter!(ax2, [up[1], dn[1]], [up[2], dn[2]]; color = :steelblue, markersize = 18, label = "given x = 0.6: two answers")
scatter!(ax2, [left[1], right[1]], [left[2], right[2]]; color = :orangered, markersize = 18, marker = :diamond, label = "given y = 0.6: the other direction")
scatter!(ax2, [off[1]], [off[2]]; color = :black, marker = :xcross, markersize = 18, label = "given x = 1.05: nearest ridge point")
for a in (ax1, ax2)
    limits!(a, -1.5, 1.5, -1.5, 1.5)
end
Legend(fig[2, 1:2], ax2; orientation = :horizontal, framevisible = false, tellheight = true)

out = joinpath(@__DIR__, "src", "assets", "readme_circle.png")
mkpath(dirname(out))
save(out, fig; px_per_unit = 2)
println("wrote ", out)
