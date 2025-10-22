using LinearAlgebra, Random, Statistics, DataFrames, CairoMakie

# --- fixed antisymmetric 3x3 base matrix ---
A = [ 0  1 -1;
     -1  0  1;
      1 -1  0 ]

function classify_case(u)
    J = Diagonal(u) * (A - I(3))
    λmax = maximum(real.(eigvals(J)))
    diff = λmax + minimum(u)  # negative → more stable
    if abs(diff) < 1e-3
        return (λmax, diff, "neutral")
    elseif diff < 0
        return (λmax, diff, "stabilizing")
    else
        return (λmax, diff, "destabilizing")
    end
end

# --- sweep over heterogeneity ---
rng = MersenneTwister(42)
samples = 2000
results = []

for cv in range(0.0, 2.0, length=50)
    for _ in 1:samples
        σ = sqrt(log(1 + cv^2))
        μ = -σ^2/2
        u = rand(rng, LogNormal(μ, σ), 3)
        λ, diff, cls = classify_case(u)
        push!(results, (u_cv = std(u)/mean(u), λmax = λ, diff_res = diff, regime = cls))
    end
end

df = DataFrame(results)

# --- summary counts ---
countmap = combine(groupby(df, :regime), nrow => :count)
println("Counts per regime:")
display(countmap)

# --- visualize ---
colors = Dict("neutral"=>:red, "stabilizing"=>:deepskyblue, "destabilizing"=>:orangered)

begin
    fig = Figure(size=(800,400))
    ax = Axis(fig[1,1],
        xlabel="u_cv (abundance heterogeneity)",
        ylabel="Re(λₘₐₓ)",
        title="Three regimes for antisymmetric 3-species system")

    for (cls, color) in colors
        subset = filter(r -> r.regime == cls, eachrow(df))
        isempty(subset) && continue
        scatter!(ax, [r.u_cv for r in subset], [r.λmax for r in subset];
                color=color, alpha=0.4, markersize=5, label=cls)
    end

    lines!(ax, [minimum(df.u_cv), maximum(df.u_cv)], [-1, -1];
        color=:black, linestyle=:dash, label="-min(u) bound")
    axislegend(ax; position=:rb)
    display(fig)
end