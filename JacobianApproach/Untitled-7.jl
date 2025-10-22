
using LinearAlgebra, Random, Statistics, CairoMakie

# --- fixed antisymmetric 3x3 base matrix ---
A = [ 0  1 -1;
     -1  0  1;
      1 -1  0 ]

function classify_case(u)
    J = Diagonal(u) * (A - I)
    λmax = maximum(real.(eigvals(J)))
    diff = λmax + minimum(u)  # if negative, more stable
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
samples = 2_000
results = []

for cv in range(0.0, 2.0, length=50)
    for _ in 1:samples
        u = rand(LogNormal(-log(1+cv^2)/2, sqrt(log(1+cv^2))), 3)
        λ, diff, cls = classify_case(u)
        push!(results, (u_cv = std(u)/mean(u), λmax = λ, diff_res = diff, regime = cls))
    end
end

# --- visualize ---
df = DataFrame(results)
df = filter(row -> row.regime == "neutral", df)
colors = Dict("neutral"=>:red, "stabilizing"=>:deepskyblue, "destabilizing"=>:orangered)

begin
    fig = Figure(size=(800,400))
    ax = Axis(fig[1,1],
        xlabel="u_cv (abundance heterogeneity)",
        ylabel="Re(λ_max)",
        title="Three regimes of antisymmetric 3-species systems")

    for (cls, color) in colors
        subset = filter(r -> r.regime == cls, eachrow(df))
        if isempty(subset)
            @warn "No data for regime: $cls"
            continue
        end
        scatter!(ax,
            [r.u_cv for r in subset],
            [r.λmax for r in subset];
            color=color, alpha=0.5, markersize=5, label=cls)
    end


    lines!(ax, [minimum(df.u_cv), maximum(df.u_cv)], [-1, -1]; color=:black, linestyle=:dash, label="-min(u) bound")
    axislegend(ax; position=:rb)
    display(fig)
end
