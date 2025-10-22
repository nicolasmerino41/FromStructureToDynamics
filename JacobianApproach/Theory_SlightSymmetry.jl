using LinearAlgebra, Random, Statistics, DataFrames, CairoMakie

# --- base antisymmetric 4×4 matrix ---
A = [ 0  1  0  0;
     -1  0  0  0;
      0  0  0  1;
      0  0 -1  0 ]

rng = MersenneTwister(42)
ε = 0.1
Aε = A + ε * randn(rng, size(A))  # break antisymmetry slightly
Aε .-= Aε' ./ 2                   # keep mostly antisymmetric

function classify_case(A, u)
    J = Diagonal(u) * (A - I(4))
    λmax = maximum(real.(eigvals(J)))
    diff = λmax + minimum(u)
    if abs(diff) < 1e-3
        return (λmax, diff, "neutral")
    elseif diff < 0
        return (λmax, diff, "stabilizing")
    else
        return (λmax, diff, "destabilizing")
    end
end

# --- sweep ---
samples = 20_000
results = []

for _ in 1:samples
    u = rand(LogNormal(0.0, 1.0), 4)
    λ, diff, cls = classify_case(Aε, u)
    push!(results, (u_cv = std(u)/mean(u), λmax = λ, diff_res = diff, regime = cls))
end

df = DataFrame(results)

# --- counts ---
countmap = combine(groupby(df, :regime), nrow => :count)
println("Counts per regime:")
display(countmap)

# --- plot regimes ---
colors = Dict("neutral"=>:red, "stabilizing"=>:blue, "destabilizing"=>:green)

begin
    fig = Figure(size=(800,400))
    ax = Axis(fig[1,1],
        xlabel="u_cv (abundance heterogeneity)",
        ylabel="Re(λₘₐₓ)",
        title="Slightly broken antisymmetry (ε = 0.1)")

    for (cls, color) in colors
        subset = filter(r -> r.regime == cls, eachrow(df))
        isempty(subset) && continue
        scatter!(ax, [r.u_cv for r in subset], [r.λmax for r in subset];
                color=color, alpha=0.4, markersize=5, label=cls)
    end

    axislegend(ax; position=:rb)
    display(fig)
end
