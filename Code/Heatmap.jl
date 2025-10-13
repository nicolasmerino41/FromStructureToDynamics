using CairoMakie, Statistics, LinearAlgebra, Graphs, StatsBase, Printf

# --- helpers ---------------------------------------------------------------
function _ensure_structure!(df)
    need = any(!(s in names(df)) for s in (:connectance, :modularity, :degree_cv))
    need || return
    N = nrow(df)
    conn = Vector{Float64}(undef, N)
    modu = Vector{Float64}(undef, N)
    degcv = Vector{Float64}(undef, N)
    for (i,row) in enumerate(eachrow(df))
        A   = row.p_final[2]
        Adj = A .!= 0.0
        S   = size(A,1)
        conn[i] = sum(Adj) / (S*(S-1))
        g = SimpleGraph(Adj); degs = degree(g); μd = mean(degs)
        degcv[i] = μd == 0 ? 0.0 : std(degs)/μd
        k = sum(Adj, dims=2)[:]; m = sum(k)/2
        B = zeros(Float64,S,S)
        @inbounds for u in 1:S, v in 1:S
            B[u,v] = (Adj[u,v] ? 1.0 : 0.0) - (k[u]*k[v])/(2*m)
        end
        vals, vecs = eigen(Symmetric(B))
        v1 = vecs[:, argmax(vals)]
        svec = map(x -> x >= 0 ? 1.0 : -1.0, v1)
        modu[i] = (svec' * (B * svec)) / (4*m)
    end
    df.connectance = conn
    df.modularity  = modu
    df.degree_cv   = degcv
end

_errvec(df, metric::Symbol, step::Int; smape_spectral::Bool=true) = begin
    fullco = Symbol("$(metric)_full")
    stepco = Symbol("$(metric)_S$(step)")
    x = df[!, fullco]; y = df[!, stepco]
    if smape_spectral && (metric === :resilience || metric === :reactivity)
        2 .* abs.(y .- x) ./ (abs.(x) .+ abs.(y) .+ 1e-8)
    else
        abs.(y .- x) ./ (abs.(x) .+ 1e-6)
    end
end

# --- main ------------------------------------------------------------------
function plot_struct_heatmap_step(
    df::DataFrame; step::Int=1,
    structures = [:connectance, :modularity, :degree_cv],
    structure_labels = ["Connectance","Modularity","Degree CV"],
    metrics = [:resilience, :reactivity, :rt_pulse, :after_press],
    metric_labels = ["Resilience","Reactivity","Return time","Persistence"],
    smape_spectral::Bool = true,
    remove_unstable::Bool = false,
    resolution=(900, 700), px_per_unit=6.0,
    save_plot::Bool=false, filename::String="Figures/struct_heatmap_step$(step).png"
)
    _ensure_structure!(df)

    if remove_unstable
        rcol = Symbol("resilience_S$step")
        df = filter(row -> isfinite(row[rcol]) && row[rcol] < 0, df)
    end
    nrow(df) == 0 && error("No rows to plot after filtering.")

    nrows = length(metrics)
    ncols = length(structures)
    M = Matrix{Float64}(undef, nrows, ncols)
    for (ri,m) in enumerate(metrics), (ci,p) in enumerate(structures)
        x = df[!, p]
        y = _errvec(df, m, step; smape_spectral=smape_spectral)
        mask = isfinite.(x) .& isfinite.(y)
        if count(mask) < 5
            M[ri,ci] = NaN
        else
            xr = StatsBase.tiedrank(x[mask]); yr = StatsBase.tiedrank(y[mask])
            M[ri,ci] = cor(xr, yr)
        end
    end

    vals = filter(isfinite, vec(M))
    cr = isempty(vals) ? 1.0 : maximum(abs, vals)
    cr = cr == 0 ? 1e-3 : cr

    # ---- PLOT ----
    fig = Figure(size=resolution, fontsize=13)
    ax = Axis(fig[1,1];
        title = "Step $step",
        xticks = (1:ncols, structure_labels),
        yticks = (1:nrows, metric_labels),
        xticklabelrotation = π/6,
        xgridvisible=false, ygridvisible=false,
        aspect = AxisAspect(1)
    )

    # note: rows = metrics (y-axis), columns = structures (x-axis)
    heatmap!(ax, 1:ncols, 1:nrows, M;
             colormap=cgrad(:RdBu, rev=true), colorrange=(-cr, cr), interpolate=false)

    xlims!(ax, 0.5, ncols + 0.5)
    ylims!(ax, 0.5, nrows + 0.5)

    # centered text annotations
    for r in 1:nrows, c in 1:ncols
        ρ = M[r,c]
        if isfinite(ρ)
            tcolor = abs(ρ) > 0.45 ? :white : :black
            text!(ax, @sprintf("%.2f", ρ),
                  position=(c, r), align=(:center,:center),
                  color=tcolor, fontsize=14)
        end
    end

    # horizontal colorbar under the heatmap
    Colorbar(fig[2,1], colormap=cgrad(:RdBu, rev=true),
             colorrange=(-cr, cr), label="Spearman ρ",
             vertical=false, width=Relative(0.8), height=15)

    rowgap!(fig.layout, 10)

    display(fig)
    save_plot && save(filename, fig; px_per_unit=px_per_unit)
    return fig
end

# Example call:
plot_struct_heatmap_step(sim_results; step=1, save_plot=true)

plot_struct_heatmap_step(sim_results; step=2)
plot_struct_heatmap_step(sim_results; step=3)
plot_struct_heatmap_step(sim_results; step=5)