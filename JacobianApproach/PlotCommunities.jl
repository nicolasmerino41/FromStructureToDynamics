######################## Plot_Samples_FromDF.jl ########################
using CairoMakie, Random, Statistics, DataFrames

# --- small helpers ----------------------------------------------------
_cv(v) = (m = mean(v); m > 0 ? std(v) / m : NaN)

# compute degrees from a directed interaction matrix A (zero diagonal)
# outdeg[i] = count of nonzeros in row i (j ≠ i)
# indeg[i]  = count of nonzeros in column i (j ≠ i)
# undeg[i]  = count of neighbors j with A[i,j]≠0 OR A[j,i]≠0
function degree_vectors(A::AbstractMatrix{<:Real})
    S = size(A, 1)
    @assert size(A,2) == S
    outdeg = zeros(Int, S)
    indeg  = zeros(Int, S)
    undeg  = zeros(Int, S)

    @inbounds for i in 1:S
        oi = 0
        ii = 0
        ui = 0
        for j in 1:S
            i == j && continue
            if A[i,j] != 0.0; oi += 1; end
            if A[j,i] != 0.0; ii += 1; end
            if (A[i,j] != 0.0) || (A[j,i] != 0.0); ui += 1; end
        end
        outdeg[i] = oi
        indeg[i]  = ii
        undeg[i]  = ui
    end
    return (; outdeg, indeg, undeg)
end

# sample row indices safely
function _sample_rows(df::DataFrame, nsample::Int; seed::Union{Int,Nothing}=nothing)
    n = nrow(df)
    ns = min(nsample, n)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    return rand(rng, 1:n, ns)
end

# --- main entry: choose what to plot ---------------------------------
"""
plot_samples_from_df(df_main; which=:abundances, nsample=9, seed=nothing)

`which` ∈ [:abundances, :degrees_total, :degrees_split]

- :abundances     → 3×3 grid of abundance histograms (u), annotated with mean & CV.
- :degrees_total  → 3×3 grid of **undirected (total)** degrees per sampled community.
- :degrees_split  → nsample×2 grid of **out** vs **in** degree histograms (one row per sample).

Assumes `df_main` has columns: `:A::Matrix{Float64}`, `:u::Vector{Float64}`.
"""
function plot_samples_from_df(df::DataFrame; which::Symbol=:abundances, nsample::Int=9, seed=nothing)
    @assert "A" in names(df) && "u" in names(df) "df must contain columns :A and :u"
    idxs = _sample_rows(df, nsample; seed=seed)

    if which === :abundances
        ncols = 3
        nrows = ceil(Int, length(idxs) / ncols)
        fig = Figure(size=(1100, 320 * nrows))
        for (k, idx) in enumerate(idxs)
            row = div(k - 1, ncols) + 1
            col = ((k - 1) % ncols) + 1
            u = df[idx, :u]
            ax = Axis(fig[row, col];
                xlabel="u", ylabel="count",
                title = "sample $(idx) — mean=$(round(mean(u),digits=3)), cv=$(round(_cv(u),digits=3))"
            )
            hist!(ax, u; bins=30)
        end
        display(fig)

    elseif which === :degrees_total
        ncols = 3
        nrows = ceil(Int, length(idxs) / ncols)
        fig = Figure(size=(1100, 320 * nrows))
        for (k, idx) in enumerate(idxs)
            row = div(k - 1, ncols) + 1
            col = ((k - 1) % ncols) + 1
            A = df[idx, :A]
            deg = degree_vectors(A)
            ax = Axis(fig[row, col];
                xlabel="k_total (undirected neighbors)", ylabel="count",
                title = "sample $(idx) — cv=$(round(_cv(deg.undeg),digits=3))"
            )
            kmax = max(0, maximum(deg.undeg))
            bins = 0:(kmax + 1)
            hist!(ax, deg.undeg; bins=bins)
        end
        display(fig)
    else
        error("which must be one of :abundances, :degrees_total")
    end
end

# --- sugar wrappers if you prefer explicit calls ----------------------
plot_samples_from_df(df_main; which=:abundances, nsample=9)#, seed=seed)
plot_samples_from_df(df_main; which=:degrees_total, nsample=9)#, seed=seed)

# --- degrees from A by sign (no R needed) ---
function trophic_degrees_from_A(A::AbstractMatrix)
    @assert size(A,1) == size(A,2)
    S = size(A,1)

    # row-based counts (exclude diagonal just in case)
    prey   = [count(j -> j != i && A[i,j] < 0.0, 1:S) for i in 1:S]   # diet breadth
    preds  = [count(j -> j != i && A[i,j] > 0.0, 1:S) for i in 1:S]   # vulnerability

    # sanity checks: reciprocity under your mapping
    @assert preds == vec(sum(A .< 0.0, dims=1))[:]
    @assert prey  == vec(sum(A .> 0.0, dims=1))[:]

    return prey, preds
end

# --- plot: overlaid in/out (prey vs predators) for a sample of rows in df_main ---
function plot_overlaid_trophic_degrees(df_main; nsample::Int=6, seed::Int=42, title::String="In/Out Degree Distributions (overlaid)")
    @assert "A" in names(df_main) "df_main must contain column A with the interaction matrix."
    rng = MersenneTwister(seed)
    idxs = rand(rng, 1:nrow(df_main), min(nsample, nrow(df_main)))

    ncols = 3
    nrows = ceil(Int, length(idxs) / ncols)
    fig = Figure(size=(1100, 320*nrows))
    Label(fig[0,1:ncols], title; fontsize=22, font=:bold, halign=:left)

    for (k, ri) in enumerate(idxs)
        A = df_main[ri, :A]
        prey, preds = trophic_degrees_from_A(A)

        r = div(k-1, ncols) + 1
        c = mod(k-1, ncols) + 1
        ax = Axis(fig[r,c]; xlabel="Degree k", ylabel="count",
                  title=@sprintf("sample %d: cv_out=%.2f, cv_in=%.2f",
                                 ri,
                                 (m=mean(preds); m>0 ? std(preds)/m : 0.0),
                                 (m=mean(prey);  m>0 ? std(prey)/m  : 0.0)))

        maxk = max(maximum(prey; init=0), maximum(preds; init=0))
        bins = 0:(maxk+1)
        # predators (out-degree): blue; prey (in-degree): orange
        hist!(ax, preds; bins=bins, normalization=:none, color=(RGBAf(70/255,130/255,180/255,0.6)), label="Out-degree (predators)")
        hist!(ax, prey;  bins=bins, normalization=:none, color=(RGBAf(1.0,0.55,0.0,0.6)),        label="In-degree (prey)")
        axislegend(ax; position=:rt, framevisible=false)
    end

    display(fig)
end

plot_overlaid_trophic_degrees(df_main; nsample=9)