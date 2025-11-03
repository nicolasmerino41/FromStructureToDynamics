# ============================================
# MakeFigures_Main.jl  (A, B, C, E, F, G)
# Requires: your MainCode.jl + included helpers
# ============================================

using Random, Statistics, LinearAlgebra, DataFrames, Distributions, Printf
using CairoMakie
using StatsBase
# -----------------------------------------------------------
# Load your environment (builders, metrics, sweeps, etc.)
# -----------------------------------------------------------
include("MainCode.jl")                    # your full pipeline & sweeps
include("niche_model_builder.jl")         # your niche builder (used by MainCode)
# Optional (if you keep the corr(u,|α|) sweep in a separate file)
include("CorrUAlpha_Predictability.jl")   # provides run_corr_timescale_interaction_sweep and plotting
# (If the above include already happened in your session, no worries.)

# -----------------------------------------------------------
# Output dir
# -----------------------------------------------------------
const FIGDIR = "figs"
isdir(FIGDIR) || mkpath(FIGDIR)

# -----------------------------------------------------------
# Small helpers (NEW; do not clash with yours)
# -----------------------------------------------------------

# 1) Row-spread index S_row: mean row CV of |α| over existing links
function _row_spread_index(α::AbstractMatrix)
    S = size(α,1)
    cvs = Float64[]
    @inbounds for i in 1:S
        v = [abs(α[i,j]) for j in 1:S if i!=j && α[i,j] != 0.0]
        if !isempty(v)
            m = mean(v)
            push!(cvs, m>0 ? std(v)/m : 0.0)
        end
    end
    isempty(cvs) ? 0.0 : mean(cvs)
end

# 2) Build α and S_row from stored (J_full, u) in df_main
function _add_branch_summaries!(df_main::DataFrame)
    Srow = Vector{Float64}(undef, nrow(df_main))
    for (k,row) in enumerate(eachrow(df_main))
        J = row.J_full
        u = row.u
        α = alpha_off_from(J,u)
        Srow[k] = _row_spread_index(α)
    end
    df_main[!, :S_row] = Srow
    rename!(df_main, Dict(:rho_sym=>:rho))  # shorthand
    return df_main
end

# 3) Join df_t (per time point) with df_main (per community) to carry summaries
#    NOTE: keys match your run_sweep_stable fields
function _join_time_with_main(df_t::DataFrame, df_main::DataFrame)
    dfm = select(df_main, :mode, :S, :conn_target, :mean_abs, :mag_cv,
                          :u_mean_target, :u_cv_target, :degree_family, :degree_param,
                          :rho => :rho, :IS_real, :u_cv, :deg_cv_all, :S_row)
    dft = deepcopy(df_t)
    rename!(dft, Dict(:conn=>:conn_target, :u_mean=>:u_mean_target, :u_cv=>:u_cv_target))
    # join on all these columns + IS_real
    joincols = [:mode, :S, :conn_target, :mean_abs, :mag_cv, :u_mean_target, :u_cv_target,
                :degree_family, :degree_param, :rho, :IS_real]
    dt = innerjoin(dft, dfm; on=joincols, makeunique=true)
    return dt
end

# 4) Drop-one partial R² for a single response y and predictors in table X (with intercept)
#    Returns Dict(:name => ΔR2_normalized) + full R²
function _dropone_importance(y::Vector{<:Real}, X::DataFrame; cols::Vector{Symbol})
    # standardize predictors to balance scales
    Z = hcat(fill(1.0, length(y)), [zscore(collect(skipmissing(X[!, c]))) for c in cols]...)
    # replace missing with zeros after zscore (rare)
    for j in 2:size(Z,2)
        for i in 1:size(Z,1)
            if !isfinite(Z[i,j]); Z[i,j] = 0.0; end
        end
    end
    # OLS
    β = Z \ y
    ŷ = Z * β
    μy = mean(y); sst = sum((y .- μy).^2)
    sse_full = sum((y .- ŷ).^2)
    r2_full = sst == 0 ? 0.0 : 1 - sse_full/sst

    # drop-one deltas
    deltas = Float64[]
    for j in 2:size(Z,2)
        Zred = Z[:, setdiff(1:size(Z,2), j)]
        βr = Zred \ y
        ŷr = Zred * βr
        sse_red = sum((y .- ŷr).^2)
        Δ = (sse_red - sse_full) / sst  # partial R² of predictor j
        push!(deltas, max(Δ, 0.0))
    end

    s = sum(deltas)
    imp = Dict{Symbol,Float64}()
    if s > 0
        for (k,c) in enumerate(cols)
            imp[c] = deltas[k] / s
        end
    else
        for c in cols
            imp[c] = 0.0
        end
    end
    return imp, max(r2_full, 0.0)
end

# 5) Compute importances across t for a joined time table (dt)
#    predictors = [:u_cv, :S_row, :rho]
function _importance_curves(dt::DataFrame; tvals=sort(unique(dt.t)))
    out = NamedTuple[]
    for tt in tvals
        sub = dt[dt.t .== tt, :]
        y   = collect(sub.r_full)
        X   = select(sub, :u_cv, :S_row, :rho)
        imp, r2full = _dropone_importance(y, X; cols=[:u_cv, :S_row, :rho])
        push!(out, (; t=tt, r2_full=r2full,
                     imp_u=imp[:u_cv], imp_Srow=imp[:S_row], imp_rho=imp[:rho],
                     n=nrow(sub)))
    end
    return DataFrame(out)
end

# 6) Find t* threshold where a branch’s normalized importance crosses θ
function _critical_t(impcurves::DataFrame; branch::Symbol, θ::Float64=0.30)
    y = impcurves[!, branch]
    t = impcurves.t
    for i in eachindex(t)
        if isfinite(y[i]) && y[i] >= θ
            return t[i]
        end
    end
    return NaN
end

# 7) 2D binning helper: compute partial R² of `target_var` within each (x,y) bin,
#    controlling for the other two branches (drop-one within-cell).
function _phase_grid(dt::DataFrame; t_fixed::Float64, target_var::Symbol,
                     xvar::Symbol, xbins::AbstractVector,
                     yvar::Symbol, ybins::AbstractVector)
    cols = [:u_cv, :S_row, :rho]
    others = filter(c -> c!=target_var, cols)
    rows = NamedTuple[]
    df = dt[dt.t .≈ t_fixed, :]
    for i in 1:length(xbins)-1, j in 1:length(ybins)-1
        sub = df[(df[!,xvar].>=xbins[i]) .& (df[!,xvar].<xbins[i+1]) .&
                 (df[!,yvar].>=ybins[j]) .& (df[!,yvar].<ybins[j+1]), :]
        n = nrow(sub)
        if n >= 25
            yresp = collect(sub.r_full)
            X = select(sub, cols)
            imp, _ = _dropone_importance(yresp, X; cols=cols)
            push!(rows, (; x=0.5*(xbins[i]+xbins[i+1]), y=0.5*(ybins[j]+ybins[j+1]),
                           r2=imp[target_var], n))
        end
    end
    return DataFrame(rows)
end

# 8) Simple schematic (A1): three colored bands + arrows
function plot_regime_map_schematic(fname::String)
    fig = Figure(size=(1100, 320))
    ax = Axis(
        fig[1,1];
        xlabel = "time t (log)",
        ylabel = "",
        yticklabelsvisible = false,
        ygridvisible = false,
        yminorgridvisible = false,
        yticksmirrored = false,
        yticksize = 0,
        xscale = log10,
        xminorticksvisible = false,
        xlabelsize = 14,
    )
    xs = 10 .^ range(-2, 2; length=200)
    y0, h = 0.0, 1.0
    band!(ax, xs, fill(0.0, length(xs)), fill(1/3, length(xs)); color=(:skyblue, 0.5))
    band!(ax, xs, fill(1/3, length(xs)), fill(2/3, length(xs)); color=(:gold, 0.5))
    band!(ax, xs, fill(2/3, length(xs)), fill(1.0, length(xs)); color=(:plum, 0.5))
    text!(ax, "time-scales u", position=(0.015, 0.17), align=(:left,:center), space=:relative, fontsize=16)
    text!(ax, "IS distribution (within-row spread)", position=(0.38, 0.50), space=:relative, fontsize=16)
    text!(ax, "topology via antisymmetry ρ", position=(0.78, 0.83), space=:relative, fontsize=16)

    # qualitative arrows (how knobs shift boundaries)
    arrows2d!(ax, [0.15], [0.95], [0.06], [0.0]; color=:black)
    text!(ax, "↑ u_cv", position=(0.21,0.95), space=:relative, fontsize=12)

    arrows2d!(ax, [0.52], [0.05], [0.10], [0.0]; color=:black)
    text!(ax, "↑ mean IS", position=(0.63,0.06), space=:relative, fontsize=12)

    arrows2d!(ax, [0.87], [0.10], [-0.06], [0.0]; color=:black)
    text!(ax, "↑ ρ", position=(0.81,0.12), space=:relative, fontsize=12)


    hidespines!(ax, :l, :t, :r)
    display(fig)
    save(joinpath(FIGDIR, fname), fig, px_per_unit=2)
end

# -----------------------------------------------------------
# ENSEMBLES (biomass and uniform) — reuse your sweep function
# -----------------------------------------------------------
println("Running ensembles for biomass-weighted and uniform pulses...")

# Settings you used in your example (t grid length=10 inside run_sweep_stable)
COMMON = (; modes=[:TR], S_vals=[120],
           conn_vals=0.05:0.05:0.30,
           mean_abs_vals=[0.05, 0.10, 0.40, 0.80, 1.20],
           mag_cv_vals=[0.01, 0.1, 0.5, 1.0, 2.0],
           u_mean_vals=[1.0],
           u_cv_vals=[0.3,0.5,0.8,1.0,2.0,3.0],
           degree_families=[:uniform, :lognormal, :pareto],
           deg_cv_vals=[0.0, 0.5, 1.0, 2.0],
           deg_pl_alphas=[1.2, 1.5, 2.0, 3.0],
           rho_sym_vals=[0.0, 0.5, 1.0],
           reps_per_combo=4, seed=42, number_of_combinations=1000,
           margin=0.05, shrink_factor=0.9, max_shrink_iter=200, q_thresh=0.20,
           long_time_value=5.0)

dfm_bio, dft_bio = run_sweep_stable(; COMMON..., u_weighted_biomass=:biomass)
dfm_uni, dft_uni = run_sweep_stable(; COMMON..., u_weighted_biomass=:uniform)

_add_branch_summaries!(dfm_bio); _add_branch_summaries!(dfm_uni)
rename!(dft_bio, :rho_sym=>:rho);  rename!(dft_uni, :rho_sym=>:rho)
dt_bio = _join_time_with_main(dft_bio, dfm_bio)
dt_uni = _join_time_with_main(dft_uni, dfm_uni)

# -----------------------------------------
# A) Orienting schematic
# -----------------------------------------
plot_regime_map_schematic("A1_regime_map_schematic.png")

# -----------------------------------------
# B1) Importance vs t (biomass & uniform)
# -----------------------------------------
imp_bio = _importance_curves(dt_bio)
imp_uni = _importance_curves(dt_uni)

function plot_importance_curves(imp::DataFrame, title::String, fname::String)
    fig = Figure(size=(980, 360))
    ax = Axis(fig[1,1]; xscale=log10, xlabel="t (log)", ylabel="normalized importance",
              title=title, limits=((minimum(imp.t), maximum(imp.t)), (-0.02, 1.02)))
    lines!(ax, imp.t, imp.imp_u;    label="time-scales (u)", linewidth=3)
    lines!(ax, imp.t, imp.imp_Srow; label="IS distribution (row spread)", linewidth=3)
    lines!(ax, imp.t, imp.imp_rho;  label="antisymmetry (ρ)", linewidth=3)
    axislegend(ax; position=:rb, framevisible=false)
    display(fig)
    save(joinpath(FIGDIR, fname), fig, px_per_unit=2)
end

plot_importance_curves(imp_bio, "Branch importance vs t — biomass-weighted", "B1_importance_vs_t_biomass.png")
plot_importance_curves(imp_uni, "Branch importance vs t — uniform",          "B1_importance_vs_t_uniform.png")

# -----------------------------------------
# B2) Critical-t curves vs knobs (u_cv, IS; separate for each branch)
# -----------------------------------------
function _critical_t_vs(dt::DataFrame, which::Symbol; θ=0.30, nbins=8)
    # which ∈ (:u_cv, :IS_real, :rho) — knob for binning
    bins = range(minimum(dt[!,which]), maximum(dt[!,which]); length=nbins+1) |> collect
    rows = NamedTuple[]
    for b in 1:nbins
        sub = dt[(dt[!,which].>=bins[b]) .& (dt[!,which].<bins[b+1]), :]
        nsub = nrow(sub); nsub < 80 && continue
        imp = _importance_curves(sub)
        push!(rows, (; knob=0.5*(bins[b]+bins[b+1]),
                      tstar_u   = _critical_t(imp; branch=:imp_u,    θ=θ),
                      tstar_S   = _critical_t(imp; branch=:imp_Srow, θ=θ),
                      tstar_rho = _critical_t(imp; branch=:imp_rho,  θ=θ),
                      n=nsub))
    end
    return DataFrame(rows)
end

cts_bio_u  = _critical_t_vs(dt_bio, :u_cv; θ=0.30)
cts_bio_IS = _critical_t_vs(dt_bio, :IS_real; θ=0.30)
cts_uni_u  = _critical_t_vs(dt_uni, :u_cv; θ=0.30)
cts_uni_IS = _critical_t_vs(dt_uni, :IS_real; θ=0.30)

function plot_tstar(df::DataFrame, knoblabel::String, title::String, fname::String)
    fig = Figure(size=(980, 420))
    ax = Axis(fig[1,1]; xlabel=knoblabel, ylabel="t* (first time importance ≥ θ)",
              title=title, yscale=log10)
    scatterlines!(ax, df.knob, df.tstar_u;    label="u branch",    markersize=6, linewidth=2)
    scatterlines!(ax, df.knob, df.tstar_S;    label="IS-distribution", markersize=6, linewidth=2)
    scatterlines!(ax, df.knob, df.tstar_rho;  label="ρ branch",    markersize=6, linewidth=2)
    axislegend(ax; position=:rt, framevisible=false)
    display(fig)
    save(joinpath(FIGDIR, fname), fig, px_per_unit=2)
end

plot_tstar(cts_bio_u,  "abundance CV (u_cv)", "Critical t* vs u_cv — biomass",   "B2_tstar_vs_u_biomass.png")
plot_tstar(cts_bio_IS, "mean |A| (IS)",       "Critical t* vs mean IS — biomass","B2_tstar_vs_IS_biomass.png")
plot_tstar(cts_uni_u,  "abundance CV (u_cv)", "Critical t* vs u_cv — uniform",   "B2_tstar_vs_u_uniform.png")
plot_tstar(cts_uni_IS, "mean |A| (IS)",       "Critical t* vs mean IS — uniform","B2_tstar_vs_IS_uniform.png")

# -----------------------------------------
# C) Phase diagrams (C1, C2, C3)
# -----------------------------------------
short_t = minimum(unique(dt_bio.t))
mid_t   = 0.5
long_t  = 5.0

xb_u  = range(minimum(dt_bio.u_cv),  maximum(dt_bio.u_cv),  length=10) |> collect
yb_IS = range(minimum(dt_bio.IS_real), maximum(dt_bio.IS_real), length=10) |> collect

xb_S  = range(minimum(filter(!isnan, dt_bio.S_row)), maximum(filter(!isnan, dt_bio.S_row)), length=10) |> collect
yb_r  = range(minimum(dt_bio.rho),   maximum(dt_bio.rho),   length=10) |> collect

# C1: short-time, importance of u_cv over (u_cv, IS)
grid_C1 = _phase_grid(dt_bio; t_fixed=short_t, target_var=:u_cv, xvar=:u_cv, xbins=xb_u, yvar=:IS_real, ybins=yb_IS)
# C2: mid-time, importance of S_row over (S_row, IS)
grid_C2 = _phase_grid(dt_bio; t_fixed=mid_t,   target_var=:S_row, xvar=:S_row, xbins=xb_S, yvar=:IS_real, ybins=yb_IS)
# C3: long-time, importance of ρ over (ρ, u_cv)
grid_C3 = _phase_grid(dt_bio; t_fixed=long_t,  target_var=:rho,   xvar=:rho,   xbins=yb_r, yvar=:u_cv,    ybins=xb_u)

function plot_phase(grid::DataFrame; xlab::String, ylab::String, title::String, fname::String)
    fig = Figure(size=(700, 560))
    ax = Axis(fig[1,1]; xlabel=xlab, ylabel=ylab, title=title)
    if nrow(grid) > 0
        heatmap!(ax, grid.x, grid.y, grid.r2; colormap=:magma, colorrange=(0,1))
        Colorbar(fig[1,2], limits=(0,1), label="partial R² (normalized)")
    end
    display(fig)
    save(joinpath(FIGDIR, fname), fig, px_per_unit=2)
end

plot_phase(grid_C1; xlab="u_cv", ylab="mean |A| (IS)", title="C1: short-time — u branch dominance",
           fname="C1_phase_short_u_vs_IS.png")
plot_phase(grid_C2; xlab="row spread S_row", ylab="mean |A| (IS)", title="C2: mid-time — IS-distribution dominance",
           fname="C2_phase_mid_Srow_vs_IS.png")
plot_phase(grid_C3; xlab="antisymmetry ρ", ylab="u_cv", title="C3: long-time — topology (ρ) dominance",
           fname="C3_phase_long_rho_vs_u.png")

# -----------------------------------------
# E1) Transition mechanics: mid-time failure window vs IS (Δ = full - row-collapsed)
# -----------------------------------------
function plot_midtime_failure(dt::DataFrame, title::String, fname::String; nbins=5)
    bins = range(minimum(dt.IS_real), maximum(dt.IS_real); length=nbins+1) |> collect
    fig = Figure(size=(980, 420))
    ax = Axis(fig[1,1]; xscale=log10, xlabel="t (log)", ylabel="Δ(t) = r_full - r_row",
              title=title, limits=(nothing, (0, nothing)))
    for b in 1:nbins
        sub = dt[(dt.IS_real .>= bins[b]) .& (dt.IS_real .< bins[b+1]), :]
        isempty(sub) && continue
        g = combine(groupby(sub, :t), :r_full=>mean=>:mfull, :r_row=>mean=>:mrow)
        Δ = g.mfull .- g.mrow
        lines!(ax, g.t, Δ; label=@sprintf("IS≈[%.2f–%.2f)", bins[b], bins[b+1]), linewidth=3)
    end
    axislegend(ax; position=:rt, framevisible=false)
    display(fig)
    save(joinpath(FIGDIR, fname), fig, px_per_unit=2)
end

plot_midtime_failure(dt_bio, "Mid-time loss from collapsing row spread — biomass", "E1_midtime_failure_biomass.png")
plot_midtime_failure(dt_uni, "Mid-time loss from collapsing row spread — uniform", "E1_midtime_failure_uniform.png")

# -----------------------------------------
# E2) Coupling ρ_c warps boundaries (use your sweep & plotting)
# -----------------------------------------
optsCorr = CorrTimescaleSweepOptions(;
    modes = [:TR], S_vals = [120],
    conn_vals = 0.10:0.10:0.30,
    mean_abs_vals = [1.0],
    mag_cv_vals = [0.1, 0.5, 1.0],
    u_mean_vals = [1.0],
    u_cv_vals   = [0.3, 0.5, 0.8, 1.0, 2.0],
    degree_families = [:uniform, :lognormal, :pareto],
    deg_cv_vals = [0.0, 0.5, 1.0, 2.0],
    deg_pl_alphas = [1.2, 1.5, 2.0, 3.0],
    rho_sym_vals = [0.0, 0.5, 1.0],
    IS_lines = [0.05, 0.10, 0.40, 0.80, 1.20],
    reps_per_combo = 2,
    number_of_combinations = 300,
    rho_c_vals = 0.0:0.1:1.0,
    t_eval = 5.0,
    q_thresh = 0.20,
    seed = 20251030
)
df_raw_corr, df_sum_corr = run_corr_timescale_interaction_sweep(optsCorr)

# baselines from df_main (long_rmed at t=5.0) for the same steps
bas = r2_baselines_long_from_df_main(dfm_bio;
        steps = ["row","thr","reshuf","rew","uni","rarer"],
        filter_fun = d -> filter(:mode => ==(:TR), d))

# Use the provided plotting helper (draws dashed baselines)
plot_corr_timescale_grid_by_IS_with_baseline(
    df_sum_corr; baselines=bas,
    title="E2: R² vs corr(u,|α|) — baselines at r̃med(t=5)",
    steps=["row","thr","reshuf","rew","uni","rarer"]
)
save(joinpath(FIGDIR, "E2_R2_vs_corr_u_alpha_with_baselines.png"), current_figure(), px_per_unit=2)

# -----------------------------------------
# F1) Antisymmetry threshold curve (long-time partial R² of ρ vs ρ)
# -----------------------------------------
function plot_antisym_threshold(dt::DataFrame, title::String, fname::String; nbins=10)
    dfL = dt[dt.t .== 14.38449888287663, :]
    bins = range(minimum(dfL.rho), maximum(dfL.rho); length=nbins+1) |> collect
    xs = Float64[]; ys = Float64[]; ns = Int[]
    for b in 1:nbins
        sub = dfL[(dfL.rho .>= bins[b]) .& (dfL.rho .< bins[b+1]), :]
        n = nrow(sub); n < 60 && continue
        imp, _ = _dropone_importance(collect(sub.r_full), select(sub, :u_cv, :S_row, :rho); cols=[:u_cv,:S_row,:rho])
        push!(xs, 0.5*(bins[b]+bins[b+1])); push!(ys, imp[:rho]); push!(ns, n)
    end
    fig = Figure(size=(820, 420))
    ax = Axis(fig[1,1]; xlabel="ρ", ylabel="partial R² (normalized)", title=title, limits=((0,1), (0,1)))
    scatterlines!(ax, xs, ys; markersize=8, linewidth=2)
    display(fig)
    save(joinpath(FIGDIR, fname), fig, px_per_unit=2)
end

plot_antisym_threshold(dt_bio, "F1: Long-time importance of ρ vs ρ — biomass", "F1_rho_importance_vs_rho_biomass.png")
plot_antisym_threshold(dt_uni, "F1: Long-time importance of ρ vs ρ — uniform", "F1_rho_importance_vs_rho_uniform.png")

# -----------------------------------------
# F2) Degree heterogeneity is second-order
#     (partial R² of deg_cv_all vs t; show it's small vs main branches)
# -----------------------------------------
function plot_degree_importance(dt::DataFrame, title::String, fname::String)
    tvals = sort(unique(dt.t))
    vals = Float64[]
    for tt in tvals
        sub = dt[dt.t .== tt, :]
        y = collect(sub.r_full)
        # include deg_cv_all alongside the three branches
        X = select(sub, :u_cv, :S_row, :rho, :deg_cv_all)
        imp, _ = _dropone_importance(y, X; cols=[:u_cv, :S_row, :rho, :deg_cv_all])
        push!(vals, imp[:deg_cv_all])
    end
    fig = Figure(size=(980, 360))
    ax = Axis(fig[1,1]; xscale=log10, xlabel="t (log)", ylabel="normalized importance",
              title=title, limits=(nothing,(0,1)))
    lines!(ax, tvals, vals; linewidth=3, label="degree CV (all)")
    hlines!(ax, 0.2; color=:gray50, linestyle=:dash) # visual threshold guide
    axislegend(ax; position=:rt, framevisible=false)
    display(fig)
    save(joinpath(FIGDIR, fname), fig, px_per_unit=2)
end

plot_degree_importance(dt_bio, "F2: DegCV importance vs t — biomass", "F2_degree_importance_biomass.png")
plot_degree_importance(dt_uni, "F2: DegCV importance vs t — uniform", "F2_degree_importance_uniform.png")

# -----------------------------------------
# G1) Bridge: r̃_med(t) + stacked branch importances (biomass & uniform)
# -----------------------------------------
function plot_bridge(dt::DataFrame, imp::DataFrame, title::String, fname::String)
    # median r_full vs t
    g = combine(groupby(dt, :t), :r_full=>median=>:rmed)
    # stacked area from importances
    fig = Figure(size=(980, 520))
    ax1 = Axis(fig[1,1]; xscale=log10, xlabel="t (log)", ylabel="r̃_med(t)", title=title)
    lines!(ax1, g.t, g.rmed; linewidth=3)
    # second axis (stacked area)
    ax2 = Axis(fig[2,1]; xscale=log10, xlabel="t (log)", ylabel="normalized importance",
               xticklabelsvisible=true, xticksvisible=true, yminorticksvisible=false)
    # cumulative stack
    a = imp.imp_u
    b = imp.imp_u .+ imp.imp_Srow
    c = b .+ imp.imp_rho
    band!(ax2, imp.t, zeros(length(imp.t)), a; color=(:skyblue,0.6))
    band!(ax2, imp.t, a, b; color=(:gold,0.6))
    band!(ax2, imp.t, b, c; color=(:plum,0.6))
    text!(ax2, "u",    position=(0.03, 0.18), space=:relative, fontsize=12)
    text!(ax2, "IS",   position=(0.38, 0.50), space=:relative, fontsize=12)
    text!(ax2, "ρ",    position=(0.78, 0.82), space=:relative, fontsize=12)
    linkxaxes!(ax1, ax2)
    display(fig)
    save(joinpath(FIGDIR, fname), fig, px_per_unit=2)
end

plot_bridge(dt_bio, imp_bio, "G1: From pulse to resilience — biomass", "G1_bridge_biomass.png")
plot_bridge(dt_uni, imp_uni, "G1: From pulse to resilience — uniform", "G1_bridge_uniform.png")

println("\nAll figures saved under $(FIGDIR)/")
