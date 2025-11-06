############## CorrUAlpha_Predictability.jl ##############
using Random, Statistics, LinearAlgebra, DataFrames, CairoMakie, Distributions, Colors, ColorSchemes
using Base.Threads

# --- couple u to interaction distribution via Gaussian copula ---------
"""
correlate_u_with_alpha(u, α; rho_c, rng)

Targets Spearman-like correlation ρ_c between u and node interaction
intensity sᵢ = ∑ⱼ (|αᵢⱼ| + |αⱼᵢ|). Preserves the marginal distribution of u.
"""
function correlate_u_with_alpha(u::AbstractVector, α::AbstractMatrix; rho_c::Float64, rng=Random.default_rng())
    S = length(u)
    s = [sum(abs, α[i, :]) + sum(abs, α[:, i]) - 2*abs(α[i,i]) for i in 1:S]  # total (in+out) |α|
    # ranks -> standard normal scores
    ord_s = sortperm(s)
    zr_s  = similar(s, Float64)
    for (rank, idx) in enumerate(ord_s)
        p = (rank - 0.5)/S
        zr_s[idx] = quantile(Normal(), p)
    end
    # correlated normal for u-order
    ε   = randn(rng, S)
    z_u = rho_c * zr_s .+ sqrt(max(0.0, 1 - rho_c^2)) .* ε
    # map to a permutation of original u (preserve marginal)
    ord_u   = sortperm(u)              # ascending values of u
    ord_zu  = sortperm(z_u)            # desired ordering by correlated scores
    u_out   = similar(u)
    for (k, idx) in enumerate(ord_zu)
        u_out[idx] = u[ord_u[k]]
    end
    return u_out
end

# --------------------------------------------------------------------
# Options (RemoveSweep-style)
# --------------------------------------------------------------------
Base.@kwdef struct CorrTimescaleSweepOptions
    modes::Vector{Symbol} = [:TR]
    S_vals::Vector{Int} = [120]
    conn_vals::AbstractVector{Float64} = 0.05:0.05:0.30
    mean_abs_vals::Vector{Float64} = [0.5, 1.0, 2.0]
    mag_cv_vals::Vector{Float64} = [0.1, 0.5, 1.0]
    u_mean_vals::Vector{Float64} = [1.0]
    u_cv_vals::Vector{Float64} = [0.3, 0.5, 0.8, 1.0, 2.0]
    degree_families::Vector{Symbol} = [:uniform, :lognormal, :pareto]
    deg_cv_vals::Vector{Float64} = [0.0, 0.5, 1.0, 2.0]
    deg_pl_alphas::Vector{Float64} = [1.2, 1.5, 2.0, 3.0]
    rho_sym_vals::Vector{Float64} = [0.0, 0.5, 1.0]

    IS_lines::Vector{Float64} = [0.05, 0.10, 0.40, 0.80, 1.20]
    reps_per_combo::Int = 2
    number_of_combinations::Int = 200

    # Coupling grid: correlation between time-scales (u) and interaction distribution (|α|)
    rho_c_vals::Vector{Float64} = collect(0.0:0.1:1.0)

    # r̃med horizon to evaluate predictability at
    t_eval::Float64 = 10.0

    # step params
    q_thresh::Float64 = 0.20

    seed::Int = 20251030
end

# --------------------------------------------------------------------
# Sweep
# - single threaded loop over 'sel', which already includes (rep, IS_line)
# - NO helper redefinitions (uses whatever is in scope)
# - local, function-scoped helpers only (copula mapping + R²)
# --------------------------------------------------------------------
function run_corr_timescale_interaction_sweep(opts::CorrTimescaleSweepOptions)
    # --- local helpers (function-scoped; not global) ---
    @inline function _splitmix64(x::UInt64)
        x += 0x9E3779B97F4A7C15
        z = x
        z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
        z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
        return z ⊻ (z >>> 31)
    end

    # degree-family expansion like your pipeline
    deg_specs = Tuple{Symbol,Float64}[]
    for fam in opts.degree_families
        if fam === :uniform
            push!(deg_specs, (:uniform, 0.0))
        elseif fam === :lognormal
            append!(deg_specs, ((:lognormal, x) for x in opts.deg_cv_vals))
        elseif fam === :pareto
            append!(deg_specs, ((:pareto, a) for a in opts.deg_pl_alphas))
        end
    end

    # build 'sel' that ALREADY includes reps and IS_lines (single loop)
    combos = collect(Iterators.product(
        opts.modes, opts.S_vals, opts.conn_vals, opts.mean_abs_vals, opts.mag_cv_vals,
        opts.u_mean_vals, opts.u_cv_vals, deg_specs, opts.rho_sym_vals,
        1:opts.reps_per_combo, opts.IS_lines
    ))
    sel = (length(combos) > opts.number_of_combinations) ?
          sample(combos, opts.number_of_combinations; replace=false) : combos

    base = UInt64(opts.seed)
    buckets = [Vector{NamedTuple}() for _ in 1:nthreads()]

    # ---------------- SINGLE THREADED LOOP (with Threads.@threads over sel) -------------
    Threads.@threads for idx in eachindex(sel)
        (mode, S, conn, mean_abs, mag_cv, u_mean, u_cv, (deg_fam, deg_param), rho_sym, rep, IS_line) = sel[idx]

        # thread-local RNG for this *single* loop index
        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx) ⊻ UInt64(threadid())))
        rng  = Random.Xoshiro(rand(rng0, UInt64))

        # generator: niche if present, else random_trophic / nontrophic
        A0 = build_niche_trophic(
            S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
            degree_family=deg_fam, deg_param=deg_param,
            rho_sym=rho_sym, rng=rng
        )

        baseIS = realized_IS(A0)
        baseIS == 0 && continue
        β = IS_line / baseIS
        A = β .* A0

        # baseline u and α (built from baseline u)
        u0 = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
        J0 = jacobian(A, u0)
        α  = alpha_off_from(J0, u0)

        # structure-free rewiring baseline (ER trophic), scaled to same IS
        A_rew0 = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                         rho_sym=rho_sym, rng=rng)
        βr = (b -> b>0 ? IS_line/b : 1.0)(realized_IS(A_rew0))
        A_rew = βr .* A_rew0

        # scan coupling rho_c (u ↔ |α|)
        for rho_c in opts.rho_c_vals
            u_corr = correlate_u_with_alpha(u0, α; rho_c=rho_c, rng=rng)

            # FULL (keep α, change time-scales)
            J_full = build_J_from(α, u_corr)
            r_full = median_return_rate(J_full, u_corr; t=opts.t_eval, perturbation=:biomass)

            # α-steps (reuse your helpers from scope)
            α_row = op_rowmean_alpha(α)
            α_thr = op_threshold_alpha(α; q=opts.q_thresh)
            α_rsh = op_reshuffle_preserve_pairs(α; rng=rng)

            J_row = build_J_from(α_row, u_corr)
            J_thr = build_J_from(α_thr, u_corr)
            J_rsh = build_J_from(α_rsh, u_corr)

            # rewiring with SAME u_corr
            J_rew = jacobian(A_rew, u_corr)

            # abundance reshuffle (break u–α alignment)
            u_sh  = u_corr[randperm(rng, length(u_corr))]
            J_ush = build_J_from(α, u_sh)

            # rare removal
            u_rr  = remove_rarest_species(u_corr; p=0.10)
            J_rr  = build_J_from(α, u_rr)

            push!(buckets[threadid()], (;
                mode, S, conn, mean_abs, mag_cv, u_mean, u_cv,
                degree_family=deg_fam, degree_param=deg_param, rho_sym,
                rep, IS_target=IS_line, rho_c,
                r_full,
                r_row    = median_return_rate(J_row,  u_corr; t=opts.t_eval, perturbation=:biomass),
                r_thr    = median_return_rate(J_thr,  u_corr; t=opts.t_eval, perturbation=:biomass),
                r_reshuf = median_return_rate(J_rsh,  u_corr; t=opts.t_eval, perturbation=:biomass),
                r_rew    = median_return_rate(J_rew,  u_corr; t=opts.t_eval, perturbation=:biomass),
                r_uni  = median_return_rate(J_ush,  u_sh;   t=opts.t_eval, perturbation=:biomass),
                r_rarer  = median_return_rate(J_rr,   filter(!iszero, u_rr); t=opts.t_eval, perturbation=:biomass)
            ))
        end
    end

    df_raw = DataFrame(vcat(buckets...))

    # Summaries: R²(step vs full) at each (mode, step, IS_target, rho_c)
    steps = (:row,:thr,:reshuf,:rew,:uni,:rarer)
    rowsS = NamedTuple[]
    for sub in groupby(df_raw, [:mode, :IS_target, :rho_c])
        x = sub.r_full
        for s in steps
            y = sub[!, Symbol(:r_, s)]
            r2 = r2_to_identity(collect(x), collect(y))
            push!(rowsS, (;
                mode = sub.mode[1],
                IS_target = sub.IS_target[1],
                rho_c = sub.rho_c[1],
                step = String(s),
                r2, n = nrow(sub)
            ))
        end
    end
    return df_raw, DataFrame(rowsS)
end

opts = CorrTimescaleSweepOptions(;
    modes = [:TR],
    S_vals = [120],
    conn_vals = 0.10:0.10:0.30,
    mean_abs_vals = [1.0],
    mag_cv_vals = [0.1, 0.5, 1.0],
    u_mean_vals = [1.0],
    u_cv_vals = [0.3, 0.5, 0.8, 1.0, 2.0],
    degree_families = [:uniform, :lognormal, :pareto],
    deg_cv_vals = [0.0, 0.5, 1.0, 2.0],
    deg_pl_alphas = [1.2, 1.5, 2.0, 3.0],
    rho_sym_vals = [0.0, 0.5, 1.0],
    IS_lines = [0.05, 0.10, 0.40, 0.80, 1.20],
    reps_per_combo = 3,
    number_of_combinations = 500,
    rho_c_vals = 0.0:0.1:1.0,
    t_eval = 5.0,
    q_thresh = 0.20,
    seed = 20251030
)

df_raw_ualpha_corr, df_sum_ualpha_corr = run_corr_timescale_interaction_sweep(opts)
df_raw_ualpha_corr_rhoOne, df_sum_ualpha_corr_rhoOne = run_corr_timescale_interaction_sweep(opts)
################# EXTENSION WITH BASELINE #################
"""
r2_baselines_long_from_df_main(df_main; steps, filter_fun) -> Dict{String,Float64}

Compute a single baseline R² per step using df_main (no removal):
    R²( long_rmed_<step>  vs  long_rmed_full )
Default steps: ["row","thr","reshuf","rew","uni","rarer"] (match your CorrUAlpha plot).
You can pass `filter_fun` to subset df_main if needed (e.g., only :TR).
"""
function r2_baselines_long_from_df_main(df_main::DataFrame;
        steps::Vector{String} = ["row","thr","reshuf","rew","uni","rarer"],
        filter_fun::Function = identity)

    d = filter_fun(df_main)
    isempty(d) && return Dict(s => NaN for s in steps)

    x = d.long_rmed_full
    baselines = Dict{String,Float64}()
    for s in steps
        col = Symbol(:long_rmed_, s)
        @assert hasproperty(d, col) "df_main is missing column $(col)"
        y = d[!, col]
        r2 = _r2_to_identity(collect(x), collect(y))
        baselines[s] = r2
    end
    return baselines
end

# ---------- CorrUAlpha plot with per-step baseline overlay ----------
"""
plot_corr_timescale_grid_by_IS_with_baseline(summary; baselines, steps, title, modes, cmap)

Same as your plot_corr_timescale_grid_by_IS, but draws one dashed horizontal
line per panel at the step’s baseline R² (from df_main at t=5.0).

- `summary`: output from run_corr_timescale_interaction_sweep(...) summary
             with cols [:mode, :step (String), :IS_target, :rho_c, :r2]
- `baselines`: Dict{String,Float64} from r2_baselines_long_from_df_main
- `steps`: panels order (default matches your 2×3 layout)
"""
function plot_corr_timescale_grid_by_IS_with_baseline(summary::DataFrame;
        baselines::Dict{String,Float64},
        # steps::Vector{String} = ["row","thr","reshuf","rew","uni","rarer"],
        steps::Vector{String} = ["reshuf","rew","uni"],
        title::String = "Predictability (R²) vs correlation(u, |α|) — with baselines",
        modes::Vector{Symbol} = [:TR],
        cmap = :viridis)

    ISvals = sort(unique(summary.IS_target))
    pal = cgrad(cmap, length(ISvals), categorical=true)
    labels = ["IS=$(round(v, digits=2))" for v in ISvals]

    fig = Figure(size=(1100, 640))
    Label(fig[0, 1:3], title; fontsize=18, font=:bold, halign=:center)

    for (i, step) in enumerate(steps)
        r = (i - 1) ÷ 3 + 1
        c = (i - 1) % 3 + 1
        ax = Axis(fig[r, c];
                  xlabel="ρc", ylabel=(c==1 ? "R² (vs FULL)" : ""),
                  title=uppercase(step),
                  limits=((0,1), (-0.05, 1.05)),
                  ygridvisible=true, xgridvisible=false)

        ISvals = sort(unique(summary.IS_target))
        
        pal = [get(ColorSchemes.viridis, i/length(ISvals)) for i in 1:length(ISvals)]
        labels = ["IS=$(round(v, digits=2))" for v in ISvals]

        for (k, IS) in enumerate(ISvals)
            d = filter(row -> row.step==step && row.IS_target==IS && (row.mode in modes), summary)
            isempty(d) && continue
            sort!(d, :rho_c)
            x = collect(Float64, d.rho_c)
            r2s = collect(Float64, d.r2)
            r2s[r2s .< 0.0] .= 0.0
            y = r2s
            lines!(ax, x, y; color=pal[k], label=labels[k])
            scatter!(ax, x, y; color=pal[k], markersize=5)
        end

        # horizontal baseline for this step (from df_main, long_rmed)
        if haskey(baselines, step) && isfinite(baselines[step])
            h = baselines[step]
            hlines!(ax, h; color=:gray35, linestyle=:dash, linewidth=2)
            text!(ax, @sprintf("baseline=%.2f", h), position=(0.98, h),
                  align=(:right,:center), color=:gray35, fontsize=10)
        end

        if i == 1
            axislegend(ax; position=:rb, framevisible=false)
        end
    end

    display(fig)
end

# 1) After run_sweep_stable(...; long_time_value=5.0):
# df_main, df_t = run_sweep_stable(...)
bas = r2_baselines_long_from_df_main(df_main_bio;
    steps=["row","thr","reshuf","rew","uni","rarer"],
    filter_fun = d -> filter(:mode => ==("TR"), d)
)

# 2) After run_corr_timescale_interaction_sweep(...):
# df_raw_ualpha_corr, df_sum = run_corr_timescale_interaction_sweep(opts)
df_sum_ualpha_corr.rho_c = Float64.(df_sum_ualpha_corr.rho_c)
df_sum_ualpha_corr.r2    = Float64.(df_sum_ualpha_corr.r2)
plot_corr_timescale_grid_by_IS_with_baseline(
    df_sum_ualpha_corr; baselines=bas,
    title = "R² vs corr(u,|α|) — baselines at r̃med(t=5) and all rho's"
)


####### ONLY RHO ONE ########
df_sum_ualpha_corr_rhoOne.rho_c = Float64.(df_sum_ualpha_corr_rhoOne.rho_c)
df_sum_ualpha_corr_rhoOne.r2    = Float64.(df_sum_ualpha_corr_rhoOne.r2)

bas_rhoOne = r2_baselines_long_from_df_main(filter(:rho_sym => ==(1.0), df_main_bio);
    # steps=["row","thr","reshuf","rew","uni","rarer"],
    steps=["reshuf","rew","uni"],
    filter_fun = d -> filter(:mode => ==("TR"), d)
)

plot_corr_timescale_grid_by_IS_with_baseline(
    df_sum_ualpha_corr_rhoOne; baselines=bas_rhoOne,
    title = "R² vs corr(u,|α|) — baselines at r̃med(t=5) and only rho=1"
)