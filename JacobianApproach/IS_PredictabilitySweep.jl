############## IS_PredictabilitySweep.jl ##############
# Explore how interaction strength (IS) controls predictability of
# resilience, reactivity, and median return rate under different
# Jacobian simplification steps — WITHOUT stabilization shrink.
#
# Assumes the helpers from your working file are available in scope:
#   build_random_trophic, build_random_nontrophic, random_u, jacobian,
#   resilience, reactivity, alpha_off_from, build_J_from,
#   op_reshuffle_alpha, op_rowmean_alpha, op_threshold_alpha,
#   uniform_u, remove_rarest_species, realized_IS
# and your corrected median-return-rate (Arnoldi) function.
# If not, `include("MainCode.jl")` before running.
#########################################################

using Random, Statistics, LinearAlgebra, DataFrames, CairoMakie

# --- If your r̃med function has a different name, alias it here ---
if !isdefined(@__MODULE__, :median_return_rate)
    # Fallback to a safe definition; replace with your exact version if needed
    @warn "median_return_rate(J,u; t, perturbation) not found in scope — using fallback." 
    function median_return_rate(J::AbstractMatrix, u::AbstractVector; t::Real=0.01, perturbation::Symbol=:biomass)
        S = size(J,1)
        S == 0 && return NaN
        E = exp(t*J)
        if perturbation === :uniform
            num = log(tr(E*transpose(E)))
            den = log(S)
        else
            w = u .^ 2
            num = log(tr(E*Diagonal(w)*transpose(E)))
            den = log(sum(w))
        end
        return -(num - den)/(2t)
    end
end

# -------------------------------
# Core sweep (NO stabilization)
# -------------------------------
Base.@kwdef struct ISOptions
    modes::Vector{Symbol} = [:TR]               # [:TR] or [:TR,:NT]
    S::Int = 120
    conn::Float64 = 0.10
    mean_abs::Float64 = 0.10
    mag_cv::Float64 = 0.60
    degree_family::Symbol = :uniform            # :uniform | :lognormal | :pareto
    deg_param::Float64 = 0.0                    # CV for lognormal, alpha for pareto
    rho_sym::Float64 = 0.0
    u_mean::Float64 = 1.0
    u_cv::Float64 = 0.5
    IS_targets::Vector{Float64} = collect(0.02:0.02:0.20)  # desired realized mean |A| (off-diag)
    reps::Int = 100
    q_thresh::Float64 = 0.20                    # threshold q for op_threshold_alpha
    t_short::Float64 = 0.01
    t_long::Float64 = 0.50
    seed::Int = 42
end

# generator wrapper consistent with your code
function _genA(mode::Symbol, rng, S; conn, mean_abs, mag_cv, degree_family, deg_param, rho_sym)
    if mode === :NT
        return build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                       degree_family=degree_family, deg_param=deg_param,
                                       rho_sym=rho_sym, rng=rng)
    else
        return build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                    degree_family=degree_family, deg_param=deg_param,
                                    rho_sym=rho_sym, rng=rng)
    end
end

# --- Metrics helper ---------------------------------------------------
function _metrics(J::AbstractMatrix, u; t_short::Real, t_long::Real)
    (;
        res = resilience(J),
        rea = reactivity(J),
        rmed_s = median_return_rate(J, u; t=t_short, perturbation=:biomass),
        rmed_l = median_return_rate(J, u; t=t_long,  perturbation=:biomass)
    )
end

# --- R^2 to the 1:1 line (for summaries later) -----------------------
# Returns (r2, slope, intercept) where slope/intercept are from OLS y ~ x
function r2_to_identity(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == 0 && return (NaN, NaN, NaN)
    μy = mean(y)
    sst = sum((y .- μy).^2)
    ssr = sum((y .- x).^2)
    r2 = sst == 0 ? 1.0 : 1 - ssr/sst
    # OLS fit
    X = [x ones(length(x))]
    β = X \ y
    return (r2, β[1], β[2])
end

# --- Main function ----------------------------------------------------
"""
run_is_predictability(opts::ISOptions) -> (df, summary)

Build A once per (mode,rep), draw u, then for each target IS value, rescale A
by a single factor β so that realized_IS(A) == IS_target. No stabilization is
applied. For each IS level, compute metrics for Full and the 6 steps:
  reshuf, thr, row, uni, rarer, rew

Output `df` has one row per (mode,rep,IS_target). `summary` gives R², slope,
intercept of step vs full per IS level and metric.
"""
function run_is_predictability(opts::ISOptions)
    rng_global = Random.Xoshiro(opts.seed)

    rows = Vector{NamedTuple}()
    pushrow(x) = push!(rows, x)

    for mode in opts.modes
        for r in 1:opts.reps
            rng = Random.Xoshiro(rand(rng_global, UInt64))

            # build base A and u once per replicate
            A0 = _genA(mode, rng, opts.S; conn=opts.conn, mean_abs=opts.mean_abs,
                       mag_cv=opts.mag_cv, degree_family=opts.degree_family,
                       deg_param=opts.deg_param, rho_sym=opts.rho_sym)
            u  = random_u(opts.S; mean=opts.u_mean, cv=opts.u_cv, rng=rng)

            base_IS = realized_IS(A0)
            base_IS == 0 && continue   # degenerate case

            for IS_tgt in opts.IS_targets
                β = IS_tgt / base_IS
                A = β .* A0

                # FULL
                J_full = jacobian(A, u)
                metF = _metrics(J_full, u; t_short=opts.t_short, t_long=opts.t_long)

                # α-based transforms
                α = alpha_off_from(J_full, u)
                α_reshuf = op_reshuffle_alpha(α; rng=rng)
                α_row    = op_rowmean_alpha(α)
                α_thr    = op_threshold_alpha(α; q=opts.q_thresh)

                u_uni     = uniform_u(u)
                u_rarerem = remove_rarest_species(u; p=0.1)

                J_reshuf = build_J_from(α_reshuf, u)
                J_row    = build_J_from(α_row,    u)
                J_thr    = build_J_from(α_thr,    u)
                J_uni    = build_J_from(α,        u_uni)
                J_rarer  = build_J_from(α,        u_rarerem)

                # REW: redraw from the same ensemble, then rescale to same IS
                A_rew0 = _genA(mode, rng, opts.S; conn=opts.conn, mean_abs=opts.mean_abs,
                                mag_cv=opts.mag_cv, degree_family=opts.degree_family,
                                deg_param=opts.deg_param, rho_sym=opts.rho_sym)
                A_rew  = β .* A_rew0
                J_rew  = jacobian(A_rew, u)

                metRsh = _metrics(J_reshuf, u;         t_short=opts.t_short, t_long=opts.t_long)
                metThr = _metrics(J_thr,    u;         t_short=opts.t_short, t_long=opts.t_long)
                metRow = _metrics(J_row,    u;         t_short=opts.t_short, t_long=opts.t_long)
                metUni = _metrics(J_uni,    u_uni;     t_short=opts.t_short, t_long=opts.t_long)
                metRar = _metrics(J_rarer,  u_rarerem; t_short=opts.t_short, t_long=opts.t_long)
                metRew = _metrics(J_rew,    u;         t_short=opts.t_short, t_long=opts.t_long)

                pushrow((;
                    mode, rep=r, IS_target=IS_tgt,
                    # realized structure at this scale
                    conn_real = realized_connectance(A),
                    IS_real   = realized_IS(A),
                    # stability flag of the FULL system at this IS
                    res_full = metF.res, rea_full = metF.rea, rmed_s_full = metF.rmed_s, rmed_l_full = metF.rmed_l,
                    stable_full = metF.res < 0,
                    # steps
                    res_reshuf = metRsh.res, rea_reshuf = metRsh.rea, rmed_s_reshuf = metRsh.rmed_s, rmed_l_reshuf = metRsh.rmed_l,
                    res_thr    = metThr.res, rea_thr    = metThr.rea, rmed_s_thr    = metThr.rmed_s, rmed_l_thr    = metThr.rmed_l,
                    res_row    = metRow.res, rea_row    = metRow.rea, rmed_s_row    = metRow.rmed_s, rmed_l_row    = metRow.rmed_l,
                    res_uni    = metUni.res, rea_uni    = metUni.rea, rmed_s_uni    = metUni.rmed_s, rmed_l_uni    = metUni.rmed_l,
                    res_rarer  = metRar.res, rea_rarer  = metRar.rea, rmed_s_rarer  = metRar.rmed_s, rmed_l_rarer  = metRar.rmed_l,
                    res_rew    = metRew.res, rea_rew    = metRew.rea, rmed_s_rew    = metRew.rmed_s, rmed_l_rew    = metRew.rmed_l
                ))
            end
        end
    end

    df = DataFrame(rows)

    # --- Summaries: R² (and slope/intercept) vs IS per step ------------
    function summarize(df; metric_sym::Symbol, steps::Vector{Symbol}, stable_only::Bool=false)
        dat = stable_only ? filter(row -> row.stable_full, df) : df
        out = DataFrame()
        for mode in unique(dat.mode)
            dmode = dat[dat.mode .== mode, :]
            for IS in unique(dmode.IS_target)
                d = dmode[dmode.IS_target .== IS, :]
                x = d[!, metric_sym]
                for step in steps
                    y = d[!, Symbol(replace(string(metric_sym), "_full" => "_"*String(step)))]
                    r2, slope, intercept = r2_to_identity(collect(x), collect(y))
                    push!(out, (; mode, IS_target=IS, metric=String(metric_sym), step=String(step), r2, slope, intercept,
                                  n=length(x), stable_only))
                end
            end
        end
        return out
    end

    steps = [:reshuf, :thr, :row, :uni, :rarer, :rew]
    summ = vcat(
        summarize(df; metric_sym=:res_full,     steps=steps, stable_only=false),
        summarize(df; metric_sym=:rea_full,     steps=steps, stable_only=false),
        summarize(df; metric_sym=:rmed_s_full,  steps=steps, stable_only=false),
        summarize(df; metric_sym=:rmed_l_full,  steps=steps, stable_only=false),
        summarize(df; metric_sym=:res_full,     steps=steps, stable_only=true),
        summarize(df; metric_sym=:rea_full,     steps=steps, stable_only=true),
        summarize(df; metric_sym=:rmed_s_full,  steps=steps, stable_only=true),
        summarize(df; metric_sym=:rmed_l_full,  steps=steps, stable_only=true),
    )

    return df, summ
end

# -------------------------------
# Plotting helpers
# -------------------------------
function plot_predictability_vs_IS(summary; metric::String, title::String="Predictability vs IS", modes=[:TR])
    steps = unique(summary.step)
    fig = Figure(size=(1000, 420))
    cols = length(steps)
    for (j, mode) in enumerate(modes)
        ax = Axis(fig[j,1]; xlabel="IS_target (mean |A|)", ylabel="R² to y=x", title="$title — $mode — $metric",
                  ylimits=(-0.5, 1.05))
        for step in steps
            d = filter(row -> row.mode==mode && row.metric==metric && !row.stable_only && row.step==step, summary)
            isempty(d) && continue
            p = sortperm(d.IS_target)
            lines!(ax, d.IS_target[p], d.r2[p]; label=step)
            scatter!(ax, d.IS_target[p], d.r2[p])
        end
        axislegend(ax; position=:rb)
    end
    display(fig)
    return fig
end

# -------------------------------
# Example run (trophic focus)
# -------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    opts = ISOptions(; modes=[:TR], S=120, conn=0.10,
        mean_abs=0.10, mag_cv=0.60,
        degree_family=:uniform, deg_param=0.0,
        rho_sym=0.3, u_mean=1.0, u_cv=0.8,
        IS_targets=collect(0.02:0.02:0.20), reps=150,
        q_thresh=0.20, t_short=0.01, t_long=0.50, seed=20251027)

    df, summ = run_is_predictability(opts)
    println("\nRows: ", nrow(df))

    # Save and plot
    CSV.write("is_predictability_raw.csv", df)
    CSV.write("is_predictability_summary.csv", summ)

    plot_predictability_vs_IS(summ; metric="res_full",   title="Resilience predictability vs IS", modes=opts.modes)
    plot_predictability_vs_IS(summ; metric="rea_full",   title="Reactivity predictability vs IS", modes=opts.modes)
    plot_predictability_vs_IS(summ; metric="rmed_s_full",title="r̃med(t_short) predictability vs IS", modes=opts.modes)
    plot_predictability_vs_IS(summ; metric="rmed_l_full",title="r̃med(t_long) predictability vs IS",  modes=opts.modes)
end
