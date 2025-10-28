############## Rho_PredictabilitySweep.jl ##############
# Explore how trophic antisymmetry (rho_sym) controls predictability of
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

using Random, Statistics, LinearAlgebra, DataFrames, CairoMakie, CSV

# --- If your r̃med function has a different name, alias it here ---
if !isdefined(@__MODULE__, :median_return_rate)
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
Base.@kwdef struct RhoOptions
    modes::Vector{Symbol} = [:TR]               # [:TR] or [:TR,:NT]
    S::Int = 120
    conn::Float64 = 0.10
    mean_abs::Float64 = 0.10
    mag_cv::Float64 = 0.60
    degree_family::Symbol = :uniform            # :uniform | :lognormal | :pareto
    deg_param::Float64 = 0.0                    # CV for lognormal, alpha for pareto
    rho_vals::Vector{Float64} = collect(range(0.0, 1.0; length=11))
    u_mean::Float64 = 1.0
    u_cv::Float64 = 0.5
    # Optional: rescale each A to this realized IS (mean |A| off-diag) to isolate rho effects
    IS_target::Union{Nothing,Float64} = 0.10
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
    X = [x ones(length(x))]
    β = X \ y
    return (r2, β[1], β[2])
end

# --- Main function ----------------------------------------------------
"""
run_rho_predictability(opts::RhoOptions) -> (df, summary)

For each replicate and each rho in `rho_vals`, build A with that rho.
Optionally rescale A so realized_IS(A) == IS_target (if provided).
No stabilization is applied. For each rho level, compute metrics for
Full and the 6 steps: reshuf, thr, row, uni, rarer, rew.

Output `df` has one row per (mode,rep,rho). `summary` gives R², slope,
intercept of step vs full per rho and metric.
"""
function run_rho_predictability(opts::RhoOptions)
    rng_global = Random.Xoshiro(opts.seed)

    rows = Vector{NamedTuple}()
    pushrow(x) = push!(rows, x)

    for mode in opts.modes
        for r in 1:opts.reps
            rng = Random.Xoshiro(rand(rng_global, UInt64))

            # draw u once per replicate
            u = random_u(opts.S; mean=opts.u_mean, cv=opts.u_cv, rng=rng)

            for ρ in opts.rho_vals
                # build A at this rho
                A0 = _genA(mode, rng, opts.S; conn=opts.conn, mean_abs=opts.mean_abs,
                           mag_cv=opts.mag_cv, degree_family=opts.degree_family,
                           deg_param=opts.deg_param, rho_sym=ρ)

                # optional: rescale to fixed IS to isolate rho effects
                if opts.IS_target !== nothing
                    base_IS = realized_IS(A0)
                    if base_IS > 0
                        β = opts.IS_target / base_IS
                        A = β .* A0
                    else
                        A = A0
                    end
                else
                    A = A0
                end

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

                # REW: redraw from the same ensemble at the SAME rho, optional rescale to same IS
                A_rew0 = _genA(mode, rng, opts.S; conn=opts.conn, mean_abs=opts.mean_abs,
                                mag_cv=opts.mag_cv, degree_family=opts.degree_family,
                                deg_param=opts.deg_param, rho_sym=ρ)
                if opts.IS_target !== nothing
                    base_IS_rew = realized_IS(A_rew0)
                    if base_IS_rew > 0
                        βrew = opts.IS_target / base_IS_rew
                        A_rew = βrew .* A_rew0
                    else
                        A_rew = A_rew0
                    end
                else
                    A_rew = A_rew0
                end
                J_rew  = jacobian(A_rew, u)

                metRsh = _metrics(J_reshuf, u;         t_short=opts.t_short, t_long=opts.t_long)
                metThr = _metrics(J_thr,    u;         t_short=opts.t_short, t_long=opts.t_long)
                metRow = _metrics(J_row,    u;         t_short=opts.t_short, t_long=opts.t_long)
                metUni = _metrics(J_uni,    u_uni;     t_short=opts.t_short, t_long=opts.t_long)
                metRar = _metrics(J_rarer,  u_rarerem; t_short=opts.t_short, t_long=opts.t_long)
                metRew = _metrics(J_rew,    u;         t_short=opts.t_short, t_long=opts.t_long)

                pushrow((;
                    mode, rep=r, rho_sym=ρ,
                    # realized structure at this rho
                    conn_real = realized_connectance(A),
                    IS_real   = realized_IS(A),
                    # stability flag of the FULL system at this rho
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

    # --- Summaries: R² (and slope/intercept) vs rho per step ----------
    function summarize(df; metric_sym::Symbol, steps::Vector{Symbol}, stable_only::Bool=false)
        dat = stable_only ? filter(row -> row.stable_full, df) : df
        out = DataFrame()
        for mode in unique(dat.mode)
            dmode = dat[dat.mode .== mode, :]
            for ρ in unique(dmode.rho_sym)
                d = dmode[dmode.rho_sym .== ρ, :]
                x = d[!, metric_sym]
                for step in steps
                    y = d[!, Symbol(replace(string(metric_sym), "_full" => "_"*String(step)))]
                    r2, slope, intercept = r2_to_identity(collect(x), collect(y))
                    push!(out, (; mode, rho_sym=ρ, metric=String(metric_sym), step=String(step),
                                  r2, slope, intercept, n=length(x), stable_only))
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
# Plotting helpers (rho as x-axis)
# -------------------------------
function plot_predictability_vs_rho(summary; metric::String, title::String="Predictability vs ρ", modes=[:TR])
    steps = ["reshuf", "uni", "rarer", "rew"]  # focus subset; change as needed
    fig = Figure(size=(1000, 420))
    for (j, mode) in enumerate(modes)
        ax = Axis(
            fig[j,1];
            xlabel="ρ (trophic antisymmetry of magnitudes)", ylabel="R² to y=x", title="$title — $mode — $metric",
            limits=((0.0, 1.0), (-0.1, 1.1))
        )
        for step in steps
            d = filter(row -> row.mode==mode && row.metric==metric && !row.stable_only && row.step==step, summary)
            isempty(d) && continue
            d.r2_corrected = copy(d.r2)
            d.r2_corrected[d.r2_corrected .< 0] .= 0.0
            p = sortperm(d.rho_sym)
            lines!(ax, d.rho_sym[p], d.r2_corrected[p]; label=step)
            scatter!(ax, d.rho_sym[p], d.r2_corrected[p])
        end
        axislegend(ax; position=:rb)
    end
    display(fig)
end

"""
plot_r2_grid_rho(summary; metric=:res_full, steps=[:rew, :reshuf, :rarer, :uni], modes=[:TR], title="Predictability grid vs ρ")

Creates a grid of R² vs rho_sym plots, one per step.
Each subplot shows how well the chosen metric (e.g. :res_full) correlates with each step.
"""
function plot_r2_grid_rho(summary; metric::Symbol=:res_full, steps::Vector{Symbol}=[:rew, :reshuf, :rarer, :uni],
                          modes::Vector{Symbol}=[:TR], title::String="Predictability grid vs ρ")

    palette = Makie.wong_colors()[1:length(modes)]
    fig = Figure(size=(950, 650))
    Label(fig[0, 1:3], title; fontsize=18, font=:bold, halign=:center)

    for (i, step) in enumerate(steps)
        ax = Axis(
            fig[(i-1) ÷ 3 + 1, (i-1) % 3 + 1];
            xlabel="ρ", ylabel="R²",
            title=uppercase(string(step)),
            limits=((0.0, 1.0), (-0.2, 1.05)),
            ygridvisible=true, xgridvisible=false
        )

        for (mi, mode) in enumerate(modes)
            df_sub = filter(row ->
                row.metric == string(metric) &&
                row.step == string(step) &&
                row.mode == mode &&
                !row.stable_only,
                summary)

            isempty(df_sub) && continue
            p = sortperm(df_sub.rho_sym)
            lines!(ax, df_sub.rho_sym[p], df_sub.r2[p]; color=palette[mi], label=string(mode))
            scatter!(ax, df_sub.rho_sym[p], df_sub.r2[p]; color=palette[mi], markersize=6)
        end

        if i == 1
            axislegend(ax; position=:rb)
        end
    end

    display(fig)
end

# -------------------------------
# Example run (trophic focus)
# -------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    opts = RhoOptions(; modes=[:TR], S=120, conn=0.10,
        mean_abs=0.10, mag_cv=0.60,
        degree_family=:uniform, deg_param=0.0,
        rho_vals=collect(range(0,1; length=11)),
        u_mean=1.0, u_cv=0.8,
        IS_target=0.10,   # fix realized IS across rho to isolate antisymmetry
        reps=150,
        q_thresh=0.20, t_short=0.01, t_long=0.50, seed=20251027)

    df_rho, summ_rho = run_rho_predictability(opts)
    println("\nRows: ", nrow(df_rho))

    CSV.write("rho_predictability_raw.csv", df_rho)
    CSV.write("rho_predictability_summary.csv", summ_rho)

    # Optional: clip negative R² for visualization
    summ = deepcopy(summ_rho)
    summ.r2_corrected = copy(summ.r2)
    summ.r2_corrected[summ.r2_corrected .< 0] .= 0.0

    plot_predictability_vs_rho(summ; metric="res_full",    title="Resilience predictability vs ρ", modes=opts.modes)
    plot_predictability_vs_rho(summ; metric="rea_full",    title="Reactivity predictability vs ρ", modes=opts.modes)
    plot_predictability_vs_rho(summ; metric="rmed_s_full", title="r̃med(t_short) predictability vs ρ", modes=opts.modes)
    plot_predictability_vs_rho(summ; metric="rmed_l_full", title="r̃med(t_long) predictability vs ρ",  modes=opts.modes)

    plot_r2_grid_rho(summ; metric=:rmed_l_full, steps=[:rew, :thr, :row, :reshuf, :rarer, :uni], modes=[:TR],
                     title="R̃med Long Predictability (R² vs ρ)")
end
