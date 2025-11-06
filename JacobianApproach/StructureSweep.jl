###############################################
# Predictability vs Heterogeneity Sweeps
# - Sweep 1: abundance heterogeneity (u_cv)
# - Sweep 2: degree heterogeneity (deg_cv, lognormal family)
# Outputs:
#   df_u, dfk  (raw rows)
#   summ_u, summ_k  (R² per step & metric vs heterogeneity)
# Plots:
#   2x2 panels for each sweep, six lines (steps) per panel
###############################################
using Random, Statistics, LinearAlgebra, DataFrames, Distributions
using CairoMakie
using Base.Threads

# ---------- robust R² to identity ----------
# returns (r2, slope, intercept); clamp r2 at ≥0 for display
function _r2_to_identity(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == 0 && return (NaN, NaN, NaN)
    μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
    if sst == 0
        return ssr == 0 ? (1.0, 1.0, 0.0) : (0.0, NaN, NaN)
    end
    r2 = max(1 - ssr/sst, 0.0)
    β = [x ones(n)] \ y
    return (r2, β[1], β[2])
end

# ---------- RNG splitter ----------
@inline function _splitmix64(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
    z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
    return z ⊻ (z >>> 31)
end

# ---------- compute the four static metrics ----------
@inline function _metrics4(J, u; t_short=0.01, t_long=10.0)
    return (
        res = resilience(J),
        rea = reactivity(J),
        rmed_s = median_return_rate(J, u; t=t_short, perturbation=:biomass),
        rmed_l = median_return_rate(J, u; t=t_long,  perturbation=:biomass)
    )
end

# ---------- common inner: build J_full and J for 6 steps ----------
function _compute_steps(A, u; q_thresh=0.20, rng=Random.default_rng(), t_short=0.01, t_long=10.0)
    J = jacobian(A, u)
    α = alpha_off_from(J, u)

    α_row = op_rowmean_alpha(α)
    α_thr = op_threshold_alpha(α; q=q_thresh)
    α_rsh = op_reshuffle_alpha(α; rng=rng)

    # rewiring: structure-free ER trophic at same mean_abs scale (keep |A| scale via IS match)
    A_rew0 = build_random_trophic_ER(size(A,1); conn=realized_connectance(A),
                                     mean_abs=realized_IS(A), mag_cv=0.60,
                                     rho_sym=0.0, rng=rng)
    βr = (b->b>0 ? realized_IS(A)/b : 1.0)(realized_IS(A_rew0))
    A_rew = βr .* A_rew0

    # abundance reshuffle (UNI per your new definition)
    u_sh  = u[randperm(rng, length(u))]
    u_sh  = fill(mean(u), length(u))
    u_rr  = remove_rarest_species(u; p=0.9)

    J_full = J
    J_row  = build_J_from(α_row, u)
    J_thr  = build_J_from(α_thr, u)
    J_rsh  = build_J_from(α_rsh, u)
    J_rew  = jacobian(A_rew, u)
    J_ush  = build_J_from(α, u_sh)
    J_rar  = build_J_from(α, u_rr)

    mF = _metrics4(J_full, u; t_short=t_short, t_long=t_long)
    return (
        full = mF,
        row  = _metrics4(J_row, u;  t_short=t_short, t_long=t_long),
        thr  = _metrics4(J_thr, u;  t_short=t_short, t_long=t_long),
        reshuf = _metrics4(J_rsh, u; t_short=t_short, t_long=t_long),
        rew  = _metrics4(J_rew, u;  t_short=t_short, t_long=t_long),
        ushuf = _metrics4(J_ush, u_sh; t_short=t_short, t_long=t_long),
        rarer = _metrics4(J_rar, filter(!iszero, u_rr); t_short=t_short, t_long=t_long)
    )
end

# =====================================================================================
# Sweep 1: Predictability vs ABUNDANCE heterogeneity (u_cv)
# =====================================================================================
Base.@kwdef struct UHeteroOptions
    S::Int = 120
    conn::Float64 = 0.10
    mean_abs::Float64 = 0.10
    mag_cv::Float64 = 0.60
    degree_family::Symbol = :lognormal   # keep degree-family fixed for this sweep
    deg_param::Float64 = 0.5             # degree CV control (lognormal CV)
    rho_sym::Float64 = 0.5
    u_mean::Float64 = 1.0
    u_cv_vals::Vector{Float64} = [0.2, 0.4, 0.8, 1.2, 2.0, 3.0]
    IS_target::Float64 = 0.2            # rescale A to this realized mean |A|
    reps::Int = 100
    q_thresh::Float64 = 0.20
    t_short::Float64 = 0.01
    t_long::Float64 = 10.0
    seed::Int = 20251031
end

function run_predictability_vs_uhetero(opts::UHeteroOptions)
    base = _splitmix64(UInt64(opts.seed))
    bucket = [Vector{NamedTuple}() for _ in 1:nthreads()]

    Threads.@threads for idx in eachindex(opts.u_cv_vals)
        ucv = opts.u_cv_vals[idx]
        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx) ⊻ UInt64(threadid())))
        local_rows = bucket[threadid()]

        for rep in 1:opts.reps
            rng = Random.Xoshiro(rand(rng0, UInt64))
            A0 = build_niche_trophic(opts.S; conn=opts.conn, mean_abs=opts.mean_abs, mag_cv=opts.mag_cv,
                          degree_family=opts.degree_family, deg_param=opts.deg_param,
                          rho_sym=opts.rho_sym, rng=rng)
            baseIS = realized_IS(A0)
            baseIS == 0 && continue
            β = opts.IS_target / baseIS
            A = β .* A0
            u = random_u(opts.S; mean=opts.u_mean, cv=ucv, rng=rng)

            mets = _compute_steps(A, u; q_thresh=opts.q_thresh, rng=rng,
                                  t_short=opts.t_short, t_long=opts.t_long)

            push!(local_rows, (; kind=:u_cv, x=ucv,
                # full
                res_full=mets.full.res, rea_full=mets.full.rea, rmed_s_full=mets.full.rmed_s, rmed_l_full=mets.full.rmed_l,
                # steps
                res_row=mets.row.res,   rea_row=mets.row.rea,   rmed_s_row=mets.row.rmed_s,   rmed_l_row=mets.row.rmed_l,
                res_thr=mets.thr.res,   rea_thr=mets.thr.rea,   rmed_s_thr=mets.thr.rmed_s,   rmed_l_thr=mets.thr.rmed_l,
                res_reshuf=mets.reshuf.res, rea_reshuf=mets.reshuf.rea, rmed_s_reshuf=mets.reshuf.rmed_s, rmed_l_reshuf=mets.reshuf.rmed_l,
                res_rew=mets.rew.res,   rea_rew=mets.rew.rea,   rmed_s_rew=mets.rew.rmed_s,   rmed_l_rew=mets.rew.rmed_l,
                res_ushuf=mets.ushuf.res, rea_ushuf=mets.ushuf.rea, rmed_s_ushuf=mets.ushuf.rmed_s, rmed_l_ushuf=mets.ushuf.rmed_l,
                res_rarer=mets.rarer.res, rea_rarer=mets.rarer.rea, rmed_s_rarer=mets.rarer.rmed_s, rmed_l_rarer=mets.rarer.rmed_l))
        end
    end
    df = DataFrame(vcat(bucket...))

    # summarize R² vs x for each step & metric
    steps = (:row, :thr, :reshuf, :rew, :ushuf, :rarer)
    metrics = (:res, :rea, :rmed_s, :rmed_l)
    rowsS = NamedTuple[]
    for xval in sort(unique(df.x))
        sub = df[df.x .== xval, :]
        for m in metrics
            x = sub[!, Symbol(m, :_full)]
            for s in steps
                y = sub[!, Symbol(m, :_, s)]
                r2, slope, intercept = _r2_to_identity(collect(x), collect(y))
                push!(rowsS, (; kind=:u_cv, x=xval, metric=String(m), step=String(s), r2, slope, intercept, n=nrow(sub)))
            end
        end
    end
    return df, DataFrame(rowsS)
end

# =====================================================================================
# Sweep 2: Predictability vs DEGREE heterogeneity (deg_cv via lognormal family)
# =====================================================================================
Base.@kwdef struct KHeteroOptions
    S::Int = 120
    conn::Float64 = 0.10
    mean_abs::Float64 = 0.10
    mag_cv::Float64 = 0.60
    rho_sym::Float64 = 0.5
    u_mean::Float64 = 1.0
    u_cv::Float64 = 0.2                  # keep abundance heterogeneity fixed here
    deg_cv_vals::Vector{Float64} = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]  # lognormal CV control
    IS_target::Float64 = 0.2
    reps::Int = 100
    q_thresh::Float64 = 0.20
    t_short::Float64 = 0.01
    t_long::Float64 = 10.0
    seed::Int = 20251031
end

function run_predictability_vs_khetero(opts::KHeteroOptions)
    base = _splitmix64(UInt64(opts.seed))
    bucket = [Vector{NamedTuple}() for _ in 1:nthreads()]

    Threads.@threads for idx in eachindex(opts.deg_cv_vals)
        dcv = opts.deg_cv_vals[idx]
        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx) ⊻ UInt64(threadid())))
        local_rows = bucket[threadid()]

        for rep in 1:opts.reps
            rng = Random.Xoshiro(rand(rng0, UInt64))
            # A0 = build_niche_trophic(opts.S; conn=opts.conn, mean_abs=opts.mean_abs, mag_cv=opts.mag_cv,
            #               degree_family=:lognormal, deg_param=dcv,
            #               rho_sym=opts.rho_sym, rng=rng)
            A0  = build_random_trophic_ER(
                opts.S; conn=opts.conn, mean_abs=opts.mean_abs, mag_cv=opts.mag_cv,
                rho_sym=opts.rho_sym, rng=rng)
            baseIS = realized_IS(A0); baseIS == 0 && continue
            β = opts.IS_target / baseIS
            A = β .* A0
            u = random_u(opts.S; mean=opts.u_mean, cv=opts.u_cv, rng=rng)

            mets = _compute_steps(A, u; q_thresh=opts.q_thresh, rng=rng,
                                  t_short=opts.t_short, t_long=opts.t_long)

            push!(local_rows, (; kind=:deg_cv, x=dcv,
                res_full=mets.full.res, rea_full=mets.full.rea, rmed_s_full=mets.full.rmed_s, rmed_l_full=mets.full.rmed_l,
                res_row=mets.row.res,   rea_row=mets.row.rea,   rmed_s_row=mets.row.rmed_s,   rmed_l_row=mets.row.rmed_l,
                res_thr=mets.thr.res,   rea_thr=mets.thr.rea,   rmed_s_thr=mets.thr.rmed_s,   rmed_l_thr=mets.thr.rmed_l,
                res_reshuf=mets.reshuf.res, rea_reshuf=mets.reshuf.rea, rmed_s_reshuf=mets.reshuf.rmed_s, rmed_l_reshuf=mets.reshuf.rmed_l,
                res_rew=mets.rew.res,   rea_rew=mets.rew.rea,   rmed_s_rew=mets.rew.rmed_s,   rmed_l_rew=mets.rew.rmed_l,
                res_ushuf=mets.ushuf.res, rea_ushuf=mets.ushuf.rea, rmed_s_ushuf=mets.ushuf.rmed_s, rmed_l_ushuf=mets.ushuf.rmed_l,
                res_rarer=mets.rarer.res, rea_rarer=mets.rarer.rea, rmed_s_rarer=mets.rarer.rmed_s, rmed_l_rarer=mets.rarer.rmed_l))
        end
    end
    df = DataFrame(vcat(bucket...))

    steps = (:row, :thr, :reshuf, :rew, :ushuf, :rarer)
    metrics = (:res, :rea, :rmed_s, :rmed_l)
    rowsS = NamedTuple[]
    for xval in sort(unique(df.x))
        sub = df[df.x .== xval, :]
        for m in metrics
            x = sub[!, Symbol(m, :_full)]
            for s in steps
                y = sub[!, Symbol(m, :_, s)]
                r2, slope, intercept = _r2_to_identity(collect(x), collect(y))
                push!(rowsS, (; kind=:deg_cv, x=xval, metric=String(m), step=String(s), r2, slope, intercept, n=nrow(sub)))
            end
        end
    end
    return df, DataFrame(rowsS)
end

# =====================================================================================
# Sweep 3: Predictability vs INTERACTION-STRENGTH heterogeneity (mag_cv)
# =====================================================================================
Base.@kwdef struct ISHeteroOptions
    S::Int = 120
    conn::Float64 = 0.10
    mean_abs::Float64 = 0.10
    rho_sym::Float64 = 0.5
    degree_family::Symbol = :lognormal
    deg_param::Float64 = 0.5
    u_mean::Float64 = 1.0
    u_cv::Float64 = 0.6                    # fix abundance heterogeneity here
    mag_cv_vals::Vector{Float64} = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
    IS_target::Float64 = 0.2
    reps::Int = 100
    q_thresh::Float64 = 0.20
    t_short::Float64 = 0.01
    t_long::Float64 = 10.0
    seed::Int = 20251031
end

function run_predictability_vs_ishero(opts::ISHeteroOptions)
    base = _splitmix64(UInt64(opts.seed))
    bucket = [Vector{NamedTuple}() for _ in 1:nthreads()]

    Threads.@threads for idx in eachindex(opts.mag_cv_vals)
        mcv = opts.mag_cv_vals[idx]
        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx) ⊻ UInt64(threadid())))
        local_rows = bucket[threadid()]

        for rep in 1:opts.reps
            rng = Random.Xoshiro(rand(rng0, UInt64))
            A0 = build_niche_trophic(opts.S; conn=opts.conn, mean_abs=opts.mean_abs, mag_cv=mcv,
                             degree_family=opts.degree_family, deg_param=opts.deg_param,
                             rho_sym=opts.rho_sym, rng=rng)
            baseIS = realized_IS(A0)
            baseIS == 0 && continue
            β = opts.IS_target / baseIS
            A = β .* A0
            u = random_u(opts.S; mean=opts.u_mean, cv=opts.u_cv, rng=rng)

            mets = _compute_steps(A, u; q_thresh=opts.q_thresh, rng=rng,
                                  t_short=opts.t_short, t_long=opts.t_long)

            push!(local_rows, (; kind=:mag_cv, x=mcv,
                res_full=mets.full.res, rea_full=mets.full.rea, rmed_s_full=mets.full.rmed_s, rmed_l_full=mets.full.rmed_l,
                res_row=mets.row.res,   rea_row=mets.row.rea,   rmed_s_row=mets.row.rmed_s,   rmed_l_row=mets.row.rmed_l,
                res_thr=mets.thr.res,   rea_thr=mets.thr.rea,   rmed_s_thr=mets.thr.rmed_s,   rmed_l_thr=mets.thr.rmed_l,
                res_reshuf=mets.reshuf.res, rea_reshuf=mets.reshuf.rea, rmed_s_reshuf=mets.reshuf.rmed_s, rmed_l_reshuf=mets.reshuf.rmed_l,
                res_rew=mets.rew.res,   rea_rew=mets.rew.rea,   rmed_s_rew=mets.rew.rmed_s,   rmed_l_rew=mets.rew.rmed_l,
                res_ushuf=mets.ushuf.res, rea_ushuf=mets.ushuf.rea, rmed_s_ushuf=mets.ushuf.rmed_s, rmed_l_ushuf=mets.ushuf.rmed_l,
                res_rarer=mets.rarer.res, rea_rarer=mets.rarer.rea, rmed_s_rarer=mets.rarer.rmed_s, rmed_l_rarer=mets.rarer.rmed_l))
        end
    end
    df = DataFrame(vcat(bucket...))

    steps = (:row, :thr, :reshuf, :rew, :ushuf, :rarer)
    metrics = (:res, :rea, :rmed_s, :rmed_l)
    rowsS = NamedTuple[]
    for xval in sort(unique(df.x))
        sub = df[df.x .== xval, :]
        for m in metrics
            x = sub[!, Symbol(m, :_full)]
            for s in steps
                y = sub[!, Symbol(m, :_, s)]
                r2, slope, intercept = _r2_to_identity(collect(x), collect(y))
                push!(rowsS, (; kind=:mag_cv, x=xval, metric=String(m), step=String(s), r2, slope, intercept, n=nrow(sub)))
            end
        end
    end
    return df, DataFrame(rowsS)
end

# ---------- progressive palette (6 lines) ----------
# Progressive (continuous) palette: light → dark along a single colormap
function _progressive_colors(n::Int)
    cols = Makie.resample_cmap(:viridis, n + 2)  # returns a Vector of colors
    return cols[2:end-1]                         # trim extremes for nicer contrast
end

# ---------- 2×2 plot helper (four metrics, six lines for steps) ----------
function plot_predictability_2x2(summary::DataFrame; kind::Symbol, title::String)
    @assert "metric" in names(summary) && "step" in names(summary) && "x" in names(summary)
    metrics = ["res","rea","rmed_s","rmed_l"]
    # steps = ["reshuf","thr","row","rew","ushuf","rarer"]  # fixed order
    steps = ["reshuf", "rew", "ushuf"]  # fixed order
    cols = _progressive_colors(length(steps))

    fig = Figure(size=(1050, 700))
    Label(fig[0,1:2], title; fontsize=20, font=:bold, halign=:left)

    for (pi, met) in enumerate(metrics)
        ax = Axis(fig[(pi-1) ÷ 2 + 1, (pi-1) % 2 + 1];
                  xlabel = (pi ≥ 3 ? (kind==:u_cv ? "abundance CV" : kind==:mag_cv ? "magnitude CV" : "degree CV (lognormal)") : ""),
                  ylabel = (pi % 2 == 1 ? "R² vs full" : ""),
                  title  = met,
                  limits = (nothing, (-0.05, 1.05)))
        for (si, step) in enumerate(steps)
            sub = summary[(summary.metric .== met) .&& (summary.step .== step) .&& (summary.kind .== kind), :]
            isempty(sub) && continue
            sort!(sub, :x)
            lines!(ax, sub.x, sub.r2; color=cols[si], label=step)
            scatter!(ax, sub.x, sub.r2; color=cols[si], markersize=6)
        end
        if pi == 1
            axislegend(ax; position=:rb, framevisible=false, nbanks=2)
        end
    end
    display(fig)
end

# ================================
# Example minimal usage
# ================================
# -- Abundance heterogeneity sweep
uh = UHeteroOptions(; u_cv_vals=[0.2,0.4,0.8,1.2,2.0,3.0], reps=120, IS_target=0.5)
df_u, summ_u = run_predictability_vs_uhetero(uh)
plot_predictability_2x2(summ_u; kind=:u_cv, title="Predictability vs abundance heterogeneity")

# -- Degree heterogeneity sweep (lognormal)
kh = KHeteroOptions(; deg_cv_vals=[0.0,0.25,0.5,1.0,1.5,2.0], reps=120, IS_target=0.5)
df_k, summ_k = run_predictability_vs_khetero(kh)
plot_predictability_2x2(summ_k; kind=:deg_cv, title="Predictability vs degree heterogeneity")

# -- Interaction-strength heterogeneity sweep
ish = ISHeteroOptions(; mag_cv_vals=[0.0,0.3,0.6,1.0,1.5,2.0], reps=120, IS_target=0.1)
df_is, summ_is = run_predictability_vs_ishero(ish)
plot_predictability_2x2(summ_is; kind=:mag_cv, title="Predictability vs IS heterogeneity BIOMASS")
