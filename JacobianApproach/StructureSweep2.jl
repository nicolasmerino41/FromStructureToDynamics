###############################################################
# Predictability vs heterogeneity — continuous time (rmed only)
# - 3 sweeps: abundance CV (u_cv), degree CV (deg_cv), IS CV (mag_cv)
# - Each sweep produces 6 heterogeneity levels → 2×3 plot
# - Each subplot: R² vs time curves for each step
###############################################################
using Random, Statistics, LinearAlgebra, DataFrames, Distributions
using CairoMakie
using Base.Threads

# ---------- robust R² to identity ----------
function _r2_to_identity(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == 0 && return NaN
    μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
    sst == 0 && return (ssr == 0 ? 1.0 : 0.0)
    return max(1 - ssr/sst, 0.0)
end

# ---------- RNG splitter ----------
@inline function _splitmix64(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
    z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
    return z ⊻ (z >>> 31)
end

# ---------- continuous time sweep of median return rate ----------
function _compute_rmeds(J, u; t_vals, rng)
    return [median_return_rate(J, u; t=t, perturbation=:biomass) for t in t_vals]
end

# ---------- compute all 6 steps ----------
function _compute_steps_rmed(A, u; t_vals, q_thresh=0.2, rng=Random.default_rng())
    J = jacobian(A, u)
    α = alpha_off_from(J, u)

    α_row = op_rowmean_alpha(α)
    α_thr = op_threshold_alpha(α; q=q_thresh)
    α_rsh = op_reshuffle_preserve_pairs(α; rng=rng)

    # rewiring (ER trophic preserving IS scale)
    A_rew0 = build_random_trophic_ER(size(A,1); conn=realized_connectance(A),
                                     mean_abs=realized_IS(A), mag_cv=0.60,
                                     rho_sym=0.0, rng=rng)
    βr = (b->b>0 ? realized_IS(A)/b : 1.0)(realized_IS(A_rew0))
    A_rew = βr .* A_rew0

    # abundance shuffling & rare removal
    u_sh  = u[randperm(rng, length(u))]
    # u_sh = fill(mean(u), length(u))
    u_rr = remove_rarest_species(u; p=0.9)

    J_full = J
    J_row  = build_J_from(α_row, u)
    J_thr  = build_J_from(α_thr, u)
    J_rsh  = build_J_from(α_rsh, u)
    J_rew  = jacobian(A_rew, u)
    J_ush  = build_J_from(α, u_sh)
    J_rar  = build_J_from(α, u_rr)

    return (
        full = _compute_rmeds(J_full, u; t_vals=t_vals, rng=rng),
        row  = _compute_rmeds(J_row,  u; t_vals=t_vals, rng=rng),
        thr  = _compute_rmeds(J_thr,  u; t_vals=t_vals, rng=rng),
        reshuf = _compute_rmeds(J_rsh, u; t_vals=t_vals, rng=rng),
        rew  = _compute_rmeds(J_rew,  u; t_vals=t_vals, rng=rng),
        ushuf = _compute_rmeds(J_ush, u_sh; t_vals=t_vals, rng=rng),
        rarer = _compute_rmeds(J_rar, filter(!iszero, u_rr); t_vals=t_vals, rng=rng)
    )
end

# ---------- one general runner ----------
function _run_predictability_continuous(; kind::Symbol, vals::Vector{Float64},
        S=120, conn=0.1, mean_abs=0.1, mag_cv=0.6, deg_param=0.5,
        rho_sym=0.5, u_mean=1.0, u_cv=0.6, IS_target=0.2, reps=50,
        t_vals=logspace(-2, 2, 20), seed=20251105)

    base = _splitmix64(UInt64(seed))
    bucket = [Vector{NamedTuple}() for _ in 1:nthreads()]

    Threads.@threads for idx in eachindex(vals)
        xval = vals[idx]
        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx) ⊻ UInt64(threadid())))
        local_rows = bucket[threadid()]

        for rep in 1:reps
            rng = Random.Xoshiro(rand(rng0, UInt64))

            # assign heterogeneity per sweep type
            if kind == :u_cv
                ucv = xval
                dcv, mcv = deg_param, mag_cv
            elseif kind == :deg_cv
                dcv = xval
                ucv, mcv = u_cv, mag_cv
            elseif kind == :mag_cv
                mcv = xval
                ucv, dcv = u_cv, deg_param
            else
                error("unknown kind $kind")
            end

            A0 = build_niche_trophic(S; conn=conn, mean_abs=mean_abs,
                                     mag_cv=mcv, degree_family=:lognormal,
                                     deg_param=dcv, rho_sym=rho_sym, rng=rng)
            baseIS = realized_IS(A0)
            baseIS == 0 && continue
            β = IS_target / baseIS
            A = β .* A0
            u = random_u(S; mean=u_mean, cv=ucv, rng=rng)

            mets = _compute_steps_rmed(A, u; t_vals=t_vals, rng=rng)
            for (i, t) in enumerate(t_vals)
                push!(local_rows, (; kind, xval, t,
                    full=mets.full[i], row=mets.row[i], thr=mets.thr[i],
                    reshuf=mets.reshuf[i], rew=mets.rew[i],
                    ushuf=mets.ushuf[i], rarer=mets.rarer[i]))
            end
        end
    end

    df = DataFrame(vcat(bucket...))

    # --- summarise R²(t) ---
    steps = (:row, :thr, :reshuf, :rew, :ushuf, :rarer)
    rowsS = NamedTuple[]
    for xv in sort(unique(df.xval)), t in sort(unique(df.t))
        sub = df[(df.xval .== xv) .& (df.t .== t), :]
        isempty(sub) && continue
        x = sub.full
        for s in steps
            y = sub[!, s]
            r2 = _r2_to_identity(x, y)
            push!(rowsS, (; kind, xval=xv, t, step=String(s), r2))
        end
    end
    return df, DataFrame(rowsS)
end

# ---------- plotting: 2×3 subplots per heterogeneity variable ----------
function plot_r2_vs_time(summary::DataFrame; kind::Symbol, title::String)
    vals = sort(unique(summary.xval))
    @assert length(vals) == 6 "Expected 6 heterogeneity levels for plotting"
    steps = ["reshuf"]
    cols = Makie.resample_cmap(:viridis, length(steps)+2)[2:end-1]

    fig = Figure(size=(1100, 700))
    Label(fig[0,1:3], title; fontsize=20, font=:bold, halign=:left)

    for (pi, xv) in enumerate(vals)
        ax = Axis(
            fig[(pi-1) ÷ 3 + 1, (pi-1) % 3 + 1];
            xscale = log10,
            xlabel = "time t",
            ylabel = "R² vs full",
            title = "$(kind)= $(round(xv, digits=2))",
            limits = ((nothing, nothing), (-0.05, 1.05))
        )

        for (si, step) in enumerate(steps)
            sub = summary[(summary.kind .== kind) .&&
                          (summary.xval .== xv) .&&
                          (summary.step .== step), :]
            isempty(sub) && continue
            sort!(sub, :t)
            lines!(ax, sub.t, sub.r2; color=cols[si], label=step)
        end
        if pi == 1
            axislegend(ax; position=:rb, framevisible=false, nbanks=2)
        end
    end
    display(fig)
end

# ===============================
# Example usage
# ===============================
t_vals = 10 .^ range(log10(0.01), log10(100.0); length=20)

# Abundance heterogeneity
uh_vals = [0.2,0.4,0.8,1.2,2.0,3.0]
df_u, summ_u = _run_predictability_continuous(
    ; kind=:u_cv, vals=uh_vals,
    mean_abs=0.5,
    reps=60, t_vals=t_vals
)
plot_r2_vs_time(summ_u; kind=:u_cv, title="Predictability vs abundance heterogeneity")

# Degree heterogeneity
kh_vals = [0.0,0.25,0.5,1.0,1.5,2.0]
df_k, summ_k = _run_predictability_continuous(
    ; kind=:deg_cv, vals=kh_vals,
    mean_abs=0.5,
    reps=60, t_vals=t_vals
)
plot_r2_vs_time(summ_k; kind=:deg_cv, title="Predictability vs degree heterogeneity")

# IS heterogeneity
ish_vals = [0.0,0.1,0.6,1.0,1.5,2.0]
df_is, summ_is = _run_predictability_continuous(
    ; kind=:mag_cv, vals=ish_vals,
    mean_abs=0.5,
    reps=60, t_vals=t_vals
)
plot_r2_vs_time(summ_is; kind=:mag_cv, title="Predictability vs IS heterogeneity")
