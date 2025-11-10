using Random, Distributions, LinearAlgebra
using DataFrames
using CairoMakie

# ----------------------------------------------------------------------
# New builder: separate controls for magnitude and sign
# ----------------------------------------------------------------------
"""
    build_ER(S, conn, mean_abs, mag_cv, rho_mag, rho_sign; rng=MersenneTwister(42))

Construct a directed ER interaction matrix A (S×S) with per-direction connectance `conn`.

Magnitude law: LogNormal tuned to have mean(|A_ij|)=mean_abs and CV=mag_cv (in expectation).
`rho_mag ∈ [0, 0.99]` controls correlation between |A_ij| and |A_ji| via a Gaussian copula.
`rho_sign ∈ [0, 1]` controls antisymmetry: with probability `rho_sign` the pair has opposite signs (±);
otherwise the pair shares the same sign (++, --). Larger `rho_sign` ⇒ more predator–prey-like (±).

No diagonal, no stats, no rescaling—just A.
"""
function build_ER(S::Int, conn::Float64, mean_abs::Float64, mag_cv::Float64,
                  rho_mag::Float64, rho_sign::Float64; rng::AbstractRNG=MersenneTwister(42))

    @assert 0.0 ≤ conn ≤ 1.0
    @assert mean_abs > 0
    @assert mag_cv ≥ 0
    @assert 0.0 ≤ rho_mag < 1.0 "use < 1 for numerical stability"
    @assert 0.0 ≤ rho_sign ≤ 1.0

    # Lognormal parameters from mean and CV
    σ2 = log(1 + mag_cv^2)
    σ  = sqrt(σ2)
    μ  = log(mean_abs) - σ2/2
    LN = LogNormal(μ, σ)

    # Cholesky for magnitude copula
    Lmag = cholesky(Symmetric([1.0 rho_mag; rho_mag 1.0])).L
    stdN = Normal()

    A = zeros(Float64, S, S)
    for i in 1:S-1, j in i+1:S
        # Bernoulli per direction
        if rand(rng) < conn
            # coupled magnitudes
            z = randn(rng, 2)
            z .= Lmag * z
            m1 = quantile(LN, cdf(stdN, z[1]))
            m2 = quantile(LN, cdf(stdN, z[2]))

            # signs: antisymmetric with prob rho_sign, symmetric otherwise
            s1 = ifelse(rand(rng) < 0.5, 1.0, -1.0)
            s2 = ifelse(rand(rng) < rho_sign, -s1, s1)

            A[i,j] = s1 * m1
            A[j,i] = s2 * m2
        end
        # If the (i,j) direction is absent but (j,i) would have been present,
        # we still respect per-direction ER; draw separately:
        if rand(rng) < conn && A[j,i] == 0.0 && A[i,j] == 0.0
            # independent single-direction draw for leftover direction
            # (keeps pure directed ER; no pair coupling when only one side exists)
            m  = rand(rng, LN)
            s  = ifelse(rand(rng) < 0.5, 1.0, -1.0)
            # randomly pick which direction this single edge lands on
            if rand(rng) < 0.5
                A[i,j] = s * m
            else
                A[j,i] = s * m
            end
        end
    end
    return A
end

# ----------------------------------------------------------------------
# Core experiment runner for one axis
# ----------------------------------------------------------------------
"""
    run_axis(axis; levels, iters, t_vals, S, conn, mean_abs, mag_cv,
             rho_mag_base, rho_sign_base, u_mean, u_cv, IS_target, seed)

axis ∈ (:magnitude, :sign, :both). Produces a DataFrame with columns:
:axis, :level, :iter, :t, :r2
"""
using Base.Threads
using Random, DataFrames

function run_axis(; axis::Symbol, levels::AbstractVector, iters::Int, reps::Int=50,
                   t_vals::AbstractVector,
                   S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.5, mag_cv::Float64=0.60,
                   rho_mag_base::Float64=0.0, rho_sign_base::Float64=0.0,
                   u_mean::Float64=1.0, u_cv::Float64=0.6,
                   IS_target::Float64=0.5,
                   seed::Int=20251110)

    base = UInt(seed)
    nthreads_used = nthreads()
    buckets = [NamedTuple[] for _ in 1:nthreads_used]

    # Combine all (li, iter) pairs into a single vector for thread safety
    combos = collect(Iterators.product(1:length(levels), 1:iters))

    @threads for idx in 1:length(combos)
        li, iter = combos[idx]
        lvl = levels[li]
        tid = threadid()
        local_rows = buckets[tid]

        # Thread-local RNG: independent stream per (li, iter, thread)
        rng_iter = Random.Xoshiro(base ⊻ UInt(li*7919) ⊻ UInt(iter*104729) ⊻ UInt(tid*4099))

        # Choose parameters per axis
        rho_mag  = axis === :magnitude ? lvl :
                   axis === :both      ? lvl : rho_mag_base
        rho_sign = axis === :sign      ? lvl :
                   axis === :both      ? lvl : rho_sign_base

        # Loop over t-values
        for t in t_vals
            r_full = Float64[]; r_rew = Float64[]
            for rep in 1:reps
                rng = Random.Xoshiro(rand(rng_iter, UInt))

                A0 = build_ER(S, conn, mean_abs, mag_cv, rho_mag, rho_sign; rng=rng)
                A1 = build_ER(S, conn, mean_abs, mag_cv, rho_mag, rho_sign; rng=rng)

                is0 = realized_IS(A0); is1 = realized_IS(A1)
                (is0 == 0 || is1 == 0) && continue
                β0 = IS_target / is0; β1 = IS_target / is1
                A0 .*= β0; A1 .*= β1

                u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

                r0 = median_return_rate(jacobian(A0, u), u; t=t, perturbation=:biomass)
                r1 = median_return_rate(jacobian(A1, u), u; t=t, perturbation=:biomass)

                if isfinite(r0) && isfinite(r1)
                    push!(r_full, r0); push!(r_rew, r1)
                end
            end

            r2 = r2_to_identity(r_full, r_rew)
            if !isfinite(r2) || r2 < 0; r2 = 0.0; end

            push!(local_rows, (; axis, level=lvl, iter, t, r2, n=length(r_full)))
        end
        buckets[tid] = local_rows
    end

    return DataFrame(vcat(buckets...))
end

# ----------------------------------------------------------------------
# Plotting: 4×3 grid per axis (rows=levels in order, cols=iterations)
# ----------------------------------------------------------------------
function plot_axis_grid(df::DataFrame, axis::Symbol; levels, iters, title::String)
    fig = Figure(size=(1100, 900))
    Label(fig[0, 1:3], title; fontsize=20, font=:bold, halign=:left)

    # Ensure deterministic ordering
    lvls  = collect(levels)
    itids = 1:iters

    for (ri, lvl) in enumerate(lvls)
        for (ci, it) in enumerate(itids)
            ax = Axis(fig[ri, ci];
                      xscale=log10,
                      xlabel="t",
                      ylabel=(ci == 1 ? "R²" : ""),
                      title="lvl=$(round(lvl,digits=2)) • iter=$it",
                      limits=((minimum(df.t), maximum(df.t)), (-0.05, 1.05)))
            sub = df[(df.axis .== axis) .& (df.level .== lvl) .& (df.iter .== it), :]
            isempty(sub) && continue
            sort!(sub, :t)
            lines!(ax, sub.t, sub.r2, linewidth=2)
            scatter!(ax, sub.t, sub.r2)
        end
    end
    display(fig)
end

# ----------------------------------------------------------------------
# Driver: three experiments (magnitude-only, sign-only, both)
# ----------------------------------------------------------------------
# Correlation/antisymmetry ladder (4 steps)
levels = [0.00, 0.33, 0.66, 0.99]
iters  = 3
reps   = 50
t_vals = 10 .^ range(-2, 2; length=20)

# Baselines held fixed while each axis is varied
rho_mag_base  = 0.0
rho_sign_base = 0.0

# Shared parameters (edit to taste)
S = 120; conn = 0.10; mean_abs = 0.5; mag_cv = 0.60
u_mean = 1.0; u_cv = 0.6
IS_target = 0.5
seed = 20251110

df_mag = run_axis(
    ; axis=:magnitude, levels=levels, iters=iters, t_vals=t_vals,
    S=S, conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
    rho_mag_base=rho_mag_base, rho_sign_base=rho_sign_base,
    u_mean=u_mean, u_cv=u_cv, IS_target=IS_target, seed=seed,
    reps=reps
)

df_sign = run_axis(
    ; axis=:sign, levels=levels, iters=iters, t_vals=t_vals,
    S=S, conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
    rho_mag_base=rho_mag_base, rho_sign_base=rho_sign_base,
    u_mean=u_mean, u_cv=u_cv, IS_target=IS_target, seed=seed,
    reps=reps
)

df_both = run_axis(
    ; axis=:both, levels=levels, iters=iters, t_vals=t_vals,
    S=S, conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
    rho_mag_base=rho_mag_base, rho_sign_base=rho_sign_base,
    u_mean=u_mean, u_cv=u_cv, IS_target=IS_target, seed=seed,
    reps=reps
)

# Three grids: (magnitude, sign, both)
fig_mag  = plot_axis_grid(df_mag,  :magnitude; levels=levels, iters=iters,
                          title="Grid 1 — Magnitude correlation only")
fig_sign = plot_axis_grid(df_sign, :sign;      levels=levels, iters=iters,
                          title="Grid 2 — Sign antisymmetry only")
fig_both = plot_axis_grid(df_both, :both;      levels=levels, iters=iters,
                          title="Grid 3 — Both increased together")
