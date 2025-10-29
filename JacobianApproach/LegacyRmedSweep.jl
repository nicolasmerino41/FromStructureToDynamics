###################### run_rmed_time_sweep.jl ######################
using Random, Statistics, LinearAlgebra, DataFrames, Random, Printf, Distributions
using Base.Threads

# ---- robust R² vs identity (same policy you wanted) ----
function r2_to_identity(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == 0 && return (NaN, NaN, NaN)
    μy = mean(y)
    sst = sum((y .- μy).^2)
    ssr = sum((y .- x).^2)
    if sst == 0
        return ssr == 0 ? (1.0, 1.0, 0.0) : (0.0, NaN, NaN)
    end
    r2 = 1 - ssr/sst
    β = [x ones(n)] \ y  # slope, intercept
    return (r2, β[1], β[2])
end

# ---- steps (minimal) ----
function op_rowmean_alpha(α)
    S = size(α,1); out = zeros(eltype(α), S, S)
    for i in 1:S
        nz = [abs(α[i,j]) for j in 1:S if i!=j && α[i,j]!=0.0]
        isempty(nz) && continue
        mi = mean(nz)
        for j in 1:S
            if i!=j && α[i,j]!=0.0; out[i,j] = sign(α[i,j]) * mi; end
        end
    end
    out
end

function op_reshuffle_alpha(α; rng=Random.default_rng())
    S = size(α,1); out = zeros(eltype(α), S, S)
    pairs = [(i,j) for i in 1:S-1 for j in i+1:S if (α[i,j]!=0.0 || α[j,i]!=0.0)]
    vals  = [(α[i,j], α[j,i]) for (i,j) in pairs]
    perm  = randperm(rng, length(pairs))
    for (k,(i,j)) in enumerate(pairs); out[i,j], out[j,i] = vals[perm[k]]; end
    for i in 1:S; out[i,i] = α[i,i]; end
    out
end

function op_threshold_alpha(α; q=0.20)
    mags = [abs(α[i,j]) for i in 1:size(α,1), j in 1:size(α,2) if i!=j && α[i,j]!=0.0]
    τ = isempty(mags) ? 0.0 : quantile(mags, q)
    S = size(α,1); out = zeros(eltype(α), S, S)
    for i in 1:S, j in 1:S
        if i!=j && abs(α[i,j]) >= τ; out[i,j] = α[i,j]; end
    end
    out
end

reshuffle_u(u; rng=Random.default_rng()) = u[randperm(rng, length(u))]

# ---- ER trophic rewire (structure-free) ----
function build_random_trophic_ER(S; conn=0.10, mean_abs=0.10, mag_cv=0.60, rho_sym=0.0, rng=Random.default_rng())
    A = zeros(Float64, S, S)
    σm = sqrt(log(1 + mag_cv^2)); μm = log(mean_abs) - σm^2/2
    pairs = [(i,j) for i in 1:S-1 for j in i+1:S]
    K = clamp(round(Int, conn * length(pairs)), 0, length(pairs))
    sel = sample(rng, pairs, K; replace=false)
    for (i,j) in sel
        m1 = rand(rng, LogNormal(μm, σm))
        m2 = rho_sym*m1 + (1-rho_sym)*rand(rng, LogNormal(μm, σm))
        if rand(rng) < 0.5
            A[i,j] =  m1; A[j,i] = -m2
        else
            A[i,j] = -m1; A[j,i] =  m2
        end
    end
    return A
end

# ---- builder dispatch (niche preferred if you have it) ----
function _build_trophic(S; conn, mean_abs, mag_cv, degree_family, deg_param, rho_sym, rng)
    if isdefined(@__MODULE__, :build_niche_trophic)
        return build_niche_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                   degree_family=degree_family, deg_param=deg_param,
                                   rho_sym=rho_sym, rng=rng)
    else
        return build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                    degree_family=degree_family, deg_param=deg_param,
                                    rho_sym=rho_sym, rng=rng)
    end
end

# ---- main sweep (no shrink) ----
"""
run_rmed_time_sweep(; modes=[:TR],
    S_vals=[120], conn_vals=0.05:0.05:0.30,
    mean_abs_vals=[0.5, 1.0, 2.0], mag_cv_vals=[0.01, 0.1, 0.5, 1.0, 2.0],
    u_mean_vals=[1.0], u_cv_vals=[0.3,0.5,0.8,1.0,2.0,3.0],
    degree_families=[:uniform,:lognormal,:pareto],
    deg_cv_vals=[0.0,0.5,1.0,2.0], deg_pl_alphas=[1.2,1.5,2.0,3.0],
    rho_sym_vals=[0.5],
    IS_targets=collect(0.02:0.02:0.20),
    t_grid = 10 .^ range(log10(0.01), log10(10.0); length=16),
    reps_per_combo=2, seed=42, number_of_combinations=500)

Returns (df_raw, df_summary).
- df_raw: one row per (combo, rep, t) with r_full and r_<step>.
- df_summary: per combo & t, R² / slope / intercept of each step vs FULL.
"""
function run_rmed_time_sweep(; modes=[:TR],
    S_vals=[120], conn_vals=0.05:0.05:0.30,
    mean_abs_vals=[0.5, 1.0, 2.0], mag_cv_vals=[0.01, 0.1, 0.5, 1.0, 2.0],
    u_mean_vals=[1.0], u_cv_vals=[0.3,0.5,0.8,1.0,2.0,3.0],
    degree_families=[:uniform,:lognormal,:pareto],
    deg_cv_vals=[0.0,0.5,1.0,2.0], deg_pl_alphas=[1.2,1.5,2.0,3.0],
    rho_sym_vals=[0.5],
    IS_targets=collect(0.02:0.02:0.20),
    t_grid = 10 .^ range(log10(0.01), log10(10.0); length=16),
    reps_per_combo=2, seed=42, number_of_combinations=500)

    # --- local helpers kept INSIDE for thread-safety & portability ---
    # robust R^2 to identity with OLS slope/intercept; no spurious 1s
    local function _r2_to_identity(x::AbstractVector, y::AbstractVector)
        n = length(x)
        n == 0 && return (NaN, NaN, NaN)
        μy = mean(y)
        sst = sum((y .- μy).^2)
        ssr = sum((y .- x).^2)
        if sst == 0
            return ssr == 0 ? (1.0, 1.0, 0.0) : (0.0, NaN, NaN)
        end
        r2 = 1 - ssr/sst
        β = [x ones(n)] \ y
        return (r2, β[1], β[2])
    end

    # deterministic per-thread RNG seeding
    local function _splitmix64(x::UInt64)
        x += 0x9E3779B97F4A7C15
        z = x
        z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
        z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
        return z ⊻ (z >>> 31)
    end

    # degree-family expansion
    deg_specs = Tuple{Symbol,Float64}[]
    for fam in degree_families
        if fam === :uniform
            push!(deg_specs, (:uniform, 0.0))
        elseif fam === :lognormal
            append!(deg_specs, ((:lognormal, x) for x in deg_cv_vals))
        elseif fam === :pareto
            append!(deg_specs, ((:pareto, a) for a in deg_pl_alphas))
        end
    end

    combos = collect(Iterators.product(
        modes, S_vals, conn_vals, mean_abs_vals, mag_cv_vals,
        u_mean_vals, u_cv_vals, deg_specs, rho_sym_vals, IS_targets,
        1:reps_per_combo
    ))

    sel = (length(combos) > number_of_combinations) ?
          sample(combos, number_of_combinations; replace=false) : combos

    # --- threaded accumulation: per-thread buckets (no shared mutation) ---
    base = UInt64(seed)
    buckets = [Vector{NamedTuple}() for _ in 1:nthreads()]
    buckets_sum = [Vector{NamedTuple}() for _ in 1:nthreads()]
    println("Computing $(number_of_combinations) of $(length(combos)) combinations")

    Threads.@threads for idx in eachindex(sel)
        (mode, S, conn, mean_abs, mag_cv, u_mean, u_cv, (deg_fam, deg_param), rho_sym, IS_tgt, rep) = sel[idx]

        # thread-local RNG chain
        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx) ⊻ UInt64(threadid())))
        # preallocate local push! closures
        local_rows = buckets[threadid()]
        local_summ = buckets_sum[threadid()]

        rng = Random.Xoshiro(rand(rng0, UInt64))

        # builder dispatch: prefer user-defined build_niche_trophic if available
        A0 = if isdefined(@__MODULE__, :build_niche_trophic)
            build_niche_trophic(
                S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                degree_family=deg_fam, deg_param=deg_param,
                rho_sym=rho_sym, rng=rng
            )
        else
            build_random_trophic(
                S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                degree_family=deg_fam, deg_param=deg_param,
                rho_sym=rho_sym, rng=rng
            )
        end

        baseIS = realized_IS(A0)
        baseIS == 0 && continue
        β = IS_tgt / baseIS
        A = β .* A0
        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

        # full J and α
        J = jacobian(A,u)
        α = alpha_off_from(J,u)

        # α-steps
        α_row = op_rowmean_alpha(α)
        α_thr = op_threshold_alpha(α; q=0.20)
        α_rsh = op_reshuffle_alpha(α; rng=rng)

        # structure-free rewiring: ER trophic draw, rescaled to same IS
        A_rew0 = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                            rho_sym=rho_sym, rng=rng)
        βr = (b->b>0 ? IS_tgt/b : 1.0)(realized_IS(A_rew0))
        J_rew = jacobian(βr .* A_rew0, u)

        # abundance edits
        u_sh = u[randperm(rng, length(u))]
        # u_sh = fill(mean(u), S)
        u_rr = remove_rarest_species(u; p=0.10)

        # Js for α-steps / u-steps
        J_row = build_J_from(α_row, u)
        J_thr = build_J_from(α_thr, u)
        J_rsh = build_J_from(α_rsh, u)
        J_ush = build_J_from(α, u_sh)
        J_rar = build_J_from(α, u_rr)

        # collect per-t raw rows
        for t in t_grid
            r_full = median_return_rate(J, u; t=t, perturbation=:biomass)
            push!(local_rows, (;
                mode, S, conn, mean_abs, mag_cv, u_mean, u_cv,
                degree_family=deg_fam, degree_param=deg_param, rho_sym,
                IS_target=IS_tgt, rep, t,
                r_full,
                r_row    = median_return_rate(J_row,  u;    t=t, perturbation=:biomass),
                r_thr    = median_return_rate(J_thr,  u;    t=t, perturbation=:biomass),
                r_reshuf = median_return_rate(J_rsh,  u;    t=t, perturbation=:biomass),
                r_rew    = median_return_rate(J_rew,  u;    t=t, perturbation=:biomass),
                r_ushuf  = median_return_rate(J_ush,  u_sh; t=t, perturbation=:biomass),
                r_rarer  = median_return_rate(J_rar,  filter(!iszero, u_rr); t=t, perturbation=:biomass)
            ))
        end
    end

    df_raw = DataFrame(vcat(buckets...))

    # --- per-combo & t summaries (thread-safe by grouping on the final df) ---
    steps = [:row,:thr,:reshuf,:rew,:ushuf,:rarer]
    # --- replace your gcols and summary loop with this ---
    # 1) choose grouping keys that give you good sample sizes
    #    (feel free to add :mode if you want one curve per mode)
    gcols = [:t]  # or [:mode, :t] if you run TR and NT together

    rowsS = Vector{NamedTuple}()
    for sub in groupby(df_raw, gcols)
        x_full = sub.r_full

        # safety: skip tiny/degenerate groups
        n = length(x_full)
        if n < 4 || var(x_full) ≤ eps()
            continue
        end

        for s in (:row, :thr, :reshuf, :rew, :ushuf, :rarer)
            y = sub[!, Symbol(:r_, s)]

            # skip groups with ~zero variance in y (they cause the “0” spikes)
            if var(y) ≤ eps()
                continue
            end

            r2, slope, intercept = r2_to_identity(collect(x_full), collect(y))
            r2 = max(r2, 0.0)  # never negative in the plot

            # keep the grouping columns only
            base = NamedTuple{Tuple(gcols)}(sub[1, gcols])
            push!(rowsS, (; base..., step=String(s), r2, slope, intercept, n))
        end
    end

    df_summary = DataFrame(rowsS)


    return df_raw, df_summary
end

@time df_raw, df_sum = run_rmed_time_sweep(
    ; modes=[:TR],
    S_vals=[120],
    conn_vals=0.05:0.05:0.30,
    mean_abs_vals=[1.0],
    mag_cv_vals=[0.01, 0.1, 0.5, 1.0, 2.0],
    u_mean_vals=[1.0],
    u_cv_vals=[0.3,0.5,0.8,1.0,2.0,3.0],
    degree_families=[:uniform,:lognormal,:pareto],
    deg_cv_vals=[0.0,0.5,1.0,2.0],
    deg_pl_alphas=[1.2, 1.5, 2.0, 3.0],
    rho_sym_vals=[0.5],
    # ← REAL multi-IS set:
    IS_targets=[0.5, 1.0, 2.0],
    t_grid=range(0.01, 10.0; length=9),
    reps_per_combo=2,        # gives healthy n
    seed=42, number_of_combinations=1000
)
# ----------------------------- quick plot (optional) ----------------------------
function plot_rmed_time_summary(df)
    fig = Figure(size=(800, 450))
    ax = Axis(fig[1,1]; xscale=log10, xlabel="t", ylabel="R² (vs full)",
              title="Predictability of r̃med vs t")

    for s in unique(df.step)
        sub = filter(:step => ==(s), df)
        sub = filter(:r2 => x -> isfinite(x), sub)
        isempty(sub) && continue
        sort!(sub, :t)
        lines!(ax, sub.t, sub.r2; label=s)
    end

    axislegend(ax; position=:lb)
    display(fig)
end

plot_rmed_time_summary(df_sum)