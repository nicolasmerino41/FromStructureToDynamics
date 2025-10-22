############################  STRUCTURE→PREDICTABILITY SIDE-FORK  ############################
# Assumes your core code is already loaded:
#   - build_topology, choose_equilibrium, stabilize_and_recalibrate!,
#     extract_metrics, modification_ladder!, build_callbacks, check_equilibrium
# No @view used anywhere.

# ============================== small utilities ==============================

# safe numeric extraction -> Float64 or nothing
@inline function _safe_real(x)
    x isa Real || return nothing
    xf = float(x)
    isfinite(xf) ? xf : nothing
end

# linear map
@inline lerp(a::Real, b::Real, s::Real) = (1 - s) * a + s * b

# fraction-predictable indicator (composite error threshold)
@inline predictable(err::Real, τ::Real=0.20) = (isfinite(err) && err ≤ τ)

# ============================== path configuration ==============================
Base.@kwdef struct PathConfig
    # Fixed controls (hold these to avoid confounding along the path)
    conn::Float64 = 0.10         # target connectance (held ~constant)
    blocks::Int   = 3            # number of balanced blocks
    margin::Float64 = 0.05       # stabilization margin on J
    k_consumer::Float64 = 0.0    # K for consumers
    step_eval::Int = 2           # which step encodes your "coarse info" predictor

    # Disorder → structure axes (min -> max, s ∈ [0,1])
    # Modularity of bipartite wiring
    mod_min::Float64 = 0.0
    mod_max::Float64 = 0.8

    # Degree heterogeneity (targets for generator)
    cv_cons_min::Float64 = 0.3   # consumer out-degree CV, low at s=0
    cv_cons_max::Float64 = 3.0
    cv_res_min::Float64  = 0.3   # resource in-degree CV, low at s=0
    cv_res_max::Float64  = 2.2

    # Abundance heterogeneity in the equilibrium sampler
    u_cv_res_min::Float64 = 0.15
    u_cv_res_max::Float64 = 0.8
    u_cv_cons_min::Float64 = 0.2
    u_cv_cons_max::Float64 = 1.0

    # Interaction-strength scale (IS) in generator
    IS_min::Float64 = 0.02
    IS_max::Float64 = 0.6

    # Fractions of pinned specialists (degree spikes)
    frac_spec_cons_min::Float64 = 0.0
    frac_spec_cons_max::Float64 = 0.4
    frac_spec_res_min::Float64  = 0.0
    frac_spec_res_max::Float64  = 0.2
end

# Map s -> generator & abundance parameters
function params_from_s(cfg::PathConfig, s::Float64)
    mod     = lerp(cfg.mod_min,    cfg.mod_max,    s)
    cv_cons = lerp(cfg.cv_cons_min,cfg.cv_cons_max,s)
    cv_res  = lerp(cfg.cv_res_min, cfg.cv_res_max, s)
    ucv_r   = lerp(cfg.u_cv_res_min,  cfg.u_cv_res_max,  s)
    ucv_c   = lerp(cfg.u_cv_cons_min, cfg.u_cv_cons_max, s)
    IS      = lerp(cfg.IS_min,     cfg.IS_max,     s)
    fsc     = lerp(cfg.frac_spec_cons_min, cfg.frac_spec_cons_max, s)
    fsr     = lerp(cfg.frac_spec_res_min,  cfg.frac_spec_res_max,  s)
    return (mod, cv_cons, cv_res, ucv_r, ucv_c, IS, fsc, fsr)
end

# ============================== realized-structure readouts ==============================

function realized_connectance(A::AbstractMatrix, R::Int)
    S = size(A,1); C = S - R
    E = 0
    for ic in 1:C, jr in 1:R
        E += (A[R+ic, jr] != 0.0)
    end
    return (R*C == 0) ? NaN : E / (R*C)
end

function realized_within_fraction(A::AbstractMatrix, R::Int, blocks::Int)
    S = size(A,1); C = S - R
    r_block = repeat(1:blocks, inner=ceil(Int, R/blocks))[1:R]
    c_block = repeat(1:blocks, inner=ceil(Int, C/blocks))[1:C]
    tot = 0; win = 0
    for ic in 1:C, jr in 1:R
        if A[R+ic, jr] != 0.0
            tot += 1
            win += (c_block[ic] == r_block[jr]) ? 1 : 0
        end
    end
    return tot == 0 ? NaN : win / tot
end

function degcv_cons_out(A::AbstractMatrix, R::Int)
    S = size(A,1); C = S - R
    d = Vector{Int}(undef, C)
    for ic in 1:C
        cnt = 0
        for jr in 1:R
            cnt += (A[R+ic, jr] != 0.0)
        end
        d[ic] = cnt
    end
    if isempty(d); return NaN; end
    dd = Float64.(d)
    μ = mean(dd)
    return μ == 0 ? NaN : std(dd) / μ
end

# ============================== composite error (clipped) ==============================

function err_clip_mean_row(row::NamedTuple, step::Int;
    metrics=( :resilience, :reactivity, :rt_pulse, :after_press ), clip_frac=0.10)

    vals = Float64[]
    for m in metrics
        full_sym = Symbol("$(m)_full")
        step_sym = Symbol("$(m)_S", step)
        haskey(row, full_sym) || continue
        haskey(row, step_sym) || continue

        f = _safe_real(row[full_sym]); s = _safe_real(row[step_sym])
        if isnothing(f) || isnothing(s); continue; end

        τ = max(abs(f) * clip_frac, 1e-6)       # robust per-row scale
        push!(vals, abs(s - f) / max(abs(f), τ))
    end
    return isempty(vals) ? NaN : mean(vals)
end

# ============================== build community at s ==============================

function build_structured(S::Int, R::Int, s::Float64; cfg::PathConfig=PathConfig(), rng=Random.default_rng())
    mod, cv_cons, cv_res, ucv_r, ucv_c, IS, fsc, fsr = params_from_s(cfg, s)

    # 1) topology
    A, _, _ = build_topology(S, R;
        conn = cfg.conn,
        cv_cons = cv_cons,
        cv_res  = cv_res,
        modularity = mod,
        blocks = cfg.blocks,
        IS = IS,
        rng = rng,
        frac_special_cons = fsc,
        frac_special_res  = fsr
    )

    # 2) equilibrium sampler
    u0 = choose_equilibrium(S, R;
        u_cv_res = ucv_r,
        u_cv_cons = ucv_c,
        cons_scale = 1.3,
        rng = rng
    )

    # 3) stabilize & enforce K_cons = cfg.k_consumer
    K, alpha, lambda_max = stabilize_and_recalibrate!(A, u0, R;
        margin = cfg.margin, k_cons = cfg.k_consumer
    )

    # 4) callbacks & quick check
    cb = build_callbacks(S, EXTINCTION_THRESHOLD)
    ok, _ = check_equilibrium(u0, K, A; cb = cb)
    ok || return nothing

    # realized structure
    conn_real   = realized_connectance(A, R)
    within_frac = realized_within_fraction(A, R, cfg.blocks)
    degcv_out   = degcv_cons_out(A, R)

    return (A = A, K = K, u = u0,
            realized_conn = conn_real,
            realized_mod  = within_frac,
            degcv_cons_out = degcv_out,
            alpha = alpha, lambda_max = lambda_max, cb = cb)
end

# ============================== evaluate (metrics + error) ==============================

function evaluate_predictability(S::Int, R::Int, s::Float64; cfg::PathConfig=PathConfig(),
                                 steps_to_run = [cfg.step_eval], rng=Random.default_rng())

    built = build_structured(S, R, s; cfg=cfg, rng=rng)
    built === nothing && return nothing

    # base (full) metrics
    full = extract_metrics(built.u, built.K, built.A; R=R,
        tspan=(0.0,500.0), tpert=250.0, delta=0.5, cb=built.cb,
        compute_rmed=false, plot_simulation=false
    )

    # record
    rec = Pair{Symbol,Any}[
        :S => S, :R => R, :C => S-R,
        :s => s,
        :conn_target => cfg.conn,
        :realized_conn => built.realized_conn,
        :realized_mod  => built.realized_mod,
        :deg_cv_cons_out_realized => built.degcv_cons_out,
        :alpha_full => built.alpha,
        :lambda_max_full => built.lambda_max,
        :resilience_full => full.resilience_full,
        :reactivity_full => full.reactivity_full,
        :rt_pulse_full   => full.rt_pulse_full,
        :after_press_full => full.after_press_full,
        :p_final => (built.K, built.A),
        :B_eq => built.u
    ]

    # run the chosen coarse-info step(s)
    modification_ladder!(rec, built.u, built.A, built.K, S, R;
        conn = cfg.conn, cv_cons = NaN, cv_res = NaN, modularity = NaN, blocks = cfg.blocks, IS = NaN,
        tspan = (0.0,500.0), tpert = 250.0, delta = 0.5, cb = built.cb,
        compute_rmed_steps = false, rng = rng,
        steps_to_run = steps_to_run,
        plot_simulation = false,
        preserve_pair_symmetry = true,
        consumer_k = cfg.k_consumer
    )

    row = NamedTuple(rec)

    # composite error for the main coarse-info step
    errc = err_clip_mean_row(row, first(steps_to_run))
    pred = predictable(errc, 0.20)

    # spectral gap proxy on A
    ρ = maximum(abs, eigvals(built.A))
    gapA = 1.0 - ρ

    return (;
        S = S, R = R, s = s,
        conn = built.realized_conn,
        realized_mod = built.realized_mod,
        deg_cv_cons_out_realized = built.degcv_cons_out,
        gap_A = gapA,
        resilience_full = full.resilience_full,
        reactivity_full  = full.reactivity_full,
        rt_pulse_full    = full.rt_pulse_full,
        after_press_full = full.after_press_full,
        err_clip_mean    = errc,
        predictable      = pred,
        step_eval        = first(steps_to_run)
    )
end

# ============================== threaded path scan (thread-safe RNG) ==============================

# SplitMix64 mixer to derive per-job seeds deterministically
@inline function _splitmix64(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
    z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
    return z ⊻ (z >>> 31)
end

"""
    scan_structure_path(; S=150, C=60, cfg=PathConfig(),
                          s_grid=range(0,1,length=21),
                          reps_per_s=20,
                          seed=12345,
                          max_tries_per_job=6,
                          threaded=true)

Parallel scan along s ∈ [0,1]. Each (s,rep) job gets its own Random.Xoshiro
RNG seeded from (seed, job_id). Per-thread result buffers; DF built once at the end.
"""
function scan_structure_path(; S=150, C=60, cfg::PathConfig=PathConfig(),
                               s_grid = range(0.0, 1.0; length=21),
                               reps_per_s::Int = 20,
                               seed::Int = 12345,
                               max_tries_per_job::Int = 6,
                               threaded::Bool = true)

    R = S - C

    # flatten jobs
    s_list = collect(s_grid)
    jobs_s = Vector{Float64}(undef, length(s_list) * reps_per_s)
    k = 0
    for s in s_list, _ in 1:reps_per_s
        k += 1
        jobs_s[k] = float(s)
    end
    njobs = length(jobs_s)

    # per-thread buffers
    results_per_thread = [Vector{NamedTuple}() for _ in 1:nthreads()]

    base = _splitmix64(UInt64(seed))

    if threaded && nthreads() > 1
        @threads for j in 1:njobs
            tid = threadid()

            # per-job RNG (never shared)
            jobseed = _splitmix64(base ⊻ UInt64(j))
            rng = Random.Xoshiro(jobseed)

            s   = jobs_s[j]
            local_row = nothing
            tries = 0
            while local_row === nothing && tries < max_tries_per_job
                local_row = evaluate_predictability(S, R, s; cfg=cfg, steps_to_run=[cfg.step_eval], rng=rng)
                tries += 1
                if local_row === nothing
                    jobseed = _splitmix64(jobseed ⊻ 0xD1342543DE82EF95)
                    rng = Random.Xoshiro(jobseed)
                end
            end
            local_row === nothing && continue
            push!(results_per_thread[tid], local_row)
        end
    else
        rng = Random.Xoshiro(base)
        buf = results_per_thread[1]
        for j in 1:njobs
            s   = jobs_s[j]
            local_row = nothing
            tries = 0
            while local_row === nothing && tries < max_tries_per_job
                local_row = evaluate_predictability(S, R, s; cfg=cfg, steps_to_run=[cfg.step_eval], rng=rng)
                tries += 1
                if local_row === nothing
                    newseed = _splitmix64(rand(rng, UInt64))
                    rng = Random.Xoshiro(newseed)
                end
            end
            local_row === nothing && continue
            push!(buf, local_row)
        end
    end

    return DataFrame(vcat(results_per_thread...))
end

# ============================== summaries (no plotting) ==============================

# Fraction predictable vs s
function summarize_predictability(df::DataFrame; τ::Float64=0.20, min_count::Int=5)
    # ensure boolean exists (else infer from err)
    if :predictable ∉ names(df)
        pred = Vector{Bool}(undef, nrow(df))
        for i in 1:nrow(df)
            v = _safe_real(df[i, :err_clip_mean])
            pred[i] = !(isnothing(v)) && v ≤ τ
        end
        df.predictable = pred
    end
    g = groupby(df, :s)
    s_vals = Float64[]; frac = Float64[]; n = Int[]
    for sub in g
        sv = _safe_real(first(sub.s))
        ns = nrow(sub)
        if ns ≥ min_count
            push!(s_vals, sv === nothing ? NaN : sv)
            push!(frac, mean(Bool.(sub.predictable)))
            push!(n, ns)
        end
    end
    return DataFrame(; s=s_vals, fraction_predictable=frac, n=n) |> x -> sort!(x, :s)
end

# Realized-structure sanity by s (mean values)
function structure_summary(df::DataFrame)
    g = groupby(df, :s)
    rows = NamedTuple[]
    for sub in g
        push!(rows, (;
            s = first(sub.s),
            n = nrow(sub),
            mod_mean = mean(sub.realized_mod),
            cvout_mean = mean(sub.deg_cv_cons_out_realized),
            conn_mean = mean(sub.conn),
            gapA_mean = mean(sub.gap_A)
        ))
    end
    return sort!(DataFrame(rows), :s)
end

# ============================== example run (commented) ==============================
# include("Helpers.jl"); include("Callbacks.jl")   # ensure your core is loaded
cfg = PathConfig(; conn=0.10, blocks=3, step_eval=2)
df_path = scan_structure_path(; S=150, C=60, cfg=cfg, s_grid=range(0,1,length=21), reps_per_s=20,
                               seed=7, max_tries_per_job=6, threaded=true)
summary = summarize_predictability(df_path; τ=0.20, min_count=5)
struct_sum = structure_summary(df_path)
println(summary)
println(struct_sum)
#######################################################################################
