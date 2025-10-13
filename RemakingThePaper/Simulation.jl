###############
# Dependencies
###############
begin
    using Random, LinearAlgebra, Statistics
    using Distributions, StatsBase
    using DataFrames, Serialization
    using CairoMakie
    using Base.Threads
    using DifferentialEquations, SparseArrays
end
include("Helpers.jl")
include("Callbacks.jl")
const CALLBACKS = load_callback_cache("RemakingThePaper/callbacks_cache.jls")

# ---------------------------------------------------------
# Threaded RunSimulations (returns DataFrame with Symbol keys)
# ---------------------------------------------------------
function RunSimulations(;
    S::Int=120, C::Int=60,
    conn_vals=0.02:0.02:0.30,
    IS_vals=[0.05, 0.1, 0.2, 0.4],
    delta_vals=[0.5, 1.0, 2.0],
    margins=[0.05, 0.1],
    cv_cons_vals=[0.8, 1.5, 2.5, 3.0],
    cv_res_vals=[0.6, 1.0, 1.8, 2.2],
    modularity_vals=[0.0, 0.3, 0.6, 0.8],
    blocks_vals=[2,3,4],
    number_of_combinations::Int=200,
    iterations::Int=1,
    tspan=(0.0,500.0),
    tpert::Float64=0.0,
    seed::Int=123,
    abstol::Float64=1e-6,
    reltol::Float64=1e-6,
    saveat = range(tspan[1], tspan[2], length=201),
    compute_rmed_full::Bool=false,
    compute_rmed_steps::Bool=false
)
    rng_global = MersenneTwister(seed)
    R = S - C

    # Get the cached callback
    cb = get_callback(S, EXTINCTION_THRESHOLD; cache=CALLBACKS, path=nothing)

    # --- Helpers for realized structure ---
    _connectance(A, R) = begin
        S = size(A,1)
        C = S - R
        E = 0
        @inbounds for ic in 1:C, jr in 1:R
            E += (A[R+ic, jr] != 0.0)
        end
        (R*C == 0) ? NaN : E/(R*C)
    end

    _degcv_cons_out(A, R) = begin
        C = size(A,1) - R
        deg = Vector{Int}(undef, C)
        @inbounds for ic in 1:C
            deg[ic] = count(!iszero, A[R+ic, 1:R])
        end
        (isempty(deg) || mean(deg)==0) ? NaN : std(deg)/mean(deg)
    end

    _degcv_res_in(A, R) = begin
        S = size(A,1)
        deg = Vector{Int}(undef, R)
        @inbounds for jr in 1:R
            deg[jr] = count(!iszero, A[(R+1):S, jr])
        end
        (isempty(deg) || mean(deg)==0) ? NaN : std(deg)/mean(deg)
    end

    _within_fraction(A, R, blocks) = begin
        S = size(A,1)
        C = S - R
        r_block = repeat(1:blocks, inner=ceil(Int, R/blocks))[1:R]
        c_block = repeat(1:blocks, inner=ceil(Int, C/blocks))[1:C]
        tot = 0
        win = 0
        @inbounds for ic in 1:C, jr in 1:R
            if A[R+ic, jr] != 0.0
                tot += 1
                win += (c_block[ic] == r_block[jr]) ? 1 : 0
            end
        end
        tot == 0 ? NaN : win/tot
    end

    # --- Parameter combinations ---
    combos = collect(Iterators.product(conn_vals, IS_vals, delta_vals, margins,
                                       cv_cons_vals, cv_res_vals,
                                       modularity_vals, blocks_vals, 1:iterations))
    @info "Computing $(number_of_combinations) of $(length(combos)) combinations"
    combos = sample(rng_global, combos, min(length(combos), number_of_combinations); replace=false)

    results_per_thread = [Vector{NamedTuple}() for _ in 1:Threads.nthreads()]

    @threads for idx in eachindex(combos)
        (conn_tgt, IS, delta, margin,
         cv_cons, cv_res,
         target_mod, blocks, _) = combos[idx]

        rng_local = MersenneTwister(seed + threadid()*10_000 + idx)

        # Build community
        A0, _, _ = build_topology(S, R; conn=conn_tgt, cv_cons=cv_cons, cv_res=cv_res,
                                  modularity=target_mod, blocks=blocks, IS=IS, rng=rng_local)

        u0 = choose_equilibrium(S, R; u_mean=1.0, u_cv_res=0.5, u_cv_cons=0.7,
                                cons_scale=1.3, rng=rng_local)

        K0, alpha, lambda_val = stabilize_and_recalibrate!(A0, u0, R; margin=margin)

        ok, diag = check_equilibrium(u0, K0, A0; cb=cb)

        if !ok
            @info "iteration $idx failed to stabilize, skipping"
            continue
        end

        # Realized structure
        conn_real = _connectance(A0, R)
        realized_mod = _within_fraction(A0, R, blocks)
        degcv_cons_out_real = _degcv_cons_out(A0, R)
        degcv_res_in_real  = _degcv_res_in(A0, R)

        # Full metrics
        full = extract_metrics(u0, K0, A0; R=R, tspan=tspan, tpert=tpert, delta=delta,
                               cb=cb, abstol=abstol, reltol=reltol, saveat=saveat,
                               compute_rmed=compute_rmed_full)

        rec = Pair{Symbol,Any}[
            :scen => :CL,
            :S => S, :R => R, :C => S - R,
            :conn_target => conn_tgt,
            :conn => conn_real,
            :target_mod => target_mod,
            :realized_mod => realized_mod,
            :deg_cv_cons_out_realized => degcv_cons_out_real,
            :deg_cv_res_in_realized => degcv_res_in_real,
            :cv_cons => cv_cons, :cv_res => cv_res,
            :blocks => blocks,
            :IS => IS, :delta => delta, :marg => margin,
            :alpha_full => alpha, :lambda_max_full => lambda_val,
            :S_full => full.S_full,
            :resilience_full => full.resilience_full,
            :reactivity_full => full.reactivity_full,
            :collectivity_full => full.collectivity_full,
            :SL_full => full.SL_full,
            :mean_SL_full => full.mean_SL_full,
            :sigma_over_min_d_full => full.sigma_over_min_d_full,
            :rmed_full => full.rmed_full,
            :after_press_full => full.after_press_full,
            :after_pulse_full => full.after_pulse_full,
            :rt_press_full => full.rt_press_full,
            :rt_pulse_full => full.rt_pulse_full,
            :p_final => (K0, A0),
            :B_eq => u0, :Beq_cv => std(u0)/mean(u0)
        ]

        modification_ladder!(rec, u0, A0, K0, S, R;
            conn=conn_tgt, cv_cons=cv_cons, cv_res=cv_res,
            modularity=target_mod, blocks=blocks, IS=IS,
            tspan=tspan, tpert=tpert, delta=delta,
            cb=cb, abstol=abstol, reltol=reltol, saveat=saveat,
            compute_rmed_steps=compute_rmed_steps, rng=rng_local)

        push!(results_per_thread[threadid()], NamedTuple(rec))
    end

    results_all = vcat(results_per_thread...)
    results_all = [NamedTuple{Tuple(Symbol.(keys(r)))}(values(r)) for r in results_all]
    return DataFrame(results_all)
end

# ---------------------------------------------------------
# Sequential wrapper
# ---------------------------------------------------------
"""
    RunAllSimulations(;
        S_C_combinations = [(50,20), (75,30), (100,40), (125,50), (150,60),
                            (175,70), (200,80), (225,90), (250,100), (275,110), (300,120)],
        number_of_combinations_per_pair::Int = 18000
    ) -> DataFrame

Sequential wrapper over RunSimulations. Explicit parameters to RunSimulations
are set inside this function; no kwargs are forwarded. Parallelism (if any)
is handled internally by RunSimulations for its sampled combos.
"""
function RunAllSimulations(;
    S_C_combinations = [(50,20), (75,30), (100,40), (125,50), (150,60),
                        (175,70), (200,80), (225,90), (250,100), (275,110), (300,120)],
    number_of_combinations_per_pair::Int = 200
)
    all_df = DataFrame()
    base_seed = 123  # deterministic but distinct seed per (S,C) pair

    for (i, (S, C)) in enumerate(S_C_combinations)
        df = RunSimulations(;
            S = S,
            C = C,
            conn_vals         = 0.005:0.01:0.30,
            IS_vals           = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            delta_vals        = range(0.01, 0.99, length=20),
            margins           = [0.0001, 0.01, 0.05, 0.1],
            cv_cons_vals      = [0.8, 1.5, 2.5, 3.0],
            cv_res_vals       = [0.6, 1.0, 1.8, 2.2],
            modularity_vals   = [0.0, 0.3, 0.6, 0.8],
            blocks_vals       = [2, 3, 4],
            number_of_combinations = number_of_combinations_per_pair,
            iterations        = 1,
            tspan             = (0.0, 500.0),
            tpert             = 250.0,
            seed              = base_seed + 10_000 * i
        )
        all_df = vcat(all_df, df; cols=:union)
        @info "Finished computing $(nrow(df)) combinations for S=$S, C=$C"
    end

    return all_df
end

# ---------------------------------------------------------
# Execute and save
# ---------------------------------------------------------
sim_results = RunAllSimulations(;
    S_C_combinations = [(50,20), (75,30), (100,40), (125,50), (150,60),
                        (175,70), (200,80), (225,90), (250,100), (275,110), (300,120)],
    number_of_combinations_per_pair = 2
)
@time sim_results = RunAllSimulations(;
    S_C_combinations = [(50,20), (75,30), (100,40), (125,50), (150,60),
                        (175,70), (200,80), (225,90), (250,100), (275,110), (300,120)],
    number_of_combinations_per_pair = 1000
)

# serialize("Outputs/sim_results1000.jls", sim_results)
sim_results = deserialize("Outputs/sim_results1000.jls")