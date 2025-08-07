# === Main Simulation Script ===
# Load all helper functions
include("Functions.jl")

"""
RunSimulations(...)

Main simulation function for sampling network structures and applying the modification ladder.

Arguments:
- S, C: total number of species and number of consumers
- conn_vals: connectance of the network
- scenarios: network topology types (i.e., :ER, :PL, :MOD)
- IS_vals: average interaction strength
- delta_vals: perturbation intensities
- margins: distance from feasibility threshold
- iterations: number of repeats per exact combination
- number_of_combinations: given the high number of combinations, sample a random subset
- tspan: time span for simulation (no need to touch)
- tpert: time of perturbation (no need to touch)
- pareto_exponents: used for power-law networks (being 1.0 very power-law and higher values less power-law)
- pareto_minimum_degrees: min degree for power-law construction
- mod_gammas: modularity tuning values for modular graphs (being 1.0 very modular and higher values less modular)

Returns:
- A results dataframe with dynamical metrics measured per iteration, for each modification step.
"""
function RunSimulations(
    S::Int=50, C::Int=20;
    conn_vals=[0.05, 0.1, 0.2],
    scenarios=[:ER, :PL, :MOD],
    IS_vals=[0.01, 0.1, 1.0, 2.0],
    delta_vals=[1.0, 3.0],
    margins=[1.0],
    number_of_combinations::Int=100,
    iterations::Int=1,
    tspan=(0.0, 500.0),
    tpert::Float64=250.0,
    pareto_exponents=[1.25,1.75,2.0,3.0,4.0,5.0],
    pareto_minimum_degrees=[1.0],
    mod_gammas = [1.0,2.0,3.0,5.0,10.0]
)
    # Number of resources
    R = S - C
    results = Vector{NamedTuple}()
    locki = ReentrantLock()

    # Sample combinations of parameters
    combos = collect(
        Iterators.product(
            conn_vals, scenarios, IS_vals,
            delta_vals, margins, 1:iterations,
            pareto_exponents, pareto_minimum_degrees,
            mod_gammas
        )
    )
    @info "Computing $(length(combos)) combinations"

    # Build solver callbacks globally
    global cb = build_callbacks(50, 1e-6)

    # Run simulations in parallel
    @threads for (conn, scen, IS, delta, marg, ite, pex, p_min_deg, mod_gamma) in
            sample(combos, min(length(combos), number_of_combinations); replace=false)

        # === Step 1: Build network and interaction matrix ===
        A = make_A(
            zeros(S,S), R, conn, scen; IS=IS,
            pareto_exponent=pex, pareto_minimum_degree=p_min_deg, mod_gamma=mod_gamma
        )

        # === Step 2: Compute collectivity metric ===
        phi = compute_collectivity(A) # Not used in the final paper results, but interesting though

        # === Step 3: Determine feasible K and equilibrium ===
        thr_sets = generate_feasible_thresholds(A, R; margins=[marg])
        isempty(thr_sets) && continue  # Skip iteration if no thresholds found
        tset = thr_sets[1]
        K    = tset.K
        u_eq = tset.u_eq

        # === Step 4: Test survivability and local stability ===
        ok, u0 = survives!(u_eq, (K,A); cb=cb)
        !ok && continue

        J = compute_jacobian_glv(u0, (K,A))
        !is_locally_stable(J) && continue

        # === Step 5: Metrics on full system ===
        S_full = count(x->x>1e-6, u0) # This must be equivalent to S by definition
        resilience_full = compute_resilience(u0, (K,A))
        reactivity_full = compute_reactivity(u0, (K,A))

        SL_full = diag(J) .|> x -> x == 0.0 ? 0.0 : -1 / x # Self-regulation Loss
        mean_SL_full = mean(SL_full) # Mean self-regulation loss
        sigma_full = sigma_over_min_d(A, J) # sigma / min(d), a proxy of diagonal-dominance

        # 5a: Press perturbation
        rt_press_vec, before_press, after_press, _ =
            simulate_press_perturbation_glv(
                u0, (K,A), tspan, tpert, delta, R;
                cb=cb
            )
        rt_press_full = mean(skipmissing(rt_press_vec))

        # 5b: Pulse perturbation
        rt_pulse_vec, before_pulse, after_pulse, _ =
            simulate_pulse_perturbation_glv(
                u0, (K,A), tspan, tpert, delta;
                cb=cb
            )
        rt_pulse_full = mean(skipmissing(rt_pulse_vec))

        rmed_full = analytical_median_return_rate(J; t=1.0)

        # === Step 6: Modification ladder ===
        after_press_S = Dict(i=>NaN for i in 1:5)
        after_pulse_S = Dict(i=>NaN for i in 1:5)
        rt_press_S    = Dict(i=>NaN for i in 1:5)
        rt_pulse_S    = Dict(i=>NaN for i in 1:5)
        S_S                   = Dict(i=>NaN for i in 1:5)
        collectivity_S        = Dict(i=>NaN for i in 1:5)
        resilience_S          = Dict(i=>NaN for i in 1:5)
        reactivity_S          = Dict(i=>NaN for i in 1:5)
        sigma_over_min_d_S    = Dict(i=>NaN for i in 1:5)
        SL_S                  = Dict(i=>Float64[] for i in 1:5)
        mean_SL_S             = Dict(i=>NaN for i in 1:5)
        rmed_S                = Dict(i=>NaN for i in 1:5)

        @info "Running ladder"
        for step in 1:5
            A_s = copy(A)

            # Step-specific simplified models
            if step == 1 # Simple rewiring
                A_s = make_A(A_s, R, conn, scen; IS=IS)
            
            elseif step == 2 # Rewiring + ↻C
                new_conn = rand()
                while abs(new_conn - conn) < 0.4
                    new_conn = rand()
                end
                A_s = make_A(A_s, R, new_conn, scen; IS=IS)
            
            elseif step == 3 # Rewiring + ↻IS
                A_s = make_A(A_s, R, conn, scen; IS=IS*10)
            
            elseif step == 4 # Rewiring + ↻C + ↻IS
                new_conn = rand()
                while abs(new_conn - conn) < 0.4
                    new_conn = rand()
                end
                A_s = make_A(A_s, R, new_conn, scen; IS=IS*10)
            
            elseif step == 5 # Rewiring + Group reassignment
                A_s = make_A(A_s, R-5, conn, scen; IS=IS)
            end

            # For step 5, turn 5 resources into consumers by setting their K to effectively 0
            if step == 5
                K[R-5:end] .= 0.01
            end

            # Recalculate equilibrium
            u_eq_s = try
                calibrate_from_K_A(K, A_s)
            catch
                continue
            end

            ok2, u0_s = survives!(u_eq_s, (K,A_s); cb=cb)

            # Compute reduced metrics
            S_S[step] = count(x->x>1e-6, u0_s)

            rt_s_press, _, after_press_s, _ =
                simulate_press_perturbation_glv(u0_s, (K,A_s), tspan, tpert, delta, R; cb=cb)
            after_press_S[step] = after_press_s
            rt_press_S[step] = mean(skipmissing(rt_s_press))

            rt_s_pulse, _, after_pulse_s, _ =
                simulate_pulse_perturbation_glv(u0_s, (K,A_s), tspan, tpert, delta; cb=cb)
            after_pulse_S[step] = after_pulse_s
            rt_pulse_S[step] = mean(skipmissing(rt_s_pulse))

            collectivity_S[step] = compute_collectivity(A_s)
            resilience_S[step] = compute_resilience(u0_s, (K,A_s); extinct_species=false)
            reactivity_S[step] = compute_reactivity(u0_s, (K,A_s); extinct_species=false)

            J_s_sub = compute_jacobian_glv(u0_s, (K,A_s))
            rmed_S[step] = analytical_median_return_rate(J_s_sub; t=1.0)
            sigma_over_min_d_S[step] = sigma_over_min_d(A_s, J_s_sub)

            SL_S[step] = diag(J_s_sub) .|> x -> x == 0.0 ? 0.0 : -1 / x
            mean_SL_S[step] = mean(SL_S[step])
        end

        # Flatten all step metrics into a single record
        step_pairs = collect(Iterators.flatten(
            ([ 
                Symbol("after_press_S$i") => after_press_S[i],
                Symbol("after_pulse_S$i") => after_pulse_S[i],
                Symbol("rt_press_S$i") => rt_press_S[i],
                Symbol("rt_pulse_S$i") => rt_pulse_S[i],
                Symbol("S_S$i") => S_S[i],
                Symbol("collectivity_S$i") => collectivity_S[i],
                Symbol("resilience_S$i") => resilience_S[i],
                Symbol("reactivity_S$i") => reactivity_S[i],
                Symbol("sigma_over_min_d_S$i") => sigma_over_min_d_S[i],
                Symbol("SL_S$i") => SL_S[i],
                Symbol("mean_SL_S$i") => mean_SL_S[i],
                Symbol("rmed_S$i") => rmed_S[i],
             ] for i in 1:5)
        ))

        # Final simulation record
        rec = (
            conn=conn, scen=scen, IS=IS, delta=delta, marg=marg, ite=ite,
            S_full=S_full,
            resilience_full=resilience_full,
            reactivity_full=reactivity_full,
            collectivity_full=phi,
            SL_full=SL_full,
            mean_SL_full=mean_SL_full,
            sigma_over_min_d_full=sigma_full,
            rmed_full=rmed_full,
            after_press_full=after_press,
            after_pulse_full=after_pulse,
            rt_press_full=rt_press_full,
            rt_pulse_full=rt_pulse_full,
            step_pairs...,
            p_final=(K,A),
            B_eq=u0,
            Beq_cv=std(u0) / mean(u0)
        )

        # Store result with thread safety
        lock(locki) do
            push!(results, rec)
        end
    end

    @info "Finished computing $number_of_combinations combinations"
    return DataFrame(results)
end

# === Run simulations for multiple S, C values ===
function run_all_glv()
    R_all = DataFrame()
    for i in [(100, 40), (50, 20), (200, 80), (300, 120)]
        a, b = i[1], i[2]
        R = checking_recalculating_demography_glv(
            a, b;
            conn_vals=0.01:0.01:1.0,
            scenarios=[:ER, :PL, :MOD],
            IS_vals=[0.01, 0.1, 1.0, 2.0],
            delta_vals=[0.1, 0.9, 1.1, 1.5, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0],
            margins=[1.0, 2.0, 3.0, 4.0, 5.0, 0.01],
            number_of_combinations=50000,
            iterations=1,
            pareto_exponents=[1.0, 1.25, 1.75, 2.0, 3.0, 4.0, 5.0],
            pareto_minimum_degrees=[5.0, 10.0, 15.0, 20.0],
            mod_gammas=[1.0,2.0,3.0,5.0,10.0]
        )
        R_all = vcat(R_all, R; cols=:union)
    end
    return R_all
end

# === Execute and save ===
R_all = run_all_glv()
serialize("checking_glv_50000perSpeciesN.jls", R_all)

# === Load (optional) ===
R = deserialize("checking_glv_50000perSpeciesN.jls")
