# ---------------------------------------------------------
# Core gLV pieces
# ---------------------------------------------------------
const EXTINCTION_THRESHOLD = 1e-6

# Generalized Lotka-Volterra right-hand side
function gLV_rhs!(du, u, p, t)
    K, A = p
    Au = A * u
    @inbounds for i in eachindex(u)
        du[i] = u[i] * (K[i] - u[i] + Au[i])
    end
end

# Jacobian evaluated at equilibrium u*
function jacobian_at_equilibrium(A::AbstractMatrix, u::AbstractVector)
    return Diagonal(u) * (A - I)
end

# Largest absolute eigenvalue of A (collectivity)
function compute_collectivity(A::AbstractMatrix)
    return maximum(abs, eigvals(A))
end

# Ratio of standard deviation of off-diagonal A to min diagonal damping in J
function sigma_over_min_d(A::AbstractMatrix, J::AbstractMatrix)
    d = -diag(J)
    if isempty(d)
        return NaN
    end
    min_d = minimum(d)
    offs = Float64[]
    @inbounds for i in 1:size(A, 1), j in 1:size(A, 1)
        if i != j && A[i, j] != 0.0
            push!(offs, A[i, j])
        end
    end
    if isempty(offs)
        return NaN
    end
    sigma_val = std(offs)
    return sigma_val / min_d
end

# Exact species return rates -> median
function analytical_species_return_rates(J::AbstractMatrix; t::Real = 0.01)
    if any(!isfinite, J)
        return fill(NaN, size(J, 1))
    end
    E = exp(t * J)
    M = E * transpose(E)
    rates = similar(diag(J), Float64)
    @inbounds for i in 1:size(J, 1)
        rates[i] = (M[i, i] - 1.0) / (2 * t)
    end
    return -rates
end

function analytical_median_return_rate(J::AbstractMatrix; t::Real = 0.01)
    return median(analytical_species_return_rates(J; t = t))
end

# ---------------------------------------------------------
# Press / Pulse perturbations: extinction threshold callbacks
# ---------------------------------------------------------
function build_callbacks(S::Int, EXTINCTION_THRESHOLD::Float64)
    # A) Continuous threshold-based extinctions
    callbacks = []

    # 1) Always ensure positivity
    push!(callbacks, PositiveDomain())

    # 2) Herbivores: set to zero if below EXTINCTION_THRESHOLD
    for x in 1:S
        function threshold_condition(u, t, integrator)
            # event if u[x] < EXTINCTION_THRESHOLD
            return u[x] - EXTINCTION_THRESHOLD
        end
        function threshold_affect!(integrator)
            integrator.u[x] = 0.0
        end
        push!(callbacks, ContinuousCallback(threshold_condition, threshold_affect!))
    end

    # Build callback set (no forced extinction triggers)
    cb_no_trigger = CallbackSet(callbacks...)

    return cb_no_trigger
end

# ---------------------------------------------------------
# FAST single-phase PRESS (switch K at t=0; one solve)
# ---------------------------------------------------------
function simulate_press_perturbation_glv(
    u0, p, tspan, _t_perturb, delta, R;
    solver = Tsit5(),
    cb = nothing,
    abstol = 1e-6,
    reltol = 1e-6,
    saveat = range(tspan[1], tspan[2], length = 201),  # approx. 200 samples
    dense = false,
    save_everystep = false,
    save_start = false,
    save_end = true
)
    before_persistence = count(>(EXTINCTION_THRESHOLD), u0) / length(u0)

    K, A = p
    K_press = vcat(K[1:R] .* (1 .- delta), K[R + 1:end])
    p2 = (K_press, A)

    prob = ODEProblem(gLV_rhs!, u0, (tspan[1], tspan[2]), p2)
    sol = solve(prob, solver;
        callback = cb,
        abstol = abstol,
        reltol = reltol,
        saveat = saveat,
        dense = dense,
        save_everystep = save_everystep,
        save_start = save_start,
        save_end = save_end
    )

    post = sol.u[end]
    after_persistence = count(>(EXTINCTION_THRESHOLD), post) / length(post)

    # Return times: first time each species is within 10% of final
    n = length(post)
    rts = fill(Float64(NaN), n)
    @inbounds for i in 1:n
        target = post[i]
        for (t, u) in zip(sol.t, sol.u)
            if abs(u[i] - target) / (abs(target) + 1e-8) < 0.10
                rts[i] = t - tspan[1]
                break
            end
        end
    end

    return rts, before_persistence, after_persistence, post
end


# ---------------------------------------------------------
# FAST single-phase PULSE (start from pulsed state at t=0)
# ---------------------------------------------------------
function simulate_pulse_perturbation_glv(
    u0, p, tspan, _t_pulse, delta;
    solver = Tsit5(),
    cb = nothing,
    abstol = 1e-6,
    reltol = 1e-6,
    saveat = range(tspan[1], tspan[2], length = 201),
    dense = false,
    save_everystep = false,
    save_start = false,
    save_end = true
)
    before_persistence = count(>(EXTINCTION_THRESHOLD), u0) / length(u0)
    pulsed = u0 .* (1 .- delta)

    prob = ODEProblem(gLV_rhs!, pulsed, (tspan[1], tspan[2]), p)
    sol = solve(prob, solver;
        callback = cb,
        abstol = abstol,
        reltol = reltol,
        saveat = saveat,
        dense = dense,
        save_everystep = save_everystep,
        save_start = save_start,
        save_end = save_end
    )

    post = sol.u[end]
    after_persistence = count(>(EXTINCTION_THRESHOLD), post) / length(post)

    n = length(post)
    rts = fill(Float64(NaN), n)
    @inbounds for i in 1:n
        target = post[i]
        for (t, u) in zip(sol.t, sol.u)
            if abs(u[i] - target) / (abs(target) + 1e-8) < 0.10
                rts[i] = t - tspan[1]
                break
            end
        end
    end

    return rts, before_persistence, after_persistence, post
end

# ---------------------------------------------------------
# Builder helpers (bipartite Chung-Lu, u*, stabilization)
# ---------------------------------------------------------

balanced_blocks(n::Int, blocks::Int) = repeat(1:blocks, inner = ceil(Int, n / blocks))[1:n]

"""
    build_topology(S, R;
        conn, cv_cons, cv_res, modularity, blocks, IS,
        rng = Random.default_rng();
        frac_special_cons = 0.4,
        frac_special_res  = 0.2,
        min_deg_cons = 1,
        min_deg_res  = 1,
        max_retries_per_edge = 8
    ) -> (A, r_block, c_block)

Degree-corrected, block-biased bipartite generator.

- `conn` controls total edges E ~= conn * R * C (clamped to [R + C, R * C]).
- `cv_cons`, `cv_res` shape the *target* out-/in-degree variability (lognormal weights).
- `modularity` in [0, 1] is the fraction of edges within blocks (approx. block mixing).
- `blocks` is the number of balanced modules.
- `frac_special_*` sets the fraction of nodes pinned at the minimum degree
  (boosts CV and restores small degrees even when mean degree is moderate).

Guarantees:
- each consumer gets >= `min_deg_cons` prey
- each resource gets >= `min_deg_res` predators

Signs: consumer->resource positive; reciprocal negative with same magnitude.

Returns:
- `A :: Matrix{Float64}` (S x S)
- `r_block :: Vector{Int}` (length R)
- `c_block :: Vector{Int}` (length C)
"""
function build_topology(S::Int, R::Int;
    conn::Float64,
    cv_cons::Float64,
    cv_res::Float64,
    modularity::Float64,
    blocks::Int,
    IS::Float64,
    rng = Random.default_rng(),
    frac_special_cons::Float64 = 0.4,
    frac_special_res::Float64  = 0.2,
    min_deg_cons::Int = 1,
    min_deg_res::Int  = 1,
    max_retries_per_edge::Int = 8
)
    @assert 1 <= R < S "R must be in [1, S-1]."
    C = S - R
    A = zeros(Float64, S, S)

    # -----------------------------
    # Helpers (local)
    # -----------------------------
    sigma_from_cv(cv) = cv <= 1e-6 ? 1e-6 : sqrt(log(1 + cv^2))

    # Integer degree sequences with min degree + specialist fractions + fixed sum E
    function _sample_degree_sequences(R::Int, C::Int, E::Int)
        @assert E >= R * min_deg_res + C * min_deg_cons "E too small for required minima."

        # Choose specialists
        n_spec_c = clamp(round(Int, frac_special_cons * C), 0, C)
        n_spec_r = clamp(round(Int, frac_special_res  * R), 0, R)
        cons_idx = collect(1:C)
        res_idx  = collect(1:R)
        spec_c   = isempty(cons_idx) ? Int[] : sample(rng, cons_idx, n_spec_c; replace = false)
        spec_r   = isempty(res_idx)  ? Int[] : sample(rng, res_idx,  n_spec_r; replace = false)
        gen_c    = setdiff(cons_idx, spec_c)
        gen_r    = setdiff(res_idx,  spec_r)

        k_cons = fill(min_deg_cons, C)  # out-degree (consumers)
        k_res  = fill(min_deg_res,  R)  # in-degree  (resources)

        rem_c = E - sum(k_cons)   # extra stubs to consumers
        rem_r = E - sum(k_res)    # extra stubs to resources
        @assert rem_c >= 0 && rem_r >= 0

        wc = isempty(gen_c) ? Float64[] : exp.(sigma_from_cv(cv_cons) .* randn(rng, length(gen_c)))
        wr = isempty(gen_r) ? Float64[] : exp.(sigma_from_cv(cv_res)  .* randn(rng, length(gen_r)))

        # Spread a fixed integer budget (randomized rounding + greedy repair), with caps
        function spread!(k::Vector{Int}, idxs::Vector{Int}, weights::Vector{Float64},
                         budget::Int, cap_per_node::Int)
            if budget <= 0 || isempty(idxs)
                return
            end
            p = weights ./ sum(weights)
            extras = round.(Int, budget .* p)
            # Repair to exact budget
            delta = budget - sum(extras)
            while delta != 0
                j = rand(rng, 1:length(extras))
                if delta > 0 && extras[j] < cap_per_node
                    extras[j] += 1
                    delta -= 1
                elseif delta < 0 && extras[j] > 0
                    extras[j] -= 1
                    delta += 1
                end
            end
            # Apply (respect caps)
            @inbounds for (t, j) in enumerate(idxs)
                k[j] = min(k[j] + extras[t], cap_per_node)
            end
        end

        spread!(k_cons, gen_c, wc, rem_c, R)
        spread!(k_res,  gen_r, wr, rem_r, C)

        # Reconcile sums to equal E exactly (simple greedy)
        while sum(k_cons) != E
            if sum(k_cons) > E
                i = rand(rng, findall(x -> k_cons[x] > min_deg_cons, 1:C))
                k_cons[i] -= 1
            else
                i = rand(rng, 1:C)
                k_cons[i] = min(k_cons[i] + 1, R)
            end
        end
        while sum(k_res) != E
            if sum(k_res) > E
                j = rand(rng, findall(x -> k_res[x] > min_deg_res, 1:R))
                k_res[j] -= 1
            else
                j = rand(rng, 1:R)
                k_res[j] = min(k_res[j] + 1, C)
            end
        end

        @assert sum(k_cons) == sum(k_res) == E
        return k_res, k_cons
    end

    # -----------------------------
    # Total edges and blocks
    # -----------------------------
    E_target = clamp(round(Int, conn * R * C), R + C, R * C)

    r_block = repeat(1:blocks, inner = ceil(Int, R / blocks))[1:R]
    c_block = repeat(1:blocks, inner = ceil(Int, C / blocks))[1:C]

    # Degree sequences with many specialists (tune frac_special_* to widen CV)
    k_res, k_cons = _sample_degree_sequences(R, C, E_target)

    # Block mixing: allocate edges per (bc, br) pair
    within = clamp(modularity, 0.0, 1.0)
    between = 1.0 - within
    W = zeros(Int, blocks, blocks)
    for bc in 1:blocks, br in 1:blocks
        frac = (bc == br) ? within / blocks : between / (blocks * (blocks - 1))
        W[bc, br] = round(Int, frac * E_target)
    end
    deltaE = E_target - sum(W)
    while deltaE != 0
        bc = rand(rng, 1:blocks)
        br = rand(rng, 1:blocks)
        if deltaE > 0
            W[bc, br] += 1
            deltaE -= 1
        elseif W[bc, br] > 0
            W[bc, br] -= 1
            deltaE += 1
        end
    end

    # Prepare stubs split by block
    cons_in_block = [findall(==(b), c_block) for b in 1:blocks]
    res_in_block  = [findall(==(b), r_block) for b in 1:blocks]

    cons_stubs = [Int[] for _ in 1:blocks]
    @inbounds for bc in 1:blocks, i in cons_in_block[bc]
        append!(cons_stubs[bc], fill(i, k_cons[i]))
    end
    res_stubs = [Int[] for _ in 1:blocks]
    @inbounds for br in 1:blocks, j in res_in_block[br]
        append!(res_stubs[br], fill(j, k_res[j]))
    end

    # Wire edges per block pair (configuration-style, simple-graph attempt)
    for bc in 1:blocks, br in 1:blocks
        m = W[bc, br]
        if m == 0
            continue
        end
        cs = cons_stubs[bc]
        rs = res_stubs[br]
        for _ in 1:m
            # If this block empties, borrow from global pools (rare)
            if isempty(cs)
                cs = reduce(vcat, cons_stubs)
            end
            if isempty(rs)
                rs = reduce(vcat, res_stubs)
            end
            if isempty(cs) || isempty(rs)
                break
            end

            tries = 0
            while true
                ic_idx = rand(rng, 1:length(cs))
                jr_idx = rand(rng, 1:length(rs))
                i = cs[ic_idx]
                j = rs[jr_idx]
                if A[R + i, j] == 0.0  # new edge
                    w = abs(randn(rng)) * IS
                    A[R + i, j] =  w
                    A[j, R + i] = -w
                    deleteat!(cs, ic_idx)
                    deleteat!(rs, jr_idx)
                    break
                else
                    tries += 1
                    if tries >= max_retries_per_edge
                        # Give up on simple-graph constraint: consume stubs to keep sums
                        deleteat!(cs, ic_idx)
                        deleteat!(rs, jr_idx)
                        break
                    end
                end
            end
        end
    end

    # Final guard: if any consumer/resource ended at zero, patch lightly
    for ic in 1:C
        if all(iszero, A[R + ic, 1:R])
            jr = argmax(rand(rng, R))  # arbitrary; CV already handled by degrees
            w = abs(randn(rng)) * IS
            A[R + ic, jr] =  w
            A[jr, R + ic] = -w
        end
    end
    for jr in 1:R
        if all(iszero, A[(R + 1):S, jr])
            ic = argmax(rand(rng, C))
            w = abs(randn(rng)) * IS
            A[R + ic, jr] =  w
            A[jr, R + ic] = -w
        end
    end

    return A, r_block, c_block
end


# ---------------------------------------------------------
# Equilibrium generator
# ---------------------------------------------------------
function choose_equilibrium(
    S::Int, R::Int;
    u_mean::Float64 = 1.0,
    u_cv_res::Float64 = 0.5,
    u_cv_cons::Float64 = 0.7,
    cons_scale::Float64 = 1.3,
    rng = Random.default_rng()
)
    C = S - R
    function logn(n, cv)
        sigma_val = sqrt(log(1 + cv^2))
        mu_val = log(u_mean) - sigma_val^2 / 2
        return rand(rng, LogNormal(mu_val, sigma_val), n)
    end
    ures = logn(R, u_cv_res)
    ucon = logn(C, u_cv_cons) .* cons_scale
    return vcat(ures, ucon)
end

"""
    calibrate_zeroK_consumers!(A, u, R) -> K

Enforce K_cons = 0 and keep u* as equilibrium by rescaling each consumer row i so (A*u)[i] = u[i].
Mirrors the same factor on the reciprocal entries A[j,i] to preserve pairwise magnitudes.
Returns K with K[1:R] set so that g(u*) = 0 for resources, and K[R+1:end] = 0.
"""
function calibrate_zeroK_consumers!(A::AbstractMatrix, u::AbstractVector, R::Int)
    S = size(A, 1)
    C = S - R
    @assert length(u) == S

    # Scale each consumer row
    Au = A * u
    for ic in 1:C
        i = R + ic
        num = u[i]
        den = Au[i]
        if den <= 0
            # Degenerate case (should not happen with positive edges); patch with one prey
            jr = argmax(u[1:R])
            A[i, jr] = max(A[i, jr], 1e-6)
            A[jr, i] = -A[i, jr]
            den = (A[i, 1:R]' * u[1:R])
        end
        s = num / den
        # Rescale consumer->resource and reciprocal resource->consumer entries
        @inbounds for jr in 1:R
            A[i, jr] *= s
            A[jr, i] *= s
        end
    end

    # Recompute after scaling
    Au = A * u

    # K for resources: K_j = u_j - (A*u)_j
    # K for consumers: 0
    K = similar(u)
    @inbounds begin
        for j in 1:R
            K[j] = u[j] - Au[j]
        end
        for i in (R + 1):S
            K[i] = 0.0
        end
    end
    return K
end


# ---------------------------------------------------------
# Stabilization routine
# ---------------------------------------------------------
function stabilize!(A::Matrix{Float64}, u::Vector{Float64}, K::Vector{Float64};
    margin::Float64 = 0.05,
    max_iter::Int = 30,
    shrink::Float64 = 0.8
)
    alpha = 1.0
    for _ in 1:max_iter
        J = jacobian_at_equilibrium(A, u)
        lambda_val = maximum(real, eigvals(J))
        if lambda_val <= -margin
            return alpha, lambda_val
        end
        A .*= shrink
        alpha *= shrink
        K .= (I - A) * u
    end
    lambda_val = maximum(real, eigvals(jacobian_at_equilibrium(A, u)))
    return alpha, lambda_val
end


"""
    stabilize_and_recalibrate!(A, u, R; margin)

Calls `stabilize!` (which may rescale A), then re-enforces zero-K consumers
and recomputes resource K so that u* remains an exact equilibrium with K_cons = 0.
Returns (K, alpha, lambda_max).
"""
function stabilize_and_recalibrate!(A::AbstractMatrix, u::AbstractVector, R::Int; margin::Float64)
    # First pass K (used only temporarily)
    K_tmp = calibrate_zeroK_consumers!(A, u, R)
    alpha, lambda_val = stabilize!(A, u, K_tmp; margin = margin)

    # After stabilize! (A changed): enforce again
    K = calibrate_zeroK_consumers!(A, u, R)
    return K, alpha, lambda_val
end


# ---------------------------------------------------------
# Metric extraction for the "full" community
# ---------------------------------------------------------
# Fast solver configuration; rmed disabled by default
function extract_metrics(u0::Vector{Float64}, K::Vector{Float64}, A::AbstractMatrix;
                         R::Int,
                         tspan = (0.0, 200.0),
                         tpert = 0.0,
                         delta = 1.0,
                         cb = nothing,
                         abstol = 1e-6,
                         reltol = 1e-6,
                         saveat = range(tspan[1], tspan[2], length = 201),
                         compute_rmed::Bool = false)

    J = jacobian_at_equilibrium(A, u0)

    S_full = count(>(EXTINCTION_THRESHOLD), u0)
    resilience_full  = maximum(real, eigvals(Matrix(J)))              # dense eig on small J
    reactivity_full  = maximum(real, eigvals(Matrix((J + J') / 2)))   # symmetric part
    collectivity_full = compute_collectivity(Matrix(A))

    SL_full = diag(J) .|> x -> x == 0.0 ? 0.0 : -1 / x
    mean_SL_full = mean(SL_full)

    sigma_full = sigma_over_min_d(Matrix(A), Matrix(J))
    rmed_full  = compute_rmed ?
        analytical_median_return_rate(Matrix(J); t = 1.0) :
        median(-diag(J))  # proxy

    # --- Press perturbation ---
    rt_press_vec, _, after_press, _ =
        simulate_press_perturbation_glv(u0, (K, A), tspan, tpert, delta, R;
                                        cb = cb, abstol = abstol, reltol = reltol, saveat = saveat)
    rt_press_full = mean(skipmissing(rt_press_vec))

    # --- Pulse perturbation ---
    rt_pulse_vec, _, after_pulse, _ =
        simulate_pulse_perturbation_glv(u0, (K, A), tspan, tpert, delta;
                                        cb = cb, abstol = abstol, reltol = reltol, saveat = saveat)
    rt_pulse_full = mean(skipmissing(rt_pulse_vec))

    return (
        S_full = S_full,
        resilience_full = resilience_full,
        reactivity_full = reactivity_full,
        collectivity_full = collectivity_full,
        SL_full = SL_full,
        mean_SL_full = mean_SL_full,
        sigma_over_min_d_full = sigma_full,
        rmed_full = rmed_full,
        after_press_full = after_press,
        after_pulse_full = after_pulse,
        rt_press_full = rt_press_full,
        rt_pulse_full = rt_pulse_full
    )
end

"""
    modification_ladder!(records, u0, A0, K0, S, R;
        conn, cv_cons, cv_res, modularity, blocks, IS,
        tspan=(0.0,500.0), tpert=250.0, delta=1.0,
        cb=nothing, abstol=1e-6, reltol=1e-6, saveat=nothing,
        compute_rmed_steps::Bool=false, rng=Random.default_rng())

Builds 5 modified models and, for each, recomputes the equilibrium by integrating
the ODE with the same demographics: K_res held from base, K_cons=0 (steps 1–4).
Step 5 reassigns groups (R -> R2) and resamples a resource-K vector from the base distribution.

Appends *_S1..*_S5 metrics to `records`.
"""
function modification_ladder!(records::Vector{Pair{Symbol,Any}},
                              u0::Vector{Float64}, A0::Matrix{Float64}, K0::Vector{Float64},
                              S::Int, R::Int;
                              conn::Float64, cv_cons::Float64, cv_res::Float64,
                              modularity::Float64, blocks::Int, IS::Float64,
                              tspan=(0.0,500.0), tpert=250.0, delta=1.0,
                              cb=nothing, abstol=1e-6, reltol=1e-6, saveat=nothing,
                              compute_rmed_steps::Bool=false,
                              IS_factor::Float64=10.0,
                              conn_shift::Float64=0.35,
                              rng = Random.default_rng())

    C = S - R
    K_res_base = copy(K0[1:R])  # keep resource K's from the base model

    # local helper: measure all step metrics given (A_s, u*_s, K_s, R_s)
    function _step_metrics(step::Int, A_s::Matrix{Float64}, ustar_s::Vector{Float64},
                           K_s::Vector{Float64}, R_s::Int)
        J_s = jacobian_at_equilibrium(A_s, ustar_s)

        # Press perturbation
        rt_press_vec, _, after_press_s, _ =
            simulate_press_perturbation_glv(ustar_s, (K_s, A_s), tspan, tpert, delta, R_s;
                                            cb=cb, abstol=abstol, reltol=reltol, saveat=saveat)
        rt_press_S = mean(skipmissing(rt_press_vec))

        # Pulse perturbation
        rt_pulse_vec, _, after_pulse_s, _ =
            simulate_pulse_perturbation_glv(ustar_s, (K_s, A_s), tspan, tpert, delta;
                                            cb=cb, abstol=abstol, reltol=reltol, saveat=saveat)
        rt_pulse_S = mean(skipmissing(rt_pulse_vec))

        collectivity_S = compute_collectivity(A_s)
        resilience_S   = maximum(real, eigvals(J_s))
        reactivity_S   = maximum(real, eigvals((J_s + J_s')/2))
        sigma_over_min_d_S = sigma_over_min_d(A_s, J_s)
        SL_S = diag(J_s) .|> x -> x == 0.0 ? 0.0 : -1.0/x
        mean_SL_S = mean(SL_S)
        rmed_S = compute_rmed_steps ? analytical_median_return_rate(J_s; t=1.0) : NaN

        push!(records, Symbol("after_press_S$(step)") => after_press_s)
        push!(records, Symbol("after_pulse_S$(step)") => after_pulse_s)
        push!(records, Symbol("rt_press_S$(step)")    => rt_press_S)
        push!(records, Symbol("rt_pulse_S$(step)")    => rt_pulse_S)
        push!(records, Symbol("S_S$(step)")           => count(>(EXTINCTION_THRESHOLD), ustar_s))
        push!(records, Symbol("collectivity_S$(step)") => collectivity_S)
        push!(records, Symbol("resilience_S$(step)")   => resilience_S)
        push!(records, Symbol("reactivity_S$(step)")   => reactivity_S)
        push!(records, Symbol("sigma_over_min_d_S$(step)") => sigma_over_min_d_S)
        push!(records, Symbol("SL_S$(step)")          => SL_S)
        push!(records, Symbol("mean_SL_S$(step)")     => mean_SL_S)
        push!(records, Symbol("rmed_S$(step)")        => rmed_S)
    end

    # ---- Step 1: rewiring (same conn & IS) ----
    A1, _, _ = build_topology(S, R; conn=conn, cv_cons=cv_cons, cv_res=cv_res,
                              modularity=modularity, blocks=blocks, IS=IS, rng=rng)
    K1 = vcat(K_res_base, zeros(C))
    ok1, u1 = equilibrate_by_ode(u0, K1, A1; tspan=tspan, cb=cb,
                                 abstol=abstol, reltol=reltol, saveat=saveat)
    if !ok1
        u1 = fill(0.0, S)
    end
    _step_metrics(1, A1, u1, K1, R)

    # ---- Step 2: change connectance ----
    new_conn = clamp(conn + (2.0*rand(rng)-1.0)*conn_shift, 0.01, 0.95)
    A2, _, _ = build_topology(S, R; conn=new_conn, cv_cons=cv_cons, cv_res=cv_res,
                              modularity=modularity, blocks=blocks, IS=IS, rng=rng)
    K2 = vcat(K_res_base, zeros(C))
    ok2, u2 = equilibrate_by_ode(u0, K2, A2; tspan=tspan, cb=cb,
                                 abstol=abstol, reltol=reltol, saveat=saveat)
    if !ok2
        u2 = fill(0.0, S)
    end
    _step_metrics(2, A2, u2, K2, R)

    # ---- Step 3: increase IS only ----
    A3, _, _ = build_topology(S, R; conn=conn, cv_cons=cv_cons, cv_res=cv_res,
                              modularity=modularity, blocks=blocks, IS=IS*IS_factor, rng=rng)
    K3 = vcat(K_res_base, zeros(C))
    ok3, u3 = equilibrate_by_ode(u0, K3, A3; tspan=tspan, cb=cb,
                                 abstol=abstol, reltol=reltol, saveat=saveat)
    if !ok3
        u3 = fill(0.0, S)
    end
    _step_metrics(3, A3, u3, K3, R)

    # ---- Step 4: change both conn and IS ----
    new_conn4 = clamp(conn + (2.0*rand(rng)-1.0)*conn_shift, 0.01, 0.95)
    A4, _, _ = build_topology(S, R; conn=new_conn4, cv_cons=cv_cons, cv_res=cv_res,
                              modularity=modularity, blocks=blocks, IS=IS*IS_factor, rng=rng)
    K4 = vcat(K_res_base, zeros(C))
    ok4, u4 = equilibrate_by_ode(u0, K4, A4; tspan=tspan, cb=cb,
                                 abstol=abstol, reltol=reltol, saveat=saveat)
    if !ok4
        u4 = fill(0.0, S)
    end
    _step_metrics(4, A4, u4, K4, R)

    # ---- Step 5: group reassignment (R -> R2) ----
    R2 = max(1, R - round(Int, 0.1*S))
    C2 = S - R2
    A5, _, _ = build_topology(S, R2; conn=conn, cv_cons=cv_cons, cv_res=cv_res,
                              modularity=modularity, blocks=blocks, IS=IS, rng=rng)

    # resample resource K's from the base distribution to fit R2
    if R2 <= length(K_res_base)
        K_res2 = K_res_base[sample(rng, 1:length(K_res_base), R2; replace=false)]
    else
        K_res2 = K_res_base[sample(rng, 1:length(K_res_base), R2; replace=true)]
    end
    K5 = vcat(K_res2, zeros(C2))

    # same initial condition length S (use base u0)
    ok5, u5 = equilibrate_by_ode(u0, K5, A5; tspan=tspan, cb=cb,
                                 abstol=abstol, reltol=reltol, saveat=saveat)
    if !ok5
        u5 = fill(0.0, S)
    end
    _step_metrics(5, A5, u5, K5, R2)
end

# ---------------------------------------------------------
# Equilibrium checker
# ---------------------------------------------------------
"""
    check_equilibrium(u, K, A; rtol = 1e-8, tpeek = 1.0, cb = nothing)

Returns (ok, stats) where ok is true if the analytic residual is small
AND a short integration stays near u.
"""
function check_equilibrium(u::AbstractVector, K, A; rtol = 1e-8, tpeek = 0.5, cb = nothing)
    du = similar(u)
    gLV_rhs!(du, u, (K, A), 0.0)
    res = maximum(abs, du) / max(1e-12, maximum(abs, u))
    ok_res = res <= rtol

    prob = ODEProblem(gLV_rhs!, u, (0.0, tpeek), (K, A))
    sol  = solve(prob, Tsit5(); callback = cb, abstol = 1e-6, reltol = 1e-6, save_everystep = false)
    drift = maximum(abs, sol.u[end] .- u) / max(1e-12, maximum(abs, u))
    ok_drift = drift <= 1e-6
    return ok_res & ok_drift, (; res, drift)
end


# ---------------------------------------------------------
# Fixed-sum integer degree sequence generator
# ---------------------------------------------------------
"""
    _sample_degree_sequences(R, C; E, cv_res, cv_cons, rng,
                             frac_special_cons = 0.0,
                             frac_special_res  = 0.0,
                             min_deg_cons = 1,
                             min_deg_res  = 1)

Fixed-sum integer degree sequences (resources = in-degree, consumers = out-degree)
with minimum degrees and optional "specialist fractions" pinned at the min degree.
The remaining edges are distributed heavy-tailed (lognormal weights) among the rest.
"""
function _sample_degree_sequences(
    R::Int, C::Int;
    E::Int,
    cv_res::Float64, cv_cons::Float64, rng,
    frac_special_cons::Float64 = 0.0,
    frac_special_res::Float64  = 0.0,
    min_deg_cons::Int = 1,
    min_deg_res::Int  = 1
)
    @assert E >= R * min_deg_res + C * min_deg_cons "E too small for required minima."

    # How many specialists we pin at the minimum
    n_spec_c = clamp(round(Int, frac_special_cons * C), 0, C)
    n_spec_r = clamp(round(Int, frac_special_res  * R), 0, R)

    # Index sets for specialists vs generalists
    cons_idx = collect(1:C)
    res_idx  = collect(1:R)
    spec_c   = isempty(cons_idx) ? Int[] : sample(rng, cons_idx, n_spec_c; replace = false)
    spec_r   = isempty(res_idx)  ? Int[] : sample(rng, res_idx,  n_spec_r; replace = false)
    gen_c    = setdiff(cons_idx, spec_c)
    gen_r    = setdiff(res_idx,  spec_r)

    # Allocate mandatory minima
    k_cons = fill(min_deg_cons, C)
    k_res  = fill(min_deg_res,  R)

    # Remaining edges after minima
    Emin = sum(k_cons) + sum(k_res)  # counts each edge twice (fix below)
    rem_c = E - sum(k_cons)
    rem_r = E - sum(k_res)
    @assert rem_c >= 0 && rem_r >= 0

    # Heavy-tailed weights only among generalists
    sigma_from_cv(cv) = cv <= 1e-6 ? 1e-6 : sqrt(log(1 + cv^2))
    wc = isempty(gen_c) ? Float64[] : exp.(sigma_from_cv(cv_cons) .* randn(rng, length(gen_c)))
    wr = isempty(gen_r) ? Float64[] : exp.(sigma_from_cv(cv_res)  .* randn(rng, length(gen_r)))

    # Helper: spread integer budget with caps, random rounding, then greedy repair
    function spread!(k::Vector{Int}, idxs::Vector{Int}, weights::Vector{Float64},
                     budget::Int, cap_per_node::Int)
        if isempty(idxs) || budget <= 0
            return
        end
        p = weights ./ sum(weights)
        extras = round.(Int, budget .* p)
        delta = budget - sum(extras)
        while delta != 0
            j = rand(rng, 1:length(extras))
            if delta > 0 && extras[j] < cap_per_node
                extras[j] += 1; delta -= 1
            elseif delta < 0 && extras[j] > 0
                extras[j] -= 1; delta += 1
            end
        end
        for (t, j) in enumerate(idxs)
            k[j] = min(k[j] + extras[t], cap_per_node)
        end
    end

    # Each node's hard cap is the bipartite size on the opposite side
    spread!(k_cons, gen_c, wc, rem_c, R)
    spread!(k_res,  gen_r, wr, rem_r, C)

    # Reconcile sums to be equal and exactly E
    while sum(k_cons) != E
        if sum(k_cons) > E
            i = rand(rng, findall(x -> x > min_deg_cons, 1:C))
            k_cons[i] -= 1
        else
            i = rand(rng, 1:C)
            k_cons[i] = min(k_cons[i] + 1, R)
        end
    end
    while sum(k_res) != E
        if sum(k_res) > E
            j = rand(rng, findall(x -> x > min_deg_res, 1:R))
            k_res[j] -= 1
        else
            j = rand(rng, 1:R)
            k_res[j] = min(k_res[j] + 1, C)
        end
    end

    @assert sum(k_cons) == sum(k_res) == E
    return k_res, k_cons
end

"""
    equilibrate_by_ode(u_init::Vector{Float64}, K::Vector{Float64}, A::AbstractMatrix;
                       tspan=(0.0, 500.0), cb=nothing, abstol=1e-6, reltol=1e-6, saveat=nothing)

Integrate the gLV system from u_init under parameters (K,A) until tspan[2].
Returns (ok::Bool, u_final::Vector{Float64}), where ok is true if the run
completed and the final state is finite and nonnegative.
"""
function equilibrate_by_ode(u_init, K, A; tspan=(0.0,500.0), cb=nothing,
                            abstol=1e-6, reltol=1e-6, saveat=nothing)
    prob = ODEProblem(gLV_rhs!, copy(u_init), tspan, (K, A))
    sol  = solve(prob, Tsit5();
                 callback=cb, abstol=abstol, reltol=reltol,
                 saveat=saveat, dense=false, save_start=false, save_everystep=false)
    ok = (sol.retcode == ReturnCode.Success)
    uF = sol.u[end]
    ok = ok && all(isfinite, uF) && all(uF .>= 0.0)
    return ok, uF
end