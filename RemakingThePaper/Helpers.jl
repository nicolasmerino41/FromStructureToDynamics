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
# PRESS (with optional plotting, colored by role + delta in title)
# ---------------------------------------------------------
function simulate_press_perturbation_glv(
    u0, p, tspan, t_perturb::Real, delta::Real, R::Int;
    solver = Tsit5(),
    cb = nothing,
    abstol = 1e-6,
    reltol = 1e-6,
    saveat = range(tspan[1], tspan[2], length = 201),
    dense = false,
    save_everystep = false,
    save_start = false,
    save_end = true,
    plot_simulation::Bool = false,
    title::AbstractString = ""
)
    K, A = p
    before_persistence = count(>(EXTINCTION_THRESHOLD), u0) / length(u0)

    # --- PRE (only simulated when plotting is requested) ---
    u_pre_end = u0
    pre_sol = nothing
    if plot_simulation && t_perturb > tspan[1]
        prob_pre = ODEProblem(gLV_rhs!, u0, (tspan[1], t_perturb), (K, A))
        pre_sol  = solve(prob_pre, solver;
                         callback = cb, abstol = abstol, reltol = reltol,
                         saveat = saveat, dense = dense, save_everystep = save_everystep,
                         save_start = save_start, save_end = true)
        u_pre_end = pre_sol.u[end]
    end

    # --- POST (press: reduce K for resources) ---
    K_press = vcat(K[1:R] .* (1 .- delta), K[R+1:end])
    p2 = (K_press, A)
    prob_post = ODEProblem(gLV_rhs!, u_pre_end, (max(tspan[1], t_perturb), tspan[2]), p2)
    post_sol  = solve(prob_post, solver;
                      callback = cb, abstol = abstol, reltol = reltol,
                      saveat = saveat, dense = dense, save_everystep = save_everystep,
                      save_start = false, save_end = true)

    post = post_sol.u[end]
    after_persistence = count(>(EXTINCTION_THRESHOLD), post) / length(post)

    # Return times (post-phase): first time within 10% of final
    n = length(post)
    rts = fill(Float64(NaN), n)
    @inbounds for i in 1:n
        target = post[i]
        for (t, u) in zip(post_sol.t, post_sol.u)
            if abs(u[i] - target) / (abs(target) + 1e-8) < 0.10
                rts[i] = t - max(tspan[1], t_perturb)
                break
            end
        end
    end

    # --- PLOTTING ---
    if plot_simulation
        full_title = isempty(title) ? "PRESS delta=$(delta)" : "PRESS $(title) delta=$(delta)"
        S = length(u0)
        fig = Figure(size = (1100, 560))
        Label(fig[0, 1], full_title; fontsize = 18, font = :bold, halign = :left)

        # Pre panel
        ax_pre = Axis(fig[1, 1];
            title = "Pre (t  [$(tspan[1]), $(t_perturb)])",
            xlabel = "time", ylabel = "abundance",
            xgridvisible=false, ygridvisible=false
        )
        if pre_sol === nothing
            # show a point at t_perturb with group colors
            if R > 0
                scatter!(ax_pre, fill(t_perturb, R), u0[1:R];  markersize=2, color=:blue)
            end
            if R < S
                scatter!(ax_pre, fill(t_perturb, S-R), u0[(R+1):S]; markersize=2, color=:red)
            end
        else
            for s in 1:S
                ys = (uu -> uu[s]).(pre_sol.u)
                lines!(ax_pre, pre_sol.t, ys; transparency=true, alpha=0.6, color = (s <= R ? :blue : :red))
            end
        end

        # Post panel
        ax_post = Axis(fig[1, 2];
            title = "Post (press) (t  [$(max(tspan[1], t_perturb)), $(tspan[2])])",
            xlabel = "time", ylabel = "abundance",
            xgridvisible=false, ygridvisible=false
        )
        for s in 1:S
            ys = (uu -> uu[s]).(post_sol.u)
            lines!(ax_post, post_sol.t, ys; transparency=true, alpha=0.6, color = (s <= R ? :blue : :red))
        end

        display(fig)
    end

    return rts, before_persistence, after_persistence, post
end

# ---------------------------------------------------------
# PULSE (with optional plotting, colored by role + delta in title)
# (Infers R from K: resources have K>0, consumers K=0)
# ---------------------------------------------------------
function simulate_pulse_perturbation_glv(
    u0, p, tspan, t_pulse::Real, delta::Real;
    solver = Tsit5(),
    cb = nothing,
    abstol = 1e-6,
    reltol = 1e-6,
    saveat = range(tspan[1], tspan[2], length = 201),
    dense = false,
    save_everystep = false,
    save_start = false,
    save_end = true,
    plot_simulation::Bool = false,
    title::AbstractString = ""
)
    K, A = p
    S = length(u0)
    # infer R from K (>0 for resources, =0 for consumers in our framework)
    R = count(>(0.0), K)

    before_persistence = count(>(EXTINCTION_THRESHOLD), u0) / length(u0)

    # --- PRE (only simulated when plotting is requested) ---
    u_pre_end = u0
    pre_sol = nothing
    if plot_simulation && t_pulse > tspan[1]
        prob_pre = ODEProblem(gLV_rhs!, u0, (tspan[1], t_pulse), (K, A))
        pre_sol  = solve(prob_pre, solver;
                         callback = cb, abstol = abstol, reltol = reltol,
                         saveat = saveat, dense = dense, save_everystep = save_everystep,
                         save_start = save_start, save_end = true)
        u_pre_end = pre_sol.u[end]
    end

    # --- POST (pulse: instant drop at t_pulse, same K,A)
    pulsed = u_pre_end .* (1 .- delta)
    prob_post = ODEProblem(gLV_rhs!, pulsed, (max(tspan[1], t_pulse), tspan[2]), (K, A))
    post_sol  = solve(prob_post, solver;
                      callback = cb, abstol = abstol, reltol = reltol,
                      saveat = saveat, dense = dense, save_everystep = save_everystep,
                      save_start = false, save_end = true)

    post = post_sol.u[end]
    after_persistence = count(>(EXTINCTION_THRESHOLD), post) / length(post)

    # Return times (post-phase)
    n = length(post)
    rts = fill(Float64(NaN), n)
    @inbounds for i in 1:n
        target = post[i]
        for (t, u) in zip(post_sol.t, post_sol.u)
            if abs(u[i] - target) / (abs(target) + 1e-8) < 0.10
                rts[i] = t - max(tspan[1], t_pulse)
                break
            end
        end
    end

    # --- PLOTTING ---
    if plot_simulation
        full_title = isempty(title) ? "PULSE delta=$(delta)" : "PULSE $(title) delta=$(delta)"
        fig = Figure(size = (1100, 560))
        Label(fig[0, 1], full_title; fontsize = 18, font = :bold, halign = :left)

        # Pre panel
        ax_pre = Axis(fig[1, 1];
            title = "Pre (t  [$(tspan[1]), $(t_pulse)])",
            xlabel = "time", ylabel = "abundance",
            xgridvisible=false, ygridvisible=false
        )
        if pre_sol === nothing
            if R > 0
                scatter!(ax_pre, fill(t_pulse, R), u0[1:R];        markersize=2, color=:blue)
            end
            if R < S
                scatter!(ax_pre, fill(t_pulse, S-R), u0[(R+1):S]; markersize=2, color=:red)
            end
        else
            for s in 1:S
                ys = (uu -> uu[s]).(pre_sol.u)
                lines!(ax_pre, pre_sol.t, ys; transparency=true, alpha=0.6, color = (s <= R ? :blue : :red))
            end
        end

        # Post panel
        ax_post = Axis(fig[1, 2];
            title = "Post (pulse) (t [$(max(tspan[1], t_pulse)), $(tspan[2])])",
            xlabel = "time", ylabel = "abundance",
            xgridvisible=false, ygridvisible=false
        )
        for s in 1:S
            ys = (uu -> uu[s]).(post_sol.u)
            lines!(ax_post, post_sol.t, ys; transparency=true, alpha=0.6, color = (s <= R ? :blue : :red))
        end

        display(fig)
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
    calibrate_consumerK!(A, u, R) -> K

Enforce K_cons = 0 and keep u* as equilibrium by rescaling each consumer row i so (A*u)[i] = u[i].
Mirrors the same factor on the reciprocal entries A[j,i] to preserve pairwise magnitudes.
Returns K with K[1:R] set so that g(u*) = 0 for resources, and K[R+1:end] = 0.
"""
# Replaces calibrate_zeroK_consumers!
# Sets every consumer i (i > R) to have intrinsic growth k_cons (default 0.0)
# and rescales its consumer->resource row of A so that at u:
#   0 = u[i] * (k_cons - u[i] + (A*u)[i])  =>  (A*u)[i] = u[i] - k_cons
function calibrate_consumer_K!(A::AbstractMatrix, u::AbstractVector, R::Int;
                               k_cons::Float64 = 0.0, min_link::Float64 = 1e-8)
    S = size(A,1)
    C = S - R
    @assert length(u) == S
    @assert R >= 1 "R must be at least 1 (need at least one resource)."

    # Ensure each consumer has at least one prey edge
    for ic in 1:C
        i = R + ic
        if all(iszero, A[i, 1:R])
            jr = argmax(view(u, 1:R))
            w  = max(min_link, abs(randn()) * min_link)
            A[i, jr] =  w
            A[jr, i] = -w
        end
    end

    # Scale each consumer row so (A*u)[i] = u[i] - k_cons
    Au = A * u
    for ic in 1:C
        i = R + ic
        num = u[i] - k_cons
        den = Au[i]
        if den <= 0
            # fallback: only resource block contribution
            den = dot(view(A, i, 1:R), view(u, 1:R))
            if den <= 0
                jr = argmax(view(u, 1:R))
                A[i, jr] = max(A[i, jr], min_link)
                A[jr, i] = -A[i, jr]
                den = dot(view(A, i, 1:R), view(u, 1:R))
                den = den <= 0 ? min_link : den
            end
        end
        s = num / den
        @inbounds for jr in 1:R
            if A[i, jr] != 0.0
                A[i, jr] *= s
                A[jr, i] = -A[i, jr]
            end
        end
    end

    # Recompute K: resources from (I - A)u, consumers fixed to k_cons
    Au = A * u
    K  = similar(u)
    @inbounds begin
        for j in 1:R
            K[j] = u[j] - Au[j]
        end
        for i in (R+1):S
            K[i] = k_cons
        end
    end
    return K
end

# ---------------------------------------------------------
# Stabilization routine
# ---------------------------------------------------------
function stabilize!(A::Matrix{Float64}, u::Vector{Float64}, K::Vector{Float64};
    margin::Float64=0.05, max_iter::Int=50, shrink::Float64=0.7)
    alpha = 1.0
    for _ in 1:max_iter
        J = jacobian_at_equilibrium(A, u)
        lambda = maximum(real, eigvals(J))
        if lambda <= -margin
            return alpha, lambda
        end
        A .*= shrink
        alpha *= shrink
        K .= (I - A) * u
    end
    lambda = maximum(real, eigvals(jacobian_at_equilibrium(A, u)))
    return alpha, lambda
end

"""
    stabilize_and_recalibrate!(A, u, R; margin)

Calls `stabilize!` (which may rescale A), then re-enforces zero-K consumers
and recomputes resource K so that u* remains an exact equilibrium with K_cons = 0.
Returns (K, alpha, lambda_max).
"""
function stabilize_and_recalibrate!(A::AbstractMatrix, u::AbstractVector, R::Int;
                                    margin::Float64, k_cons::Float64 = 0.0,
                                    max_iter::Int = 30, shrink::Float64 = 0.8)
    # First calibrate with the requested consumer K
    K_tmp = calibrate_consumer_K!(A, u, R; k_cons=k_cons)

    # Shrink A until the Jacobian is to the left of -margin
    alpha = 1.0
    for _ in 1:max_iter
        J = Diagonal(u) * (A - I)
        lambda = maximum(real, eigvals(J))
        if lambda <= -margin
            # Final recalc of K under the achieved A
            K = calibrate_consumer_K!(A, u, R; k_cons=k_cons)
            return K, alpha, lambda
        end
        A .*= shrink
        alpha *= shrink
        # keep K consistent with current A and target k_cons
        K_tmp = calibrate_consumer_K!(A, u, R; k_cons=k_cons)
    end
    # Give up but still return current lambda, alpha, and K
    J = Diagonal(u) * (A - I)
    lambda = maximum(real, eigvals(J))
    K = calibrate_consumer_K!(A, u, R; k_cons=k_cons)
    return K, alpha, lambda
end

function compute_resilience(u::AbstractVector, A::AbstractMatrix;
                            extinct_threshold::Real=EXTINCTION_THRESHOLD,
                            extant_only::Bool=true)
    J = Diagonal(u) * (A - I)
    if extant_only
        Iext = findall(>(extinct_threshold), u)
        isempty(Iext) && return 0.0  # degenerate: everything dead
        J = @view J[Iext, Iext]
    end
    return maximum(real, eigvals(J))
end

function compute_reactivity(u::AbstractVector, A::AbstractMatrix;
                            extinct_threshold::Real=EXTINCTION_THRESHOLD,
                            extant_only::Bool=true)
    J = Diagonal(u) * (A - I)
    if extant_only
        Iext = findall(>(extinct_threshold), u)
        isempty(Iext) && return 0.0
        J = @view J[Iext, Iext]
    end
    Jsym = (J + J')/2
    return maximum(real, eigvals(Jsym))
end

# ---------------------------------------------------------
# Metric extraction for the "full" community
# ---------------------------------------------------------
# Fast solver configuration; rmed disabled by default
function extract_metrics(
    u0::Vector{Float64}, K::Vector{Float64}, A::AbstractMatrix;
    R::Int,
    tspan = (0.0, 200.0),
    tpert = 0.0,
    delta = 1.0,
    cb = nothing,
    abstol = 1e-6,
    reltol = 1e-6,
    saveat = range(tspan[1], tspan[2], length = 201),
    compute_rmed::Bool = false,
    plot_simulation::Bool = false,
    title::AbstractString = "Full"
)

    J = jacobian_at_equilibrium(A, u0)

    S_full = count(>(EXTINCTION_THRESHOLD), u0)
    resilience_full  = compute_resilience(u0, A; extant_only=false)
    resilienceE_ext   = compute_resilience(u0, A; extant_only=true) 
    
    reactivity_full  = compute_reactivity(u0, A; extant_only=false)
    reactivityE_full   = compute_reactivity(u0, A; extant_only=true)

    collectivity_full = compute_collectivity(Matrix(A))

    SL_full = diag(J) .|> x -> x == 0.0 ? 0.0 : -1 / x
    mean_SL_full = mean(SL_full)

    sigma_full = sigma_over_min_d(Matrix(A), Matrix(J))
    rmed_full  = compute_rmed ?
        analytical_median_return_rate(Matrix(J); t = 1.0) :
        median(-diag(J))  # proxy

    # --- Press perturbation ---
    rt_press_vec, _, after_press, _ =
        simulate_press_perturbation_glv(
            u0, (K, A), tspan, tpert, delta, R;
            cb = cb, abstol = abstol, reltol = reltol, saveat = saveat,
            plot_simulation = plot_simulation, title = "Press simulations: " * title
        )
    rt_press_full = mean(skipmissing(rt_press_vec))

    # --- Pulse perturbation ---
    rt_pulse_vec, _, after_pulse, _ =
        simulate_pulse_perturbation_glv(
            u0, (K, A), tspan, tpert, delta;
            cb = cb, abstol = abstol, reltol = reltol, saveat = saveat,
            plot_simulation = plot_simulation, title = "Pulse simulations: " * title
        )
    rt_pulse_full = mean(skipmissing(rt_pulse_vec))

    return (
        S_full = S_full,
        resilience_full = resilience_full,
        resilienceE_full = resilienceE_ext,
        reactivity_full = reactivity_full,
        reactivityE_full = reactivityE_full,
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
the ODE with the same demographics: K_res held from base, K_cons=0 (steps 1-4).
Step 5 reassigns groups (R -> R2) and resamples a resource-K vector from the base distribution.

Appends *_S1..*_S5 metrics to `records`.
"""
function modification_ladder!(
    records::Vector{Pair{Symbol,Any}},
    ustar::Vector{Float64}, A::AbstractMatrix, K::Vector{Float64},
    S::Int, R::Int;
    # Topology controls
    conn::Float64, cv_cons::Float64, cv_res::Float64,
    modularity::Float64, blocks::Int, IS::Float64,
    # ODE / measurement controls
    tspan=(0.0, 500.0), tpert=0.0, delta=1.0,
    cb=nothing, abstol=1e-6, reltol=1e-6, saveat=range(tspan[1], tspan[2], length=201),
    compute_rmed_steps::Bool=false,
    rng = Random.default_rng(),
    # NEW: choose which step numbers to run (from 1..8). Kept as their original ids.
    steps_to_run::AbstractVector{Int} = 1:8,
    plot_simulation::Bool = false,
    title::AbstractString = "Step",
    preserve_pair_symmetry = false,
    consumer_k::Float64 = 0.0
)
    EXT = EXTINCTION_THRESHOLD
    S0  = S

    # ---------- helpers ----------
    settle_equilibrium = function(u_init::Vector{Float64}, A_step::AbstractMatrix, K_step::Vector{Float64})
        # returns (u_s, analytic::Bool)
        return equilibrium_for(A_step, K_step, u_init, tspan, cb; abstol=abstol, reltol=reltol, saveat=saveat)
    end
    # settle_equilibrium = (u_init, true)

    function step_metrics!(step::Int, u_s::Vector{Float64}, K_s::Vector{Float64}, A_s::AbstractMatrix, R_s::Int)
        J_s = jacobian_at_equilibrium(A_s, u_s)

        # ---- pre-perturbation persistence at this step
        init_survival = count(>(EXT), u_s) / S0
        push!(records, Symbol("init_survival_S$(step)") => init_survival)

        # full vs extant-only stability/reactivity
        res_full = compute_resilience(u_s, A_s; extant_only=false)
        res_ext  = compute_resilience(u_s, A_s; extant_only=true)
        rea_full = compute_reactivity(u_s, A_s; extant_only=false)
        rea_ext  = compute_reactivity(u_s, A_s; extant_only=true)

        # PRESS: mean RT over species extant at start
        rt_press_vec, _, after_press_s, _ =
            simulate_press_perturbation_glv(
                u_s, (K_s, A_s), tspan, tpert, delta, R_s;
                cb=cb, abstol=abstol, reltol=reltol, saveat=saveat,
                plot_simulation=plot_simulation, title="Press simulations: " * title * " $(step)"
            )
        mask_press = u_s .> EXT
        rt_press_vals = [rt_press_vec[i] for i in eachindex(u_s)
                         if mask_press[i] && isfinite(rt_press_vec[i]) && !ismissing(rt_press_vec[i])]
        rt_press_S = isempty(rt_press_vals) ? NaN : mean(rt_press_vals)

        # PULSE: mean RT over species extant at start
        rt_pulse_vec, _, after_pulse_s, _ =
            simulate_pulse_perturbation_glv(
                u_s, (K_s, A_s), tspan, tpert, delta;
                cb=cb, abstol=abstol, reltol=reltol, saveat=saveat,
                plot_simulation=plot_simulation, title="Pulse simulations: " * title * " $(step)"
            )
        mask_pulse = u_s .> EXT
        rt_pulse_vals = [rt_pulse_vec[i] for i in eachindex(u_s)
                         if mask_pulse[i] && isfinite(rt_pulse_vec[i]) && !ismissing(rt_pulse_vec[i])]
        rt_pulse_S = isempty(rt_pulse_vals) ? NaN : mean(rt_pulse_vals)

        S_S = count(>(EXT), u_s)
        collectivity_S = compute_collectivity(Matrix(A_s))
        sigma_over_min_d_S = sigma_over_min_d(Matrix(A_s), Matrix(J_s))
        SL_S = diag(J_s) .|> x -> x == 0.0 ? 0.0 : -1/x
        mean_SL_S = mean(SL_S)
        rmed_S = compute_rmed_steps ? analytical_median_return_rate(Matrix(J_s); t=1.0) : median(-diag(J_s))

        push!(records, Symbol("after_press_S$(step)")      => after_press_s)
        push!(records, Symbol("after_pulse_S$(step)")      => after_pulse_s)
        push!(records, Symbol("rt_press_S$(step)")         => rt_press_S)
        push!(records, Symbol("rt_pulse_S$(step)")         => rt_pulse_S)
        push!(records, Symbol("S_S$(step)")                => S_S)
        push!(records, Symbol("collectivity_S$(step)")     => collectivity_S)
        push!(records, Symbol("resilience_S$(step)")       => res_full)
        push!(records, Symbol("resilienceE_S$(step)")      => res_ext)
        push!(records, Symbol("reactivity_S$(step)")       => rea_full)
        push!(records, Symbol("reactivityE_S$(step)")      => rea_ext)
        push!(records, Symbol("sigma_over_min_d_S$(step)") => sigma_over_min_d_S)
        push!(records, Symbol("SL_S$(step)")               => SL_S)
        push!(records, Symbol("mean_SL_S$(step)")          => mean_SL_S)
        push!(records, Symbol("rmed_S$(step)")             => rmed_S)
        return nothing
    end

    # magnitude field applier (keeps CR sign pairing if requested)
    function apply_magnitudes_with_sign!(Aout::AbstractMatrix, Ain::AbstractMatrix, M::AbstractMatrix, R::Int;
                                         preserve_pairs::Bool=false)
        S = size(Ain,1)
        fill!(Aout, 0.0)
        if preserve_pairs
            for i in (R+1):S, j in 1:R
                if Ain[i,j] != 0.0 || Ain[j,i] != 0.0
                    m = (M[i,j] + M[j,i]) / 2
                    if m > 0
                        Aout[i,j] =  +m
                        Aout[j,i] =  -m
                    end
                end
            end
        else
            @inbounds for i in 1:S, j in 1:S
                if Ain[i,j] != 0.0
                    Aout[i,j] = sign(Ain[i,j]) * M[i,j]
                end
            end
        end
        return Aout
    end

    # ---------- step constructors (always derive A_s from ORIGINAL A) ----------
    function A_step1_rowmean(A::AbstractMatrix)
        S = size(A,1); M = zeros(eltype(A), S, S)
        @inbounds for i in 1:S
            mags = Float64[]
            for j in 1:S
                if A[i,j] != 0.0; push!(mags, abs(A[i,j])); end
            end
            if !isempty(mags)
                m_mean = mean(mags)
                for j in 1:S
                    if A[i,j] != 0.0; M[i,j] = m_mean; end
                end
            end
        end
        A1 = similar(A); apply_magnitudes_with_sign!(A1, A, M, R; preserve_pairs=preserve_pair_symmetry)
    end

    function A_step2_rolemean(A::AbstractMatrix)
        S = size(A,1); M = zeros(eltype(A), S, S)
        # consumer -> resource magnitudes
        mags_cons = Float64[]
        @inbounds for i in (R+1):S, j in 1:R
            if A[i,j] != 0.0; push!(mags_cons, abs(A[i,j])); end
        end
        mC = isempty(mags_cons) ? 0.0 : mean(mags_cons)
        # resource -> consumer magnitudes
        mags_res = Float64[]
        @inbounds for i in 1:R, j in (R+1):S
            if A[i,j] != 0.0; push!(mags_res, abs(A[i,j])); end
        end
        mR = isempty(mags_res) ? 0.0 : mean(mags_res)
        @inbounds begin
            for i in (R+1):S, j in 1:R
                if A[i,j] != 0.0; M[i,j] = mC; end
            end
            for i in 1:R, j in (R+1):S
                if A[i,j] != 0.0; M[i,j] = mR; end
            end
        end
        A2 = similar(A); apply_magnitudes_with_sign!(A2, A, M, R; preserve_pairs=preserve_pair_symmetry)
    end

    function A_step3_globalmean(A::AbstractMatrix)
        S = size(A,1)
        mags_all = Float64[]
        @inbounds for i in 1:S, j in 1:S
            if A[i,j] != 0.0; push!(mags_all, abs(A[i,j])); end
        end
        mG = isempty(mags_all) ? 0.0 : mean(mags_all)
        M = zeros(eltype(A), S, S)
        @inbounds for i in 1:S, j in 1:S
            if A[i,j] != 0.0; M[i,j] = mG; end
        end
        A3 = similar(A); apply_magnitudes_with_sign!(A3, A, M, R; preserve_pairs=preserve_pair_symmetry)
    end

    # rewiring variants use the generator, always from scratch (original params)
    function A_step4_rewire_same()
        A4, _, _ = build_topology(S, R; conn=conn, cv_cons=cv_cons, cv_res=cv_res,
                                  modularity=modularity, blocks=blocks, IS=IS, rng=rng)
        A4
    end
    function A_step5_connshift()
        conn_shift = 0.35
        new_conn = clamp(conn + (2rand(rng)-1)*conn_shift, 0.01, 0.99)
        A5, _, _ = build_topology(S, R; conn=new_conn, cv_cons=cv_cons, cv_res=cv_res,
                                  modularity=modularity, blocks=blocks, IS=IS, rng=rng)
        A5
    end
    function A_step6_ISboost()
        IS_factor = 10.0
        A6, _, _ = build_topology(S, R; conn=conn, cv_cons=cv_cons, cv_res=cv_res,
                                  modularity=modularity, blocks=blocks, IS=IS*IS_factor, rng=rng)
        A6
    end
    function A_step7_connshift_ISboost()
        conn_shift = 0.35; IS_factor = 10.0
        new_conn = clamp(conn + (2rand(rng)-1)*conn_shift, 0.01, 0.99)
        A7, _, _ = build_topology(S, R; conn=new_conn, cv_cons=cv_cons, cv_res=cv_res,
                                  modularity=modularity, blocks=blocks, IS=IS*IS_factor, rng=rng)
        A7
    end
    function step8_reassign_roles()
        R8 = max(1, R - round(Int, 0.1*S))
        A8, _, _ = build_topology(S, R8; conn=conn, cv_cons=cv_cons, cv_res=cv_res,
                                modularity=modularity, blocks=blocks, IS=IS, rng=rng)
        # Build a K8 with same stats for resources, and the chosen consumer_k for consumers
        K8 = similar(K, S)
        muK, sigmaK = mean(K[1:R]), std(K[1:R])
        for j in 1:R8
            val = max(1e-6, muK + sigmaK * randn(rng))
            K8[j] = val
        end
        # HERE: use the global choice (thread it in as an argument to modification_ladder!)
        for i in (R8+1):S
            K8[i] = consumer_k    # was 0.0 before
        end
        return A8, K8, R8
    end


    # ---------- run the requested steps (in numeric order), keeping their step ids ----------
    A_base = A
    u_last, K_last, R_last = ustar, K, R

    for step in sort(collect(steps_to_run))
        if step == 1
            A_s = A_step1_rowmean(A_base); K_s = K_last; R_s = R_last
        elseif step == 2
            A_s = A_step2_rolemean(A_base); K_s = K_last; R_s = R_last
        elseif step == 3
            A_s = A_step3_globalmean(A_base); K_s = K_last; R_s = R_last
        elseif step == 4
            A_s = A_step4_rewire_same();     K_s = K_last; R_s = R_last
        elseif step == 5
            A_s = A_step5_connshift();       K_s = K_last; R_s = R_last
        elseif step == 6
            A_s = A_step6_ISboost();         K_s = K_last; R_s = R_last
        elseif step == 7
            A_s = A_step7_connshift_ISboost(); K_s = K_last; R_s = R_last
        elseif step == 8
            A_s, K_s, R_s = step8_reassign_roles()
        else
            @warn "Unknown step $step - skipping."
            continue
        end

        # u_s, analytic = settle_equilibrium(u_last, A_s, K_s)
        u_s, analytic = u_last, true

        # record whether we settled analytically (true) or via ODE (false)
        push!(records, Symbol("analytic_settle_S", step) => analytic)

        step_metrics!(step, u_s, K_s, A_s, R_s)

        # advance initial condition for next requested step
        # u_last, K_last, R_last = u_s, K_s, R_s

    end
    return nothing
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

# algebraic equilibrium (throws on failure)
function algebraic_equilibrium(A::AbstractMatrix, K::AbstractVector)
    u = (I - A) \ K
    if any(!isfinite, u) || any(u .<= 0.0)
        error("bad equilibrium")
    end
    return u
end

# robust: try algebraic first; fallback to short ODE settle
# returns (u, analytic::Bool)
function equilibrium_for(A::AbstractMatrix, K::AbstractVector, u_init::AbstractVector,
                         tspan, cb; abstol=1e-6, reltol=1e-6, saveat=nothing)
    try
        return algebraic_equilibrium(A, K), true
    catch
        prob = ODEProblem(gLV_rhs!, u_init, tspan, (K, A))
        sol  = solve(prob, Tsit5(); callback=cb, abstol=abstol, reltol=reltol,
                     saveat=saveat, dense=false, save_everystep=false,
                     save_start=false, save_end=true)
        return sol.u[end], false
    end
end

# mean |A| over existing links (overall)
mean_abs_interaction(A::AbstractMatrix) = begin
    nz = [abs(A[i,j]) for i in axes(A,1), j in axes(A,2) if i!=j && A[i,j] != 0.0]
    isempty(nz) ? 0.0 : mean(nz)
end

# For bipartite matrices with R resources (optional but handy)
mean_abs_CR(A::AbstractMatrix, R::Int) = begin
    S = size(A,1)
    nz = [abs(A[i,j]) for i in (R+1):S, j in 1:R if A[i,j] != 0.0]
    isempty(nz) ? 0.0 : mean(nz)
end

mean_abs_RC(A::AbstractMatrix, R::Int) = begin
    S = size(A,1)
    nz = [abs(A[i,j]) for i in 1:R, j in (R+1):S if A[i,j] != 0.0]
    isempty(nz) ? 0.0 : mean(nz)
end
