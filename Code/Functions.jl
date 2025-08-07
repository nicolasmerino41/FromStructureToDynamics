# === Helper Functions Script ===
# --------------------------------------------------------------------------------
# 1) gLV dynamics
# --------------------------------------------------------------------------------
# du[i] = u[i] * (K[i] - u[i] - (A*u)[i])
function gLV_rhs!(du, u, p, t)
    K, A = p
    Au = A * u
    @inbounds for i in eachindex(u)
        du[i] = u[i] * (K[i] - u[i] + Au[i])
    end
end

# --------------------------------------------------------------------------------
# 2) Create network
# --------------------------------------------------------------------------------
function make_network(
    A::AbstractMatrix, R::Int, conn::Float64, scenario::Symbol;
    IS = 1.0,
    pareto_exponent::Float64=1.75,
    pareto_minimum_degree=1.0,
    mod_gamma::Float64=5.0,
    B_term::Bool=false,
    B_term_IS::Float64=0.1
)
    S = size(A,1)
    C = S - R
    fill!(A, 0.0)

    if scenario == :ER
        for i in (R+1):S, j in 1:R
            if rand() < conn && i != j
                A[i,j] = abs(rand(Normal(0.0, IS)))
                A[j,i] = -abs(rand(Normal(0.0, IS)))
            end
        end
        
    elseif scenario == :PL
        raw = rand(Pareto(pareto_minimum_degree, pareto_exponent), C)
        ks  = clamp.(floor.(Int, raw), 1, R)
        for (idx,k) in enumerate(ks)
            i = R + idx
            for j in sample(1:R, min(k,R); replace=false)
                if i != j
                    A[i,j] = abs(rand(Normal(0.0, IS)))
                    A[j,i] = -abs(rand(Normal(0.0, IS)))
                end
            end
        end

    elseif scenario == :MOD
        halfR, halfC = fld(R,2), fld(C,2)
        res1, res2   = 1:halfR, (halfR+1):R
        con1, con2   = (R+1):(R+halfC), (R+halfC+1):S

        for i in (R+1):S, j in 1:R
            same = (i in con1 && j in res1) || (i in con2 && j in res2)
            p    = same ? conn*mod_gamma : conn/mod_gamma
            if rand() < clamp(p,0,1) && i != j
                A[i,j] = abs(rand(Normal(0.0, IS)))
                A[j,i] = -abs(rand(Normal(0.0, IS)))
            end
        end

    else
        error("Unknown scenario $scenario")
    end

    # optionally sprinkle in consumer-consumer predation
    if B_term
        for i in (R+1):S, j in (R+1):S
            if i != j && rand() < conn
                A[i,j] = abs(rand(Normal(0.0, B_term_IS)))
                A[j,i] = -abs(rand(Normal(0.0, B_term_IS)))
            end
        end
    end

    return A
end

# --------------------------------------------------------------------------------
# 3) Compute SL = B ./ K
# --------------------------------------------------------------------------------
function compute_SL(A::Matrix{Float64}, K::Vector{Float64})
    B = (I + A) 
            K    # equilibrium: (I + A) * B = K
    return B ./ K
end

# --------------------------------------------------------------------------------
# 4) Extract sigma/min(d) for Jacobian J
# --------------------------------------------------------------------------------
function sigma_over_min_d(A, J)
    d = -diag(J)
    if isempty(d)
        return NaN
    end
    min_d = minimum(d)
    offs = [A[i,j] for i in 1:size(A,1), j in 1:size(A,1) if i != j]
    if isempty(offs)
        return NaN
    end
    sigma = std(offs)
    return sigma / min_d
end

# --------------------------------------------------------------------------------
# 5) rewire in place
# --------------------------------------------------------------------------------
function rewire_A!(A, mask, sigma)
    A[mask] .= randn(sum(mask)) .* sigma
end

# --------------------------------------------------------------------------------
# 6) Equilibrium and feasibility
# --------------------------------------------------------------------------------
function calibrate_from_K_A(K::Vector{<:Real}, A::AbstractMatrix)
    # Solve (I + A) * u = K
    u = (I - A) \ K
    return u
end

function generate_feasible_thresholds(A::AbstractMatrix, R; margins=[1.0])
    S = size(A,1)
    out = NamedTuple[]
    for marg in margins
        K = abs.(rand(Normal(2.0, 1.0), S) .* marg)
        K[R+1:end] .= 0.01
        u_eq = try
            calibrate_from_K_A(K, A)
        catch
            continue
        end
        if any(!isfinite, u_eq) || any(u_eq .<= 0)
            continue
        end
        # stability check
        J = compute_jacobian_glv(u_eq, (K, A))
        # J = D * M
        if is_locally_stable(J)
            push!(out, (K=K, margin=marg, u_eq=copy(u_eq)))
        end
    end
    return out
end

# --------------------------------------------------------------------------------
# 7) Stability check
# --------------------------------------------------------------------------------
function is_locally_stable(J::AbstractMatrix)
    if any(!isfinite, J)
        return false
    end
    mu = eigvals(J)
    maximum(real.(mu)) <= 0
end

# --------------------------------------------------------------------------------
# 8) Survival simulation
# --------------------------------------------------------------------------------
function survives!(fixed, p; tspan=(0.,500.), cb)
    prob = ODEProblem(gLV_rhs!, fixed, tspan, p)
    sol  = solve(prob, Tsit5(); callback=cb, abstol=1e-8, reltol=1e-8)
    return sol.t[end]<tspan[2] ? (false, sol.u[end]) : (all(sol.u[end] .> 1e-6), sol.u[end])
end

# --------------------------------------------------------------------------------
# 9) Jacobian, resilience, reactivity
# --------------------------------------------------------------------------------
function compute_jacobian_glv(Bstar::AbstractVector, p)
    # g(u) returns f(u)the time‐derivative at state u
    g(u) = begin
        du = similar(u)
        gLV_rhs!(du, u, p, 0.0)
        return du
    end

    # allocate output
    S = length(Bstar)
    J = Matrix{Float64}(undef, S, S)

    # fill J in‐place
    ForwardDiff.jacobian!(J, g, Bstar)

    return J
end

# Resilience: maximum eigenvalue of the Jacobian.
function compute_resilience(u, p; extinct_species = false)
    if !extinct_species 
        J = compute_jacobian_glv(u, p)
        ev = eigvals(J)
        return maximum(real.(ev))
    else
        extant = findall(bi -> bi > EXTINCTION_THRESHOLD, u)
        J = compute_jacobian_glv(u, p)
        Jsub = J[extant, extant]
        ev = eigvals(Jsub)
        return maximum(real.(ev))
    end
end

# Reactivity: maximum eigenvalue of the symmetric part of the Jacobian.
function compute_reactivity(B, p; extinct_species = false)
    if !extinct_species 
        J = compute_jacobian_glv(B, p)
        J_sym = (J + J') / 2
        ev_sym = eigvals(J_sym)
        return maximum(real.(ev_sym))
    else
        extant = findall(bi -> bi > EXTINCTION_THRESHOLD, B)
        J = compute_jacobian_glv(B, p)
        Jsub = J[extant, extant]
        J_sym = (Jsub + Jsub') / 2
        ev_sym = eigvals(J_sym)
        return maximum(real.(ev_sym))
    end
end

# --------------------------------------------------------------------------------
# 10) Compute collectivity
# --------------------------------------------------------------------------------
function compute_collectivity(A::AbstractMatrix)
    vals = eigvals(A)
    return maximum(abs, vals)
end

# --------------------------------------------------------------------------------
# 11) Press and pulse perturbation functions
# --------------------------------------------------------------------------------
# Press perturbation
function simulate_press_perturbation_glv(
    u0, p, tspan, t_perturb, delta, R;
    solver=Tsit5(), plot=false, cb=nothing
)
    # Phase 1: pre-perturb
    prob1 = ODEProblem(gLV_rhs!, u0, (tspan[1], t_perturb), p)
    sol1  = solve(prob1, solver; callback=cb, abstol=1e-8, reltol=1e-8)
    pre_state = sol1.u[end]
    before_persistence = count(x -> x > 1e-6, pre_state) / length(pre_state)
    
    # Phase 2: perturb K by (1 - delta)
    K, A = p
    K_press = vcat(K[1:R] .* (1 .- delta), K[R+1:end])
    p_press = (K_press, A)
    prob2 = ODEProblem(gLV_rhs!, pre_state, (t_perturb, tspan[2]), p_press)
    sol2  = solve(prob2, solver; callback=cb, abstol=1e-8, reltol=1e-8)
    new_equil = sol2.u[end]
    after_persistence = count(x -> x > 1e-6, new_equil) / length(new_equil)

    # Return times
    n = length(new_equil)
    return_times = fill(NaN, n)
    for i in 1:n
        target = new_equil[i]
        for (t, state) in zip(sol2.t, sol2.u)
            if abs(state[i] - target) / (abs(target) + 1e-8) < 0.10
                return_times[i] = t - t_perturb
                break
            end
        end
    end

    return return_times, before_persistence, after_persistence, new_equil
end

# Pulse perturbation
function simulate_pulse_perturbation_glv(
    u0, p, tspan, t_pulse, delta;
    solver=Tsit5(), plot=false, cb=nothing
)
    # Phase 1: pre-pulse
    prob1 = ODEProblem(gLV_rhs!, u0, (tspan[1], t_pulse), p)
    sol1  = solve(prob1, solver; callback=cb, abstol=1e-8, reltol=1e-8)
    pre_state = sol1.u[end]
    before_persistence = count(x -> x > 1e-6, pre_state) / length(pre_state)

    # Apply pulse: u -> u * (1 + delta)
    pulsed = pre_state .* (1 .- delta)
    prob2 = ODEProblem(gLV_rhs!, pulsed, (t_pulse, tspan[2]), p)
    sol2  = solve(prob2, solver; callback=cb, abstol=1e-8, reltol=1e-8)
    eq_state = sol2.u[end]
    after_persistence = count(x -> x > 1e-6, eq_state) / length(eq_state)

    # Return times
    n = length(eq_state)
    return_times = fill(NaN, n)
    for i in 1:n
        target = eq_state[i]
        for (t, u) in zip(sol2.t, sol2.u)
            if abs(u[i] - target) / (abs(target) + 1e-8) < 0.10
                return_times[i] = t - t_pulse
                break
            end
        end
    end

    return return_times, before_persistence, after_persistence, eq_state
end

# --------------------------------------------------------------------------------
# 12) Callback builder
# --------------------------------------------------------------------------------
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

    # Build callback set => no forced extinction
    cb_no_trigger = CallbackSet(callbacks...)

    return cb_no_trigger
end

# --------------------------------------------------------------------------------
# 13) Analytical Median Return Rate
# --------------------------------------------------------------------------------
"""
    analytical_species_return_rates(J::AbstractMatrix; t::Real=0.01)

Computes the exact, species-level return rate vector
r_i(t) = [exp(t * J) * exp(t * J)']_{i,i} / (2 * t),
for i = 1:S, where S = size(J,1).  Returns an S-element Vector{Float64}.

In the short-time limit (t -> 0), this should approach -diag(J).
"""
function analytical_species_return_rates(J::AbstractMatrix; t::Real = 0.01)
    S = size(J, 1)
    # If J contains any non-finite values, return NaNs
    if any(!isfinite, J)
        return fill(NaN, S)
    end

    # Compute exp(t * J)
    E = exp(t * J)

    # Compute the product exp(tJ) * (exp(tJ))'
    M = E * transpose(E)

    # Extract the diagonal and divide by 2*t
    rates = Vector{Float64}(undef, S)
    for i in 1:S
        rates[i] = (M[i,i] - 1.0) / (2*t)
    end
    return -rates # flip the sign to make a positive return rate?? Is this correct?
end

# Computes the median across species of the analytical species-level return-rates.
function analytical_median_return_rate(J::AbstractMatrix; t::Real=0.01)
    rates = analytical_species_return_rates(J; t=t)
    return median(rates)
end
