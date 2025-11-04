# =============================================================================
# (1) SIMPLE NICHE-MODEL TROPHIC BUILDER (with degree-family control)
# =============================================================================
# We assign each species a niche value n in (0,1). Consumers eat contiguous
# ranges of prey with smaller niche. Range lengths are drawn from a base
# distribution then shaped by a degree-family weight to match heterogeneity.
# Every realized consumer-prey edge is converted into a *pairwise trophic
# interaction* with opposite signs (prey->predator +, predator->prey -) and
# rho-controlled magnitude correlation across directions. We ensure the directed
# graph is connected.

# helper: degree-family weights (same semantics as your _node_weights)
function _deg_weights(S; degree_family::Symbol, deg_param::Float64, rng)
    if degree_family === :uniform
        return ones(Float64, S)
    elseif degree_family === :lognormal
        sigma = deg_param <= 1e-12 ? 0.0 : sqrt(log(1 + deg_param^2))
        mu = -sigma^2/2
        return rand(rng, LogNormal(mu, sigma), S)  # mean ~ 1
    elseif degree_family === :pareto
        alpha = max(deg_param, 1.01)
        return rand(rng, Pareto(1.0, alpha), S)
    else
        error("Unknown degree_family = $degree_family")
    end
end

"""
    build_niche_trophic(
        S; conn=0.10, mean_abs=0.10, mag_cv=0.60,
        degree_family::Symbol=:uniform, deg_param::Float64=0.0,
        rho_sym::Float64=0.0, rng=Random.default_rng())

Simplest niche-model food web:
- Draw niche values n_i ~ U(0,1) and sort by n (higher niche eats lower niche).
- Draw a *feeding range length* r_i for each consumer from Beta-like control,
  modulated by degree-family weights to introduce out-degree heterogeneity.
- For each consumer i, choose a contiguous prey range on (0, n_i], place edges to
  all prey j whose niche falls in the chosen window.
- Convert each consumer-prey edge into a *pairwise antisymmetric* interaction:
  predator impact on prey: -m1, prey impact on predator: +m2 with |m2| correlated
  to |m1| via rho_sym (0 -> independent magnitudes, 1 -> identical magnitudes).
- Ensure no species is isolated and the directed graph is (weakly) connected by
  stitching minimal extra links if needed (nearest-niche neighbor).
"""
function build_niche_trophic(
    S; conn=0.10, mean_abs=0.10, mag_cv=0.60,
    degree_family::Symbol=:uniform, deg_param::Float64=0.0,
    rho_sym::Float64=0.0, rng=Random.default_rng()
)
    @assert 0.0 <= rho_sym <= 1.0
    # 1) niches and ordering
    niche = sort(rand(rng, S))
    order = sortperm(niche)            # already sorted; keep explicit for clarity

    # 2) target number of unordered consumer-prey pairs
    pairs_target = round(Int, conn * (S * (S - 1) / 2))

    # 3) consumer range lengths driven by degree-family heterogeneity
    w = _deg_weights(S; degree_family=degree_family, deg_param=deg_param, rng=rng)
    w ./= mean(w)
    base_r = rand(rng, Beta(2, 8), S)         # skewed to short ranges by default
    r_len  = clamp.(base_r .* w, 0.0, 1.0)    # desired fraction of (0, n_i]

    # 4) place contiguous ranges and assemble unordered pairs set
    pairs = Tuple{Int,Int}[]  # store as (pred, prey) indices in niche order
    for i in 2:S  # lowest-niche species (i=1) cannot be a consumer
        # choose a range on (0, niche[i]] of length r_len[i]*niche[i]
        L = niche[i] * r_len[i]
        L == 0 && continue
        a = rand(rng) * max(niche[i] - L, 0)    # left boundary
        b = a + L
        for j in 1:i-1
            if a < niche[j] && niche[j] <= b
                push!(pairs, (i, j))           # i consumes j
            end
        end
    end

    # 5) adjust density toward target by thinning/augmenting
    if length(pairs) > pairs_target
        pairs = sample(rng, pairs, pairs_target; replace=false)
    elseif length(pairs) < pairs_target
        # augment by linking nearest-lower-niche prey where missing
        needed = pairs_target - length(pairs)
        exists = Set(pairs)
        addeds = 0
        for i in 2:S
            if addeds >= needed
                break
            end
            # attach to nearest lower j not yet linked
            j = i - 1
            if j >= 1 && !((i, j) in exists)
                push!(pairs, (i, j))
                push!(exists, (i, j))
                addeds += 1
            end
        end
    end

    # 6) ensure each species participates; stitch if isolated
    indeg = zeros(Int, S)
    outdeg = zeros(Int, S)
    for (i, j) in pairs
        outdeg[i] += 1
        indeg[j] += 1
    end
    for i in 1:S
        if indeg[i] + outdeg[i] == 0
            # connect to nearest neighbor by niche
            if i == 1
                push!(pairs, (2, 1))
                outdeg[2] += 1
                indeg[1] += 1
            else
                push!(pairs, (i, i - 1))
                outdeg[i] += 1
                indeg[i - 1] += 1
            end
        end
    end

    # 7) assign magnitudes with lognormal stats and rho-correlated pairwise abs
    sigma_m = sqrt(log(1 + mag_cv^2))
    mu_m = log(mean_abs) - sigma_m^2 / 2
    A = zeros(Float64, S, S)
    for (i, j) in pairs
        m1 = rand(rng, LogNormal(mu_m, sigma_m))  # |pred->prey|
        m2 = rho_sym * m1 + (1 - rho_sym) * rand(rng, LogNormal(mu_m, sigma_m))  # |prey->pred|
        # prey -> predator is positive; predator -> prey is negative
        A[i, j] = -m1
        A[j, i] = +m2
    end

    return A
end

# Convenience wrappers to plug into your pipeline
build_trophic = build_niche_trophic

# =============================================================================
# (5) UPDATED STEPS: REW = RANDOM NETWORK; UNI = ABUNDANCE RESHUFFLE
# =============================================================================
# Random network for REW: ignore niche (pure ensemble), maintain trophic signs
function build_random_trophic_ER(S; conn=0.10, mean_abs=0.10, mag_cv=0.60, rho_sym=0.0, rng=Random.default_rng())
    A = zeros(Float64, S, S)
    sigma_m = sqrt(log(1 + mag_cv^2))
    mu_m = log(mean_abs) - sigma_m^2 / 2
    # sample unordered pairs uniformly
    pairs = [(i, j) for i in 1:S for j in (i + 1):S]
    K = round(Int, conn * length(pairs))
    K = clamp(K, 0, length(pairs))
    sel = sample(rng, pairs, K; replace=false)
    for (i, j) in sel
        m1 = rand(rng, LogNormal(mu_m, sigma_m))
        m2 = rho_sym * m1 + (1 - rho_sym) * rand(rng, LogNormal(mu_m, sigma_m))
        if rand(rng) < 0.5
            A[i, j] = -m1
            A[j, i] = +m2  # i predator of j
        else
            A[i, j] = +m2
            A[j, i] = -m1  # j predator of i
        end
    end
    return A
end

# Abundance reshuffle step (replaces uniform_u)
reshuffle_u(u; rng=Random.default_rng()) = u[randperm(rng, length(u))]
