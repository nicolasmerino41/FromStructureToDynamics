using Random, Statistics
using Distributions

# ------------------------------------------------------------
# Weighted sampling without replacement (simple, robust)
# ------------------------------------------------------------
function weighted_sample_without_replacement(items::Vector{Int}, weights::Vector{Float64}, k::Int;
        rng=Random.default_rng(), exclude::Union{Nothing,Int}=nothing)

    @assert length(items) == length(weights)
    k <= 0 && return Int[]

    # filter exclude
    if exclude !== nothing
        keep = [x != exclude for x in items]
        items = items[keep]
        weights = weights[keep]
    end

    n = length(items)
    k = min(k, n)
    k == 0 && return Int[]

    chosen = Int[]
    items_work = copy(items)
    w = copy(weights)

    for _ in 1:k
        sw = sum(w)
        sw <= 0 && break
        w ./= sw
        idx = rand(rng, Distributions.Categorical(w))
        push!(chosen, items_work[idx])
        deleteat!(items_work, idx)
        deleteat!(w, idx)
    end
    return chosen
end


# ------------------------------------------------------------
# FIXED PPM (Preferential Preying Model) with exact L links
# ------------------------------------------------------------
"""
    ppm(S, B, L, T; rng=...)

Preferential Preying Model adjacency matrix A (predator i eats prey j => A[i,j]=1).

Key properties:
- Basal species: 1:B (no prey)
- Consumers: B+1:S (eat only among 1:(i-1)) => acyclic by construction
- T controls prey similarity around the first prey using *provisional trophic levels*
- Enforces EXACTLY L links by adjusting after initial construction
"""
function ppm(S, B, L, T; rng=Random.default_rng())
    @assert 1 ≤ B < S
    @assert L ≥ (S - B)  "Need at least 1 prey per consumer (L ≥ S-B)."
    @assert T > 0

    A = zeros(Int, S, S)

    # provisional trophic levels during construction
    # basal start at 1
    s_prov = ones(Float64, S)

    consumers = (B+1):S
    ncons = length(consumers)

    # target mean prey per consumer (initial pass)
    k̄ = L / ncons

    total_links = 0

    # --- initial construction ---
    for i in consumers
        existing = collect(1:(i-1))
        n_i = length(existing)

        # choose prey count for this consumer (noisy around k̄, bounded)
        k_i = clamp(rand(rng, Poisson(k̄)), 1, n_i)

        # ensure feasibility: leave ≥1 link for each remaining consumer
        remaining_consumers = S - i
        remaining_links = L - total_links
        min_needed = remaining_consumers  # 1 each
        max_for_i = max(1, remaining_links - min_needed)
        k_i = min(k_i, max_for_i, n_i)

        # 1) first prey uniformly among existing
        j = rand(rng, existing)
        A[i, j] = 1
        prey_list = [j]

        # 2) additional prey biased by trophic-level distance to first prey
        if k_i > 1
            probs = [exp(-abs(s_prov[ℓ] - s_prov[j]) / T) for ℓ in existing]
            chosen = weighted_sample_without_replacement(existing, probs, k_i-1; rng=rng, exclude=j)
            for ℓ in chosen
                A[i, ℓ] = 1
            end
            append!(prey_list, chosen)
        end

        # update provisional trophic level of i (simple recursion)
        s_prov[i] = 1 + mean(s_prov[prey_list])

        total_links += k_i
    end

    # --- adjust to EXACTLY L links ---

    # add links if short
    while total_links < L
        i = rand(rng, consumers)
        existing = 1:(i-1)
        # candidates not already prey
        candidates = Int[]
        for ℓ in existing
            A[i, ℓ] == 0 && push!(candidates, ℓ)
        end
        isempty(candidates) && continue

        # anchor = one existing prey if any, else random
        preys = findall(@view(A[i, 1:(i-1)]) .== 1)
        anchor = isempty(preys) ? rand(rng, existing) : rand(rng, preys)

        weights = [exp(-abs(s_prov[ℓ] - s_prov[anchor]) / T) for ℓ in candidates]
        sw = sum(weights)
        sw <= 0 && continue
        weights ./= sw

        ℓ = candidates[rand(rng, Distributions.Categorical(weights))]
        A[i, ℓ] = 1
        total_links += 1

        # update s_prov[i]
        preys2 = findall(@view(A[i, 1:(i-1)]) .== 1)
        s_prov[i] = 1 + mean(s_prov[preys2])
    end

    # remove links if too many (keep ≥1 prey per consumer)
    while total_links > L
        i = rand(rng, consumers)
        preys = findall(@view(A[i, 1:(i-1)]) .== 1)
        length(preys) <= 1 && continue
        ℓ = rand(rng, preys)
        A[i, ℓ] = 0
        total_links -= 1

        # update s_prov[i]
        preys2 = findall(@view(A[i, 1:(i-1)]) .== 1)
        s_prov[i] = 1 + mean(s_prov[preys2])
    end

    # final, correct TL computation (your exact definition)
    s = trophic_levels(A)

    return A, s
end
