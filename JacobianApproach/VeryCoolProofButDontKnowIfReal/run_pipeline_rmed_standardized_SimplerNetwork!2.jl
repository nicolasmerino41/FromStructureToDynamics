function run_pipeline_rmed_standardized_SimplerNetwork!(
        B, S, L;
        q_targets = range(0.01, 1.5, length=9),    # still only sets number of bins
        replicates = 30,
        mag_abs = 1.0,
        mag_cv = 0.5,
        corr = 0.0,
        u_cv = 0.5,
        t_vals = 10 .^ range(log10(0.01), log10(100.0), length=60),
        u,
        connectance = 0.1,                         # central value
        connectance_range = (0.02, 0.3),           # NEW: variability in C per network
        oversample_factor = 4,                     # NEW: how many extra networks to draw
        rng = Random.default_rng()
    )

    # --- containers as before ---
    results   = Dict{Float64, Vector{Vector{Float64}}}()   # rmed curves
    Js        = Dict{Float64, Matrix{Float64}}()           # one Jacobian per bin
    τ_axes    = Dict{Float64, Vector{Float64}}()           # tau axes
    t95_vals  = Dict{Float64, Float64}()                   # t95 per bin

    n_bins = length(q_targets)

    # --- choose connectance regime ---
    Cmin, Cmax = connectance_range
    if Cmin > Cmax
        Cmin, Cmax = Cmax, Cmin
    end

    # --- Step 1: oversample networks with varying C to get a broad q distribution ---
    n_networks = oversample_factor * n_bins * replicates

    As = Vector{Matrix{Float64}}(undef, n_networks)
    Qs = Vector{Float64}(undef, n_networks)

    for k in 1:n_networks
        Ck = Cmin + (Cmax - Cmin) * rand(rng)
        A  = build_random_network(S; C=Ck, rng=rng)

        A_binary = Int.(A .> 0)
        s = trophic_levels(A_binary)
        q = trophic_coherence(A_binary, s)   # from Johnson 2014 implementation
        As[k] = A
        Qs[k] = q
    end

    q_min = minimum(Qs)
    q_max = maximum(Qs)

    # --- Step 2: define equal-width q-bins across the observed range ---
    edges  = collect(range(q_min, q_max; length = n_bins + 1))
    q_bins = edges[1:end-1]          # lowest value in each bin → label

    # For each bin, store indices of networks in that bin
    bin_members = [Int[] for _ in 1:n_bins]

    for k in 1:n_networks
        q = Qs[k]
        # bin index: last edge <= q, clamped to [1, n_bins]
        idx = searchsortedlast(edges, q)
        idx = clamp(idx, 1, n_bins)
        push!(bin_members[idx], k)
    end

    # --- Step 3: for each bin, pick up to `replicates` networks and run the usual pipeline ---
    for b in 1:n_bins
        q_label = q_bins[b]
        idxs = bin_members[b]

        if isempty(idxs)
            @warn "Bin $b (q_label ≈ $q_label) has no networks; skipping this bin."
            continue
        end

        # If we have more than `replicates` networks in this bin, subsample
        if length(idxs) > replicates
            idxs = shuffle(rng, idxs)[1:replicates]
        end

        curves_q = Vector{Vector{Float64}}()
        J_rep    = nothing

        for k in idxs
            A = As[k]

            # Interaction matrix (unchanged)
            W = build_interaction_matrix(A;
                mag_abs      = mag_abs,
                mag_cv       = mag_cv,
                corr_aij_aji = corr,
                rng          = rng
            )

            # Jacobian
            J = jacobian(W, u)

            if J_rep === nothing
                J_rep = J
                Js[q_label] = J   # representative Jacobian for this bin
            end

            # rmed curve as before
            push!(curves_q, compute_rmed_curve(J, u, t_vals))
        end

        # safety: if somehow no curves, skip
        if J_rep === nothing || isempty(curves_q)
            @warn "Bin $b (q_label ≈ $q_label) produced no curves; skipping."
            continue
        end

        # Distance curve and t95 from the representative J
        D_curve = compute_distance_curve(J_rep, u, t_vals)
        t95 = compute_t95(D_curve, t_vals)
        t95_vals[q_label] = t95
        τ_axes[q_label]   = compute_tau_axis(t_vals, t95)

        results[q_label] = curves_q

        println("Finished bin $b: q_label = $q_label, n_networks = $(length(curves_q)), t95 = $t95")
    end

    params = (
        mag_abs     = mag_abs,
        mag_cv      = mag_cv,
        corr        = corr,
        connectance = connectance,
        connectance_range = connectance_range,
    )
    return results, τ_axes, Js, t_vals, t95_vals, params
end
