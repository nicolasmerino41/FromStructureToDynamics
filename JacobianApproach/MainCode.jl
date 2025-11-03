############################
# 1) Helper functions
############################
using Random, Statistics, LinearAlgebra, DataFrames, Distributions, Printf
using CairoMakie
using Base.Threads
include("niche_model_builder.jl")
# ===================== Degree-controlled A builders =====================
# degree_family ∈ (:uniform, :lognormal, :pareto)
# :uniform    -> ER-like
# :lognormal  -> node propensities with given CV
# :pareto     -> propensities Pareto(x_min=1, α=deg_param)  (smaller α => heavier tail)

# -- helper: node weights per family
function _node_weights(S; degree_family::Symbol, deg_param::Float64, rng)
    if degree_family === :uniform
        return ones(Float64, S)
    elseif degree_family === :lognormal
        σ = deg_param <= 1e-12 ? 0.0 : sqrt(log(1 + deg_param^2))
        μ = -σ^2/2
        return rand(rng, LogNormal(μ, σ), S)  # mean ≈ 1
    elseif degree_family === :pareto
        α = max(deg_param, 1.01)             # finite mean
        return rand(rng, Pareto(1.0, α), S)  # mean = α/(α-1)
    else
        error("Unknown degree_family = $degree_family")
    end
end

# -- directed, non-trophic: P[i,j] ∝ w_out[i] * w_in[j], i≠j
# Directed, non-trophic with pairwise magnitude correlation:
# - Edge existence i→j and j→i is sampled independently (Chung–Lu style).
# - If BOTH directions exist, magnitudes are drawn with Corr(|Aij|,|Aji|)=rho_sym.
# - Signs are independent ±1 (no constraint).
function build_random_nontrophic(
    S; conn=0.10, mean_abs=0.10, mag_cv=0.60,
    degree_family::Symbol=:uniform, deg_param::Float64=0.0,
    rho_sym::Float64=0.0,
    rng=Random.default_rng()
)
    @assert 0.0 <= rho_sym <= 1.0 "rho_sym must be in [0,1]"

    A = zeros(Float64, S, S)
    # target expected number of directed edges
    E_target = clamp(round(Int, conn * S*(S-1)), 0, S*(S-1))

    # node propensities
    w_out = _node_weights(S; degree_family=degree_family, deg_param=deg_param, rng=rng)
    w_in  = _node_weights(S; degree_family=degree_family, deg_param=deg_param, rng=rng)

    # unnormalized probs (zero diagonal) -> scale to match E_target in expectation
    P = w_out .* transpose(w_in)
    @inbounds for i in 1:S; P[i,i] = 0.0; end
    s = (sum(P) > 0) ? (E_target / sum(P)) : 0.0
    @. P = min(1.0, s*P)

    # lognormal magnitude parameters
    σm = sqrt(log(1 + mag_cv^2))
    μm = log(mean_abs) - σm^2/2

    # map target corr of magnitudes -> corr of underlying normals
    # Corr(LogNormal) = (exp(σ^2*rZ) - 1) / (exp(σ^2) - 1)
    # => rZ = log(ρ*(e^{σ^2}-1)+1) / σ^2
    rZ = if σm == 0.0
        0.0
    else
        expσ2 = exp(σm^2)
        clamp(log(rho_sym * (expσ2 - 1) + 1) / σm^2, -1.0, 1.0)
    end

    # iterate unordered pairs; sample two directed edges per pair
    for i in 1:S-1, j in (i+1):S
        p_ij = P[i,j]
        p_ji = P[j,i]

        has_ij = (rand(rng) < p_ij)
        has_ji = (rand(rng) < p_ji)

        if !has_ij && !has_ji
            continue
        end

        if σm == 0.0
            # constant magnitudes
            m1 = mean_abs
            m2 = mean_abs
        elseif has_ij && has_ji && rho_sym > 0.0
            # draw correlated normals for magnitudes
            z1 = randn(rng)
            z2 = rZ*z1 + sqrt(max(0.0, 1 - rZ^2)) * randn(rng)
            m1 = exp(μm + σm*z1)
            m2 = exp(μm + σm*z2)
        else
            # at most one direction present or rho_sym==0: draw independent
            m1 = exp(μm + σm*randn(rng))
            m2 = exp(μm + σm*randn(rng))
        end

        # independent random signs (no constraint)
        if has_ij
            A[i,j] = (rand(rng) < 0.5 ? +m1 : -m1)
        end
        if has_ji
            A[j,i] = (rand(rng) < 0.5 ? +m2 : -m2)
        end
    end

    return A
end

# -- trophic (antisymmetric on unordered pairs): Pr(edge {i,j}) ∝ w[i]*w[j]
# ---------------------------------------------------------------
# build_random_trophic WITH trophic symmetry coefficient (rho_sym)
# ---------------------------------------------------------------
# -- trophic (antisymmetric on unordered pairs): Pr(edge {i,j}) ∝ w[i]*w[j]
# ρ_sym is the TARGET Pearson correlation of the two magnitudes |A_ij| and |A_ji|.
function build_random_trophic(
    S; conn=0.10, mean_abs=0.10, mag_cv=0.60,
    degree_family::Symbol=:uniform, deg_param::Float64=0.0,
    rho_sym::Float64 = 0.0,     # 0 => independent magnitudes, 1 => identical magnitudes
    rng=Random.default_rng()
)
    @assert 0.0 <= rho_sym <= 1.0 "rho_sym must be in [0,1]"

    A = zeros(Float64, S, S)

    # unordered pairs and target number of active pairs
    pairs = [(i,j) for i in 1:S for j in (i+1):S]
    E_target = clamp(round(Int, conn * length(pairs)), 0, length(pairs))

    # pair sampling weights (degree heterogeneity)
    w = _node_weights(S; degree_family=degree_family, deg_param=deg_param, rng=rng)
    W = [w[i]*w[j] for (i,j) in pairs]
    ZW = sum(W)
    ZW == 0 && return A
    s_pair = E_target / ZW  # global scale to hit expected connectance on unordered pairs

    # lognormal magnitude parameters
    σm = sqrt(log(1 + mag_cv^2))
    μm = log(mean_abs) - σm^2/2

    # If σm == 0 the magnitudes are constant; set them equal (any ρ is effectively satisfied)
    if σm == 0.0
        for (idx, (i,j)) in enumerate(pairs)
            p = min(1.0, s_pair * W[idx])
            if rand(rng) < p
                m = mean_abs
                if rand(rng) < 0.5
                    A[i,j] =  m;  A[j,i] = -m
                else
                    A[i,j] = -m;  A[j,i] =  m
                end
            end
        end
        return A
    end

    # Map target corr of magnitudes (rho_sym) -> corr of the underlying normals (rZ)
    # Corr(LogNormal) = (exp(σ^2*rZ) - 1) / (exp(σ^2) - 1)  =>  rZ = log(ρ*(e^{σ^2}-1)+1) / σ^2
    expσ2 = exp(σm^2)
    rZ = log(rho_sym * (expσ2 - 1) + 1) / σm^2
    rZ = clamp(rZ, -1.0, 1.0)  # numeric safety; for rho_sym∈[0,1] this is [0,1]

    # draw per-pair correlated lognormal magnitudes with Corr(|A_ij|,|A_ji|)=rho_sym
    for (idx, (i,j)) in enumerate(pairs)
        p = min(1.0, s_pair * W[idx])
        if rand(rng) < p
            # correlated standard normals
            z1 = randn(rng)
            z2 = rZ*z1 + sqrt(max(0.0, 1 - rZ^2)) * randn(rng)

            # lognormal magnitudes with matching marginals
            m1 = exp(μm + σm*z1)
            m2 = exp(μm + σm*z2)

            # random trophic direction (strict antisymmetry of signs)
            if rand(rng) < 0.5
                A[i,j] =  m1;  A[j,i] = -m2
            else
                A[i,j] = -m1;  A[j,i] =  m2
            end
        end
    end

    return A
end

# ----- u* generator -----
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    σ = sqrt(log(1 + cv^2)); μ = log(mean) - σ^2/2
    rand(rng, LogNormal(μ,σ), S)
end

# ----- Jacobian & metrics -----
jacobian(A,u) = Diagonal(u) * (A - I)
resilience(J) = maximum(real, eigvals(J))
reactivity(J) = maximum(real, eigvals((J + J')/2))

# Arnoldi median average return rate at time t
# C model: :uniform  -> C ∝ I
#          :biomass  -> C_ii ∝ (u_i)^2   (needs u)
function median_return_rate(
    J::AbstractMatrix, u::AbstractVector;
    t::Real=0.01, perturbation::Symbol=:biomass
)
    S = size(J,1)
    if S == 0 || any(!isfinite, J)
        return NaN
    end
    E = exp(t*J)
    if perturbation === :uniform
        # constants cancel in the log-ratio; use C = I for simplicity
        num = log(tr(E * transpose(E)))
        den = log(S)
    elseif perturbation === :biomass
        @assert u !== nothing "u is required for perturbation=:biomass"
        w = u .^ 2
        C = Diagonal(w)
        num = log(tr(E * C * transpose(E)))
        den = log(sum(w))
    else
        error("Unknown perturbation model: $perturbation")
    end
    return -(num - den) / (2t)
end

# ----- Stabilize by shrinking A (keep u* fixed) -----
"""
stabilize_shrink!(A,u; margin=0.05, factor=0.9, max_iter=200)

Scales A ← factor*A until λmax(Diag(u)*(A-I)) ≤ -margin (or iterations exhausted).
Returns (A_stab, shrink_alpha, lambda_max).
"""
function stabilize_shrink!(A::AbstractMatrix, u::AbstractVector; margin=0.05, factor=0.9, max_iter=200)
    @assert 0 < factor < 1
    S = size(A,1); @assert length(u) == S
    α = 1.0
    for _ in 1:max_iter
        J = jacobian(A, u)
        λ = maximum(real, eigvals(J))
        if λ <= -margin
            return (A, α, λ)
        end
        A .*= factor
        α *= factor
    end
    # final check/return even if not reached margin
    J = jacobian(A, u)
    λ = maximum(real, eigvals(J))
    return (A, α, λ)
end

# ----- α_off from J, build J from α & u -----
function alpha_off_from(J,u)
    S = length(u); α = zeros(Float64, S, S)
    @inbounds for i in 1:S, j in 1:S
        if i!=j && J[i,j] != 0.0
            α[i,j] = J[i,j] / u[i]
        end
    end
    α
end

function build_J_from(α::AbstractMatrix, u::AbstractVector)
    # indices of non-zero abundances
    nonzero_idx = findall(!iszero, u)
    n = length(nonzero_idx)

    # if all u are zero, return an empty 0×0 matrix
    if n == 0
        return zeros(Float64, 0, 0)
    end

    # extract corresponding submatrix and subvector
    α_sub = α[nonzero_idx, nonzero_idx]
    u_sub = u[nonzero_idx]

    # build Jacobian on the reduced system
    J = zeros(Float64, n, n)
    @inbounds for i in 1:n
        J[i,i] = -u_sub[i]
        for j in 1:n
            if i != j && α_sub[i,j] != 0.0
                J[i,j] = u_sub[i] * α_sub[i,j]
            end
        end
    end

    return J
end

# 1. Variance-preserving reshuffling
function op_reshuffle_alpha(α::AbstractMatrix; rng=Random.default_rng())
    S = size(α,1)
    nonzeros = [(i,j) for i in 1:S for j in 1:S if i != j && α[i,j] != 0.0]
    vals = [α[i,j] for (i,j) in nonzeros]
    perm = randperm(rng, length(vals))
    α_new = zeros(Float64, S, S)
    for (k, (i,j)) in enumerate(nonzeros)
        α_new[i,j] = vals[perm[k]]
    end
    α_new
end

"""
Variance-preserving pairwise reshuffle of off-diagonal α while preserving symmetry/antisymmetry.

Each interaction pair (i,j) and (j,i) is treated as a unit. Their joint pattern is shuffled
across all other pairs, ensuring that α'[i,j] = old[k,l] and α'[j,i] = old[l,k].

This preserves the global symmetry/antisymmetry structure.
"""
function op_reshuffle_alpha(α::AbstractMatrix; rng=Random.default_rng())
    S = size(α, 1)
    α_new = zeros(Float64, S, S)

    # Collect upper-triangular (i<j) pairs that have at least one nonzero
    pairs = [(i,j) for i in 1:S-1 for j in i+1:S if (α[i,j] != 0.0 || α[j,i] != 0.0)]

    # Extract original values for each pair as a tuple (αᵢⱼ, αⱼᵢ)
    vals = [(α[i,j], α[j,i]) for (i,j) in pairs]

    # Permute the order of these pairs
    perm = randperm(rng, length(pairs))

    # Reassign pairs according to permutation
    for (k, (i,j)) in enumerate(pairs)
        α_new[i,j], α_new[j,i] = vals[perm[k]]
    end

    # Keep diagonal unchanged
    for i in 1:S
        α_new[i,i] = α[i,i]
    end

    return α_new
end

# 2. Row mean averaging
function op_rowmean_alpha(α::AbstractMatrix)
    S = size(α,1)
    out = zeros(Float64, S, S)
    for i in 1:S
        nz = [abs(α[i,j]) for j in 1:S if i != j && α[i,j] != 0.0]
        if !isempty(nz)
            mi = mean(nz)
            for j in 1:S
                if i != j && α[i,j] != 0.0
                    out[i,j] = sign(α[i,j]) * mi
                end
            end
        end
    end
    out
end

# 3. Thresholding (remove weakest q%)
function op_threshold_alpha(α::AbstractMatrix; q=0.2)
    mags = [abs(α[i,j]) for i in 1:size(α,1), j in 1:size(α,2) if i != j && α[i,j] != 0.0]
    τ = isempty(mags) ? 0.0 : quantile(mags, q)
    S = size(α,1)
    out = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        if i != j && abs(α[i,j]) >= τ
            out[i,j] = α[i,j]
        end
    end
    out
end

# 4. Uniform abundances
uniform_u(u) = fill(mean(u), length(u))

# 5. Remove rarest species (lowest fraction p)
function remove_rarest_species(u::Vector{Float64}; p::Float64=0.1)
    u_cutoff = quantile(u, p)
    u_masked = copy(u)
    u_masked[u .< u_cutoff] .= 0.0
    u_masked
end

# ----- realized structure (post-stabilization) -----
function realized_connectance(A)
    S = size(A,1)
    nz = count(!iszero, A) - count(!iszero, diag(A))
    nz / (S*(S-1))
end
function realized_IS(A)
    mags = [abs(A[i,j]) for i in 1:size(A,1), j in 1:size(A,2) if i!=j && A[i,j]!=0.0]
    isempty(mags) ? 0.0 : mean(mags)
end
function degree_CVs(A)
    S = size(A,1)
    outdeg = [count(j->(j!=i && A[i,j]!=0.0), 1:S) for i in 1:S]
    indeg  = [count(i->(i!=j && A[i,j]!=0.0), 1:S) for j in 1:S]
    und    = falses(S,S)
    @inbounds for i in 1:S, j in 1:S
        if i!=j && (A[i,j]!=0.0 || A[j,i]!=0.0); und[i,j] = true; end
    end
    undeg = [count(und[i,:]) for i in 1:S]
    cv(v) = (m=mean(v); m>0 ? std(v)/m : NaN)
    (deg_cv_out=cv(outdeg), deg_cv_in=cv(indeg), deg_cv_all=cv(undeg))
end

# RNG splitter
@inline function _splitmix64(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
    z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
    z ⊻ (z >>> 31)
end

############################
# 2) Threaded sweep (stable communities)
############################
"""
run_sweep_stable(; ...)

Build A,u*, then **stabilize_shrink!** to ensure λmax(J) ≤ -margin.
Returns a DataFrame with realized structure (post-stabilization),
shrink_alpha, lambda_max, and res/rea for full + 6 steps.
"""
function run_sweep_stable(
    ; modes = [:TR],
      S_vals = [150],
      conn_vals = 0.05:0.05:0.30,
      mean_abs_vals = [0.05, 0.10, 0.20],
      mag_cv_vals   = [0.4, 0.6, 1.0],
      u_mean_vals   = [1.0],
      u_cv_vals     = [0.01, 0.25, 0.5, 1.0, 2.0],
      degree_families = [:uniform, :lognormal, :pareto],
      deg_cv_vals   = [0.0, 0.5, 1.0, 2.0],
      deg_pl_alphas = [1.2, 1.5, 2.0, 3.0],
      rho_sym_vals  = [0.0, 0.25, 0.5, 0.75, 1.0],
      margin = 0.05, shrink_factor = 0.9, max_shrink_iter = 200,
      reps_per_combo = 2,
      seed = 1234, number_of_combinations = 10_000,
      q_thresh = 0.20,
      long_time_value = 0.5,
      u_weighted_biomass = :biomass
)
    genA(mode, rho_sym, rng, conn, mean_abs, mag_cv, deg_fam, deg_param, S) = mode === :NT ? 
        build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                degree_family=deg_fam, deg_param=deg_param, 
                                rho_sym=rho_sym, rng=rng) :
        build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                             degree_family=deg_fam, deg_param=deg_param,
                             rho_sym=rho_sym, rng=rng)

    # expand degree specs
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
        u_mean_vals, u_cv_vals, deg_specs, 1:reps_per_combo,
        rho_sym_vals
    ))
    println("Computing $(number_of_combinations) of $(length(combos)) combinations")

    sel = (length(combos) > number_of_combinations) ?
          sample(combos, number_of_combinations; replace=false) : combos
    
    base = _splitmix64(UInt64(seed))
    bucket_main = [Vector{NamedTuple}() for _ in 1:nthreads()]
    bucket_t    = [Vector{NamedTuple}() for _ in 1:nthreads()]

    Threads.@threads for idx in eachindex(sel)
        (mode, S, conn, mean_abs, mag_cv, u_mean, u_cv, (deg_fam, deg_param), _, rho_sym) = sel[idx]
        
        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx)))
        rng  = Random.Xoshiro(_splitmix64(rand(rng0, UInt64)))

        # A = genA(mode, rho_sym, rng, conn, mean_abs, mag_cv, deg_fam, deg_param, S)

        A0 = build_niche_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                             degree_family=deg_fam, deg_param=deg_param,
                             rho_sym=rho_sym, rng=rng)

        baseIS = realized_IS(A0)
        baseIS == 0 && continue
        β = mean_abs / baseIS
        A = β .* A0

        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
        
        # Stabilize and compute metrics
        # A, αshrink, λmax = stabilize_shrink!(A, u; margin=margin, factor=shrink_factor)
        αshrink = 1.0
        λmax = 0.1
        J = jacobian(A, u)

        # realized structure AFTER stabilization
        conn_real = realized_connectance(A)
        IS_real   = realized_IS(A)
        degs      = degree_CVs(A)
        ucv_real  = (mean(u)>0 ? std(u)/mean(u) : NaN)

        # --- Compute α and transformations
        α = alpha_off_from(J, u)

        α_reshuf = op_reshuffle_alpha(α; rng=rng)
        α_row    = op_rowmean_alpha(α)
        α_thr    = op_threshold_alpha(α; q=q_thresh)

        # u_uni     = uniform_u(u)
        u_uni  = reshuffle_u(u; rng=rng)
        u_rarerem = remove_rarest_species(u; p=0.2)

        # ---- rewiring step (same generator & params, new arrangement) ----
        # A_rew = genA(mode, rho_sym, rng, conn, mean_abs, mag_cv, deg_fam, deg_param, S)
        # A_rew .*= αshrink               # keep post-shrink magnitude scale "as is"
        A_rew  = build_random_trophic_ER(
            S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
            rho_sym=rho_sym, rng=rng
        )
        βr = realized_IS(A_rew)
        J_rew = jacobian(βr .* A_rew, u)
        
        # J variants
        J_full   = J
        J_reshuf = build_J_from(α_reshuf, u)
        J_row    = build_J_from(α_row, u)
        J_thr    = build_J_from(α_thr, u)
        J_uni = build_J_from(α,u_uni)
        J_rarer  = build_J_from(α, u_rarerem)
        J_rew  = jacobian(A_rew, u)

        push!(bucket_main[threadid()], (;
            mode, S,
            conn_target=conn, mean_abs, mag_cv,
            u_mean_target=u_mean, u_cv_target=u_cv,
            degree_family = deg_fam, degree_param = deg_param,
            # realized post-stabilization
            conn_real=conn_real, IS_real=IS_real, u_cv=ucv_real,
            deg_cv_in=degs.deg_cv_in, deg_cv_out=degs.deg_cv_out, deg_cv_all=degs.deg_cv_all,
            rho_sym = rho_sym,
            shrink_alpha = αshrink, lambda_max = λmax,
            
            # full (stable)
            res_full = resilience(J_full), min_u = minimum(u), diff_res_min_u = -resilience(J_full) - minimum(u),
            rea_full = reactivity(J_full),
            rmed_full = median_return_rate(J_full, u; t=0.01, perturbation=u_weighted_biomass),
            long_rmed_full = median_return_rate(J_full, u; t=long_time_value, perturbation=u_weighted_biomass),
            
            res_rel_to_min_u_full = resilience(J_full) / minimum(u),
            rea_rel_to_min_u_full = reactivity(J_full) / minimum(u),
            rmed_rel_to_min_u_full = median_return_rate(J_full, u; t=0.01, perturbation=u_weighted_biomass) / minimum(u),

            # ---- STEPS ----
            res_reshuf = resilience(J_reshuf), rea_reshuf = reactivity(J_reshuf),
            rmed_reshuf = median_return_rate(J_reshuf, u; t=0.01, perturbation=u_weighted_biomass),
            long_rmed_reshuf = median_return_rate(J_reshuf, u; t=long_time_value, perturbation=u_weighted_biomass),
            
            res_thr = resilience(J_thr), rea_thr = reactivity(J_thr),
            rmed_thr = median_return_rate(J_thr, u; t=0.01, perturbation=u_weighted_biomass),
            long_rmed_thr = median_return_rate(J_thr, u; t=long_time_value, perturbation=u_weighted_biomass),
            
            res_row = resilience(J_row), rea_row = reactivity(J_row), 
            rmed_row = median_return_rate(J_row, u; t=0.01, perturbation=u_weighted_biomass),
            long_rmed_row = median_return_rate(J_row, u; t=long_time_value, perturbation=u_weighted_biomass),
            
            res_uni = resilience(J_uni), rea_uni = reactivity(J_uni),
            rmed_uni = median_return_rate(J_uni, u_uni; t=0.01, perturbation=u_weighted_biomass),
            long_rmed_uni = median_return_rate(J_uni, u_uni; t=long_time_value, perturbation=u_weighted_biomass),
            
            res_rarer = resilience(J_rarer), rea_rarer = reactivity(J_rarer),
            rmed_rarer = median_return_rate(J_rarer, filter(!iszero, u_rarerem); t=0.01, perturbation=u_weighted_biomass),
            long_rmed_rarer = median_return_rate(J_rarer, filter(!iszero, u_rarerem); t=long_time_value, perturbation=u_weighted_biomass),
            
            res_rew = resilience(J_rew), rea_rew = reactivity(J_rew),
            rmed_rew = median_return_rate(J_rew, u; t=0.01, perturbation=u_weighted_biomass),
            long_rmed_rew = median_return_rate(J_rew, u; t=long_time_value, perturbation=u_weighted_biomass),

            A=A, u=u, J_full=J_full
        ))

        # collect per-t raw rows
        for t in 10 .^ range(log10(0.01), log10(100.0); length=20)
            r_full = median_return_rate(J, u; t=t, perturbation=u_weighted_biomass)
            push!(bucket_t[threadid()], (;
                mode, S, conn, mean_abs, mag_cv, u_mean, u_cv,
                degree_family=deg_fam, degree_param=deg_param, rho_sym,
                IS_real=IS_real, t=t,
                r_full,
                r_row    = median_return_rate(J_row,  u;    t=t, perturbation=u_weighted_biomass),
                r_thr    = median_return_rate(J_thr,  u;    t=t, perturbation=u_weighted_biomass),
                r_reshuf = median_return_rate(J_reshuf,  u;    t=t, perturbation=u_weighted_biomass),
                r_rew    = median_return_rate(J_rew,  u;    t=t, perturbation=u_weighted_biomass),
                r_ushuf  = median_return_rate(J_uni,  u_uni; t=t, perturbation=u_weighted_biomass),
                r_rarer  = median_return_rate(J_rarer,  filter(!iszero, u_rarerem); t=t, perturbation=u_weighted_biomass)
            ))
        end
    end
    flat_main = reduce(vcat, (b for b in bucket_main if !isempty(b)); init=NamedTuple[])
    flat_t    = reduce(vcat, (b for b in bucket_t if !isempty(b)); init=NamedTuple[])

    df_main = isempty(flat_main) ? DataFrame() : DataFrame(flat_main)
    df_t    = isempty(flat_t)    ? DataFrame() : DataFrame(flat_t)

    return df_main, df_t
end

# --------------------------
# Minimal example run
# --------------------------
@time df_main, df_t = run_sweep_stable(
    ; modes=[:TR], S_vals=[120], conn_vals=0.05:0.05:0.30,
      mean_abs_vals=[0.05, 0.10, 0.40, 0.80, 1.20],
      mag_cv_vals=[0.01, 0.1, 0.5, 1.0, 2.0],
      u_mean_vals=[1.0], 
      u_cv_vals=[0.3,0.5,0.8,1.0,2.0,3.0],
      degree_families = [:uniform, :lognormal, :pareto],
      deg_cv_vals   = [0.0, 0.5, 1.0, 2.0],
      deg_pl_alphas = [1.2, 1.5, 2.0, 3.0],
      rho_sym_vals  = [0.0, 0.5, 1.0], # range(0, 1, length=10),
      reps_per_combo=4, seed=42, number_of_combinations=2000,
      margin=0.05, shrink_factor=0.9, max_shrink_iter=200, q_thresh=0.20,
      long_time_value=5.0,
      u_weighted_biomass=:biomass
)
print_structure_summary(df_tr)
# ----------------------------- plotting: correlations ----------------------------
"""
plot_correlations(df; steps=1:6, metrics=[:res, :rea])

Scatter of Full vs Step k with 1:1 line and R² to y=x.
"""
function plot_correlations(
    df::DataFrame;
    steps=["reshuf", "thr", "row", "uni", "rarer", "rew"],
    metrics=[:res, :rea, :rmed, long_rmed], title=""
)
    labels = Dict(:res=>"Resilience", :rea=>"Reactivity", :rmed=>"Rmed (t=0.01)", :long_rmed=>"Rmed (t=10.0)")
    colors = [:steelblue, :orangered, :seagreen, :purple]

    fig = Figure(size=(1100, 725))
    Label(fig[0, 2:6], title; fontsize=18, font=:bold, halign=:left)

    for (mi, m) in enumerate(metrics)
        xname = Symbol(m, :_full)
        for (si, s) in enumerate(steps)
            yname = Symbol(m, :_, s)

            xs = df[!, xname]   |> collect
            ys = df[!, yname]   |> collect
            x  = Float64[]; y = Float64[]
            @inbounds for i in eachindex(xs)
                xi=xs[i]
                yi=ys[i]
                if xi isa Real && yi isa Real && isfinite(xi) && isfinite(yi)
                    push!(x, float(xi)); push!(y, float(yi))
                end
            end
            if isempty(x)
                Axis(
                    fig[mi, si];
                    title="$(labels[m]) — S$s",
                    xgridvisible=false, ygridvisible=false)
                continue
            end

            mn = min(minimum(x), minimum(y))
            mx = max(maximum(x), maximum(y))
            if !(isfinite(mn) && isfinite(mx)) || mn == mx
                c = isfinite(mn) ? mn : 0.0
                pad = max(abs(c)*0.1, 1.0)
                mn, mx = c - pad, c + pad
            end

            ax = Axis(
                fig[mi, si];
                title="$(labels[m]) — $s",
                xlabel=string(xname), ylabel=string(yname),
                limits=((mn, mx), (mn, mx)),
                xgridvisible=false, ygridvisible=false,
                xticklabelsize=10, yticklabelsize=10,
                xlabelsize=12, ylabelsize=12,
                titlesize=11
            )

            scatter!(ax, x, y; color=colors[mi], markersize=4, alpha=0.35)
            lines!(ax, [mn, mx], [mn, mx]; color=:black, linestyle=:dash)

            μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
            r2 = sst == 0 ? NaN : 1 - ssr/sst
            isfinite(r2) && text!(ax, "R²=$(round(r2, digits=3))";
                                  position=(mx, mn), align=(:right,:bottom))
        end
    end
    display(fig)
end

# ----------------------------- run both modes, show plots ------------------------
df_tr_stable = filter(row -> row.shrink_alpha > 0.5, df_tr)
# Trophic
plot_correlations(
    df_tr_trial; metrics=[:res, :rea, :rmed, :long_rmed],
    title="Trophic (heterogeneus abundances, ρ=0.5) maintaining antisymmetry" # — Full vs 6-steps Jacobian simplifications")
)

# Trophic sweep
df_nt_stable = filter(row -> row.shrink_alpha > 0.5, df_nt)
plot_correlations(df_nt; metrics=[:res, :rea, :rmed, :long_rmed],
                  title="Non-trophic (heterogeneus abundances)")
###################################################################################
function plot_stability_metrics(df; title="Stability metrics vs symmetry coefficient")
    fig = Figure(size=(1000, 400))
    for (i, metric) in enumerate([:res_full, :rea_full, :rmed_full])
    # for (i, metric) in enumerate([:res_rel_to_min_u_full, :rea_rel_to_min_u_full, :rmed_rel_to_min_u_full])
        ax = Axis(fig[1, i],
            title=replace(string(metric), "_full" => ""),
            xlabel="Symmetry coefficient (ρ)",
            # ylabel = metric == :rmed_rel_to_min_u_full ? "Median Return Rate" :
            #           metric == :rea_rel_to_min_u_full ? "Reactivity" : "Resilience")
            ylabel = metric == :rmed_full ? "Median Return Rate" :
                      metric == :rea_full ? "Reactivity" : "Resilience")

        for mode in unique(df.mode)
            sub = df[df.mode .== mode, :]
            scatter!(ax, sub.rho_sym, sub[!, metric];
                     label=mode, alpha=0.4, markersize=5)
        end
        # axislegend(ax; position=:rb)
    end
    Label(fig[0, 1:3], title; fontsize=18, halign=:center)
    display(fig)
end

plot_stability_metrics(df_main; title="Stability metrics vs symmetry coefficient")

############################
# 3) Structural summary (unchanged)
############################
function print_structure_summary(df::DataFrame)
    cols = [
        (:conn_real,  "connectance"),
        (:IS_real,    "IS(mean|A|)"),
        (:u_cv,       "abundance CV"),
        (:deg_cv_in,  "degree CV (in)"),
        (:deg_cv_out, "degree CV (out)"),
        (:deg_cv_all, "degree CV (undirected)")
    ]
    groups = (:mode in names(df)) ? groupby(df, :mode) : [df]
    for g in groups
        hdr = (g isa SubDataFrame) ? "mode=$(only(unique(g.mode)))" : "All"
        println("\n--- ", hdr, " ---")
        for (c,label) in cols
            x = collect(skipmissing(g[!, c])) |> x->filter(isfinite, x)
            if isempty(x)
                println(rpad(label, 26), ": (no data)")
            else
                q = quantile(x, (0.10,0.50,0.90))
                println(rpad(label, 26), ": mean=$(round(mean(x),sigdigits=5))  ",
                        "sd=$(round(std(x),sigdigits=5))  ",
                        "p10=$(round(q[1],sigdigits=5))  med=$(round(q[2],sigdigits=5))  p90=$(round(q[3],sigdigits=5))")
            end
        end
    end
end

print_structure_summary(df_main)