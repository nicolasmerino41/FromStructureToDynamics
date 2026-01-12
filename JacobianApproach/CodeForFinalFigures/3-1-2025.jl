################################################################################
# FIXED-BASE, MANY-P DIRECTIONS TEST
#
# Goal:
#   Compare "old" ΔG(ω) approach vs Jeff sensitivity approach on the same base
#   system by varying perturbation directions P (rewire-like).
#
# Base dynamics:
#   J = diag(u) * Abar, with Abar_ii = -1
#   Generalized resolvent: R(ω) = (i*ω*T - Abar)^(-1), T = diag(1/u)
#
# Time metric:
#   biomass-weighted rmed(t) using C = diag(u^2)
#   bump = max_t | rmed_base(t) - rmed_pert(t) |
#
# Old frequency diagnostic:
#   G(ω) = || R(ω) * diag(u) ||_F^2 / sum(u^2)
#   ΔG(ω) = |G_base(ω) - G_pert(ω)| ; old_metric = peak_ω ΔG(ω)
#
# Jeff frequency diagnostic (single-system sensitivity):
#   S(ω) = ε^2 * || R(ω) * P * R(ω) * diag(u) ||_F^2 / sum(u^2)
#   jeff_metric = peak_ω S(ω)
#
# Perturbations:
#   P is derived from rewiring the off-diagonal part of Abar (diag fixed),
#   then normalized: P = (Abar_rew - Abar)/||Abar_rew - Abar||_F
#
# ε:
#   ε is fixed across all P for a given base, chosen as:
#     ε = eps_rel * ||offdiag(Abar)||_F
#   If a particular P makes perturbed system unstable, that sample is rejected
#   (NaN), so ε remains constant as desired.
################################################################################

# ---------------------------
# Helpers
# ---------------------------
meanfinite(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigen(J).values))

function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# Permute off-diagonal entries (including zeros), keeping diagonal fixed
function reshuffle_offdiagonal(M::AbstractMatrix; rng=Random.default_rng())
    S = size(M, 1)
    M2 = copy(Matrix(M))

    vals = Float64[]
    idxs = Tuple{Int,Int}[]
    for i in 1:S, j in 1:S
        if i != j
            push!(vals, float(M2[i,j]))
            push!(idxs, (i,j))
        end
    end

    perm = randperm(rng, length(vals))
    for k in 1:length(vals)
        (i,j) = idxs[k]
        M2[i,j] = vals[perm[k]]
    end
    return M2
end

# Extract off-diagonal part (diag -> 0)
function offdiag_part(M::AbstractMatrix)
    S = size(M,1)
    O = copy(Matrix(M))
    for i in 1:S
        O[i,i] = 0.0
    end
    return O
end

# ---------------------------
# One knob η: directionality / non-normality
# ---------------------------
"""
Build sparse random M, then
  Oη = U + (1-η)*L   (U upper, L lower)
η=0 ~ more bidirectional; η=1 ~ feedforward-ish.
"""
function make_O_eta(S::Int, η::Real; p::Real=0.05, σ::Real=1.0, rng=Random.default_rng())
    @assert 0.0 <= η <= 1.0
    M = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        i == j && continue
        rand(rng) < p && (M[i,j] = randn(rng) * σ)
    end
    U = triu(M, 1)
    L = tril(M, -1)
    return U + (1.0 - float(η)) * L
end

# ---------------------------
# Stabilize base by scaling interaction strength
# Base: Abar = -I + s*O  (diag -1)
# J = diag(u)*Abar = -diag(u) + s*diag(u)*O
# ---------------------------
function find_stable_scale(O::AbstractMatrix, u::AbstractVector;
                           s0::Real=1.0, margin::Real=1e-3, max_shrinks::Int=60)
    s = float(s0)
    J = -Diagonal(u) + s * (Diagonal(u) * O)
    α = spectral_abscissa(J)
    k = 0
    while !(isfinite(α) && α < -margin) && k < max_shrinks
        s *= 0.5
        J = -Diagonal(u) + s * (Diagonal(u) * O)
        α = spectral_abscissa(J)
        k += 1
    end
    return (isfinite(α) && α < -margin) ? s : NaN
end

# ---------------------------
# Biomass-weighted rmed(t)
# ---------------------------
function rmed_biomass(J::AbstractMatrix, u::AbstractVector; t::Real)
    E = exp(float(t) * J)
    any(!isfinite, E) && return NaN
    w = u .^ 2
    C = Diagonal(w)
    Ttr = tr(E * C * transpose(E))
    (!isfinite(Ttr) || Ttr <= 0) && return NaN
    r = -(log(Ttr) - log(sum(w))) / (2*float(t))
    return isfinite(r) ? r : NaN
end

function rmed_curve(J, u, tvals)
    r = Vector{Float64}(undef, length(tvals))
    for (i,t) in enumerate(tvals)
        r[i] = rmed_biomass(J, u; t=t)
    end
    return r
end

function bump_from_curves(tvals, r_base, r_pert)
    Δ = similar(r_base)
    for i in eachindex(r_base)
        Δ[i] = (isfinite(r_base[i]) && isfinite(r_pert[i])) ? abs(r_base[i]-r_pert[i]) : NaN
    end
    mask = map(isfinite, Δ)
    !any(mask) && return (bump=NaN, tstar=NaN)
    idx = argmax(Δ .* map(x -> isfinite(x) ? 1.0 : 0.0, Δ))
    return (bump=Δ[idx], tstar=float(tvals[idx]))
end

# ---------------------------
# Frequency diagnostics
# R(ω) = (i*ω*T - Abar)^(-1),  T=diag(1/u)
# Use solves rather than explicit inverse
# ---------------------------

# OLD gain: G(ω) = ||R diag(u)||_F^2 / sum(u^2)
function G_old_biomass(Abar::Matrix{Float64}, u::Vector{Float64}, ω::Float64)
    T = Diagonal(1.0 ./ u)
    Mω = Matrix{ComplexF64}(im*ω*T - Abar)
    U  = Matrix{ComplexF64}(Diagonal(u))   # diag(u) = sqrt(C)
    Y  = Mω \ U
    val = (norm(Y)^2) / sum(u.^2)
    return (isfinite(val) && val > 0) ? val : NaN
end

# Jeff sensitivity spectrum: S(ω) = ε^2 * || R P R diag(u) ||_F^2 / sum(u^2)
function S_jeff_biomass(Abar::Matrix{Float64}, P::Matrix{Float64},
                        u::Vector{Float64}, ω::Float64, ε::Float64)
    T = Diagonal(1.0 ./ u)
    Mω = Matrix{ComplexF64}(im*ω*T - Abar)

    U  = Matrix{ComplexF64}(Diagonal(u))
    Y  = Mω \ U                 # Y = R * diag(u)
    Z  = Matrix{ComplexF64}(P) * Y
    X  = Mω \ Z                 # X = R * P * R * diag(u)

    val = (ε^2) * (norm(X)^2) / sum(u.^2)
    return (isfinite(val) && val > 0) ? val : NaN
end

function peak_on_grid(ωvals::AbstractVector, g::AbstractVector)
    mask = map(i -> isfinite(g[i]) && g[i] > 0, eachindex(g))
    idxs = findall(mask)
    isempty(idxs) && return (peak=NaN, ωstar=NaN)
    gsub = g[idxs]; ωsub = ωvals[idxs]
    imax = argmax(gsub)
    return (peak=gsub[imax], ωstar=ωsub[imax])
end

# ---------------------------
# Build base systems (η × base_reps)
# ---------------------------
struct BaseSystem
    η::Float64
    u::Vector{Float64}
    Abar::Matrix{Float64}     # diag -1
    rmed_base::Vector{Float64}
    eps::Float64              # fixed for this base
    Gbase::Vector{Float64}    # G_old on ω grid for base (so old ΔG is faster)
end

function build_bases(; S::Int,
    η_grid, base_reps::Int,
    seed::Int,
    u_mean::Float64, u_cv::Float64,
    p::Float64, σ::Float64,
    margin::Float64,
    eps_rel::Float64,
    tvals::Vector{Float64},
    ωvals::Vector{Float64}
)
    nη = length(η_grid)
    bases = Vector{BaseSystem}()
    for (iη, η0) in enumerate(η_grid)
        η = float(η0)
        for b in 1:base_reps
            rng = MersenneTwister(seed + 1_000_000*iη + 10_003*b)

            u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))
            O = make_O_eta(S, η; p=p, σ=σ, rng=rng)

            s = find_stable_scale(O, u; s0=1.0, margin=margin)
            isfinite(s) || continue

            Abar = -Matrix{Float64}(I, S, S) + s * O
            J = Diagonal(u) * Abar
            rbase = rmed_curve(J, u, tvals)

            offA = offdiag_part(Abar)   # diag set to 0
            eps = eps_rel * norm(offA)

            # precompute base G(ω) once
            Gbase = Vector{Float64}(undef, length(ωvals))
            for (k, ω) in enumerate(ωvals)
                Gbase[k] = G_old_biomass(Abar, u, ω)
            end

            push!(bases, BaseSystem(η, u, Abar, rbase, eps, Gbase))
        end
    end
    return bases
end

# ---------------------------
# Perturbation generator: many P directions for a fixed base
# Rewire off-diagonal part of Abar while keeping diag -1.
# Then Pdir = (Abar_rew - Abar)/||...||_F
# ---------------------------
function sample_Pdir_from_rewire(Abar::Matrix{Float64}; rng=Random.default_rng())
    S = size(Abar,1)
    off = Abar + Matrix{Float64}(I, S, S)     # remove diag -1 => offdiag part (diag 0)
    off_rew = reshuffle_offdiagonal(off; rng=rng)
    Abar_rew = -Matrix{Float64}(I, S, S) + off_rew

    Δ = Abar_rew - Abar
    # Δ has diag 0 by construction
    nΔ = norm(Δ)
    nΔ == 0 && return nothing
    return Δ / nΔ
end

# ---------------------------
# One perturbation evaluation on a fixed base
# ---------------------------
############################
# MODIFIED eval_one_P (FULL FUNCTION)
# Replace your existing eval_one_P with this version.
############################
function eval_one_P(base::BaseSystem, Pdir::Matrix{Float64};
                    tvals::Vector{Float64}, ωvals::Vector{Float64}, margin::Float64)

    u = base.u
    Abar = base.Abar
    eps = base.eps

    # perturbed system (eps FIXED by design)
    Abarp = Abar + eps * Pdir
    Jp = Diagonal(u) * Abarp

    # reject if unstable (do not shrink eps here)
    αp = spectral_abscissa(Jp)
    if !(isfinite(αp) && αp < -margin)
        return nothing
    end

    # ---- time domain ----
    rpert = rmed_curve(Jp, u, tvals)
    Δt = delta_rmed_curve(base.rmed_base, rpert)

    # bump (as before)
    b = bump_from_curves(tvals, base.rmed_base, rpert)

    # robust timescales
    t50   = t50_from_delta(tvals, Δt; tail_frac=0.2, level=0.5)
    tslope = t_slope_from_delta(tvals, Δt)

    # ---- frequency domain ----
    # OLD: ΔG(ω) between two systems
    Gp = Vector{Float64}(undef, length(ωvals))
    for (k, ω) in enumerate(ωvals)
        Gp[k] = G_old_biomass(Abarp, u, ω)
    end
    ΔG = abs.(base.Gbase .- Gp)
    pm_old = peak_on_grid(ωvals, ΔG)
    ωbar_old = omega_centroid(ωvals, ΔG; frac_of_max=0.8)

    # JEFF: sensitivity spectrum S(ω) from single base
    Sω = Vector{Float64}(undef, length(ωvals))
    for (k, ω) in enumerate(ωvals)
        Sω[k] = S_jeff_biomass(Abar, Pdir, u, ω, eps)
    end
    pm_jeff = peak_on_grid(ωvals, Sω)
    ωbar_jeff = omega_centroid(ωvals, Sω; frac_of_max=0.8)

    return (
        bump=b.bump,
        tbump=b.tstar,     # keep if you still want argmax time
        t50=t50,
        tslope=tslope,

        old_peak=pm_old.peak,
        old_ω=pm_old.xstar,
        old_ωbar=ωbar_old,

        jeff_peak=pm_jeff.peak,
        jeff_ω=pm_jeff.xstar,
        jeff_ωbar=ωbar_jeff
    )
end

############################
# NEW HELPERS (TIMING + ω-CENTROID)
############################

"""
Return Δ(t) = |r_base(t) - r_pert(t)|, NaN where either is non-finite.
"""
function delta_rmed_curve(r_base::AbstractVector, r_pert::AbstractVector)
    @assert length(r_base) == length(r_pert)
    Δ = Vector{Float64}(undef, length(r_base))
    for i in eachindex(r_base)
        Δ[i] = (isfinite(r_base[i]) && isfinite(r_pert[i])) ? abs(r_base[i] - r_pert[i]) : NaN
    end
    return Δ
end

"""
Robust timescale: t50 = first time when Δ(t) reaches `level` of its late-time plateau.

Plateau estimate Δ∞ is the mean of the last `tail_frac` fraction of finite Δ values
(along the *existing* grid order).

Returns NaN if not enough finite points or plateau is non-positive.
"""
function t50_from_delta(tvals::AbstractVector, Δ::AbstractVector;
                        tail_frac::Real=0.2, level::Real=0.5)
    @assert length(tvals) == length(Δ)
    @assert 0 < tail_frac <= 1
    @assert 0 < level < 1

    # keep finite points (in original order)
    idx = findall(i -> isfinite(tvals[i]) && tvals[i] > 0 && isfinite(Δ[i]), eachindex(Δ))
    length(idx) < 5 && return NaN

    # plateau from tail
    m = max(2, ceil(Int, tail_frac * length(idx)))
    tail_idx = idx[end-m+1:end]
    Δ∞ = mean(Δ[tail_idx])
    (!isfinite(Δ∞) || Δ∞ <= 0) && return NaN

    target = level * Δ∞

    # first crossing with linear interpolation in t
    for k in 2:length(idx)
        i1 = idx[k-1]; i2 = idx[k]
        d1 = Δ[i1]; d2 = Δ[i2]
        (d1 < target && d2 >= target) || continue

        t1 = float(tvals[i1]); t2 = float(tvals[i2])
        d2 == d1 && return t2
        return t1 + (target - d1) * (t2 - t1) / (d2 - d1)
    end

    # never reaches target within grid
    return NaN
end

"""
Alternative robust timescale: time of maximum slope of Δ vs log(t).
This captures where "structure sensitivity turns on fastest".
Returns NaN if insufficient finite points.
"""
function t_slope_from_delta(tvals::AbstractVector, Δ::AbstractVector)
    @assert length(tvals) == length(Δ)
    idx = findall(i -> isfinite(tvals[i]) && tvals[i] > 0 && isfinite(Δ[i]), eachindex(Δ))
    length(idx) < 3 && return NaN

    best = -Inf
    tbest = NaN
    for k in 1:(length(idx)-1)
        i1 = idx[k]; i2 = idx[k+1]
        t1 = float(tvals[i1]); t2 = float(tvals[i2])
        d1 = Δ[i1]; d2 = Δ[i2]
        (t1 > 0 && t2 > 0) || continue

        s = (d2 - d1) / (log(t2) - log(t1))
        if isfinite(s) && s > best
            best = s
            tbest = sqrt(t1 * t2)  # geometric mid-point in time
        end
    end
    return isfinite(tbest) ? tbest : NaN
end

"""
Frequency summary that avoids boundary-pinned argmax:

Compute a geometric-mean "centroid" ω̄ of the top band of g(ω).
Band is defined by g(ω) >= frac_of_max * max(g).

Weights are g(ω) itself (so higher gain contributes more).
Returns NaN if insufficient points.
"""
function omega_centroid(ωvals::AbstractVector, g::AbstractVector;
                        frac_of_max::Real=0.8)
    @assert length(ωvals) == length(g)
    @assert 0 < frac_of_max <= 1

    # finite positive
    idx = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(g[i]) && g[i] > 0, eachindex(g))
    isempty(idx) && return NaN

    gmax = maximum(g[idx])
    (!isfinite(gmax) || gmax <= 0) && return NaN
    thresh = frac_of_max * gmax

    band = filter(i -> g[i] >= thresh, idx)
    length(band) < 2 && return NaN

    # weighted geometric mean
    wsum = 0.0
    lsum = 0.0
    for i in band
        wi = float(g[i])
        wsum += wi
        lsum += wi * log(float(ωvals[i]))
    end
    (wsum <= 0) && return NaN
    return exp(lsum / wsum)
end

# ---------------------------
# Main experiment: fixed bases, many P directions
# Thread over (base_index, P_rep)
# ---------------------------
############################
# MODIFIED run_fixedbase_manyP (FULL FUNCTION)
# Only changes are: store the new fields returned by eval_one_P.
############################
function run_fixedbase_manyP(;
    S::Int=120,
    η_grid = collect(range(0.0, 1.0; length=7)),
    base_reps::Int=5,
    P_reps::Int=50,
    seed::Int=1234,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    p::Real=0.05,
    σ::Real=1.0,
    margin::Real=1e-3,
    eps_rel::Real=0.10,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=25),
    ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=40)
)
    tvals = collect(float.(tvals))
    ωvals = collect(float.(ωvals))

    bases = build_bases(
        S=S, η_grid=η_grid, base_reps=base_reps, seed=seed,
        u_mean=float(u_mean), u_cv=float(u_cv),
        p=float(p), σ=float(σ), margin=float(margin),
        eps_rel=float(eps_rel),
        tvals=tvals, ωvals=ωvals
    )
    nb = length(bases)
    @info "Built $nb base systems (η × base_reps after stability)."

    bump      = fill(NaN, nb, P_reps)
    tbump     = fill(NaN, nb, P_reps)
    t50       = fill(NaN, nb, P_reps)
    tslope    = fill(NaN, nb, P_reps)

    old_peak  = fill(NaN, nb, P_reps)
    old_ωbar  = fill(NaN, nb, P_reps)

    jeff_peak = fill(NaN, nb, P_reps)
    jeff_ωbar = fill(NaN, nb, P_reps)

    rejected = fill(0, nb)
    accepted = fill(0, nb)

    njobs = nb * P_reps
    Threads.@threads for job in 1:njobs
        bi = (job - 1) ÷ P_reps + 1
        pr = (job - 1) % P_reps + 1

        base = bases[bi]
        rng = MersenneTwister(seed + 9_000_000*bi + 13_007*pr)

        Pdir = sample_Pdir_from_rewire(base.Abar; rng=rng)
        if Pdir === nothing
            @inbounds rejected[bi] += 1
            continue
        end

        out = eval_one_P(base, Pdir; tvals=tvals, ωvals=ωvals, margin=float(margin))
        if out === nothing
            @inbounds rejected[bi] += 1
            continue
        end

        bump[bi, pr]   = out.bump
        tbump[bi, pr]  = out.tbump
        t50[bi, pr]    = out.t50
        tslope[bi, pr] = out.tslope

        old_peak[bi, pr] = out.old_peak
        old_ωbar[bi, pr] = out.old_ωbar

        jeff_peak[bi, pr] = out.jeff_peak
        jeff_ωbar[bi, pr] = out.jeff_ωbar

        @inbounds accepted[bi] += 1
    end

    return (
        bases=bases, η_grid=η_grid,
        tvals=tvals, ωvals=ωvals,
        bump=bump, tbump=tbump,
        t50=t50, tslope=tslope,
        old_peak=old_peak, old_ωbar=old_ωbar,
        jeff_peak=jeff_peak, jeff_ωbar=jeff_ωbar,
        accepted=accepted, rejected=rejected
    )
end

# Atomic add helper for Int vectors (safe across threads)
function atomic_add!(v::Vector{Int}, idx::Int, val::Int)
    @inbounds Base.Threads.atomic_add!(Base.Threads.Atomic{Int}(pointer(v, idx)), val)
    return nothing
end

# ---------------------------
# Plot + compare correlations
# ---------------------------
############################
# MODIFIED summarize_and_plot (FULL FUNCTION)
# Adds timing correlations + timing scatterplots using ω-centroid and t50/tslope.
############################
function summarize_and_plot(res; figsize=(1800, 1100))
    bases = res.bases
    ηs = [b.η for b in bases]
    uniqη = sort(unique(ηs))

    bump = vec(res.bump)

    oldm  = vec(res.old_peak)
    jeffm = vec(res.jeff_peak)

    # ---- magnitude correlations (as before, but keep both) ----
    m_old  = map(i -> isfinite(bump[i]) && bump[i] > 0 && isfinite(oldm[i])  && oldm[i]  > 0, eachindex(bump))
    m_jeff = map(i -> isfinite(bump[i]) && bump[i] > 0 && isfinite(jeffm[i]) && jeffm[i] > 0, eachindex(bump))

    ρ_old  = (count(m_old)  >= 6) ? cor(log.(bump[m_old]),  log.(oldm[m_old]))  : NaN
    ρ_jeff = (count(m_jeff) >= 6) ? cor(log.(bump[m_jeff]), log.(jeffm[m_jeff])) : NaN

    @info "Pooled cor(log bump, log old_peakΔG)  = $ρ_old  (N=$(count(m_old)))"
    @info "Pooled cor(log bump, log jeff_peakS)  = $ρ_jeff (N=$(count(m_jeff)))"

    # ---- acceptance by η ----
    for η in uniqη
        idx = findall(x -> x == η, ηs)
        acc = sum(res.accepted[idx])
        rej = sum(res.rejected[idx])
        @info "η=$(round(η,digits=2))  accepted=$acc  rejected=$rej  (acc rate=$(acc/(acc+rej+1e-9)))"
    end

    # ---- mean bump by η ----
    mean_bump_by_eta = Dict{Float64, Float64}()
    for η in uniqη
        idx = findall(x -> x == η, ηs)
        vals = Float64[]
        for bi in idx
            append!(vals, vec(res.bump[bi, :]))
        end
        vals = filter(x -> isfinite(x) && x > 0, vals)
        mean_bump_by_eta[η] = isempty(vals) ? NaN : mean(vals)
    end

    # ---- timing metrics ----
    t50   = vec(res.t50)
    tslope = vec(res.tslope)

    ωbar_old  = vec(res.old_ωbar)
    ωbar_jeff = vec(res.jeff_ωbar)

    # use inverse frequency as a "time scale"
    invω_old = similar(ωbar_old, Float64)
    for i in eachindex(ωbar_old)
        invω_old[i] =
            (isfinite(ωbar_old[i]) && ωbar_old[i] > 0) ? 1.0 / ωbar_old[i] : NaN
    end

    invω_jeff = similar(ωbar_jeff, Float64)
    for i in eachindex(ωbar_jeff)
        invω_jeff[i] =
            (isfinite(ωbar_jeff[i]) && ωbar_jeff[i] > 0) ? 1.0 / ωbar_jeff[i] : NaN
    end

    # correlations for t50 and tslope
    mask_t50_old  = map(i -> isfinite(t50[i]) && t50[i] > 0 && isfinite(invω_old[i])  && invω_old[i]  > 0, eachindex(t50))
    mask_t50_jeff = map(i -> isfinite(t50[i]) && t50[i] > 0 && isfinite(invω_jeff[i]) && invω_jeff[i] > 0, eachindex(t50))

    mask_ts_old  = map(i -> isfinite(tslope[i]) && tslope[i] > 0 && isfinite(invω_old[i])  && invω_old[i]  > 0, eachindex(tslope))
    mask_ts_jeff = map(i -> isfinite(tslope[i]) && tslope[i] > 0 && isfinite(invω_jeff[i]) && invω_jeff[i] > 0, eachindex(tslope))

    ρ_t50_old  = (count(mask_t50_old)  >= 6) ? cor(log.(t50[mask_t50_old]),  log.(invω_old[mask_t50_old]))   : NaN
    ρ_t50_jeff = (count(mask_t50_jeff) >= 6) ? cor(log.(t50[mask_t50_jeff]), log.(invω_jeff[mask_t50_jeff])) : NaN

    ρ_ts_old  = (count(mask_ts_old)  >= 6) ? cor(log.(tslope[mask_ts_old]),  log.(invω_old[mask_ts_old]))   : NaN
    ρ_ts_jeff = (count(mask_ts_jeff) >= 6) ? cor(log.(tslope[mask_ts_jeff]), log.(invω_jeff[mask_ts_jeff])) : NaN

    @info "Timing cor(log t50,   log 1/ω̄): old = $ρ_t50_old   (N=$(count(mask_t50_old)))"
    @info "Timing cor(log t50,   log 1/ω̄): jeff= $ρ_t50_jeff  (N=$(count(mask_t50_jeff)))"
    @info "Timing cor(log tslope,log 1/ω̄): old = $ρ_ts_old    (N=$(count(mask_ts_old)))"
    @info "Timing cor(log tslope,log 1/ω̄): jeff= $ρ_ts_jeff   (N=$(count(mask_ts_jeff)))"

    # ---- plots ----
    fig = Figure(size=figsize)

    ax1 = Axis(fig[1,1];
        xlabel="η",
        ylabel="mean bump",
        yscale=log10,
        title="Time-domain mean effect (fixed bases, many P)"
    )
    xs = uniqη
    ys = [mean_bump_by_eta[η] for η in xs]
    scatter!(ax1, xs, ys, markersize=10)
    lines!(ax1, xs, ys, linewidth=3)

    ax2 = Axis(fig[1,2];
        xscale=log10, yscale=log10,
        xlabel="bump",
        ylabel="old: peak ΔG(ω)",
        title="Old predictor across many P"
    )
    scatter!(ax2, bump[m_old], oldm[m_old], markersize=6)

    ax3 = Axis(fig[2,1];
        xscale=log10, yscale=log10,
        xlabel="bump",
        ylabel="Jeff: peak sensitivity",
        title="Jeff predictor across many P"
    )
    scatter!(ax3, bump[m_jeff], jeffm[m_jeff], markersize=6)

    # Timing: t50 vs 1/ω̄ (old vs jeff)
    ax4 = Axis(fig[2,2];
        xscale=log10, yscale=log10,
        xlabel="t50 (half-plateau time)",
        ylabel="1/ω̄ (centroid timescale)",
        title="Timing: t50 vs resolvent timescale"
    )
    scatter!(ax4, t50[mask_t50_old],  invω_old[mask_t50_old],  markersize=5)
    scatter!(ax4, t50[mask_t50_jeff], invω_jeff[mask_t50_jeff], markersize=5)

    # Timing: tslope vs 1/ω̄
    ax5 = Axis(fig[3,1];
        xscale=log10, yscale=log10,
        xlabel="tslope (max turn-on slope time)",
        ylabel="1/ω̄ (centroid timescale)",
        title="Timing: tslope vs resolvent timescale"
    )
    scatter!(ax5, tslope[mask_ts_old],  invω_old[mask_ts_old],  markersize=5)
    scatter!(ax5, tslope[mask_ts_jeff], invω_jeff[mask_ts_jeff], markersize=5)

    # Predictor divergence (same as before but using peaks)
    m_both = map(i -> isfinite(oldm[i]) && oldm[i] > 0 && isfinite(jeffm[i]) && jeffm[i] > 0, eachindex(oldm))
    ax6 = Axis(fig[3,2];
        xscale=log10, yscale=log10,
        xlabel="old peak ΔG(ω)",
        ylabel="Jeff peak sensitivity",
        title="Do the predictors diverge?"
    )
    scatter!(ax6, oldm[m_both], jeffm[m_both], markersize=6)

    display(fig)
end

# ---------------------------
# MAIN RUN
# ---------------------------
tvals = 10 .^ range(log10(0.01), log10(100.0); length=100)
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=100)

res = run_fixedbase_manyP(
    S=120,
    η_grid=collect(range(0.0, 1.0; length=7)),
    base_reps=5,        # increase once everything works
    P_reps=50,          # lots of perturbation directions per base
    seed=1234,
    u_mean=1.0,
    u_cv=0.5,
    p=0.05,
    σ=1.0,
    margin=1e-3,
    eps_rel=0.5,       # if too many rejects, reduce to 0.05
    tvals=tvals,
    ωvals=ωvals
)

summarize_and_plot(res)