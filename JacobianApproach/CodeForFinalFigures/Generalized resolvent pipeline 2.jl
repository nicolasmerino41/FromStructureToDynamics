###############################################################
# THREADED + HIGH-ACCEPTANCE PIPELINE
#  - One knob η ∈ [0,1] controlling directionality / non-normality
#  - Biomass-weighted rmed(t) ONLY (C = diag(u^2))
#  - Weighted generalized resolvent diagnostic matching C:
#      G_C(ω) = tr(R C R*) / tr(C),  R=(iωT - Ahat)^(-1)
#  - Rewiring = pure off-diagonal permutation
#  - Stability = symmetric common scaling:
#      compute s_orig, s_rew that make each stable,
#      then set s_common = min(s_orig, s_rew) and apply to BOTH
#  - Thread-friendly: parallel over all jobs (η × reps)
#
# Requires: Random, LinearAlgebra, Statistics, Distributions, CairoMakie
###############################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie
using Base.Threads

# ----------------------------
# Helpers
# ----------------------------
meanfinite(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigen(J).values))

function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# Pure rewiring: permute off-diagonal entries (including zeros)
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

# ----------------------------
# Single knob η: directionality / non-normality
# ----------------------------
"""
Make one sparse weight matrix M (diag=0), then form:
  O(η) = U + (1-η) L
where U = upper(M), L = lower(M).

η=0 => more bidirectional, η=1 => fully feedforward (upper triangular only).
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

# ----------------------------
# Stability scaling (fast, no rejection loops)
# ----------------------------
"""
Find a scaling s in (0, s0] such that J = -Diag(u) + s*O is stable with margin:
  α(J) < -margin
Uses halving; returns NaN if not found (should be rare with enough shrinks).
"""
function find_stable_scale(O::AbstractMatrix, u::AbstractVector;
                           s0::Real=1.0, margin::Real=1e-3, max_shrinks::Int=60)
    s = float(s0)
    J = -Diagonal(u) + s * O
    α = spectral_abscissa(J)
    k = 0
    while !(isfinite(α) && α < -margin) && k < max_shrinks
        s *= 0.5
        J = -Diagonal(u) + s * O
        α = spectral_abscissa(J)
        k += 1
    end
    return (isfinite(α) && α < -margin) ? s : NaN
end

# ----------------------------
# Biomass-weighted rmed(t) ONLY
# ----------------------------
function rmed_biomass(J::AbstractMatrix, u::AbstractVector; t::Real=0.01)
    E = exp(float(t) * J)
    any(!isfinite, E) && return NaN
    w = u .^ 2
    C = Diagonal(w)
    Ttr = tr(E * C * transpose(E))
    (!isfinite(Ttr) || Ttr <= 0) && return NaN
    r = -(log(Ttr) - log(sum(w))) / (2*float(t))
    return isfinite(r) ? r : NaN
end

function bump_from_rmed(tvals::AbstractVector, r1::AbstractVector, r2::AbstractVector)
    Δ = similar(r1)
    for i in eachindex(r1)
        Δ[i] = (isfinite(r1[i]) && isfinite(r2[i])) ? abs(r1[i] - r2[i]) : NaN
    end
    mask = map(i -> isfinite(Δ[i]), eachindex(Δ))
    !any(mask) && return (bump=NaN, tstar=NaN)
    idx = argmax(Δ .* map(x -> isfinite(x) ? 1.0 : 0.0, Δ))
    return (bump=Δ[idx], tstar=float(tvals[idx]))
end

# ----------------------------
# Jeff weighted generalized resolvent diagnostic
# ----------------------------
"""
Descriptor form consistent with your J:

Take T = diag(1/u) and Ahat = diag(1/u) * J  (so diag(Ahat) = -1).
Then R(iω) = (iω T - Ahat)^(-1).
"""
function descriptor_from_J(J::AbstractMatrix, u::AbstractVector)
    T = Diagonal(1.0 ./ u)
    Ahat = Diagonal(1.0 ./ u) * Matrix(J)
    return T, Ahat
end

"""
Weighted resolvent energy:
  G_C(ω) = tr(R C R*) / tr(C),  C=diag(u^2)
Compute via solves: tr(R C R*) = || R * diag(u) ||_F^2
So solve (iωT - Ahat) X = diag(u), then G = ||X||_F^2 / sum(u^2).
"""
function weighted_GC(J::AbstractMatrix, u::AbstractVector, ω::Real)
    T, Ahat = descriptor_from_J(J, u)
    M = Matrix{ComplexF64}(im*float(ω)*T - Ahat)
    X = M \ Matrix{ComplexF64}(Diagonal(u))
    val = (norm(X)^2) / sum(u.^2)
    return (isfinite(real(val)) && val > 0) ? real(val) : NaN
end

"""
Coarse→refine peak search for:
  Go(ω), Gr(ω), ΔG(ω)=|Go-Gr|
Returns:
  peakGo, peakΔG, ωstarΔG
"""
function peak_weighted_resolvent_metrics(J::AbstractMatrix, Jrew::AbstractMatrix, u::AbstractVector;
                                         ω_coarse, refine_factor::Real=30.0, n_refine::Int=60)
    # coarse scan
    Go = similar(ω_coarse, Float64)
    Gr = similar(ω_coarse, Float64)
    for (k, ω) in enumerate(ω_coarse)
        Go[k] = weighted_GC(J, u, ω)
        Gr[k] = weighted_GC(Jrew, u, ω)
    end
    Δ = abs.(Go .- Gr)

    # peakGo on coarse
    maskGo = map(i -> isfinite(Go[i]) && Go[i] > 0, eachindex(Go))
    peakGo = any(maskGo) ? maximum(Go[maskGo]) : NaN

    # find coarse peak of Δ to decide refinement window
    maskΔ = map(i -> isfinite(Δ[i]) && Δ[i] > 0, eachindex(Δ))
    if !any(maskΔ)
        return (peakGo=peakGo, peakΔG=NaN, ωstarΔG=NaN)
    end
    i0 = argmax(Δ .* map(x -> isfinite(x) ? 1.0 : 0.0, Δ))
    ω0 = float(ω_coarse[i0])

    # refine around ω0 on log-scale
    ωlo = ω0 / refine_factor
    ωhi = ω0 * refine_factor
    ω_ref = 10 .^ range(log10(ωlo), log10(ωhi); length=n_refine)

    Go2 = similar(ω_ref, Float64)
    Gr2 = similar(ω_ref, Float64)
    for (k, ω) in enumerate(ω_ref)
        Go2[k] = weighted_GC(J, u, ω)
        Gr2[k] = weighted_GC(Jrew, u, ω)
    end
    Δ2 = abs.(Go2 .- Gr2)

    maskΔ2 = map(i -> isfinite(Δ2[i]) && Δ2[i] > 0, eachindex(Δ2))
    peakΔG = any(maskΔ2) ? maximum(Δ2[maskΔ2]) : NaN
    ωstar  = any(maskΔ2) ? ω_ref[argmax(Δ2 .* map(x -> isfinite(x) ? 1.0 : 0.0, Δ2))] : NaN

    # improve peakGo estimate by including refined window too
    maskGo2 = map(i -> isfinite(Go2[i]) && Go2[i] > 0, eachindex(Go2))
    if any(maskGo2)
        peakGo = isfinite(peakGo) ? max(peakGo, maximum(Go2[maskGo2])) : maximum(Go2[maskGo2])
    end

    return (peakGo=peakGo, peakΔG=peakΔG, ωstarΔG=float(ωstar))
end

# ----------------------------
# Threaded pipeline across η × reps
# ----------------------------
"""
High acceptance because we do:
  s_orig = stable scale for (Oη,u)
  s_rew  = stable scale for (Orew,u)
  s_common = min(s_orig, s_rew)
and then build BOTH J and Jrew using s_common.

This keeps fairness and avoids long rejection runs.
"""
function run_eta_pipeline_threaded(;
    S::Int=120,
    η_grid = collect(range(0.0, 1.0; length=7)),
    reps_per_eta::Int=20,
    seed::Int=1234,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    p::Real=0.05,
    σ::Real=1.0,
    margin::Real=1e-3,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=25),
    ω_coarse = 10 .^ range(log10(1e-4), log10(1e4); length=40),
    refine_factor::Real=30.0,
    n_refine::Int=60
)
    nη = length(η_grid)
    nt = length(tvals)

    # outputs: (η index, rep index)
    bump   = fill(NaN, nη, reps_per_eta)
    tbump  = fill(NaN, nη, reps_per_eta)
    relpeak = fill(NaN, nη, reps_per_eta)   # peakΔG / peakGo
    ωstarΔ = fill(NaN, nη, reps_per_eta)

    # mean |Δrmed(t)| profiles by η (computed after)
    # store per-rep Δrmed(t) to average later
    absΔ_rmed = fill(NaN, nη, reps_per_eta, nt)

    # job indexing
    njobs = nη * reps_per_eta

    Threads.@threads for job in 1:njobs
        iη = (job - 1) ÷ reps_per_eta + 1
        rep = (job - 1) % reps_per_eta + 1
        η = float(η_grid[iη])

        # thread-safe RNG per job
        rng = MersenneTwister(seed + 10_000*iη + 97*rep)

        # draw u, build Oη and rewired Orew
        u = random_u(S; mean=float(u_mean), cv=float(u_cv), rng=rng)
        Oη = make_O_eta(S, η; p=p, σ=σ, rng=rng)
        Orew = reshuffle_offdiagonal(Oη; rng=rng)

        # find stable scales for both, then use common scale
        s1 = find_stable_scale(Oη,   u; s0=1.0, margin=margin)
        s2 = find_stable_scale(Orew, u; s0=1.0, margin=margin)
        if !(isfinite(s1) && isfinite(s2))
            continue
        end
        s = min(s1, s2)

        J    = -Diagonal(u) + s * Oη
        Jrew = -Diagonal(u) + s * Orew

        # compute rmed curves and absΔ(t)
        r1 = Vector{Float64}(undef, nt)
        r2 = Vector{Float64}(undef, nt)
        for (ti, t) in enumerate(tvals)
            r1[ti] = rmed_biomass(J,   u; t=t)
            r2[ti] = rmed_biomass(Jrew,u; t=t)
            absΔ_rmed[iη, rep, ti] = (isfinite(r1[ti]) && isfinite(r2[ti])) ? abs(r1[ti]-r2[ti]) : NaN
        end

        b = bump_from_rmed(tvals, r1, r2)
        bump[iη, rep]  = b.bump
        tbump[iη, rep] = b.tstar

        # weighted resolvent metrics (coarse→refine)
        pm = peak_weighted_resolvent_metrics(J, Jrew, u;
                                             ω_coarse=ω_coarse,
                                             refine_factor=refine_factor,
                                             n_refine=n_refine)
        if isfinite(pm.peakGo) && pm.peakGo > 0 && isfinite(pm.peakΔG) && pm.peakΔG > 0
            relpeak[iη, rep] = pm.peakΔG / pm.peakGo
        end
        ωstarΔ[iη, rep] = pm.ωstarΔG
    end

    # compute mean |Δ rmed(t)| by η
    mean_absΔ_by_eta = Dict{Float64, Vector{Float64}}()
    for (iη, η) in enumerate(η_grid)
        m = Vector{Float64}(undef, nt)
        for ti in 1:nt
            m[ti] = meanfinite(vec(absΔ_rmed[iη, :, ti]))
        end
        mean_absΔ_by_eta[float(η)] = m
    end

    return (
        S=S, η_grid=η_grid, reps_per_eta=reps_per_eta,
        tvals=tvals,
        bump=bump, tbump=tbump, relpeak=relpeak, ωstarΔ=ωstarΔ,
        mean_absΔ_by_eta=mean_absΔ_by_eta
    )
end

# ----------------------------
# Plotting
# ----------------------------
function plot_eta_results(res; figsize=(1500, 900))
    tvals = res.tvals
    ηs = float.(res.η_grid)
    nη = length(ηs)

    # mean bump vs η
    mean_bump = [meanfinite(filter(x -> isfinite(x) && x > 0, vec(res.bump[i, :]))) for i in 1:nη]

    # pooled scatters
    bump = vec(res.bump)
    tbump = vec(res.tbump)
    relpeak = vec(res.relpeak)
    ωstar = vec(res.ωstarΔ)
    tchar = 1.0 ./ ωstar

    m1 = map(i -> isfinite(bump[i]) && bump[i] > 0 && isfinite(relpeak[i]) && relpeak[i] > 0, eachindex(bump))
    m2 = map(i -> isfinite(tbump[i]) && tbump[i] > 0 && isfinite(tchar[i]) && tchar[i] > 0, eachindex(tbump))

    ρ_b = (count(m1) >= 6) ? cor(log.(bump[m1]),  log.(relpeak[m1])) : NaN
    ρ_t = (count(m2) >= 6) ? cor(log.(tbump[m2]), log.(tchar[m2]))  : NaN
    @info "pooled cor(log bump, log relpeak) = $ρ_b (N=$(count(m1)))"
    @info "pooled cor(log t_bump, log 1/ω*)  = $ρ_t (N=$(count(m2)))"

    fig = Figure(size=figsize)

    # (1) mean |Δrmed(t)| profiles for each η
    ax1 = Axis(fig[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="mean |Δ rmed(t)| (weighted)",
        title="Structure sensitivity profile vs time (by η)"
    )
    for η in sort(ηs)
        lines!(ax1, tvals, res.mean_absΔ_by_eta[η], linewidth=3, label="η=$(round(η,digits=2))")
    end
    axislegend(ax1, position=:rb)

    # (2) mean bump vs η
    ax2 = Axis(fig[1,2];
        xlabel="η (directionality knob)",
        ylabel="mean bump",
        yscale=log10,
        title="Does bump strength increase with η?"
    )
    scatter!(ax2, ηs, mean_bump, markersize=10)
    lines!(ax2, ηs, mean_bump, linewidth=3)

    # (3) pooled bump vs relpeak
    ax3 = Axis(fig[2,1];
        xscale=log10, yscale=log10,
        xlabel="bump = max_t |Δ rmed|",
        ylabel="relpeak = peakΔG / peakGo",
        title="Bump vs weighted resolvent diagnostic (pooled)"
    )
    scatter!(ax3, bump[m1], relpeak[m1], markersize=8)

    # (4) pooled time matching
    ax4 = Axis(fig[2,2];
        xscale=log10, yscale=log10,
        xlabel="t_bump",
        ylabel="1/ω*_ΔG",
        title="Bump timescale vs resolvent timescale (pooled)"
    )
    scatter!(ax4, tbump[m2], tchar[m2], markersize=8)

    display(fig)
end

# ----------------------------
# Main run
# ----------------------------
# Tip: if you already use Julia's multithreading, consider avoiding BLAS oversubscription:
#   using LinearAlgebra; BLAS.set_num_threads(1)
# But only do that if you see CPU thrashing.

tvals = 10 .^ range(log10(0.01), log10(100.0); length=25)
ω_coarse = 10 .^ range(log10(1e-4), log10(1e4); length=40)

res = run_eta_pipeline_threaded(
    S=120,
    η_grid=collect(range(0.0, 1.0; length=7)),
    reps_per_eta=50,
    seed=1234,
    u_mean=1.0,
    u_cv=0.5,
    p=0.05,
    σ=1.0,
    margin=1e-3,
    tvals=tvals,
    ω_coarse=ω_coarse,
    refine_factor=30.0,
    n_refine=60
)

plot_eta_results(res)
