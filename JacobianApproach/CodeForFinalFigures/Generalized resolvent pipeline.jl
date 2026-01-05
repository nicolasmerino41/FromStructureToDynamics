###############################################
# Generalized resolvent pipeline (Jeff’s idea)
#   central object: R(iω) = (iω T - A)^(-1)
#   where T = diag(time scales) and A_ii = -1
#
# This script:
#  1) builds stable communities with broad non-normality (your builder)
#  2) creates a rewired counterpart (off-diagonal shuffle)
#  3) computes rmed(t) curves and “bump strength” between original vs rewired
#  4) computes resolvent gain curves vs ω and extracts peak metrics
#  5) plots associations: bump ↔ resolvent peak features; t* ↔ 1/ω*
###############################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# ============================================================
# 0) Small utilities
# ============================================================

meanfinite(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))

function normalize_probs(dict::Dict{Symbol,Float64})
    ks = collect(keys(dict))
    ps = [dict[k] for k in ks]
    s = sum(ps)
    s <= 0 && error("family_probs must sum to > 0")
    return ks, ps ./ s
end

spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigen(J).values))

# ============================================================
# 1) rmed(t) and t95 (as in your work)
# ============================================================

function median_return_rate(J::AbstractMatrix, u::AbstractVector; t::Real=0.01, perturbation::Symbol=:biomass)
    S = size(J,1)
    if S == 0 || any(!isfinite, J)
        return NaN
    end
    E = exp(float(t)*J)
    if any(!isfinite, E)
        return NaN
    end

    if perturbation === :uniform
        Ttr = tr(E * transpose(E))
        if !isfinite(Ttr) || Ttr <= 0
            return NaN
        end
        num = log(Ttr)
        den = log(S)
    elseif perturbation === :biomass
        w = u .^ 2
        C = Diagonal(w)
        Ttr = tr(E * C * transpose(E))
        if !isfinite(Ttr) || Ttr <= 0
            return NaN
        end
        num = log(Ttr)
        den = log(sum(w))
    else
        error("Unknown perturbation model: $perturbation")
    end

    r = -(num - den) / (2*float(t))
    return isfinite(r) ? r : NaN
end

function t95_from_rmed(tvals::AbstractVector, rmed::AbstractVector; target::Real=0.05)
    @assert length(tvals) == length(rmed)
    @assert 0 < target < 1

    y_prev = NaN
    t_prev = NaN

    for i in eachindex(tvals)
        ti = float(tvals[i])
        ri = float(rmed[i])
        if !(isfinite(ti) && isfinite(ri) && ti > 0)
            continue
        end
        yi = exp(-ri * ti)
        if !isfinite(yi); continue; end

        if yi <= target
            if !(isfinite(y_prev) && isfinite(t_prev))
                return ti
            end
            # log-linear interpolation (more stable)
            ℓ1, ℓ2, ℓt = log(y_prev), log(yi), log(float(target))
            if ℓ2 == ℓ1
                return ti
            end
            return t_prev + (ℓt - ℓ1) * (ti - t_prev) / (ℓ2 - ℓ1)
        end

        y_prev = yi
        t_prev = ti
    end

    return Inf
end

# ============================================================
# 2) Your “family builders” for diverse non-normality
#    (returns OFFDIAGONAL matrix O)
# ============================================================

function offdiag_part(M::AbstractMatrix)
    S = size(M,1)
    O = copy(Matrix(M))
    for i in 1:S
        O[i,i] = 0.0
    end
    return O
end

function stabilize_by_shrinking_offdiag(O::AbstractMatrix, u::AbstractVector;
                                        s0::Real=1.0, margin::Real=1e-3,
                                        max_shrinks::Int=40)
    S = length(u)
    @assert size(O,1)==S && size(O,2)==S
    @assert all(u .> 0)

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
    return J, s, α
end

function O_symmetric_like(S; p=0.05, rng=Random.default_rng())
    B = zeros(Float64, S, S)
    for i in 1:S, j in i+1:S
        if rand(rng) < p
            v = randn(rng)
            B[i,j] = v
            B[j,i] = v
        end
    end
    return B
end

function O_asymmetric(S; p=0.05, rng=Random.default_rng())
    B = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        i == j && continue
        rand(rng) < p && (B[i,j] = randn(rng))
    end
    return B
end

function O_feedforward(S; p=0.05, rng=Random.default_rng())
    B = zeros(Float64, S, S)
    for i in 1:S-1, j in i+1:S
        rand(rng) < p && (B[i,j] = randn(rng))
    end
    return B
end

function O_jordan(S)
    B = zeros(Float64, S, S)
    for i in 1:S-1
        B[i, i+1] = 1.0
    end
    return B
end

function O_block_feedforward(S; nblocks=4, pin=0.08, pout=0.02, rng=Random.default_rng())
    B = zeros(Float64, S, S)
    cuts = round.(Int, range(0, S; length=nblocks+1))
    cuts[1] = 0; cuts[end] = S
    blocks = [(cuts[k]+1):(cuts[k+1]) for k in 1:nblocks]

    for b in 1:nblocks
        idx = blocks[b]
        for i in idx, j in idx
            i == j && continue
            rand(rng) < pin && (B[i,j] = randn(rng))
        end
    end

    for b1 in 1:nblocks-1
        for b2 in b1+1:nblocks
            for i in blocks[b1], j in blocks[b2]
                rand(rng) < pout && (B[i,j] = randn(rng))
            end
        end
    end

    for i in 1:S
        B[i,i] = 0.0
    end
    return B
end

# κ metrics (optional)
function mode_kappas(J::AbstractMatrix)
    S = size(J,1)
    try
        F = eigen(J)
        V = F.vectors
        Y = inv(V)'   # Y'V = I
        κ = [norm(V[:,i]) * norm(Y[:,i]) for i in 1:S]
        return κ
    catch
        return fill(NaN, S)
    end
end

function kappa_mean_max_sum(κ::AbstractVector)
    vals = filter(x -> isfinite(x) && x > 0, κ)
    isempty(vals) && return (NaN, NaN, NaN)
    return (mean(vals), maximum(vals), sum(vals))
end

# Random u
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

"""
Build a random community A with broad range of non-normality/conditioning.
Returns:
- A (diag(A)=0) such that J = Diagonal(u)*(A - I) is stable (if stabilize=true)
- meta NamedTuple
"""
function build_A_diverse(u::AbstractVector;
    family_probs = Dict(
        :symmetric_like => 0.20,
        :asymmetric     => 0.25,
        :feedforward    => 0.25,
        :jordan         => 0.15,
        :block_ff       => 0.15
    ),
    p::Real=0.05,
    nblocks::Int=4,
    amp_log10_range = (-2.0, 2.0),
    stabilize::Bool=true,
    margin::Real=1e-3,
    rng=Random.default_rng()
)
    S = length(u)
    @assert S > 1
    @assert all(u .> 0)

    fams, probs = normalize_probs(family_probs)
    fam = rand(rng, Distributions.Categorical(probs)) |> i -> fams[i]

    O = if fam == :symmetric_like
        O_symmetric_like(S; p=p, rng=rng)
    elseif fam == :asymmetric
        O_asymmetric(S; p=p, rng=rng)
    elseif fam == :feedforward
        O_feedforward(S; p=p, rng=rng)
    elseif fam == :jordan
        O_jordan(S)
    elseif fam == :block_ff
        O_block_feedforward(S; nblocks=nblocks, pin=2p, pout=p/2, rng=rng)
    else
        error("Unknown family $fam")
    end

    lo, hi = amp_log10_range
    amp_draw = 10.0^(rand(rng)*(hi-lo) + lo)
    O .*= amp_draw

    if stabilize
        J, amp_used, α = stabilize_by_shrinking_offdiag(O, u; s0=1.0, margin=margin)
        Dinv = Diagonal(1.0 ./ u)
        A = Dinv * J + I
        for i in 1:S
            A[i,i] = 0.0
        end
        κ = mode_kappas(J)
        kmean, kmax, _ = kappa_mean_max_sum(κ)
        return A, (family=fam, amp_draw=amp_draw, amp_used=amp_used, alpha=α, kmean=kmean, kmax=kmax)
    else
        J = -Diagonal(u) + O
        α = spectral_abscissa(J)
        Dinv = Diagonal(1.0 ./ u)
        A = Dinv * J + I
        for i in 1:S
            A[i,i] = 0.0
        end
        κ = mode_kappas(J)
        kmean, kmax, _ = kappa_mean_max_sum(κ)
        return A, (family=fam, amp_draw=amp_draw, amp_used=1.0, alpha=α, kmean=kmean, kmax=kmax)
    end
end

# ============================================================
# 3) Rewiring move (off-diagonals only)
# ============================================================

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

# ============================================================
# 4) Your propagator-aligned metrics (given)
# ============================================================

function peak_singular_gain(J::AbstractMatrix, tvals::AbstractVector)
    best = -Inf
    tbest = float(tvals[1])
    for t in tvals
        E = exp(float(t) * J)
        g2 = opnorm(E)^2
        if isfinite(g2) && g2 > best
            best = g2
            tbest = float(t)
        end
    end
    return best, tbest
end

function numerical_abscissa(J::AbstractMatrix)
    return maximum(eigen(Symmetric((J + J')/2)).values)
end

# ============================================================
# 5) Jeff’s generalized resolvent on imaginary axis
#     R(iω) = (iω T - A)^(-1), with A_ii=-1
# ============================================================

"""
Make (T, Ahat) consistent with your J = Diagonal(u)*(A - I).

Your A here has diag(A)=0.
Define Ahat = A - I, so diag(Ahat) = -1.
Define T = Diagonal(1 ./ u), giving descriptor form:
    T xdot = Ahat x
which matches Jeff’s (zT - Ahat)^-1.
"""
function descriptor_matrices(A::AbstractMatrix, u::AbstractVector)
    S = size(A,1)
    @assert size(A,2) == S
    @assert length(u) == S
    Ahat = Matrix(A) - I       # diag = -1
    T = Diagonal(1.0 ./ u)
    return T, Ahat
end

"""
Compute resolvent gains over ω grid using singular values of M(ω)=iωT-Ahat.

Returns:
- g2[ω]  = ||R(iω)||_2^2   = 1 / σ_min(M)^2
- gf2[ω] = ||R(iω)||_F^2   = Σ_i 1/σ_i(M)^2   (no explicit inverse)
"""
function resolvent_gain_curves(T::Diagonal, Ahat::AbstractMatrix, ωvals::AbstractVector)
    nω = length(ωvals)
    g2  = fill(NaN, nω)
    gf2 = fill(NaN, nω)

    for (k, ω) in enumerate(ωvals)
        M = Matrix{ComplexF64}(im*float(ω)*T - Ahat)
        σ = svdvals(M)  # real, ≥ 0
        if any(!isfinite, σ) || isempty(σ)
            continue
        end
        σmin = minimum(σ)
        if !(isfinite(σmin) && σmin > 0)
            continue
        end
        g2[k]  = 1.0 / (σmin^2)
        gf2[k] = sum(1.0 ./ (σ .^ 2))
    end

    return g2, gf2
end

"""
Extract peak metrics from a gain curve (vs ω):
- peak value
- ω* at peak
- “area” in log-ω (rough measure of bandwidth/importance)
"""
function peak_metrics(ωvals::AbstractVector, g::AbstractVector)
    mask = map(i -> isfinite(g[i]) && g[i] > 0 && isfinite(ωvals[i]) && ωvals[i] > 0, eachindex(g))
    idxs = findall(mask)
    isempty(idxs) && return (peak=NaN, ωstar=NaN, logarea=NaN)

    gsub = g[idxs]
    ωsub = ωvals[idxs]

    imax = argmax(gsub)
    peak = gsub[imax]
    ωstar = ωsub[imax]

    # area in log ω (trapezoid on log-scale)
    x = log.(ωsub)
    y = gsub
    ord = sortperm(x)
    x = x[ord]; y = y[ord]
    logarea = sum( (x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1]) ) / 2

    return (peak=peak, ωstar=ωstar, logarea=logarea)
end

# ============================================================
# 6) A clean “difference signal” in time: bump strength
# ============================================================

"""
Compute:
- Δ(t) = |rmed1(t) - rmed2(t)|
- bump_strength = max_t Δ(t)
- t_at_bump
"""
function bump_from_rmed(tvals::AbstractVector, r1::AbstractVector, r2::AbstractVector)
    @assert length(tvals) == length(r1) == length(r2)
    Δ = similar(r1)
    for i in eachindex(r1)
        Δ[i] = (isfinite(r1[i]) && isfinite(r2[i])) ? abs(r1[i] - r2[i]) : NaN
    end
    mask = map(i -> isfinite(Δ[i]), eachindex(Δ))
    if !any(mask)
        return (Δ=Δ, bump=NaN, tstar=NaN)
    end
    idx = argmax(Δ .* map(x -> isfinite(x) ? 1.0 : 0.0, Δ))  # robust-ish
    return (Δ=Δ, bump=Δ[idx], tstar=float(tvals[idx]))
end

# ============================================================
# 7) Main pipeline: build → rewire → rmed → resolvent → analyze
# ============================================================

"""
Run an ensemble comparing original vs rewired.

Key outputs:
- bump_strength (max_t |Δ rmed(t)|) and t_at_bump
- resolvent peak metrics (2-norm-based and Fro-based) for original and rewired
- ω* at peak, compare to 1/t*
"""
function run_resolvent_pipeline(;
    S::Int=120,
    n::Int=60,
    seed::Int=1234,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    perturbation::Symbol=:biomass,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=35),
    ωvals = 10 .^ range(log10(1e-2), log10(1e2); length=80),
    stabilize_rewired::Bool=true,
    margin::Real=1e-3,
    family_probs = Dict(
        :symmetric_like => 0.20,
        :asymmetric     => 0.25,
        :feedforward    => 0.25,
        :jordan         => 0.15,
        :block_ff       => 0.15
    )
)
    rng = MersenneTwister(seed)

    nt = length(tvals)

    # store curves (optional)
    rmed_orig = fill(NaN, n, nt)
    rmed_rew  = fill(NaN, n, nt)

    bump      = fill(NaN, n)
    tbump     = fill(NaN, n)

    # resolvent peak metrics
    peakR2_o   = fill(NaN, n);  peakR2_r   = fill(NaN, n)
    ωstar2_o   = fill(NaN, n);  ωstar2_r   = fill(NaN, n)
    area2_o    = fill(NaN, n);  area2_r    = fill(NaN, n)

    peakRF_o   = fill(NaN, n);  peakRF_r   = fill(NaN, n)
    ωstarF_o   = fill(NaN, n);  ωstarF_r   = fill(NaN, n)
    areaF_o    = fill(NaN, n);  areaF_r    = fill(NaN, n)

    # other diagnostics (optional)
    G2_o       = fill(NaN, n);  G2_r       = fill(NaN, n)
    tG_o       = fill(NaN, n);  tG_r       = fill(NaN, n)
    R_o        = fill(NaN, n);  R_r        = fill(NaN, n)
    kmean_o    = fill(NaN, n);  kmean_r    = fill(NaN, n)
    kmax_o     = fill(NaN, n);  kmax_r     = fill(NaN, n)
    family     = Vector{Symbol}(undef, n)

    # helper: rebuild J from A,u (your convention)
    jacobian(A,u) = Diagonal(u) * (A - I)

    for k in 1:n
        u = random_u(S; mean=float(u_mean), cv=float(u_cv), rng=rng)

        A, meta = build_A_diverse(u; family_probs=family_probs, rng=rng, stabilize=true, margin=margin)
        family[k] = meta.family

        J = jacobian(A, u)

        # rewired A (off-diagonals only, diag stays 0)
        Arew = reshuffle_offdiagonal(A; rng=rng)
        Jrew = jacobian(Arew, u)

        # if rewired unstable, optionally stabilize by shrinking its offdiag (keeping diag=-u)
        if stabilize_rewired
            Orew = offdiag_part(Jrew)
            Jrew2, _, αrew = stabilize_by_shrinking_offdiag(Orew, u; s0=1.0, margin=margin)
            Jrew = Jrew2

            # back out a consistent Arew with diag 0 (so descriptor form stays coherent)
            Dinv = Diagonal(1.0 ./ u)
            Arew = Dinv * Jrew + I
            for i in 1:S
                Arew[i,i] = 0.0
            end
        end

        # rmed curves
        for (ti, t) in enumerate(tvals)
            rmed_orig[k, ti] = median_return_rate(J,    u; t=t, perturbation=perturbation)
            rmed_rew[k,  ti] = median_return_rate(Jrew, u; t=t, perturbation=perturbation)
        end

        # bump signal
        b = bump_from_rmed(tvals, vec(rmed_orig[k,:]), vec(rmed_rew[k,:]))
        bump[k]  = b.bump
        tbump[k] = b.tstar

        # propagator diagnostics (optional)
        G2_o[k], tG_o[k] = peak_singular_gain(J, tvals)
        G2_r[k], tG_r[k] = peak_singular_gain(Jrew, tvals)
        R_o[k] = numerical_abscissa(J)
        R_r[k] = numerical_abscissa(Jrew)

        κo = mode_kappas(J)
        κr = mode_kappas(Jrew)
        kmean_o[k], kmax_o[k], _ = kappa_mean_max_sum(κo)
        kmean_r[k], kmax_r[k], _ = kappa_mean_max_sum(κr)

        # generalized resolvent diagnostics
        To, Ahat_o = descriptor_matrices(A, u)
        Tr, Ahat_r = descriptor_matrices(Arew, u)

        g2o, gf2o = resolvent_gain_curves(To, Ahat_o, ωvals)
        g2r, gf2r = resolvent_gain_curves(Tr, Ahat_r, ωvals)

        pm2o = peak_metrics(ωvals, g2o)
        pm2r = peak_metrics(ωvals, g2r)
        peakR2_o[k] = pm2o.peak;  ωstar2_o[k] = pm2o.ωstar;  area2_o[k] = pm2o.logarea
        peakR2_r[k] = pm2r.peak;  ωstar2_r[k] = pm2r.ωstar;  area2_r[k] = pm2r.logarea

        pmFo = peak_metrics(ωvals, gf2o)
        pmFr = peak_metrics(ωvals, gf2r)
        peakRF_o[k] = pmFo.peak;  ωstarF_o[k] = pmFo.ωstar;  areaF_o[k] = pmFo.logarea
        peakRF_r[k] = pmFr.peak;  ωstarF_r[k] = pmFr.ωstar;  areaF_r[k] = pmFr.logarea
    end

    # ensemble means (for curves)
    mean_rmed_o = [meanfinite(view(rmed_orig, :, ti)) for ti in 1:nt]
    mean_rmed_r = [meanfinite(view(rmed_rew,  :, ti)) for ti in 1:nt]
    mean_absΔ   = [meanfinite(abs.(view(rmed_orig, :, ti) .- view(rmed_rew, :, ti))) for ti in 1:nt]

    return (
        S=S, n=n, seed=seed,
        tvals=tvals, ωvals=ωvals,
        rmed_orig=rmed_orig, rmed_rew=rmed_rew,
        mean_rmed_orig=mean_rmed_o, mean_rmed_rew=mean_rmed_r, mean_absΔ=mean_absΔ,
        bump=bump, tbump=tbump,
        peakR2_o=peakR2_o, peakR2_r=peakR2_r, ωstar2_o=ωstar2_o, ωstar2_r=ωstar2_r, area2_o=area2_o, area2_r=area2_r,
        peakRF_o=peakRF_o, peakRF_r=peakRF_r, ωstarF_o=ωstarF_o, ωstarF_r=ωstarF_r, areaF_o=areaF_o, areaF_r=areaF_r,
        G2_o=G2_o, G2_r=G2_r, tG_o=tG_o, tG_r=tG_r,
        R_o=R_o, R_r=R_r,
        kmean_o=kmean_o, kmean_r=kmean_r, kmax_o=kmax_o, kmax_r=kmax_r,
        family=family
    )
end

# ============================================================
# 8) Plotting & quick association checks
# ============================================================

function analyze_and_plot_resolvent(results; figsize=(1400, 900))
    tvals = results.tvals
    ωvals = results.ωvals

    bump  = results.bump
    tbump = results.tbump

    # deltas
    ΔpeakR2 = abs.(results.peakR2_o .- results.peakR2_r) ./ results.peakR2_o
    ΔpeakRF = abs.(results.peakRF_o .- results.peakRF_r) ./ results.peakRF_o

    # map ω* -> characteristic time (two conventions shown)
    ωstar = results.ωstar2_o
    tchar_1 = 1.0 ./ ωstar
    tchar_2pi = 2π ./ ωstar

    # masks
    m1 = map(i -> isfinite(bump[i]) && bump[i] > 0 &&
                 isfinite(ΔpeakR2[i]) && ΔpeakR2[i] > 0, eachindex(bump))

    m2 = map(i -> isfinite(tbump[i]) && tbump[i] > 0 &&
                 isfinite(ωstar[i]) && ωstar[i] > 0, eachindex(bump))

    # correlations (log-log)
    ρ_bump_ΔR2 = (count(m1) >= 4) ? cor(log.(bump[m1]), log.(ΔpeakR2[m1])) : NaN
    ρ_bump_ΔRF = (count(m1) >= 4) ? cor(log.(bump[m1]), log.(ΔpeakRF[m1])) : NaN

    ρ_time_1 = (count(m2) >= 4) ? cor(log.(tbump[m2]), log.(tchar_1[m2])) : NaN
    ρ_time_2 = (count(m2) >= 4) ? cor(log.(tbump[m2]), log.(tchar_2pi[m2])) : NaN

    @info "cor(log bump, log Δpeak||R||2^2)  = $ρ_bump_ΔR2   (N=$(count(m1)))"
    @info "cor(log bump, log Δpeak||R||F^2)  = $ρ_bump_ΔRF   (N=$(count(m1)))"
    @info "cor(log t_bump, log (1/ω*))       = $ρ_time_1     (N=$(count(m2)))"
    @info "cor(log t_bump, log (2π/ω*))      = $ρ_time_2     (N=$(count(m2)))"

    fig = Figure(size=figsize)

    # mean curves
    ax0 = Axis(fig[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="rmed(t)",
        title="Mean rmed(t): original vs rewired"
    )
    lines!(ax0, tvals, results.mean_rmed_orig, linewidth=3)
    lines!(ax0, tvals, results.mean_rmed_rew,  linewidth=3)
    # axislegend(ax0, ["orig", "rew"], position=:rb)

    ax1 = Axis(fig[1,2];
        xscale=log10,
        xlabel="t",
        ylabel="mean |Δ rmed(t)|",
        title="Mean absolute rmed difference (time profile)"
    )
    lines!(ax1, tvals, results.mean_absΔ, linewidth=3)

    # bump vs Δ resolvent peak
    ax2 = Axis(fig[2,1];
        xscale=log10, yscale=log10,
        xlabel="bump = max_t |Δ rmed(t)|",
        ylabel="Δ peak resolvent gain (2-norm): |Δpeak|/peak",
        title="Bump strength vs resolvent-peak change"
    )
    scatter!(ax2, bump[m1], ΔpeakR2[m1], markersize=9)

    ax3 = Axis(fig[2,2];
        xscale=log10, yscale=log10,
        xlabel="bump = max_t |Δ rmed(t)|",
        ylabel="Δ peak resolvent gain (Fro): |Δpeak|/peak",
        title="Bump strength vs Fro-resolvent-peak change"
    )
    scatter!(ax3, bump[m1], ΔpeakRF[m1], markersize=9)

    # time matching: t_bump vs 1/ω*
    ax4 = Axis(fig[3,1];
        xscale=log10, yscale=log10,
        xlabel="t_bump (time of max |Δ rmed|)",
        ylabel="1/ω*  (from resolvent peak)",
        title="Does the resolvent peak predict the bump timescale?"
    )
    scatter!(ax4, tbump[m2], tchar_1[m2], markersize=9)

    ax5 = Axis(fig[3,2];
        xscale=log10, yscale=log10,
        xlabel="t_bump",
        ylabel="2π/ω*",
        title="Alternative mapping: period scale"
    )
    scatter!(ax5, tbump[m2], tchar_2pi[m2], markersize=9)

    # optional: family labels (quick sanity check)
    ax6 = Axis(fig[1,3];
        xlabel="family index",
        ylabel="bump",
        yscale=log10,
        title="Bump by sampled family (rough check)"
    )
    fams = unique(results.family)
    fam_to_i = Dict(f => i for (i,f) in enumerate(fams))
    xi = [fam_to_i[f] for f in results.family]
    maskfam = map(i -> isfinite(bump[i]) && bump[i] > 0, eachindex(bump))
    scatter!(ax6, xi[maskfam], bump[maskfam], markersize=8)
    ax6.xticks = (1:length(fams), string.(fams))

    display(fig)
end

# ============================================================
# 9) Run
# ============================================================

tvals = 10 .^ range(log10(0.01), log10(100.0); length=35)
ωvals = 10 .^ range(log10(1e-2), log10(1e2); length=80)

results = run_resolvent_pipeline(
    S=120,
    n=60,
    seed=1234,
    u_mean=1.0,
    u_cv=0.5,
    perturbation=:biomass,
    tvals=tvals,
    ωvals=ωvals,
    stabilize_rewired=true,
    margin=1e-3,
    # family_probs = Dict(
    #     :symmetric_like => 0.0,
    #     :asymmetric     => 1.0,
    #     :feedforward    => 0.0,
    #     :jordan         => 0.0,
    #     :block_ff       => 0.0
    # )
)

analyze_and_plot_resolvent(results)