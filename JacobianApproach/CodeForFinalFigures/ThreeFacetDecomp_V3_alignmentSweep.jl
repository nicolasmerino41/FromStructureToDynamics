################################################################################
# CONTROLLED ALIGNMENT SWEEP: "ALIGNMENT ORGANIZES WHEN"
#
# What this script does (minimal, but complete):
#
#  (1) Builds ONE stable interaction matrix A (off-diagonal) and Abar = -I + A.
#      Uses the same construction you have been using (trophic-ish O + scaling).
#
#  (2) Draws ONE multiset of timescales u (same values always), then generates
#      MANY permutations of the assignment of u_i to nodes (row time-scales).
#      This changes ALIGNMENT between (A) and (u) without changing:
#        - the distribution of u
#        - A itself
#        - connectance / strength distribution / reciprocity of A
#
#  (3) For each permutation, computes:
#        - energy spectrum e(ω) = tr(R C0 R†) / tr(C0)
#        - fragility spectrum G(ω) = mean_P |g(ω;P)|
#        - combined "response-mass" m(ω) = e(ω) * G(ω)
#      and extracts WHERE m(ω) sits using:
#        - ω_m50    (median location)
#        - ω_m_ctr  (log-centroid, more sensitive than the median)
#        - ω_m_q10, ω_m_q90
#
#      All locations are reported on a DIMENSIONLESS frequency axis:
#        ω̃ = ω * Tbar   with Tbar = mean(1/u)
#      so "WHEN" is comparable across systems.
#
#  (4) Separately, computes HOW MUCH per-unit structural displacement:
#        d = ||ΔAbar||_F   (here d = eps because ||P||_F=1)
#        realized:   (ΔV/V)/d   via Lyapunov (OU stationary covariance)
#        predicted:  H_real = mean_P | ∫ g e / ∫ e |
#
#  (5) Also computes the same predicted H but with HOMOGENEOUS timescales:
#        u_hom = 1/Tbar  (so Tbar is preserved exactly)
#      to separate "resolvent geometry of A" from "timescale modulation".
#
#  (6) Produces the key plots:
#        - Aidx (alignment) vs WHERE (log ω̃_m50, log ω̃_centroid)
#        - ω̃_pred from strength-weighted u vs WHERE
#        - predicted per-unit magnitude vs realized per-unit magnitude
#        - compare H_hom vs realized per-unit magnitude
#        - example spectra for extreme alignments
#
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# -----------------------------
# Small utilities
# -----------------------------
function trapz(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    n = length(x)
    n < 2 && return NaN
    s = 0.0
    for i in 1:(n-1)
        x1, x2 = float(x[i]), float(x[i+1])
        y1, y2 = float(y[i]), float(y[i+1])
        if isfinite(x1) && isfinite(x2) && isfinite(y1) && isfinite(y2)
            s += 0.5 * (y1 + y2) * (x2 - x1)
        end
    end
    return s
end

spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigvals(Matrix(J))))

function logsafe_vec(y::AbstractVector; eps=1e-18)
    z = similar(y, Float64)
    for i in eachindex(y)
        yi = y[i]
        z[i] = (isfinite(yi) && yi > eps) ? yi : NaN
    end
    return z
end

# -----------------------------
# Random u (timescale rates)
# -----------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# -----------------------------
# Trophic-ish off-diagonal generator O (diag=0)
# -----------------------------
function trophic_O(S::Int;
    connectance::Float64,
    trophic_align::Float64,
    reciprocity::Float64,
    σ::Float64,
    rng=Random.default_rng()
)
    @assert 0.0 <= connectance <= 1.0
    @assert 0.0 <= trophic_align <= 1.0
    @assert 0.0 <= reciprocity <= 1.0

    h = rand(rng, S)
    O = zeros(Float64, S, S)

    for i in 1:S-1, j in i+1:S
        rand(rng) < connectance || continue

        if rand(rng) < reciprocity
            O[i,j] = randn(rng) * σ
            O[j,i] = randn(rng) * σ
        else
            low, high = (h[i] <= h[j]) ? (i, j) : (j, i)
            aligned = rand(rng) < trophic_align
            if aligned
                O[low, high] = randn(rng) * σ
            else
                O[high, low] = randn(rng) * σ
            end
        end
    end

    for i in 1:S
        O[i,i] = 0.0
    end
    return O
end

function normalize_offdiag!(O::Matrix{Float64})
    n = norm(O)
    n == 0 && return false
    O ./= n
    return true
end

# -----------------------------
# Find scale s so α(J) ≈ target_alpha (<0)
# J = diag(u) * (-I + s O)
# -----------------------------
function find_scale_to_target_alpha(O::Matrix{Float64}, u::Vector{Float64};
    target_alpha::Float64=-0.20,
    max_grow::Int=40,
    max_iter::Int=60
)
    @assert target_alpha < 0
    Du = Diagonal(u)

    α0 = spectral_abscissa(-Du)
    isfinite(α0) || return NaN

    s_hi = 1.0
    α_hi = spectral_abscissa(-Du + s_hi*(Du*O))
    k = 0
    while (isfinite(α_hi) && α_hi < target_alpha) && k < max_grow
        s_hi *= 2.0
        α_hi = spectral_abscissa(-Du + s_hi*(Du*O))
        k += 1
    end
    if !(isfinite(α_hi)) || α_hi < target_alpha
        return NaN
    end

    s_lo = 0.0
    α_lo = α0
    if α_lo > target_alpha
        return 0.0
    end

    for _ in 1:max_iter
        s_mid = 0.5*(s_lo + s_hi)
        α_mid = spectral_abscissa(-Du + s_mid*(Du*O))
        if !isfinite(α_mid)
            s_hi = s_mid
            continue
        end
        if α_mid < target_alpha
            s_lo = s_mid
        else
            s_hi = s_mid
        end
    end
    return 0.5*(s_lo + s_hi)
end

# -----------------------------
# Structural perturbation directions P
# (diag=0, ||P||_F = 1)
# -----------------------------
function sample_Pdir_allfree(S::Int; sparsity_p::Float64=1.0, rng=Random.default_rng())
    P = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        i == j && continue
        rand(rng) < sparsity_p || continue
        P[i,j] = randn(rng)
    end
    nP = norm(P)
    nP == 0 && return nothing
    P ./= nP
    return P
end

function sample_Pdir_edgesonly(A::Matrix{Float64}; rng=Random.default_rng())
    S = size(A,1)
    P = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        i == j && continue
        A[i,j] == 0.0 && continue
        P[i,j] = randn(rng)
    end
    nP = norm(P)
    nP == 0 && return nothing
    P ./= nP
    return P
end

# -----------------------------
# Frequency-domain estimators
#
# R(ω) = (i ω T - Abar)^(-1),  T=diag(1/u)
# e(ω) = tr(R C0 R†)/tr(C0)
# g(ω;P) ≈ 2 Re tr(R P Ĉ) / tr(Ĉ),   Ĉ = R C0 R†
#
# Hutchinson probes:
#   tr(Ĉ) = E_v || R sqrt(C0) v ||^2
#   Re tr(R P Ĉ) = E_v Re[ x† (R P x) ] with x = R sqrt(C0) v
# -----------------------------
function estimate_energy_and_g_at_ω!(
    F::LU{ComplexF64, Matrix{ComplexF64}},
    sqrtc::Vector{Float64},
    trC0::Float64,
    Pdirs::Vector{Matrix{Float64}};
    nprobe::Int=10,
    rng=Random.default_rng()
)
    S = length(sqrtc)
    nP = length(Pdirs)

    # Rademacher probes
    probes = Vector{Vector{Float64}}(undef, nprobe)
    for k in 1:nprobe
        probes[k] = rand(rng, (-1.0, 1.0), S)
    end

    x_list = Vector{Vector{ComplexF64}}(undef, nprobe)
    xnorm2 = zeros(Float64, nprobe)

    for k in 1:nprobe
        rhs = ComplexF64.(sqrtc .* probes[k])
        x = F \ rhs
        x_list[k] = x
        xnorm2[k] = real(dot(conj.(x), x))
    end

    trChat = mean(xnorm2)
    eω = (isfinite(trChat) && trC0 > 0) ? (trChat / trC0) : NaN

    gvec = fill(NaN, nP)
    for (pidx, P) in enumerate(Pdirs)
        inn = zeros(Float64, nprobe)
        Pc = ComplexF64.(P)
        for k in 1:nprobe
            x = x_list[k]
            y = Pc * x
            z = F \ y
            inn[k] = real(dot(conj.(x), z))
        end
        num = mean(inn)
        if isfinite(num) && isfinite(trChat) && trChat > 0
            gvec[pidx] = 2.0 * (num / trChat)
        else
            gvec[pidx] = NaN
        end
    end

    return eω, gvec
end

function estimate_energy_and_fragility_spectra(
    Abar::Matrix{Float64},
    u::Vector{Float64},
    C0diag::Vector{Float64},
    ωvals::Vector{Float64},
    Pdirs::Vector{Matrix{Float64}};
    nprobe::Int=10,
    rng=Random.default_rng()
)
    S = length(u)
    @assert length(C0diag) == S
    sqrtc = sqrt.(C0diag)
    trC0 = sum(C0diag)
    Tmat = Diagonal(1.0 ./ u)

    nω = length(ωvals)
    nP = length(Pdirs)

    eω = fill(NaN, nω)
    Gabs = fill(NaN, nω)
    gmat = fill(NaN, nP, nω)

    for (k, ω0) in enumerate(ωvals)
        ω = float(ω0)
        Mω = Matrix{ComplexF64}(im*ω*Tmat - Abar)
        F = lu(Mω)

        ek, gvec = estimate_energy_and_g_at_ω!(F, sqrtc, trC0, Pdirs; nprobe=nprobe, rng=rng)
        eω[k] = ek
        for p in 1:nP
            gmat[p,k] = gvec[p]
        end

        goodg = filter(isfinite, gvec)
        Gabs[k] = isempty(goodg) ? NaN : mean(abs.(goodg))
    end

    return (eω=eω, Gabs=Gabs, gmat=gmat)
end

# -----------------------------
# Variability via Lyapunov (OU stationary covariance)
#
# Dynamics:  T xdot = Abar x + ξ,  E[ξ ξᵀ] = C0 δ
# Standard form: xdot = J x + η,  J = diag(u) Abar,  η = diag(u) ξ
# So Q = E[η ηᵀ] = diag(u) C0 diag(u)
# Solve: JΣ + ΣJᵀ + Q = 0
# V = tr(Σ) / tr(C0)
# -----------------------------
function variability_time_domain(Abar::Matrix{Float64}, u::Vector{Float64}, C0diag::Vector{Float64})
    S = length(u)
    Du = Diagonal(u)
    J = Du * Abar
    Q = Du * Diagonal(C0diag) * Du

    α = spectral_abscissa(J)
    (isfinite(α) && α < 0) || return NaN

    Σ = lyap(Matrix(J), Matrix(Q))
    trC0 = sum(C0diag)
    return (isfinite(tr(Σ)) && trC0 > 0) ? (tr(Σ) / trC0) : NaN
end

# -----------------------------
# WHERE statistics for a nonnegative spectrum y(ω)
# - median (m50), quantiles (q10,q90) using cumulative integral
# - log-centroid: exp( ∫ log ω y / ∫ y )
# -----------------------------
function spectrum_location_stats(ω::Vector{Float64}, y::Vector{Float64})
    idx = findall(i -> isfinite(ω[i]) && ω[i] > 0 && isfinite(y[i]) && y[i] >= 0, eachindex(y))
    length(idx) < 4 && return (m50=NaN, q10=NaN, q90=NaN, lctr=NaN, mass=NaN)

    w = ω[idx]
    v = y[idx]
    total = trapz(w, v)
    (isfinite(total) && total > 0) || return (m50=NaN, q10=NaN, q90=NaN, lctr=NaN, mass=NaN)

    cum = zeros(Float64, length(w))
    for i in 2:length(w)
        cum[i] = cum[i-1] + 0.5*(v[i-1] + v[i]) * (w[i] - w[i-1])
    end

    function interp_at(frac)
        tgt = frac * total
        j = findfirst(cum .>= tgt)
        isnothing(j) && return NaN
        j == 1 && return w[1]
        w1, w2 = w[j-1], w[j]
        c1, c2 = cum[j-1], cum[j]
        if c2 == c1
            return w2
        else
            return w1 + (tgt - c1) * (w2 - w1) / (c2 - c1)
        end
    end

    # log-centroid
    num = 0.0
    den = 0.0
    for i in 1:(length(w)-1)
        w1, w2 = w[i], w[i+1]
        v1, v2 = v[i], v[i+1]
        dw = (w2 - w1)
        if !(isfinite(w1) && isfinite(w2) && isfinite(v1) && isfinite(v2) && w1 > 0 && w2 > 0)
            continue
        end
        # trapezoid for y and y*log(w)
        den += 0.5*(v1 + v2) * dw
        num += 0.5*(v1*log(w1) + v2*log(w2)) * dw
    end
    lctr = (den > 0) ? exp(num / den) : NaN

    return (m50=interp_at(0.50), q10=interp_at(0.10), q90=interp_at(0.90), lctr=lctr, mass=total)
end

# -----------------------------
# Alignment metrics: strength <-> timescales coupling
#
# Strength per node (incident absolute interaction strength):
#   s_i = sum_j |A_ij| + sum_j |A_ji|
#
# Timescales:
#   T_i = 1/u_i,  use logT_i
#
# Alignment index (strength-weighted logT minus mean logT):
#   w_i = s_i / sum s_i
#   Aidx = sum_i w_i logT_i  - mean(logT)
#
# This is "how much strong interaction mass sits on slow nodes" (Aidx > 0)
#
# Strength-weighted u (geometric mean):
#   u_str = exp( sum_i w_i log(u_i) )
# Predictor for WHERE in normalized frequency:
#   ω̃_pred = u_str * Tbar ,  where Tbar = mean(1/u)
# -----------------------------
function node_strength_abs(A::Matrix{Float64})
    S = size(A,1)
    out = vec(sum(abs.(A), dims=2))
    inn = vec(sum(abs.(A), dims=1))  # row vector -> vec
    return out .+ inn
end

function alignment_metrics(A::Matrix{Float64}, u::Vector{Float64})
    s = node_strength_abs(A)
    stot = sum(s)
    w = (stot > 0) ? (s ./ stot) : fill(1/length(s), length(s))

    T = 1.0 ./ u
    logT = log.(T)

    Aidx = dot(w, logT) - mean(logT)
    u_str = exp(dot(w, log.(u)))

    Tbar = mean(T)
    wtilde_pred = u_str * Tbar

    # optional alternative: corr(s, logT)
    cidx = (std(s) > 0 && std(logT) > 0) ? cor(s, logT) : NaN

    return (Aidx=Aidx, Aidx_corr=cidx, u_str=u_str, Tbar=Tbar, wtilde_pred=wtilde_pred)
end

# -----------------------------
# Evaluate one (A,u) system:
# - choose ω grid using a fixed ωtilde grid: ω = ωtilde / Tbar
# - compute e(ω), G(ω), m(ω)=eG
# - compute WHERE stats for m on ωtilde axis
# - compute per-unit magnitude: realized vs predicted
# - compute H_hom with u_hom preserving Tbar
# -----------------------------
function eval_system!(
    Abar::Matrix{Float64},
    A::Matrix{Float64},
    u::Vector{Float64};
    ωtilde_vals::Vector{Float64},
    C0_mode::Symbol=:I,             # :I or :u2
    P_mode::Symbol=:edges_only,     # :edges_only or :all_free
    P_reps::Int=12,
    P_sparsity::Float64=1.0,
    nprobe::Int=10,
    eps_rel::Float64=0.03,          # eps = eps_rel * norm(A)
    margin::Float64=1e-6,
    seed::Int=1
)
    rng = MersenneTwister(seed)
    S = length(u)

    # alignment + normalization scale
    am = alignment_metrics(A, u)
    Tbar = am.Tbar
    ωvals = ωtilde_vals ./ Tbar

    # noise covariance diagonal
    C0diag = if C0_mode == :I
        ones(Float64, S)
    elseif C0_mode == :u2
        u.^2
    else
        error("Unknown C0_mode. Use :I or :u2.")
    end

    # structural displacement scale
    eps = eps_rel * norm(A)   # this is d when ||P||=1 and we perturb Abar by eps P
    isfinite(eps) && eps > 0 || return nothing

    # sample P ensemble (filter those that keep perturbed system stable)
    Du = Diagonal(u)
    Pdirs = Matrix{Float64}[]
    for k in 1:P_reps
        P = if P_mode == :edges_only
            sample_Pdir_edgesonly(A; rng=rng)
        elseif P_mode == :all_free
            sample_Pdir_allfree(S; sparsity_p=P_sparsity, rng=rng)
        else
            error("Unknown P_mode. Use :edges_only or :all_free.")
        end
        P === nothing && continue

        # stability filter for perturbed
        Jp = Du * (Abar + eps * P)
        αp = spectral_abscissa(Jp)
        (isfinite(αp) && αp < -margin) || continue
        push!(Pdirs, P)
    end
    length(Pdirs) >= max(6, Int(floor(P_reps/2))) || return nothing

    # base variability
    V0 = variability_time_domain(Abar, u, C0diag)
    isfinite(V0) && V0 > 0 || return nothing

    # spectra on real u
    sp = estimate_energy_and_fragility_spectra(Abar, u, C0diag, ωvals, Pdirs; nprobe=nprobe, rng=rng)
    eω = sp.eω
    Gω = sp.Gabs
    mω = eω .* Gω

    # WHERE of m in normalized frequency axis (ωtilde)
    mstats = spectrum_location_stats(ωtilde_vals, mω)
    wtilde_m50  = mstats.m50
    wtilde_q10  = mstats.q10
    wtilde_q90  = mstats.q90
    wtilde_lctr = mstats.lctr

    # predicted per-unit magnitude (and signed slope)
    idxE = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω[i]) && eω[i] >= 0, eachindex(eω))
    length(idxE) >= 4 || return nothing
    denom = trapz(ωvals[idxE], eω[idxE])
    isfinite(denom) && denom > 0 || return nothing

    gmat = sp.gmat
    slopes = Float64[]
    slopes_abs = Float64[]
    for p in 1:size(gmat,1)
        gω = vec(gmat[p, :])
        good = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω[i]) && eω[i] >= 0 && isfinite(gω[i]),
                       eachindex(eω))
        length(good) >= 4 || continue
        num = trapz(ωvals[good], gω[good] .* eω[good])
        sgn = num / denom             # predicted (ΔV/V)/eps at first order
        isfinite(sgn) || continue
        push!(slopes, sgn)
        push!(slopes_abs, abs(sgn))
    end
    isempty(slopes_abs) && return nothing
    H_real = mean(slopes_abs)         # predicted per-unit magnitude
    slope_mean = mean(slopes)         # predicted signed mean per-unit

    # realized per-unit magnitude from Lyapunov
    realized_abs = Float64[]
    realized_sgn = Float64[]
    for P in Pdirs
        Vp = variability_time_domain(Abar + eps*P, u, C0diag)
        isfinite(Vp) && Vp > 0 || continue
        push!(realized_abs, abs(Vp - V0) / V0 / eps)
        push!(realized_sgn, (Vp - V0) / V0 / eps)
    end
    isempty(realized_abs) && return nothing
    R_abs = mean(realized_abs)
    R_sgn = mean(realized_sgn)

    # Homogeneous-timescale predicted facet (same Tbar exactly)
    u_hom = fill(1.0 / Tbar, S)
    C0_hom = if C0_mode == :I
        ones(Float64, S)
    else
        u_hom.^2
    end
    sp_h = estimate_energy_and_fragility_spectra(Abar, u_hom, C0_hom, ωvals, Pdirs; nprobe=nprobe, rng=rng)
    eω_h = sp_h.eω
    gmat_h = sp_h.gmat
    idxEh = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω_h[i]) && eω_h[i] >= 0, eachindex(eω_h))
    denom_h = trapz(ωvals[idxEh], eω_h[idxEh])
    H_hom = NaN
    if isfinite(denom_h) && denom_h > 0
        slopes_abs_h = Float64[]
        for p in 1:size(gmat_h,1)
            gωh = vec(gmat_h[p, :])
            good = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω_h[i]) && eω_h[i] >= 0 && isfinite(gωh[i]),
                           eachindex(eω_h))
            length(good) >= 4 || continue
            num = trapz(ωvals[good], gωh[good] .* eω_h[good])
            push!(slopes_abs_h, abs(num / denom_h))
        end
        H_hom = isempty(slopes_abs_h) ? NaN : mean(slopes_abs_h)
    end

    return (
        # alignment / predictors
        Aidx=am.Aidx,
        Aidx_corr=am.Aidx_corr,
        u_str=am.u_str,
        Tbar=Tbar,
        wtilde_pred=am.wtilde_pred,

        # WHERE
        wtilde_m50=wtilde_m50,
        wtilde_q10=wtilde_q10,
        wtilde_q90=wtilde_q90,
        wtilde_lctr=wtilde_lctr,

        # HOW MUCH per-unit
        H_real=H_real,
        H_hom=H_hom,
        R_abs=R_abs,
        R_sgn=R_sgn,
        slope_mean=slope_mean,

        # keep spectra for optional plotting
        ωtilde=ωtilde_vals,
        ω=ωvals,
        eω=eω,
        Gω=Gω,
        mω=mω,
        eω_h=eω_h,
        Gω_h=sp_h.Gabs,
        mω_h=(sp_h.eω .* sp_h.Gabs),

        # eps and ensemble size
        eps=eps,
        nP=length(Pdirs),
        V0=V0
    )
end

# -----------------------------
# Build ONE base Abar = -I + A and ONE u-multiset.
# Then run many u-permutations to sweep alignment.
# -----------------------------
struct BasePack
    A::Matrix{Float64}
    Abar::Matrix{Float64}
    uvals::Vector{Float64}     # multiset of u values (will be permuted)
end

function build_base_pack(; S::Int=120, seed::Int=1234,
    u_mean::Float64=1.0, u_cv::Float64=0.5,
    connectance::Float64=0.06,
    trophic_align::Float64=0.85,
    reciprocity::Float64=0.10,
    σ::Float64=1.0,
    target_alpha::Float64=-0.20
)
    rng = MersenneTwister(seed)

    u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))

    O = trophic_O(S; connectance=connectance, trophic_align=trophic_align,
                 reciprocity=reciprocity, σ=σ, rng=rng)
    normalize_offdiag!(O) || error("Failed to normalize O")

    s = find_scale_to_target_alpha(O, u; target_alpha=target_alpha)
    isfinite(s) || error("Failed to find scaling")

    A = s * O
    Abar = -Matrix{Float64}(I, S, S) + A

    # check base stability
    α = spectral_abscissa(Diagonal(u) * Abar)
    (isfinite(α) && α < 0) || error("Base system not stable. α=$α")

    return BasePack(A, Abar, u)
end

# -----------------------------
# Main alignment sweep
# -----------------------------
function run_alignment_sweep(; S::Int=120, seed::Int=1234,
    nperm::Int=120,
    ωtilde_vals = 10 .^ range(log10(1e-4), log10(1e-1); length=70),
    C0_mode::Symbol=:I,
    P_mode::Symbol=:edges_only,
    P_reps::Int=12,
    nprobe::Int=10,
    eps_rel::Float64=0.03
)
    ωtilde_vals = collect(float.(ωtilde_vals))

    base = build_base_pack(S=S, seed=seed)
    A, Abar, uvals = base.A, base.Abar, base.uvals

    # generate permutations (include a few structured ones to widen Aidx range)
    rng = MersenneTwister(seed + 99)
    perms = Vector{Vector{Int}}()

    # random perms
    for k in 1:(nperm-4)
        push!(perms, randperm(rng, S))
    end

    # extreme: strongest nodes get slowest timescales vs fastest
    svec = node_strength_abs(A)
    ord_str = sortperm(svec; rev=true)                    # strong -> weak
    ord_u_slow = sortperm(uvals; rev=false)               # small u = slow
    ord_u_fast = sortperm(uvals; rev=true)                # large u = fast

    perm_str_slow = similar(ord_str)
    perm_str_fast = similar(ord_str)
    # perm is mapping new_index -> old_index; we want u_new[i] = uvals[perm[i]]
    # easiest: build u_new by assignment and then derive perm index list:
    u_new_slow = similar(uvals)
    u_new_fast = similar(uvals)
    for k in 1:S
        u_new_slow[ord_str[k]] = uvals[ord_u_slow[k]]
        u_new_fast[ord_str[k]] = uvals[ord_u_fast[k]]
    end
    # derive perm arrays by matching (stable because u are continuous, but do a safe match)
    # Instead, just store u_new directly for these extremes.
    # We'll handle them specially as "u cases".
    u_cases = Vector{Vector{Float64}}()
    push!(u_cases, u_new_slow)
    push!(u_cases, u_new_fast)

    results = NamedTuple[]
    kept = 0

    # evaluate structured cases first
    for (k, ucase) in enumerate(u_cases)
        out = eval_system!(Abar, A, ucase;
            ωtilde_vals=ωtilde_vals,
            C0_mode=C0_mode,
            P_mode=P_mode,
            P_reps=P_reps,
            nprobe=nprobe,
            eps_rel=eps_rel,
            seed=seed + 10_000 + k
        )
        out === nothing && continue
        push!(results, out)
        kept += 1
    end

    # evaluate random permutations
    for (k, perm) in enumerate(perms)
        uperm = uvals[perm]
        out = eval_system!(Abar, A, uperm;
            ωtilde_vals=ωtilde_vals,
            C0_mode=C0_mode,
            P_mode=P_mode,
            P_reps=P_reps,
            nprobe=nprobe,
            eps_rel=eps_rel,
            seed=seed + 20_000 + k
        )
        out === nothing && continue
        push!(results, out)
        kept += 1
    end

    @info "Alignment sweep: kept $kept / $(length(perms) + length(u_cases)) stable/evaluable systems."

    return (base=base, ωtilde=ωtilde_vals, res=results)
end

# -----------------------------
# Plot and interpret core diagnostics
# -----------------------------
function plot_alignment_sweep(run; figsize=(1650, 950))
    res = run.res
    N = length(res)
    N == 0 && error("No results to plot.")

    Aidx        = [r.Aidx for r in res]
    w_m50       = [r.wtilde_m50 for r in res]
    w_lctr      = [r.wtilde_lctr for r in res]
    w_pred      = [r.wtilde_pred for r in res]
    H_real      = [r.H_real for r in res]
    H_hom       = [r.H_hom for r in res]
    R_abs       = [r.R_abs for r in res]

    # masks
    m_when50 = findall(i -> isfinite(Aidx[i]) && isfinite(w_m50[i]) && w_m50[i] > 0, 1:N)
    m_whenc  = findall(i -> isfinite(Aidx[i]) && isfinite(w_lctr[i]) && w_lctr[i] > 0, 1:N)
    m_predw  = findall(i -> isfinite(w_pred[i]) && w_pred[i] > 0 && isfinite(w_m50[i]) && w_m50[i] > 0, 1:N)
    m_mag    = findall(i -> isfinite(H_real[i]) && H_real[i] > 0 && isfinite(R_abs[i]) && R_abs[i] > 0, 1:N)
    m_mag_h  = findall(i -> isfinite(H_hom[i]) && H_hom[i] > 0 && isfinite(R_abs[i]) && R_abs[i] > 0, 1:N)

    ρ_when50 = (length(m_when50) >= 6) ? cor(Aidx[m_when50], log.(w_m50[m_when50])) : NaN
    ρ_whenc  = (length(m_whenc)  >= 6) ? cor(Aidx[m_whenc],  log.(w_lctr[m_whenc])) : NaN
    ρ_predw  = (length(m_predw)  >= 6) ? cor(log.(w_pred[m_predw]), log.(w_m50[m_predw])) : NaN
    ρ_mag    = (length(m_mag)    >= 6) ? cor(log.(H_real[m_mag]), log.(R_abs[m_mag])) : NaN
    ρ_mag_h  = (length(m_mag_h)  >= 6) ? cor(log.(H_hom[m_mag_h]),  log.(R_abs[m_mag_h])) : NaN

    @info "WHEN: corr(Aidx, log ωtilde_m50)  = $(round(ρ_when50,digits=3))   N=$(length(m_when50))"
    @info "WHEN: corr(Aidx, log ωtilde_ctr)  = $(round(ρ_whenc,digits=3))   N=$(length(m_whenc))"
    @info "WHEN: corr(log ωtilde_pred, log ωtilde_m50) = $(round(ρ_predw,digits=3))   N=$(length(m_predw))"
    @info "HOW MUCH: corr(log H_real, log realized per-unit) = $(round(ρ_mag,digits=3))   N=$(length(m_mag))"
    @info "HOW MUCH: corr(log H_hom,  log realized per-unit) = $(round(ρ_mag_h,digits=3))  N=$(length(m_mag_h))"

    # sanity: u_str*Tbar scale
    wpred_rng = (minimum(filter(isfinite, w_pred)), maximum(filter(isfinite, w_pred)))
    wm50_rng  = (minimum(filter(x -> isfinite(x) && x>0, w_m50)), maximum(filter(x -> isfinite(x) && x>0, w_m50)))
    @info "Range ωtilde_pred = $(wpred_rng)"
    @info "Range ωtilde_m50  = $(wm50_rng)"
    @info "Range Aidx        = ($(minimum(filter(isfinite,Aidx))), $(maximum(filter(isfinite,Aidx))))"

    fig = Figure(size=figsize)

    ax1 = Axis(fig[1,1];
        xlabel="alignment Aidx (strength-weighted logT minus mean)",
        ylabel="log10(ωtilde_m50)  (WHERE m(ω) sits)",
        title="A) Alignment vs WHEN (median location)"
    )
    scatter!(ax1, Aidx[m_when50], log10.(w_m50[m_when50]), markersize=7)
    text!(ax1, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(Aidx, log ωtilde_m50) = $(round(ρ_when50,digits=3))   N=$(length(m_when50))")

    ax2 = Axis(fig[1,2];
        xlabel="alignment Aidx",
        ylabel="log10(ωtilde_ctr)  (log-centroid)",
        title="B) Alignment vs WHEN (log-centroid, more sensitive)"
    )
    scatter!(ax2, Aidx[m_whenc], log10.(w_lctr[m_whenc]), markersize=7)
    text!(ax2, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(Aidx, log ωtilde_ctr) = $(round(ρ_whenc,digits=3))   N=$(length(m_whenc))")

    ax3 = Axis(fig[1,3];
        xscale=log10, yscale=log10,
        xlabel="ωtilde_pred = u_str * Tbar  (strength-weighted u)",
        ylabel="ωtilde_m50",
        title="C) Strength-weighted predictor vs WHEN"
    )
    scatter!(ax3, w_pred[m_predw], w_m50[m_predw], markersize=7)
    text!(ax3, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log) = $(round(ρ_predw,digits=3))   N=$(length(m_predw))")

    ax4 = Axis(fig[2,1];
        xscale=log10, yscale=log10,
        xlabel="predicted per-unit magnitude H_real",
        ylabel="realized per-unit magnitude mean(|ΔV|/V)/eps",
        title="D) HOW MUCH per unit: prediction vs realized"
    )
    scatter!(ax4, H_real[m_mag], R_abs[m_mag], markersize=7)
    text!(ax4, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log) = $(round(ρ_mag,digits=3))   N=$(length(m_mag))")

    ax5 = Axis(fig[2,2];
        xscale=log10, yscale=log10,
        xlabel="H_hom (timescales homogenized, Tbar preserved)",
        ylabel="realized per-unit magnitude",
        title="E) HOW MUCH: dynamic facet with timescales removed"
    )
    scatter!(ax5, H_hom[m_mag_h], R_abs[m_mag_h], markersize=7)
    text!(ax5, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log) = $(round(ρ_mag_h,digits=3))   N=$(length(m_mag_h))")

    ax6 = Axis(fig[2,3];
        xlabel="Aidx",
        ylabel="count",
        title="F) Did we excite the alignment axis?"
    )
    hist!(ax6, filter(isfinite, Aidx), bins=25)

    # Example spectra: pick extremes by Aidx
    idx_ok = findall(i -> isfinite(res[i].Aidx) && isfinite(res[i].wtilde_m50), 1:N)
    if length(idx_ok) >= 4
        imin = idx_ok[argmin(Aidx[idx_ok])]
        imax = idx_ok[argmax(Aidx[idx_ok])]

        ax7 = Axis(fig[3,1:3];
            xscale=log10, yscale=log10,
            xlabel="ωtilde = ω * Tbar",
            ylabel="value",
            title="G) Example spectra at extreme alignment: e(ω), G(ω), m(ω)=eG (real vs hom)"
        )

        function plot_one(r; lab="")
            w = r.ωtilde
            lines!(ax7, w, logsafe_vec(r.eω), linewidth=2, label="e(ω) "*lab)
            lines!(ax7, w, logsafe_vec(r.Gω), linewidth=2, linestyle=:dash, label="G(ω) "*lab)
            lines!(ax7, w, logsafe_vec(r.mω), linewidth=3, linestyle=:dot, label="m(ω) "*lab)

            # hom (lighter)
            lines!(ax7, w, logsafe_vec(r.eω_h), linewidth=2, linestyle=:dashdot, label="e_hom "*lab)
            lines!(ax7, w, logsafe_vec(r.mω_h), linewidth=2, linestyle=:dash, label="m_hom "*lab)
        end

        plot_one(res[imin]; lab="(low Aidx)")
        plot_one(res[imax]; lab="(high Aidx)")
        axislegend(ax7; position=:rb, nbanks=2)
    end

    display(fig)
    return nothing
end

# -----------------------------
# MAIN
# -----------------------------
# Use a fixed normalized-frequency grid ωtilde across all permutations.
# Typical relevant range for ωtilde depends on your ensemble; start conservative.
ωtilde_vals = 10 .^ range(log10(1e-4), log10(2e-1); length=70)

run = run_alignment_sweep(
    S=120,
    seed=1234,
    nperm=120,
    ωtilde_vals=ωtilde_vals,
    C0_mode=:I,              # isolate resolvent geometry + timescale placement
    P_mode=:edges_only,      # perturb existing edges only
    P_reps=14,
    nprobe=10,
    eps_rel=0.03
)

plot_alignment_sweep(run)
################################################################################
