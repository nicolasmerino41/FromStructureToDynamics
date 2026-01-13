################################################################################
# FREQUENCY-RESOLVED STRUCTURAL SENSITIVITY: NUMERICAL PROOFS (FULL PIPELINE)
#
# What this script is meant to show (across many heterogeneous stable bases):
#
# (1) Total time-domain structural error matches total frequency-domain sensitivity.
#     - Time error from rmed:   Err_tot  = ∫ Δ(τ) d log τ
#     - Sensitivity:            Sens_tot = ∫ S(ω) dω
#
# (2) Relevant-window match (defined ONLY by t95; no arbitrary extra times):
#     - Relevant window in time: τ = t/t95, with τ ∈ (0,1]
#       Err_rel  = ∫_{τ≤1} Δ(τ) d log τ
#     - Relevant frequencies: ω ≥ ω95 = 1/t95
#       Sens_rel = ∫_{ω≥ω95} S(ω) dω
#
# (3) “How early indirect effects can matter” via a cutoff frequency ωc from a
#     collectivity-like K(ω) built from the resolvent factorisation:
#       (i ω T - Abar)^(-1) = (I - A_ω)^(-1) (i ω T + I)^(-1),
#       A_ω = (i ω T + I)^(-1) A, with A = Abar + I (off-diagonal part).
#     Define K(ω) = ρ(|A_ω|). Then ωc is where K(ω) drops below 1.
#     Interpret τc = (1/ωc)/t95 as a minimal normalised time for “indirect-effect
#     amplification” to become possible.
#
# (4) Typical vs worst-case vs aligned perturbations:
#     - Typical S(ω): average over random Fro-normalised uncertainty directions P.
#     - Worst-case bound Swc(ω): using σ_min of M(ω) and ||R(ω)diag(u)||_2.
#     - Aligned P*(ω_star): rank-1 direction built from singular vectors of R(ω_star) and
#       R(ω_star)diag(u), showing worst-case can be “excited” when aligned.
#
# Core objects:
# - Biomass-weighted rmed(t):
#     rmed(t) = -(1/(2t)) [ log tr(E C E') - log tr(C) ],  E=exp(tJ), C=diag(u^2).
# - Structural error curve:
#     Δ(t) = | rmed_base(t) - rmed_pert(t) |, and τ = t/t95(base).
# - Generalised resolvent sensitivity:
#     R(ω) = (i ω T - Abar)^(-1),   T = diag(1/u)
#     S(ω;P) = eps^2 * || R(ω) P R(ω) diag(u) ||_F^2 / sum(u^2)
################################################################################
using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie
using Base.Threads

# -----------------------------
# Utilities
# -----------------------------
meanfinite(v) = (x = filter(isfinite, v); isempty(x) ? NaN : mean(x))
medianfinite(v) = (x = filter(isfinite, v); isempty(x) ? NaN : median(x))

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

# Integrate y(x) d log x = ∫ y(x) (dx/x)
function trapz_logx(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    n = length(x)
    n < 2 && return NaN
    s = 0.0
    for i in 1:(n-1)
        x1, x2 = float(x[i]), float(x[i+1])
        y1, y2 = float(y[i]), float(y[i+1])
        if isfinite(x1) && isfinite(x2) && x1 > 0 && x2 > 0 && isfinite(y1) && isfinite(y2)
            dlogx = log(x2) - log(x1)
            s += 0.5 * (y1 + y2) * dlogx
        end
    end
    return s
end

spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigvals(Matrix(J))))

# Nonnegative-matrix spectral radius (power iteration)
function spectral_radius_power(M::AbstractMatrix{<:Real}; iters::Int=80)
    n = size(M,1)
    v = ones(Float64, n)
    v ./= norm(v)
    λ = 0.0
    for _ in 1:iters
        w = M * v
        nw = norm(w)
        nw == 0 && return 0.0
        v = w / nw
        λ = dot(v, M*v) / dot(v,v)
    end
    return isfinite(λ) ? max(λ, 0.0) : NaN
end

# Interpolate cutoff in log ω (more stable on log-grids)
function cutoff_frequency_from_K(ωvals::Vector{Float64}, Kvals::Vector{Float64}; κ::Float64=1.0)
    @assert length(ωvals) == length(Kvals)
    idx = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(Kvals[i]), eachindex(ωvals))
    length(idx) < 2 && return NaN
    ω = ωvals[idx]; K = Kvals[idx]

    # find first crossing K <= κ as ω increases
    j = findfirst(K .<= κ)
    if isnothing(j)
        return NaN
    elseif j == 1
        return ω[1]
    else
        ω1, ω2 = ω[j-1], ω[j]
        K1, K2 = K[j-1], K[j]
        if K2 == K1
            return ω2
        end
        x1, x2 = log(ω1), log(ω2)
        x = x1 + (κ - K1) * (x2 - x1) / (K2 - K1)
        return exp(x)
    end
end

# -----------------------------
# Biomass weights u
# -----------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    abs.(rand(rng, LogNormal(mu, sigma), S))
end

# -----------------------------
# Trophic-like heterogeneous off-diagonal generator O (diag=0)
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
# Choose scale s so α(J) ≈ target_alpha (<0)
# Base: Abar = -I + s*O,  J = diag(u)*Abar
# -----------------------------
function find_scale_to_target_alpha(O::Matrix{Float64}, u::Vector{Float64};
    target_alpha::Float64=-0.05,
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
# Biomass-weighted rmed(t) and t95
# -----------------------------
function rmed_biomass(J::AbstractMatrix, u::AbstractVector; t::Real)
    tt = float(t)
    tt <= 0 && return NaN
    E = exp(tt * Matrix(J))
    any(!isfinite, E) && return NaN
    w = u .^ 2
    C = Diagonal(w)
    Ttr = tr(E * C * transpose(E))
    (!isfinite(Ttr) || Ttr <= 0) && return NaN
    r = -(log(Ttr) - log(sum(w))) / (2*tt)
    return isfinite(r) ? r : NaN
end

function rmed_curve(J, u, tvals::Vector{Float64})
    r = Vector{Float64}(undef, length(tvals))
    for (i,t) in enumerate(tvals)
        r[i] = rmed_biomass(J, u; t=t)
    end
    return r
end

function t95_from_rmed_curve(t_vals::AbstractVector, rmed::AbstractVector; target::Real=0.05)
    @assert length(t_vals) == length(rmed)
    y = @. exp(-rmed * t_vals)
    idx = findfirst(y .<= target)
    isnothing(idx) && return Inf
    idx == 1 && return float(t_vals[1])

    t1, t2 = float(t_vals[idx-1]), float(t_vals[idx])
    y1, y2 = float(y[idx-1]), float(y[idx])
    y2 == y1 && return t2
    return t1 + (target - y1) * (t2 - t1) / (y2 - y1)
end

function delta_curve(r_base::Vector{Float64}, r_pert::Vector{Float64})
    @assert length(r_base) == length(r_pert)
    Δ = Vector{Float64}(undef, length(r_base))
    for i in eachindex(r_base)
        Δ[i] = (isfinite(r_base[i]) && isfinite(r_pert[i])) ? abs(r_base[i] - r_pert[i]) : NaN
    end
    return Δ
end

# -----------------------------
# Simple uncertainty directions P (noise on off-diagonals), ||P||_F=1
# -----------------------------
function sample_noise_Pdir(S::Int; sparsity_p::Float64=1.0, rng=Random.default_rng())
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

# -----------------------------
# Non-normality proxy: peak transient gain (base only)
# Gpeak = max_t ||exp(Jt)||_2^2
# -----------------------------
function peak_transient_gain(J::AbstractMatrix, tvals::Vector{Float64})
    best = -Inf
    for t in tvals
        E = exp(float(t) * Matrix(J))
        g2 = opnorm(E)^2
        if isfinite(g2) && g2 > best
            best = g2
        end
    end
    return isfinite(best) ? best : NaN
end

# -----------------------------
# Time-domain error integrals and timing quantile (normalised time τ=t/t95)
# Err_tot = ∫ Δ(τ) dlogτ (over all τ sampled)
# Err_rel = ∫_{τ≤1} Δ(τ) dlogτ
# τqΔ    = time-quantile from Δ-mass in relevant window (τ≤1)
# -----------------------------
function quantile_time_from_mass(τ::Vector{Float64}, y::Vector{Float64};
    q::Float64=0.5, τmax::Float64=1.0
)
    @assert 0 < q < 1
    @assert length(τ) == length(y)

    idx = findall(i -> isfinite(τ[i]) && τ[i] > 0 && τ[i] <= τmax && isfinite(y[i]) && y[i] >= 0, eachindex(τ))
    length(idx) < 3 && return NaN

    τx = τ[idx]
    yx = y[idx]

    # total mass on log τ
    tot = trapz_logx(τx, yx)
    (isfinite(tot) && tot > 0) || return NaN

    # cumulative trapezoids on log τ
    cum = zeros(Float64, length(τx))
    for i in 2:length(τx)
        dlogτ = log(τx[i]) - log(τx[i-1])
        cum[i] = cum[i-1] + 0.5*(yx[i-1] + yx[i]) * dlogτ
    end
    target = q * tot
    j = findfirst(cum .>= target)
    isnothing(j) && return NaN
    j == 1 && return τx[1]

    # interpolate in cum vs log τ
    τ1, τ2 = τx[j-1], τx[j]
    c1, c2 = cum[j-1], cum[j]
    if c2 == c1
        return τ2
    else
        x1, x2 = log(τ1), log(τ2)
        x = x1 + (target - c1) * (x2 - x1) / (c2 - c1)
        return exp(x)
    end
end

function time_errors_normalised(tvals::Vector{Float64}, rbase::Vector{Float64}, rpert::Vector{Float64};
    qΔ::Float64=0.5
)
    @assert length(tvals) == length(rbase) == length(rpert)

    t95 = t95_from_rmed_curve(tvals, rbase; target=0.05)
    (isfinite(t95) && t95 > 0) || return nothing

    τ = tvals ./ t95
    Δ = delta_curve(rbase, rpert)

    good_all = findall(i -> isfinite(τ[i]) && τ[i] > 0 && isfinite(Δ[i]) && Δ[i] >= 0, eachindex(τ))
    length(good_all) < 2 && return nothing
    Err_tot = trapz_logx(τ[good_all], Δ[good_all])

    good_rel = findall(i -> isfinite(τ[i]) && τ[i] > 0 && τ[i] <= 1.0 && isfinite(Δ[i]) && Δ[i] >= 0, eachindex(τ))
    Err_rel = (length(good_rel) >= 2) ? trapz_logx(τ[good_rel], Δ[good_rel]) : NaN

    τqΔ = quantile_time_from_mass(τ, Δ; q=qΔ, τmax=1.0)

    return (t95=t95, τ=τ, Δ=Δ, Err_tot=Err_tot, Err_rel=Err_rel, τqΔ=τqΔ)
end

# -----------------------------
# Frequency: typical S(ω), q90, and worst-case bound Swc(ω)
# R(ω) = (i ω T - Abar)^(-1),  T = diag(1/u)
# S(ω;P) = eps^2 * || R P R diag(u) ||_F^2 / sum(u^2)
# Swc(ω) bound: eps^2/sum(u^2) * ||R||_2^2 * ||R diag(u)||_2^2
# -----------------------------
function sensitivity_spectra_typ_q90_wc(Abar::Matrix{Float64}, u::Vector{Float64},
    eps::Float64, ωvals::Vector{Float64}, Pdirs::Vector{Matrix{Float64}}
)
    T = Diagonal(1.0 ./ u)
    U = Matrix{ComplexF64}(Diagonal(u))
    denom = sum(u.^2)

    nω = length(ωvals)
    Smean = fill(NaN, nω)
    Sq90  = fill(NaN, nω)
    Swc   = fill(NaN, nω)

    for (k, ω0) in enumerate(ωvals)
        ω = float(ω0)
        Mω = Matrix{ComplexF64}(im*ω*T - Abar)
        F = lu(Mω)

        Y = F \ U  # Y = R diag(u)

        vals = Float64[]
        for P in Pdirs
            Z = Matrix{ComplexF64}(P) * Y
            X = F \ Z
            v = (eps^2) * (norm(X)^2) / denom
            (isfinite(v) && v >= 0) && push!(vals, v)
        end

        if !isempty(vals)
            Smean[k] = mean(vals)
            Sq90[k]  = quantile(vals, 0.90)
        end

        # worst-case bound
        svalsM = svdvals(Mω)
        σmin = svalsM[end]
        Rop = (isfinite(σmin) && σmin > 0) ? (1.0/σmin) : NaN
        Yop = opnorm(Y)
        vwc = (eps^2) * (Rop^2) * (Yop^2) / denom
        Swc[k] = (isfinite(vwc) && vwc >= 0) ? vwc : NaN
    end

    return (Smean=Smean, Sq90=Sq90, Swc=Swc)
end

function integrate_S_tot(ωvals::Vector{Float64}, Sω::Vector{Float64})
    idx = findall(i -> isfinite(ωvals[i]) && isfinite(Sω[i]) && ωvals[i] > 0, eachindex(Sω))
    length(idx) < 2 && return NaN
    return trapz(ωvals[idx], Sω[idx])
end

function integrate_S_relevant(ωvals::Vector{Float64}, Sω::Vector{Float64}, ω95::Float64)
    if !(isfinite(ω95) && ω95 > 0)
        return NaN
    end
    idx = findall(i -> isfinite(ωvals[i]) && isfinite(Sω[i]) && ωvals[i] >= ω95, eachindex(Sω))
    length(idx) < 2 && return NaN
    return trapz(ωvals[idx], Sω[idx])
end

# Cutoff time from S-mass above ω95: find ωq such that fraction q is accumulated
function cutoff_time_from_S(ωvals::Vector{Float64}, Sω::Vector{Float64}, ω95::Float64; q::Float64=0.5)
    @assert 0 < q < 1
    if !(isfinite(ω95) && ω95 > 0)
        return (ωq=NaN, tq=NaN)
    end
    idx = findall(i -> isfinite(ωvals[i]) && ωvals[i] >= ω95 && ωvals[i] > 0 && isfinite(Sω[i]) && Sω[i] >= 0, eachindex(Sω))
    length(idx) < 3 && return (ωq=NaN, tq=NaN)

    ω = ωvals[idx]
    S = Sω[idx]

    Stot = trapz(ω, S)
    (isfinite(Stot) && Stot > 0) || return (ωq=NaN, tq=NaN)

    cum = zeros(Float64, length(ω))
    for i in 2:length(ω)
        cum[i] = cum[i-1] + 0.5*(S[i-1] + S[i])*(ω[i]-ω[i-1])
    end

    target = q * Stot
    j = findfirst(cum .>= target)
    isnothing(j) && return (ωq=NaN, tq=NaN)
    j == 1 && return (ωq=ω[1], tq=1.0/ω[1])

    ω1, ω2 = ω[j-1], ω[j]
    c1, c2 = cum[j-1], cum[j]
    if c2 == c1
        ωq = ω2
    else
        ωq = ω1 + (target - c1) * (ω2 - ω1) / (c2 - c1)
    end
    tq = (isfinite(ωq) && ωq > 0) ? (1.0/ωq) : NaN
    return (ωq=ωq, tq=tq)
end

# -----------------------------
# Collectivity spectrum K(ω) and ωc
# Abar = -I + A, with A = Abar + I (zero diagonal).
# A_ω = (i ω T + I)^(-1) A   (row-scaled by 1/(1 + i ω T_i))
# K(ω) = ρ(|A_ω|)
# -----------------------------
function collectivity_spectrum(Abar::Matrix{Float64}, u::Vector{Float64}, ωvals::Vector{Float64};
    iters::Int=80
)
    S = size(Abar,1)
    A = Abar + Matrix{Float64}(I, S, S)  # remove diag -1 => offdiag matrix A, diag ~0
    Tvec = 1.0 ./ u                      # T_i

    K = fill(NaN, length(ωvals))
    for (k, ω0) in enumerate(ωvals)
        ω = float(ω0)
        denom = 1 .+ im*ω .* Tvec
        Aω = A ./ reshape(denom, S, 1)      # row scaling
        M = abs.(Aω)                        # nonnegative real matrix
        K[k] = spectral_radius_power(M; iters=iters)
    end
    return K
end

# -----------------------------
# Worst-case “aligned” perturbation at one ω_star:
# max_{||P||_F=1} ||R P Y||_F = ||R||_2 ||Y||_2, achieved by rank-1 P = vR * uY'
# (right singular vector of R) times (left singular vector of Y).
# -----------------------------
function aligned_P_at_ω(Abar::Matrix{Float64}, u::Vector{Float64}, ω_star::Float64)
    S = length(u)
    T = Diagonal(1.0 ./ u)
    U = Matrix{ComplexF64}(Diagonal(u))

    Mω = Matrix{ComplexF64}(im * ω_star * T - Abar)
    R = inv(Mω)
    Y = R * U

    svR = svd(R)
    vR = svR.V[:,1]             # right singular vector of R for σmax

    svY = svd(Y)
    uY = svY.U[:,1]             # left singular vector of Y for σmax

    P = real.(vR * uY')         # rank-1 real direction
    for i in 1:S
        P[i,i] = 0.0
    end
    nP = norm(P)
    nP == 0 && return nothing
    P ./= nP
    return (P=P, R=R, Y=Y)
end

function S_at_ω_for_P(Abar::Matrix{Float64}, u::Vector{Float64}, eps::Float64, ω::Float64, P::Matrix{Float64})
    T = Diagonal(1.0 ./ u)
    U = Matrix{ComplexF64}(Diagonal(u))
    denom = sum(u.^2)
    Mω = Matrix{ComplexF64}(im*ω*T - Abar)
    F = lu(Mω)
    Y = F \ U
    Z = Matrix{ComplexF64}(P) * Y
    X = F \ Z
    v = (eps^2) * (norm(X)^2) / denom
    return (isfinite(v) && v >= 0) ? v : NaN
end

# -----------------------------
# Base systems
# -----------------------------
struct BaseSys
    u::Vector{Float64}
    Abar::Matrix{Float64}   # diag -1
    rbase::Vector{Float64}
    t95::Float64
    eps::Float64
    Gpeak::Float64
end

function build_bases(; S::Int, base_reps::Int, seed::Int,
    tvals::Vector{Float64},
    u_mean::Float64=1.0, u_cv::Float64=1.0,
    connectance_rng=(0.03, 0.12),
    trophic_align_rng=(0.55, 0.98),
    reciprocity_rng=(0.00, 0.20),
    σ_rng=(0.3, 1.5),
    target_alpha::Float64=-0.05,
    eps_rel::Float64=0.20
)
    bases = BaseSys[]
    for b in 1:base_reps
        rng = MersenneTwister(seed + 10007*b)

        u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))

        c  = rand(rng, Uniform(connectance_rng[1], connectance_rng[2]))
        γ  = rand(rng, Uniform(trophic_align_rng[1], trophic_align_rng[2]))
        ρr = rand(rng, Uniform(reciprocity_rng[1], reciprocity_rng[2]))
        σ  = rand(rng, Uniform(σ_rng[1], σ_rng[2]))

        O = trophic_O(S; connectance=c, trophic_align=γ, reciprocity=ρr, σ=σ, rng=rng)
        normalize_offdiag!(O) || continue

        s = find_scale_to_target_alpha(O, u; target_alpha=target_alpha)
        isfinite(s) || continue

        Abar = -Matrix{Float64}(I, S, S) + s * O
        J = Diagonal(u) * Abar

        rbase = rmed_curve(J, u, tvals)
        t95 = t95_from_rmed_curve(tvals, rbase; target=0.05)
        (isfinite(t95) && t95 > 0) || continue

        offA = Abar + Matrix{Float64}(I, S, S)
        eps = eps_rel * norm(offA)
        (isfinite(eps) && eps > 0) || continue

        Gpk = peak_transient_gain(J, tvals)

        push!(bases, BaseSys(u, Abar, rbase, t95, eps, Gpk))
    end
    return bases
end

# -----------------------------
# Evaluate one base:
# - sample P ensemble, keep stable perturbations
# - compute time errors and τqΔ
# - compute S(ω) typical/q90 and worst-case bound
# - compute integrated sensitivities (total + relevant)
# - compute τqS from sensitivity mass
# - compute K(ω), ωc, τc, and static K0
# -----------------------------
function eval_base(base::BaseSys, tvals::Vector{Float64}, ωvals::Vector{Float64};
    P_reps::Int=20,
    P_sparsity::Float64=1.0,
    margin::Float64=1e-3,
    qS::Float64=0.5,
    qΔ::Float64=0.5,
    κK::Float64=1.0,
    seed::Int=1
)
    S = length(base.u)
    rng = MersenneTwister(seed)

    Du = Diagonal(base.u)
    Abar = base.Abar

    Pdirs = Matrix{Float64}[]
    Err_tot_list = Float64[]
    Err_rel_list = Float64[]
    τqΔ_list = Float64[]

    for k in 1:P_reps
        P = sample_noise_Pdir(S; sparsity_p=P_sparsity, rng=rng)
        P === nothing && continue

        Abarp = Abar + base.eps * P
        Jp = Du * Abarp
        αp = spectral_abscissa(Jp)
        (isfinite(αp) && αp < -margin) || continue

        rpert = rmed_curve(Jp, base.u, tvals)
        tm = time_errors_normalised(tvals, base.rbase, rpert; qΔ=qΔ)
        tm === nothing && continue

        push!(Pdirs, P)
        push!(Err_tot_list, tm.Err_tot)
        push!(Err_rel_list, tm.Err_rel)
        push!(τqΔ_list, tm.τqΔ)
    end

    length(Pdirs) < 6 && return nothing

    # spectra
    sp = sensitivity_spectra_typ_q90_wc(Abar, base.u, base.eps, ωvals, Pdirs)

    # total & relevant sensitivity
    Sens_tot = integrate_S_tot(ωvals, sp.Smean)
    ω95 = 1.0 / base.t95
    Sens_rel = integrate_S_relevant(ωvals, sp.Smean, ω95)

    # τqS from S mass above ω95
    ctS = cutoff_time_from_S(ωvals, sp.Smean, ω95; q=qS)
    τqS = (isfinite(ctS.tq) && ctS.tq > 0) ? (ctS.tq / base.t95) : NaN

    # collectivity spectrum K(ω), ωc and τc
    Kω = collectivity_spectrum(Abar, base.u, ωvals; iters=60)
    ωc = cutoff_frequency_from_K(ωvals, Kω; κ=κK)
    τc = (isfinite(ωc) && ωc > 0) ? ((1.0/ωc) / base.t95) : NaN

    # static collectivity K0
    Sdim = size(Abar,1)
    A = Abar + Matrix{Float64}(I, Sdim, Sdim)
    K0 = spectral_radius_power(abs.(A); iters=80)

    return (
        nP=length(Pdirs),

        Err_tot=meanfinite(Err_tot_list),
        Err_rel=meanfinite(Err_rel_list),
        τqΔ=meanfinite(τqΔ_list),

        Smean=sp.Smean,
        Sq90=sp.Sq90,
        Swc=sp.Swc,

        Sens_tot=Sens_tot,
        Sens_rel=Sens_rel,

        ω95=ω95,
        ωqS=ctS.ωq,
        τqS=τqS,

        Kω=Kω,
        ωc=ωc,
        τc=τc,
        K0=K0,

        Gpeak=base.Gpeak,
        t95=base.t95,

        # keep Pdirs for an optional example analysis
        Pdirs=Pdirs
    )
end

# -----------------------------
# Run experiment across many bases (threaded over bases)
# -----------------------------
function run_experiment(; S::Int=70, base_reps::Int=70, P_reps::Int=18,
    seed::Int=1234,
    tvals = 10 .^ range(log10(0.01), log10(200.0); length=45),
    ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=80),
    target_alpha::Float64=-0.05,
    eps_rel::Float64=0.20,
    margin::Float64=1e-3,
    qS::Float64=0.5,
    qΔ::Float64=0.5,
    κK::Float64=1.0
)
    tvals = collect(float.(tvals))
    ωvals = collect(float.(ωvals))

    bases = build_bases(
        S=S, base_reps=base_reps, seed=seed,
        tvals=tvals,
        target_alpha=target_alpha,
        eps_rel=eps_rel
    )
    @info "Built $(length(bases)) stable bases (out of $base_reps attempts)."

    nB = length(bases)

    Err_tot  = fill(NaN, nB)
    Err_rel  = fill(NaN, nB)
    τqΔ      = fill(NaN, nB)

    Sens_tot = fill(NaN, nB)
    Sens_rel = fill(NaN, nB)

    τqS      = fill(NaN, nB)
    τc       = fill(NaN, nB)

    ωc       = fill(NaN, nB)
    K0       = fill(NaN, nB)

    Gpeak    = fill(NaN, nB)
    t95      = fill(NaN, nB)
    nPacc    = fill(0, nB)

    # store one example (first valid)
    example = Ref{Any}(nothing)

    Threads.@threads for i in 1:nB
        out = eval_base(bases[i], tvals, ωvals;
            P_reps=P_reps,
            margin=margin,
            qS=qS,
            qΔ=qΔ,
            κK=κK,
            seed=seed + 900_000*i
        )
        out === nothing && continue

        Err_tot[i]  = out.Err_tot
        Err_rel[i]  = out.Err_rel
        τqΔ[i]      = out.τqΔ

        Sens_tot[i] = out.Sens_tot
        Sens_rel[i] = out.Sens_rel

        τqS[i]      = out.τqS
        τc[i]       = out.τc

        ωc[i]       = out.ωc
        K0[i]       = out.K0

        Gpeak[i]    = out.Gpeak
        t95[i]      = out.t95
        nPacc[i]    = out.nP

        if example[] === nothing && isfinite(out.Sens_rel) && isfinite(out.ωc) && isfinite(out.K0)
            example[] = (i=i, out=out, base=bases[i])
        end
    end

    return (
        tvals=tvals, ωvals=ωvals,
        bases=bases,
        Err_tot=Err_tot, Err_rel=Err_rel, τqΔ=τqΔ,
        Sens_tot=Sens_tot, Sens_rel=Sens_rel,
        τqS=τqS, τc=τc,
        ωc=ωc, K0=K0,
        Gpeak=Gpeak, t95=t95,
        nPacc=nPacc,
        example=example[]
    )
end

# -----------------------------
# Plotting + summary
# -----------------------------
function corr_loglog(x::Vector{Float64}, y::Vector{Float64})
    idx = findall(i -> isfinite(x[i]) && x[i] > 0 && isfinite(y[i]) && y[i] > 0, eachindex(x))
    length(idx) < 6 && return (NaN, 0, idx)
    return (cor(log.(x[idx]), log.(y[idx])), length(idx), idx)
end

function summarize_and_plot(res; figsize=(1700, 1100), q_align=0.5)
    ωvals = res.ωvals

    ρ1, N1, m1 = corr_loglog(res.Sens_tot, res.Err_tot)
    @info "Test 1: cor(log Sens_tot, log Err_tot) = $ρ1 (N=$N1)"

    ρ2, N2, m2 = corr_loglog(res.Sens_rel, res.Err_rel)
    @info "Test 2: cor(log Sens_rel, log Err_rel) = $ρ2 (N=$N2)"

    ρ3, N3, m3 = corr_loglog(res.Sens_rel, res.τqS)
    @info "Timing from S (τqS) vs Sens_rel: cor(log,log) = $ρ3 (N=$N3)"

    ρ4, N4, m4 = corr_loglog(res.τqS, res.τqΔ)
    @info "Shape alignment: τqΔ vs τqS: cor(log,log) = $ρ4 (N=$N4)"

    ρ5, N5, m5 = corr_loglog(res.τc, res.τqΔ)
    @info "Collectivity cutoff link: τqΔ vs τc: cor(log,log) = $ρ5 (N=$N5)"

    ρ6, N6, m6 = corr_loglog(res.K0, res.ωc)
    @info "Static→dynamic collectivity: ωc vs K0: cor(log,log) = $ρ6 (N=$N6)"

    ρ7, N7, m7 = corr_loglog(res.Gpeak, res.τqΔ)
    @info "Error timing vs non-normality: τqΔ vs Gpeak: cor(log,log) = $ρ7 (N=$N7)"

    fig = Figure(size=figsize)

    axA = Axis(fig[1,1], xscale=log10, yscale=log10,
        xlabel="Sens_tot = ∫ S(ω) dω",
        ylabel="Err_tot = ∫ Δ(τ) d log τ",
        title="A) Total match"
    )
    scatter!(axA, res.Sens_tot[m1], res.Err_tot[m1], markersize=7)
    text!(axA, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log)=$(round(ρ1,digits=3))  N=$N1")

    axB = Axis(fig[1,2], xscale=log10, yscale=log10,
        xlabel="Sens_rel = ∫_{ω≥1/t95} S(ω) dω",
        ylabel="Err_rel = ∫_{τ≤1} Δ(τ) d log τ",
        title="B) Relevant-window match"
    )
    scatter!(axB, res.Sens_rel[m2], res.Err_rel[m2], markersize=7)
    text!(axB, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log)=$(round(ρ2,digits=3))  N=$N2")

    axG = Axis(fig[1,3], xscale=log10, yscale=log10,
        xlabel="Gpeak = max_t ||exp(Jt)||_2^2",
        ylabel="τqΔ",
        title="G) Error timing vs non-normality proxy"
    )
    scatter!(axG, res.Gpeak[m7], res.τqΔ[m7], markersize=7)
    text!(axG, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log)=$(round(ρ7,digits=3))  N=$N7")

    axC = Axis(fig[2,1], xscale=log10, yscale=log10,
        xlabel="Sens_rel",
        ylabel="τqS (from S-mass)",
        title="C) Sensitivity timing proxy (from S)"
    )
    scatter!(axC, res.Sens_rel[m3], res.τqS[m3], markersize=7)
    text!(axC, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log)=$(round(ρ3,digits=3))  N=$N3")

    axD = Axis(fig[2,2], xscale=log10, yscale=log10,
        xlabel="τqS (from S)",
        ylabel="τqΔ (from Δ)",
        title="D) Shape alignment (time quantile vs freq quantile)"
    )
    scatter!(axD, res.τqS[m4], res.τqΔ[m4], markersize=7)
    text!(axD, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log)=$(round(ρ4,digits=3))  N=$N4")

    axH = Axis(fig[2,3], xscale=log10, yscale=log10,
        xlabel="ω",
        ylabel="S(ω)",
        title="H) Example S(ω): typical vs worst-case bound"
    )
    if res.example !== nothing
        out = res.example.out
        lines!(axH, ωvals, out.Smean, linewidth=3)
        lines!(axH, ωvals, out.Sq90, linewidth=3, linestyle=:dash)
        lines!(axH, ωvals, out.Swc,  linewidth=4)
        vlines!(axH, [out.ω95], linestyle=:dot, linewidth=2)
        if isfinite(out.ωc)
            vlines!(axH, [out.ωc], linestyle=:dashdot, linewidth=2)
        end
    end

    axE = Axis(fig[3,1], xscale=log10, yscale=log10,
        xlabel="τc = (1/ωc)/t95 (from K(ω))",
        ylabel="τqΔ (from Δ-mass, τ≤1)",
        title="E) Cutoff frequency → earlier structural divergence"
    )
    scatter!(axE, res.τc[m5], res.τqΔ[m5], markersize=7)
    text!(axE, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log)=$(round(ρ5,digits=3))  N=$N5")

    axF = Axis(fig[3,2], xscale=log10, yscale=log10,
        xlabel="K0 = ρ(|A|)",
        ylabel="ωc (K(ω)=1 cutoff)",
        title="F) Static collectivity predicts cutoff frequency"
    )
    scatter!(axF, res.K0[m6], res.ωc[m6], markersize=7)
    text!(axF, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log)=$(round(ρ6,digits=3))  N=$N6")

    axI = Axis(fig[3,3],
        xlabel="",
        ylabel="S(ω_star; P)",
        title="I) Typical vs aligned vs bound at ω_star"
    )
    if res.example !== nothing
        out = res.example.out
        base = res.example.base
        ω_star = isfinite(out.ωc) ? out.ωc : ωvals[argmax(replace(out.Swc, NaN=>-Inf))]
        ω_star = (isfinite(ω_star) && ω_star > 0) ? ω_star : ωvals[div(length(ωvals),2)]

        Svals = Float64[]
        for P in out.Pdirs
            v = S_at_ω_for_P(base.Abar, base.u, base.eps, ω_star, P)
            (isfinite(v) && v >= 0) && push!(Svals, v)
        end
        if !isempty(Svals)
            x = rand(length(Svals)) .* 0.2 .+ 0.9
            scatter!(axI, x, Svals, markersize=6)
        end

        s_mean = isempty(Svals) ? NaN : mean(Svals)
        s_q90  = isempty(Svals) ? NaN : quantile(Svals, 0.90)

        align = aligned_P_at_ω(base.Abar, base.u, ω_star)
        s_align = (align === nothing) ? NaN : S_at_ω_for_P(base.Abar, base.u, base.eps, ω_star, align.P)

        k_star = argmin(abs.(ωvals .- ω_star))
        s_bound = out.Swc[k_star]

        isfinite(s_mean)  && hlines!(axI, [s_mean],  linewidth=3)
        isfinite(s_q90)   && hlines!(axI, [s_q90],   linewidth=3, linestyle=:dash)
        isfinite(s_align) && hlines!(axI, [s_align], linewidth=4, linestyle=:dashdot)
        isfinite(s_bound) && hlines!(axI, [s_bound], linewidth=4)

        txt = "ω_star=$(round(ω_star,digits=3))\nmean=$(round(s_mean,digits=3))  q90=$(round(s_q90,digits=3))\naligned=$(round(s_align,digits=3))  bound=$(round(s_bound,digits=3))"
        text!(axI, 0.05, 0.95, space=:relative, align=(:left,:top), text=txt)
    end

    display(fig)

    # -------------------------
    # NEW TEST / PLOT:
    # K0 vs τc  (dimensionless comparability check)
    # -------------------------
    ρ8, N8, m8 = corr_loglog(res.K0, res.τc)
    @info "Comparability check: τc vs K0: cor(log,log) = $ρ8 (N=$N8)"

    fig2 = Figure(size=(650, 500))
    axJ = Axis(fig2[1,1], xscale=log10, yscale=log10,
        xlabel="K0 = ρ(|A|)",
        ylabel="τc = (1/ωc)/t95",
        title="NEW) Is cutoff comparable across systems?"
    )
    scatter!(axJ, res.K0[m8], res.τc[m8], markersize=7)
    text!(axJ, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log)=$(round(ρ8,digits=3))  N=$N8")
    display(fig2)

end

# -----------------------------
# MAIN
# -----------------------------
tvals = 10 .^ range(log10(0.01), log10(200.0); length=45)
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=80)

res_hetero_timescales = run_experiment(
    S=120,              # raise later (e.g. 90–120) once happy with behaviour
    base_reps=80,
    P_reps=18,
    seed=1234,
    tvals=tvals,
    ωvals=ωvals,
    target_alpha=-0.05,
    eps_rel=0.20,
    margin=1e-3,
    qS=0.5,            # sensitivity timing quantile
    qΔ=0.5,            # error timing quantile
    κK=1.0             # collectivity cutoff threshold
)

summarize_and_plot(res_hetero_timescales)
################################################################################
