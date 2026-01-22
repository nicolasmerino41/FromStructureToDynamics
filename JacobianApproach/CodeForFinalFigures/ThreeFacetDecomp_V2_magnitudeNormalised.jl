################################################################################
# NORMALISED MAGNITUDE (ΔV/V per perturbation size) + NORMALISED WHEN (ω,t scaled)
#
# Goal:
#   Re-run the structural-sensitivity / variability pipeline but:
#     (1) Magnitude is normalised by raw perturbation size d = ||A' - A||_F,
#         so we study "response per unit structural change".
#     (2) WHEN metrics (frequency/time location of fragility mass) are normalised
#         by a system time unit, here mean time scale T̄ = mean(T_i)=mean(1/u_i),
#         so ω̃ = ω * T̄ and t̃ = t / T̄ are comparable across systems.
#
# Outputs:
#   - Realised response (Lyapunov): ΔV/V and (ΔV/V)/d for each perturbation.
#   - Predicted slope per unit change from frequency theory:
#         H(P) = [∫ g(ω;P) e(ω) dω] / [∫ e(ω) dω]     (dimensionless)
#     so for small ε:  ΔV/V ≈ ε H(P) and (ΔV/V)/d ≈ H(P) if d≈ε.
#   - "WHEN" via medians of densities over ω:
#       energy density:            e(ω)
#       leverage spectrum:         L(ω) = mean_P |g(ω;P)|
#       fragility mass density:    m(ω) = e(ω) * L(ω)
#     Location metrics (medians): ω_e50, ω_L50, ω_m50, then normalised:
#         ω̃_* = ω_* * T̄    and    t̃_* = (1/ω_*) / T̄ = 1/ω̃_*
#
# Minimal ecological “organisers” to test:
#   - Alignment proxy: strength sits on slow vs fast species
#   - Strength-weighted timescale: u_str = exp( Σ w_i log u_i ), w_i ∝ strength_i
#   - Non-normality proxy: Henrici departure of J_off
#
# Notes:
#   - This script is designed to be simple and explicit (not ultra-optimised).
#   - For big runs: reduce base_reps, ω grid length, nprobe, P_reps first.
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# -----------------------------
# Helpers
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
            s += 0.5*(y1+y2)*(x2-x1)
        end
    end
    return s
end

spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigvals(Matrix(J))))

function spectral_radius_complex(M::AbstractMatrix{ComplexF64})
    vals = eigvals(Matrix(M))
    return maximum(abs.(vals))
end

function logsafe(x; eps=1e-18)
    y = similar(x, Float64)
    for i in eachindex(x)
        v = x[i]
        y[i] = (isfinite(v) && v > eps) ? v : NaN
    end
    return y
end

# cumulative integral array (same grid)
function cumtrapz(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    c = zeros(Float64, n)
    for i in 2:n
        if isfinite(x[i]) && isfinite(x[i-1]) && isfinite(y[i]) && isfinite(y[i-1])
            c[i] = c[i-1] + 0.5*(y[i-1]+y[i])*(x[i]-x[i-1])
        else
            c[i] = c[i-1]
        end
    end
    return c
end

# quantile location for a nonnegative density dens(ω) over ω grid
function quantile_location(ω::Vector{Float64}, dens::Vector{Float64}; q::Float64=0.5)
    @assert 0.0 < q < 1.0
    idx = findall(i -> isfinite(ω[i]) && ω[i] > 0 && isfinite(dens[i]) && dens[i] ≥ 0, eachindex(ω))
    length(idx) < 3 && return NaN
    w = ω[idx]
    d = dens[idx]
    total = trapz(w, d)
    (isfinite(total) && total > 0) || return NaN
    cum = cumtrapz(w, d)
    target = q * total
    j = findfirst(cum .>= target)
    isnothing(j) && return NaN
    j == 1 && return w[1]
    w1, w2 = w[j-1], w[j]
    c1, c2 = cum[j-1], cum[j]
    if c2 == c1
        return w2
    else
        return w1 + (target - c1) * (w2 - w1) / (c2 - c1)
    end
end

# -----------------------------
# Random u (rates) and time scales
# -----------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
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
# Base: Abar = -I + s*O  (diag -1), J = diag(u)*Abar
# -----------------------------
function find_scale_to_target_alpha(O::Matrix{Float64}, u::Vector{Float64};
    target_alpha::Float64 = -0.05,
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
# Structural directions P (diag=0, ||P||_F=1)
# -----------------------------
function sample_Pdir(S::Int; sparsity_p::Float64=1.0, only_edges::Union{Nothing,BitMatrix}=nothing,
                     sign_only::Bool=false, rng=Random.default_rng())
    P = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        i == j && continue
        if only_edges !== nothing
            only_edges[i,j] || continue
        end
        rand(rng) < sparsity_p || continue
        x = randn(rng)
        if sign_only
            x = sign(x) == 0 ? 1.0 : sign(x)
        end
        P[i,j] = x
    end
    nP = norm(P)
    nP == 0 && return nothing
    P ./= nP
    return P
end

# -----------------------------
# Frequency-domain estimators
#   R(ω) = (i ω T - Abar)^(-1),  T = diag(1/u)
#   e(ω) = tr(R C0 R†)/tr(C0)
#   g(ω;P) ≈ 2 Re tr( R P Ĉ ) / tr(Ĉ),  Ĉ = R C0 R†
#
# Hutchinson probes:
#   tr(R C0 R†) = E_v || R sqrt(C0) v ||^2
#   Re tr(R P Ĉ) = E_v Re [ x† (R P x) ] , with x = R sqrt(C0) v
# -----------------------------
function estimate_energy_and_g_at_ω!(
    F::LU{ComplexF64, Matrix{ComplexF64}},
    sqrtc::Vector{Float64},
    trC0::Float64,
    Pdirs::Vector{Matrix{Float64}};
    nprobe::Int=12,
    rng=Random.default_rng()
)
    S = length(sqrtc)
    nP = length(Pdirs)

    # Rademacher probes
    probes = Vector{Vector{Float64}}(undef, nprobe)
    for k in 1:nprobe
        probes[k] = rand(rng, (-1.0, 1.0), S)
    end

    # x_k = R sqrt(C0) v_k
    x_list = Vector{Vector{ComplexF64}}(undef, nprobe)
    xnorm2 = zeros(Float64, nprobe)
    for k in 1:nprobe
        rhs = ComplexF64.(sqrtc .* probes[k])
        x = F \ rhs
        x_list[k] = x
        xnorm2[k] = real(dot(conj.(x), x))
    end

    trChat_est = mean(xnorm2)
    eω = (isfinite(trChat_est) && trC0 > 0) ? (trChat_est / trC0) : NaN

    gvec = fill(NaN, nP)
    for (pidx, P) in enumerate(Pdirs)
        inners = zeros(Float64, nprobe)
        for k in 1:nprobe
            x = x_list[k]
            y = ComplexF64.(P) * x
            z = F \ y
            inners[k] = real(dot(conj.(x), z))
        end
        num_est = mean(inners)
        if isfinite(num_est) && isfinite(trChat_est) && trChat_est > 0
            gvec[pidx] = 2.0 * (num_est / trChat_est)
        end
    end

    return eω, gvec
end

function estimate_spectra(
    Abar::Matrix{Float64},
    u::Vector{Float64},
    C0diag::Vector{Float64},
    ωvals::Vector{Float64},
    Pdirs::Vector{Matrix{Float64}};
    nprobe::Int=12,
    rng=Random.default_rng()
)
    S = length(u)
    @assert length(C0diag) == S
    sqrtc = sqrt.(C0diag)
    trC0  = sum(C0diag)
    Tmat = Diagonal(1.0 ./ u)

    nω = length(ωvals)
    nP = length(Pdirs)

    eω   = fill(NaN, nω)
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
    end

    return (eω=eω, gmat=gmat)
end

# -----------------------------
# Variability via Lyapunov
#   T ẋ = Abar x + ξ,   E[ξ ξᵀ] = C0 δ
# Standard form: ẋ = J x + η,   J = diag(u) Abar,   η = diag(u) ξ
# Q = diag(u) C0 diag(u)
# Solve: JΣ + ΣJᵀ + Q = 0
# V = tr(Σ)/tr(C0)
# -----------------------------
function variability_time_domain(Abar::Matrix{Float64}, u::Vector{Float64}, C0diag::Vector{Float64})
    S = length(u)
    Du = Diagonal(u)
    J  = Du * Abar
    Q  = Du * Diagonal(C0diag) * Du
    α  = spectral_abscissa(J)
    (isfinite(α) && α < 0) || return NaN
    Σ = lyap(Matrix(J), Matrix(Q))
    trC0 = sum(C0diag)
    return (isfinite(tr(Σ)) && trC0 > 0) ? (tr(Σ) / trC0) : NaN
end

# -----------------------------
# Simple structural metrics to organise WHEN/HOW
# -----------------------------
# Strength vector on A (interaction part): s_i = sum_j (|A_ij| + |A_ji|)
function node_strengths(A::Matrix{Float64})
    S = size(A,1)
    s = zeros(Float64, S)
    for i in 1:S
        s[i] = sum(abs.(A[i,:])) + sum(abs.(A[:,i]))
    end
    return s
end

# Alignment index: weighted mean log(T) minus unweighted mean log(T)
# positive => strength sits on slow species (large T) => later effects expected
function alignment_index(A::Matrix{Float64}, u::Vector{Float64})
    T = 1.0 ./ u
    logT = log.(T)
    s = node_strengths(A)
    if !isfinite(sum(s)) || sum(s) <= 0
        return NaN
    end
    w = s ./ sum(s)
    return sum(w .* logT) - mean(logT)
end

# Strength-weighted geometric mean of u: u_str = exp( Σ w_i log u_i )
function strength_weighted_u(A::Matrix{Float64}, u::Vector{Float64})
    s = node_strengths(A)
    if !isfinite(sum(s)) || sum(s) <= 0
        return NaN
    end
    w = s ./ sum(s)
    return exp(sum(w .* log.(u)))
end

# Henrici departure from normality (normalised) for a real matrix M:
# d_H = sqrt(||M||_F^2 - Σ |λ_i|^2) / ||M||_F
function henrici_departure(M::Matrix{Float64})
    nF = norm(M)
    nF == 0 && return 0.0
    λ = eigvals(Matrix(M))
    s2 = sum(abs2.(λ))
    val = max(nF^2 - s2, 0.0)
    return sqrt(val) / nF
end

# -----------------------------
# Base system container + builder
# -----------------------------
struct BaseSystem
    u::Vector{Float64}
    Abar::Matrix{Float64}     # -I + A
    A::Matrix{Float64}        # interaction-only (diag=0), so Abar = -I + A
    C0_mode::Symbol
    C0diag::Vector{Float64}
end

function build_bases(; S::Int=80, base_reps::Int=40, seed::Int=1234,
    # u distribution
    u_mean::Float64=1.0, u_cv::Float64=0.5,
    # O generator
    connectance_rng=(0.03, 0.12),
    trophic_align_rng=(0.55, 0.98),
    reciprocity_rng=(0.00, 0.20),
    σ_rng=(0.3, 1.5),
    # standardise stability
    target_alpha::Float64=-0.05,
    # noise covariance mode in ξ-space
    C0_mode::Symbol=:u2        # :u2 or :I
)
    bases = BaseSystem[]
    for b in 1:base_reps
        rng = MersenneTwister(seed + 10007*b)

        u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))

        c  = rand(rng, Uniform(connectance_rng[1], connectance_rng[2]))
        γ  = rand(rng, Uniform(trophic_align_rng[1], trophic_align_rng[2]))
        rr = rand(rng, Uniform(reciprocity_rng[1], reciprocity_rng[2]))
        σ  = rand(rng, Uniform(σ_rng[1], σ_rng[2]))

        O = trophic_O(S; connectance=c, trophic_align=γ, reciprocity=rr, σ=σ, rng=rng)
        normalize_offdiag!(O) || continue

        s = find_scale_to_target_alpha(O, u; target_alpha=target_alpha)
        isfinite(s) || continue

        A = s * O
        Abar = -Matrix{Float64}(I, S, S) + A

        α = spectral_abscissa(Diagonal(u) * Abar)
        (isfinite(α) && α < 0) || continue

        C0diag = if C0_mode == :u2
            u.^2
        elseif C0_mode == :I
            ones(Float64, S)
        else
            error("Unknown C0_mode. Use :u2 or :I.")
        end

        push!(bases, BaseSystem(u, Abar, A, C0_mode, C0diag))
    end
    return bases
end

# -----------------------------
# Evaluate one system (one base)
#   - sample P dirs (stable at ε_max)
#   - compute spectra for real u and homogeneous u_hom
#   - compute "WHEN" (ω medians) and normalised (ω̃ = ω*T̄)
#   - compute realised magnitude at a chosen ε0:
#        ΔV/V and (ΔV/V)/d
#   - compute predicted per-unit-change slopes (H_real, H_hom) and M_T
# -----------------------------
function eval_system(
    base::BaseSystem,
    ωvals::Vector{Float64};
    # perturbation magnitudes: choose ε0 for realised response
    eps0_rel::Float64 = 0.10,         # ε0 = eps0_rel * ||A||_F
    # choose also ε_max for P stability filtering
    eps_max_rel::Float64 = 0.20,      # accept P if stable at ε_max = eps_max_rel * ||A||_F
    P_reps::Int=14,
    P_sparsity::Float64=1.0,
    margin::Float64=1e-4,
    nprobe::Int=12,
    seed::Int=1
)
    rng = MersenneTwister(seed)
    S = length(base.u)

    # time scale normaliser: mean T
    Tbar = mean(1.0 ./ base.u)
    isfinite(Tbar) || return nothing

    # homogeneous timescales preserving mean T
    u_hom = fill(1.0 / Tbar, S)
    C0_hom = if base.C0_mode == :u2
        u_hom.^2
    else
        ones(Float64, S)
    end

    # build P ensemble, stable at eps_max
    eps_max = eps_max_rel * norm(base.A)   # ||A||_F scales with interaction magnitude
    eps0    = eps0_rel    * norm(base.A)

    Du = Diagonal(base.u)
    Pdirs = Matrix{Float64}[]
    for k in 1:P_reps
        P = sample_Pdir(S; sparsity_p=P_sparsity, rng=rng)
        P === nothing && continue
        Abarp = base.Abar + eps_max * P
        αp = spectral_abscissa(Du * Abarp)
        (isfinite(αp) && αp < -margin) || continue
        push!(Pdirs, P)
    end
    length(Pdirs) < max(6, Int(floor(P_reps/2))) && return nothing

    # baseline V
    V0 = variability_time_domain(base.Abar, base.u, base.C0diag)
    isfinite(V0) || return nothing

    # realised response at eps0
    DV_list = Float64[]
    DVd_list = Float64[]
    DV_signed_list = Float64[]
    DVd_signed_list = Float64[]
    d_list = Float64[]

    for P in Pdirs
        Abarp = base.Abar + eps0 * P
        Vp = variability_time_domain(Abarp, base.u, base.C0diag)
        isfinite(Vp) || continue
        DV = (Vp - V0) / V0
        d  = norm(Abarp - base.Abar) # raw structural displacement in the same space (Abar)
        isfinite(d) && d > 0 || continue

        push!(DV_list, abs(DV))
        push!(DV_signed_list, DV)
        push!(DVd_list, abs(DV)/d)
        push!(DVd_signed_list, DV/d)
        push!(d_list, d)
    end
    isempty(DV_list) && return nothing

    # spectra for real u
    sp = estimate_spectra(base.Abar, base.u, base.C0diag, ωvals, Pdirs; nprobe=nprobe, rng=rng)
    eω = sp.eω
    gmat = sp.gmat

    # spectra for homogeneous u
    spH = estimate_spectra(base.Abar, u_hom, C0_hom, ωvals, Pdirs; nprobe=nprobe, rng=rng)
    eω_h = spH.eω
    gmat_h = spH.gmat

    # build L(ω) and m(ω) using |g|
    nP = size(gmat, 1)
    Lω = fill(NaN, length(ωvals))
    for k in eachindex(ωvals)
        gg = Float64[]
        for p in 1:nP
            v = gmat[p,k]
            isfinite(v) && push!(gg, abs(v))
        end
        Lω[k] = isempty(gg) ? NaN : mean(gg)
    end
    mω = Lω .* eω

    # hom leverage/mass (optional)
    nPh = size(gmat_h, 1)
    Lω_h = fill(NaN, length(ωvals))
    for k in eachindex(ωvals)
        gg = Float64[]
        for p in 1:nPh
            v = gmat_h[p,k]
            isfinite(v) && push!(gg, abs(v))
        end
        Lω_h[k] = isempty(gg) ? NaN : mean(gg)
    end
    mω_h = Lω_h .* eω_h

    # WHEN locations (medians) in ω, then normalise to ω̃ = ω*Tbar
    ω_e50  = quantile_location(ωvals, eω; q=0.5)
    ω_L50  = quantile_location(ωvals, Lω; q=0.5)
    ω_m50  = quantile_location(ωvals, mω; q=0.5)

    ω̃_e50 = isfinite(ω_e50) ? ω_e50 * Tbar : NaN
    ω̃_L50 = isfinite(ω_L50) ? ω_L50 * Tbar : NaN
    ω̃_m50 = isfinite(ω_m50) ? ω_m50 * Tbar : NaN

    t̃_m50 = isfinite(ω̃_m50) && ω̃_m50 > 0 ? 1.0/ω̃_m50 : NaN

    # Predicted per-unit-change slopes:
    # H(P) = ∫ g e / ∫ e
    idxE = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω[i]) && eω[i] ≥ 0, eachindex(eω))
    denom = length(idxE) >= 3 ? trapz(ωvals[idxE], eω[idxE]) : NaN
    H_real = Float64[]
    H_hom  = Float64[]
    if isfinite(denom) && denom > 0
        for p in 1:nP
            gω = vec(gmat[p,:])
            good = findall(i -> i in idxE && isfinite(gω[i]), eachindex(eω))
            length(good) < 3 && continue
            num = trapz(ωvals[good], gω[good] .* eω[good])
            push!(H_real, num/denom)
        end
    end
    # hom
    idxEh = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω_h[i]) && eω_h[i] ≥ 0, eachindex(eω_h))
    denom_h = length(idxEh) >= 3 ? trapz(ωvals[idxEh], eω_h[idxEh]) : NaN
    if isfinite(denom_h) && denom_h > 0
        for p in 1:nPh
            gω = vec(gmat_h[p,:])
            good = findall(i -> i in idxEh && isfinite(gω[i]), eachindex(eω_h))
            length(good) < 3 && continue
            num = trapz(ωvals[good], gω[good] .* eω_h[good])
            push!(H_hom, num/denom_h)
        end
    end

    Habs_real = isempty(H_real) ? NaN : mean(abs.(H_real))
    Habs_hom  = isempty(H_hom)  ? NaN : mean(abs.(H_hom))
    M_T = (isfinite(Habs_real) && isfinite(Habs_hom) && Habs_hom > 0) ? (Habs_real / Habs_hom) : NaN

    # Structural/time-scale organisers
    Aidx = alignment_index(base.A, base.u)
    u_str = strength_weighted_u(base.A, base.u)
    ω̃_pred = (isfinite(u_str) ? (u_str * Tbar) : NaN)  # predicted dimensionless frequency
    # non-normality of interaction-only Jacobian part J_off = diag(u)*A (diag 0)
    Joff = Diagonal(base.u) * base.A
    for i in 1:S
        Joff[i,i] = 0.0
    end
    hen = henrici_departure(Matrix(Joff))

    # Summaries
    return (
        nP = length(Pdirs),
        Tbar = Tbar,

        # realised magnitude
        d_mean = mean(d_list),
        DV_abs_mean = mean(DV_list),
        DV_abs_perd_mean = mean(DVd_list),
        DV_signed_mean = mean(DV_signed_list),
        DV_signed_perd_mean = mean(DVd_signed_list),

        # predicted per-unit-change magnitude
        Habs_real = Habs_real,
        Habs_hom  = Habs_hom,
        M_T = M_T,

        # WHEN (raw and normalised)
        ω_e50 = ω_e50,
        ω_L50 = ω_L50,
        ω_m50 = ω_m50,
        ω̃_e50 = ω̃_e50,
        ω̃_L50 = ω̃_L50,
        ω̃_m50 = ω̃_m50,
        t̃_m50 = t̃_m50,

        # organisers
        Aidx = Aidx,
        ω̃_pred = ω̃_pred,
        hen = hen,

        # keep spectra for optional examples
        eω = eω, Lω = Lω, mω = mω,
        eω_h = eω_h, Lω_h = Lω_h, mω_h = mω_h
    )
end

# -----------------------------
# Run experiment across many bases
# -----------------------------
function run_experiment_normalised(;
    S::Int=120,
    base_reps::Int=60,
    seed::Int=1234,
    ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=70),
    target_alpha::Float64=-0.05,
    C0_mode::Symbol=:u2,
    # perturbation settings
    eps0_rel::Float64=0.10,
    eps_max_rel::Float64=0.20,
    P_reps::Int=14,
    P_sparsity::Float64=1.0,
    margin::Float64=1e-4,
    nprobe::Int=12
)
    ωvals = collect(float.(ωvals))

    bases = build_bases(
        S=S, base_reps=base_reps, seed=seed,
        target_alpha=target_alpha, C0_mode=C0_mode
    )
    @info "Built $(length(bases)) stable bases (attempted $base_reps)."

    outs = Any[]
    example = nothing

    for (i, base) in enumerate(bases)
        out = eval_system(base, ωvals;
            eps0_rel=eps0_rel,
            eps_max_rel=eps_max_rel,
            P_reps=P_reps,
            P_sparsity=P_sparsity,
            margin=margin,
            nprobe=nprobe,
            seed=seed + 900_000*i
        )
        out === nothing && continue
        push!(outs, out)
        if example === nothing
            example = out
        end
    end

    return (ωvals=ωvals, outs=outs, example=example)
end

# -----------------------------
# Optional: epsilon linearity check (per-unit-change should be ~constant for small ε)
# -----------------------------
function epsilon_sweep_check(base::BaseSystem, ωvals;
    eps_rel_list = 10 .^ range(-2.0, -0.5; length=6),
    eps_max_rel::Float64 = 0.25,
    P_reps::Int=14,
    nprobe::Int=12,
    seed::Int=7
)
    rng = MersenneTwister(seed)
    S = length(base.u)
    Tbar = mean(1.0 ./ base.u)
    u_hom = fill(1.0 / Tbar, S)
    C0_hom = (base.C0_mode == :u2) ? u_hom.^2 : ones(Float64,S)

    # accept P at eps_max
    eps_max = eps_max_rel * norm(base.A)
    Du = Diagonal(base.u)
    Pdirs = Matrix{Float64}[]
    for k in 1:P_reps
        P = sample_Pdir(S; rng=rng)
        P === nothing && continue
        Abarp = base.Abar + eps_max * P
        αp = spectral_abscissa(Du * Abarp)
        (isfinite(αp) && αp < -1e-4) || continue
        push!(Pdirs, P)
    end
    length(Pdirs) < max(6, Int(floor(P_reps/2))) && return nothing

    V0 = variability_time_domain(base.Abar, base.u, base.C0diag)
    isfinite(V0) || return nothing

    # predicted per-unit-change slope (Habs_real)
    sp = estimate_spectra(base.Abar, base.u, base.C0diag, ωvals, Pdirs; nprobe=nprobe, rng=rng)
    eω = sp.eω
    gmat = sp.gmat
    idxE = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω[i]) && eω[i] ≥ 0, eachindex(eω))
    denom = length(idxE) >= 3 ? trapz(ωvals[idxE], eω[idxE]) : NaN
    H = Float64[]
    if isfinite(denom) && denom > 0
        for p in 1:size(gmat,1)
            gω = vec(gmat[p,:])
            good = findall(i -> i in idxE && isfinite(gω[i]), eachindex(eω))
            length(good) < 3 && continue
            num = trapz(ωvals[good], gω[good] .* eω[good])
            push!(H, num/denom)
        end
    end
    Habs = isempty(H) ? NaN : mean(abs.(H))

    # realised DV/eps across eps list
    eps_abs = eps_rel_list .* norm(base.A)
    y = Float64[]
    for eps in eps_abs
        dv = Float64[]
        for P in Pdirs
            Abarp = base.Abar + eps * P
            Vp = variability_time_domain(Abarp, base.u, base.C0diag)
            isfinite(Vp) || continue
            DV = abs((Vp - V0)/V0)
            d  = norm(Abarp - base.Abar)
            isfinite(d) && d > 0 || continue
            push!(dv, DV/d)
        end
        push!(y, isempty(dv) ? NaN : mean(dv))
    end

    return (eps=eps_abs, y=y, Habs=Habs)
end

# -----------------------------
# Plotting summary
# -----------------------------
function summarize_and_plot(res; figsize=(1700, 1000))
    outs = res.outs
    n = length(outs)
    n == 0 && error("No outputs to plot.")

    # vectors
    d_mean   = [o.d_mean for o in outs]
    DV_abs   = [o.DV_abs_mean for o in outs]
    DV_perd  = [o.DV_abs_perd_mean for o in outs]

    Habs_real = [o.Habs_real for o in outs]
    Habs_hom  = [o.Habs_hom for o in outs]
    M_T       = [o.M_T for o in outs]

    ω̃_m50 = [o.ω̃_m50 for o in outs]
    t̃_m50 = [o.t̃_m50 for o in outs]
    Aidx  = [o.Aidx for o in outs]
    ω̃_pred = [o.ω̃_pred for o in outs]
    hen = [o.hen for o in outs]

    # masks
    m1 = findall(i -> isfinite(DV_abs[i]) && DV_abs[i] > 0 && isfinite(d_mean[i]) && d_mean[i] > 0, 1:n)
    m2 = findall(i -> isfinite(DV_perd[i]) && DV_perd[i] > 0 && isfinite(Habs_real[i]) && Habs_real[i] > 0, 1:n)
    m3 = findall(i -> isfinite(DV_perd[i]) && DV_perd[i] > 0 && isfinite(Habs_hom[i]) && Habs_hom[i] > 0, 1:n)
    m4 = findall(i -> isfinite(ω̃_m50[i]) && ω̃_m50[i] > 0 && isfinite(Aidx[i]), 1:n)
    m5 = findall(i -> isfinite(ω̃_m50[i]) && ω̃_m50[i] > 0 && isfinite(ω̃_pred[i]) && ω̃_pred[i] > 0, 1:n)
    m6 = findall(i -> isfinite(DV_perd[i]) && DV_perd[i] > 0 && isfinite(hen[i]) && hen[i] > 0, 1:n)

    # correlations (log where appropriate)
    ρB = length(m1) >= 6 ? cor(log.(d_mean[m1]), log.(DV_abs[m1])) : NaN
    ρA = length(m2) >= 6 ? cor(log.(Habs_real[m2]), log.(DV_perd[m2])) : NaN
    ρC = length(m3) >= 6 ? cor(log.(Habs_hom[m3]), log.(DV_perd[m3])) : NaN
    ρW1 = length(m4) >= 6 ? cor(Aidx[m4], log.(ω̃_m50[m4])) : NaN
    ρW2 = length(m5) >= 6 ? cor(log.(ω̃_pred[m5]), log.(ω̃_m50[m5])) : NaN
    ρN  = length(m6) >= 6 ? cor(log.(hen[m6]), log.(DV_perd[m6])) : NaN

    @info "Raw scaling check: corr(log d, log ΔV/V) = $(round(ρB,digits=3))  N=$(length(m1))"
    @info "Per-unit prediction: corr(log H_real, log (ΔV/V)/d) = $(round(ρA,digits=3))  N=$(length(m2))"
    @info "Dyn facet only: corr(log H_hom, log (ΔV/V)/d) = $(round(ρC,digits=3))  N=$(length(m3))"
    @info "WHEN vs alignment: corr(Aidx, log ω̃_m50) = $(round(ρW1,digits=3))  N=$(length(m4))"
    @info "WHEN vs ω̃_pred: corr(log ω̃_pred, log ω̃_m50) = $(round(ρW2,digits=3))  N=$(length(m5))"
    @info "Magnitude per unit vs nonnormality: corr(log hen, log (ΔV/V)/d) = $(round(ρN,digits=3))  N=$(length(m6))"

    fig = Figure(size=figsize)

    ax1 = Axis(fig[1,1];
        xscale=log10, yscale=log10,
        xlabel="d = mean ||A' - A||_F",
        ylabel="mean |ΔV|/V",
        title="A) Raw scaling (before normalising by d)"
    )
    scatter!(ax1, d_mean[m1], DV_abs[m1], markersize=6)
    text!(ax1, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log)=$(round(ρB,digits=3))  N=$(length(m1))")

    ax2 = Axis(fig[1,2];
        xscale=log10, yscale=log10,
        xlabel="H_real = mean_P |∫ g e / ∫ e|",
        ylabel="mean |ΔV|/(V d)",
        title="B) Per-unit-change magnitude: prediction vs realised"
    )
    scatter!(ax2, Habs_real[m2], DV_perd[m2], markersize=6)
    text!(ax2, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log)=$(round(ρA,digits=3))  N=$(length(m2))")

    ax3 = Axis(fig[1,3];
        xscale=log10, yscale=log10,
        xlabel="H_hom (homogeneous timescales)",
        ylabel="mean |ΔV|/(V d)",
        title="C) Dynamical facet alone (timescales removed)"
    )
    scatter!(ax3, Habs_hom[m3], DV_perd[m3], markersize=6)
    text!(ax3, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log)=$(round(ρC,digits=3))  N=$(length(m3))")

    ax4 = Axis(fig[2,1];
        yscale=log10,
        xlabel="alignment index Aidx (strength on slow minus mean)",
        ylabel="ω̃_m50 = ω_m50 * T̄",
        title="D) WHEN (dimensionless) organised by alignment"
    )
    scatter!(ax4, Aidx[m4], ω̃_m50[m4], markersize=6)
    text!(ax4, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(Aidx, log ω̃)=$(round(ρW1,digits=3))  N=$(length(m4))")

    ax5 = Axis(fig[2,2];
        xscale=log10, yscale=log10,
        xlabel="ω̃_pred = (u_str) * T̄",
        ylabel="ω̃_m50",
        title="E) WHEN: strength-weighted timescale predictor"
    )
    scatter!(ax5, ω̃_pred[m5], ω̃_m50[m5], markersize=6)
    text!(ax5, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log)=$(round(ρW2,digits=3))  N=$(length(m5))")

    ax6 = Axis(fig[2,3];
        xscale=log10, yscale=log10,
        xlabel="Henrici departure (J_off)",
        ylabel="mean |ΔV|/(V d)",
        title="F) HOW MUCH per unit change vs non-normality"
    )
    scatter!(ax6, hen[m6], DV_perd[m6], markersize=6)
    text!(ax6, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log)=$(round(ρN,digits=3))  N=$(length(m6))")

    # Example spectra (dimensionless frequency ω̃)
    ex = res.example
    if ex !== nothing
        ω = res.ωvals
        Tbar = ex.Tbar
        ω̃ = ω .* Tbar

        ax7 = Axis(fig[3,1:3];
            xscale=log10, yscale=log10,
            xlabel="ω̃ = ω * T̄",
            ylabel="density (arbitrary scale)",
            title="G) Example spectra on normalised frequency: e(ω), L(ω), m(ω)=e·L (real vs hom)"
        )
        lines!(ax7, ω̃, logsafe(ex.eω), linewidth=3)
        lines!(ax7, ω̃, logsafe(ex.Lω), linewidth=3, linestyle=:dash)
        lines!(ax7, ω̃, logsafe(ex.mω), linewidth=3, linestyle=:dot)

        lines!(ax7, ω̃, logsafe(ex.eω_h), linewidth=2)
        lines!(ax7, ω̃, logsafe(ex.Lω_h), linewidth=2, linestyle=:dash)
        lines!(ax7, ω̃, logsafe(ex.mω_h), linewidth=2, linestyle=:dot)

        # mark ω̃_m50
        if isfinite(ex.ω̃_m50)
            vlines!(ax7, [ex.ω̃_m50], linewidth=2)
        end
    end

    display(fig)
    return nothing
end

# -----------------------------
# MAIN
# -----------------------------
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=70)

res = run_experiment_normalised(
    S=120,
    base_reps=90,
    seed=1234,
    ωvals=ωvals,
    target_alpha=-0.05,
    C0_mode=:u2,

    # magnitude: realised at eps0_rel * ||A||_F
    eps0_rel=0.10,
    # P ensemble stability tested at eps_max_rel * ||A||_F
    eps_max_rel=0.20,

    P_reps=14,
    P_sparsity=1.0,
    margin=1e-4,
    nprobe=12
)

summarize_and_plot(res)

# -----------------------------
# OPTIONAL: epsilon sweep linearity (per-unit-change should flatten at small ε)
# -----------------------------
# Pick one of the base systems by rebuilding one base; simplest: reuse the first
# stable base from the same builder settings (so results are consistent).
bases = build_bases(S=120, base_reps=5, seed=1234, target_alpha=-0.05, C0_mode=:u2)

sweep = epsilon_sweep_check(bases[1], collect(float.(ωvals));
    eps_rel_list = 10 .^ range(-2.0, -0.5; length=6),
    eps_max_rel = 0.25,
    P_reps=14,
    nprobe=12,
    seed=77
)
begin
    fig = Figure(size=(900, 450))
    ax = Axis(fig[1,1];
        xscale=log10, yscale=log10,
        xlabel="ε (absolute, here ε_rel * ||A||_F)",
        ylabel="mean |ΔV|/(V d)",
        title="Linearity check: (ΔV/V)/d vs ε (should be flat when linear)"
    )
    scatter!(ax, sweep.eps, sweep.y, markersize=8)
    if isfinite(sweep.Habs)
        lines!(ax, sweep.eps, fill(sweep.Habs, length(sweep.eps)), linewidth=2, linestyle=:dash)
    end
    display(fig)
end
################################################################################