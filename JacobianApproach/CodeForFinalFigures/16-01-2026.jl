################################################################################
# FREQUENCY-RESOLVED VARIABILITY + STRUCTURAL FRAGILITY PIPELINE
#
# Key changes vs your previous rmed-centric pipelines:
#
# (1) Work with stochastic-variability stability:
#       Dynamics (generalized):   T ẋ = Abar x + ξ(t)
#       with white noise ξ,  E[ξ(t) ξ(t')ᵀ] = C0 δ(t-t')
#
#     Frequency response:         x̂(ω) = R(ω) ξ̂(ω)
#       R(ω) = (i ω T - Abar)^(-1)
#
#     Spectral energy density:    e(ω) = tr( R C0 R† ) / tr(C0)
#     Normalized variability:     V = (1/2π) ∫ e(ω) dω      (≈ tr(Σ)/tr(C0))
#
# (2) Structural fragility (first-order, per frequency):
#     Abar → Abar + ε P,   with ||P||_F = 1, diag(P)=0
#     Fractional sensitivity of energy at ω:
#       g(ω;P) ≈ 2 Re tr( R(ω) P Ĉ(ω) ) / tr(Ĉ(ω)),
#       Ĉ(ω) = R C0 R†.
#
#     We compute:
#       - Gabs(ω) = mean_P |g(ω;P)|
#       - predicted fractional change in V:
#           δV/V ≈ ε * | ∫ g(ω;P) e(ω) dω  / ∫ e(ω) dω |
#       - actual fractional change via Lyapunov on perturbed system.
#
# (3) Cutoffs without arbitrary thresholds:
#     (a) Baseline (no-interaction) cutoff ωc0 from self-filter:
#         Abar0 = -I  ⇒  R0(ω) = (I + i ω T)^(-1)
#         e0(ω) = tr(R0 C0 R0†)/tr(C0), with e0(0)=1
#         ωc0 is defined by e0(ωc0)=1/2.
#
#     (b) Energetic window [ωL, ωH] from the interacting system:
#         choose quantiles qL,qH of cumulative ∫ e(ω)dω
#         so that [ωL,ωH] contains (qH-qL) of the variability “mass”.
#
#     Then evaluate “interaction availability” at these ω:
#         A = Abar + I
#         Dω = diag( 1/(1+i ω T_i) )
#         Aω = A * Dω
#         report ρ(Aω) and ||Aω||_2 at ωc0, ωL, ωH
#
# Notes:
# - Uses Hutchinson-style probe vectors to estimate traces cheaply.
# - Keeps everything as an independent unit; only dependencies below.
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# -----------------------------
# Utilities
# -----------------------------
meanfinite(v)   = (x = filter(isfinite, v); isempty(x) ? NaN : mean(x))
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

# Complex spectral radius ρ(M) = max |eig|
function spectral_radius_complex(M::AbstractMatrix{ComplexF64})
    vals = eigvals(Matrix(M))
    return maximum(abs.(vals))
end

# -----------------------------
# Biomass / time-scale vector u
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
# Base: Abar = -I + s*O  (diag -1)
# J = diag(u) * Abar
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
# Noise/structure directions P (diag=0, ||P||_F=1)
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
# Core resolvent objects
# R(ω) = (i ω T - Abar)^(-1), T=diag(1/u)
# Using Hutchinson probes:
#   tr(R C0 R†) = E_v || R sqrt(C0) v ||^2
#   tr(R P Ĉ)   = E_v ( (R sqrt(C0) v)† (R P R sqrt(C0) v) )
# -----------------------------
"""
Estimate (for one ω, fixed LU of Mω):
  e(ω) = tr(R C0 R†)/tr(C0)
and for each P in Pdirs:
  g(ω;P) ≈ 2 Re tr(R P Ĉ)/tr(Ĉ)
Returns:
  eω (Float64),
  gvec (Vector{Float64}, length = nP)  [fractional sensitivity at ω]
"""
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

    # probe vectors (Rademacher)
    probes = Vector{Vector{Float64}}(undef, nprobe)
    for k in 1:nprobe
        v = rand(rng, (-1.0, 1.0), S)
        probes[k] = v
    end

    # Precompute x_k = R sqrt(C0) v_k  for each probe
    x_list = Vector{Vector{ComplexF64}}(undef, nprobe)
    xnorm2 = zeros(Float64, nprobe)

    for k in 1:nprobe
        rhs = ComplexF64.(sqrtc .* probes[k])  # sqrt(C0)*v
        x = F \ rhs                            # x = R rhs
        x_list[k] = x
        xnorm2[k] = real(dot(conj.(x), x))
    end

    # trace(Ĉ) estimator
    trChat_est = mean(xnorm2)                  # ≈ tr(R C0 R†)
    eω = (isfinite(trChat_est) && trC0 > 0) ? (trChat_est / trC0) : NaN

    # For each P: estimate tr(R P Ĉ) via probes:
    # inner_k = x_k† (R P x_k)
    gvec = fill(NaN, nP)

    for (pidx, P) in enumerate(Pdirs)
        inners = zeros(Float64, nprobe)

        for k in 1:nprobe
            x = x_list[k]
            y = ComplexF64.(P) * x           # y = P x
            z = F \ y                        # z = R P x
            inner = dot(conj.(x), z)         # x† (R P x)
            inners[k] = real(inner)
        end

        num_est = mean(inners)               # ≈ Re tr(R P Ĉ)
        if isfinite(num_est) && isfinite(trChat_est) && trChat_est > 0
            gvec[pidx] = 2.0 * (num_est / trChat_est)
        else
            gvec[pidx] = NaN
        end
    end

    return eω, gvec
end

"""
Compute e(ω) across ωvals for a base system (Abar,u,C0).
Also returns:
  - Gabs(ω) = mean_P |g(ω;P)|
  - gmat[P, ω] storing g(ω;P) (used to predict δV/V)
"""
function estimate_energy_and_fragility_spectra(
    Abar::Matrix{Float64}, u::Vector{Float64}, C0diag::Vector{Float64},
    ωvals::Vector{Float64}, Pdirs::Vector{Matrix{Float64}};
    nprobe::Int=12,
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
# Variability in time domain via Lyapunov (exact, given dynamics)
# T ẋ = Abar x + ξ ,  E[ξ ξᵀ]=C0 δ
# Equivalent standard form: ẋ = J x + η,   J = diag(u)Abar,   η = diag(u)ξ
# So Q = E[η ηᵀ] = diag(u) C0 diag(u)
# Solve JΣ + ΣJᵀ + Q = 0
# Variability V = tr(Σ)/tr(C0)
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
# Baseline cutoff ωc0 from no-interaction filter:
# Abar0=-I ⇒ R0(ω)=(I+iωT)^(-1)
# e0(ω)=Σ c_i/(1+(ωT_i)^2) / Σ c_i
# Find ω where e0 crosses 1/2
# -----------------------------
function baseline_energy_curve(u::Vector{Float64}, C0diag::Vector{Float64}, ωvals::Vector{Float64})
    T = 1.0 ./ u
    trC0 = sum(C0diag)
    e0 = Vector{Float64}(undef, length(ωvals))
    for (k, ω0) in enumerate(ωvals)
        ω = float(ω0)
        denom = @. 1.0 + (ω*T)^2
        e0[k] = sum(C0diag ./ denom) / trC0
    end
    return e0
end

function cutoff_from_curve_halfpower(ωvals::Vector{Float64}, y::Vector{Float64}; target::Float64=0.5)
    idx = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(y[i]), eachindex(y))
    length(idx) < 3 && return NaN
    ω = ωvals[idx]
    v = y[idx]

    # assume v decreases with ω (true for baseline); if not, still pick first crossing
    for j in 2:length(ω)
        if v[j-1] > target && v[j] <= target
            x1, x2 = log(ω[j-1]), log(ω[j])
            y1, y2 = v[j-1], v[j]
            if y2 == y1
                return ω[j]
            end
            x = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
            return exp(x)
        end
    end
    return NaN
end

# -----------------------------
# Energetic window [ωL, ωH] from cumulative ∫ e(ω) dω
# -----------------------------
function energy_quantile_band(ωvals::Vector{Float64}, eω::Vector{Float64}; qL::Float64=0.05, qH::Float64=0.95)
    @assert 0.0 < qL < qH < 1.0
    idx = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω[i]) && eω[i] >= 0, eachindex(eω))
    length(idx) < 3 && return (ωL=NaN, ωH=NaN)

    ω = ωvals[idx]
    e = eω[idx]

    total = trapz(ω, e)
    (isfinite(total) && total > 0) || return (ωL=NaN, ωH=NaN)

    # cumulative
    cum = zeros(Float64, length(ω))
    for i in 2:length(ω)
        cum[i] = cum[i-1] + 0.5*(e[i-1]+e[i])*(ω[i]-ω[i-1])
    end

    function interp_at(frac)
        target = frac * total
        j = findfirst(cum .>= target)
        isnothing(j) && return NaN
        j == 1 && return ω[1]
        ω1, ω2 = ω[j-1], ω[j]
        c1, c2 = cum[j-1], cum[j]
        if c2 == c1
            return ω2
        else
            return ω1 + (target - c1) * (ω2 - ω1) / (c2 - c1)
        end
    end

    return (ωL=interp_at(qL), ωH=interp_at(qH))
end

# -----------------------------
# Interaction availability metrics at a given ω:
# A = Abar + I
# Dω = diag(1/(1+i ω T_i)),  T_i=1/u_i
# Aω = A Dω
# Report ρ(Aω), ||Aω||_2
# -----------------------------
function Aomega_metrics(Abar::Matrix{Float64}, u::Vector{Float64}, ω::Float64)
    S = size(Abar,1)
    (isfinite(ω) && ω > 0) || return (rho=NaN, opn=NaN)

    A = Abar + Matrix{Float64}(I, S, S)
    T = 1.0 ./ u
    Dω = Diagonal(ComplexF64.(1.0 ./ (1.0 .+ im*ω*T)))
    Aω = Matrix{ComplexF64}(A) * Dω

    rho = spectral_radius_complex(Aω)
    opn = opnorm(Aω)
    return (rho=rho, opn=opn)
end

# -----------------------------
# Base system container and builder
# -----------------------------
struct BaseSys2
    u::Vector{Float64}
    Abar::Matrix{Float64}           # -I + A
    eps_struct::Float64             # perturbation magnitude on Abar
    C0diag::Vector{Float64}         # noise covariance in ξ-space (diagonal)
    K0::Float64                     # static collectivity proxy: ρ(A)
end

function build_bases(; S::Int=80, base_reps::Int=40, seed::Int=1234,
    # u distribution
    u_mean::Float64=1.0, u_cv::Float64=0.5,
    # trophic generator heterogeneity
    connectance_rng=(0.03, 0.12),
    trophic_align_rng=(0.55, 0.98),
    reciprocity_rng=(0.00, 0.20),
    σ_rng=(0.3, 1.5),
    # stability standardization
    target_alpha::Float64=-0.05,
    # structural uncertainty amplitude
    eps_rel::Float64=0.15,
    # C0 choice
    C0_mode::Symbol=:u2       # :u2 or :I
)
    bases = BaseSys2[]
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
        J = Diagonal(u) * Abar
        α = spectral_abscissa(J)
        (isfinite(α) && α < 0) || continue

        # structural noise magnitude relative to interaction scale ||A||
        eps_struct = eps_rel * norm(A)
        (isfinite(eps_struct) && eps_struct > 0) || continue

        # C0 diagonal
        C0diag = if C0_mode == :u2
            u.^2
        elseif C0_mode == :I
            ones(Float64, S)
        else
            error("Unknown C0_mode. Use :u2 or :I.")
        end

        # static collectivity K0 = ρ(A)
        K0 = maximum(abs.(eigvals(Matrix(A))))
        isfinite(K0) || continue

        push!(bases, BaseSys2(u, Abar, eps_struct, C0diag, K0))
    end
    return bases
end

# -----------------------------
# Evaluate one base:
# - Build stable P ensemble
# - Compute variability: time (Lyapunov) and frequency (integration)
# - Compute structural sensitivity: actual ΔV/V from perturbed Lyapunov
#                                and predicted δV/V from g(ω;P)
# - Compute ωc0 (baseline half-power) and energetic band [ωL, ωH]
# - Compute interaction metrics at ωc0, ωL, ωH
# -----------------------------
function eval_base(base::BaseSys2, ωvals::Vector{Float64};
    P_reps::Int=12,
    P_sparsity::Float64=1.0,
    margin::Float64=1e-4,
    nprobe::Int=12,
    qL::Float64=0.05,
    qH::Float64=0.95,
    seed::Int=1
)
    rng = MersenneTwister(seed)
    S = length(base.u)
    Du = Diagonal(base.u)

    # Sample P directions and keep those preserving stability
    Pdirs = Matrix{Float64}[]
    for k in 1:P_reps
        P = sample_noise_Pdir(S; sparsity_p=P_sparsity, rng=rng)
        P === nothing && continue
        Abarp = base.Abar + base.eps_struct * P
        Jp = Du * Abarp
        αp = spectral_abscissa(Jp)
        (isfinite(αp) && αp < -margin) || continue
        push!(Pdirs, P)
    end
    length(Pdirs) < max(6, Int(floor(P_reps/2))) && return nothing

    # Variability in time domain for base
    V_time = variability_time_domain(base.Abar, base.u, base.C0diag)
    isfinite(V_time) || return nothing

    # Estimate e(ω) and g(ω;P)
    sp = estimate_energy_and_fragility_spectra(base.Abar, base.u, base.C0diag, ωvals, Pdirs;
                                               nprobe=nprobe, rng=rng)
    eω = sp.eω
    gmat = sp.gmat

    # Frequency-domain variability (normalized): V = (1/2π) ∫ e(ω) dω
    idxE = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω[i]) && eω[i] >= 0, eachindex(eω))
    length(idxE) < 3 && return nothing
    V_freq = (1.0/(2π)) * trapz(ωvals[idxE], eω[idxE])

    # Baseline cutoff ωc0 from no-interaction curve
    e0 = baseline_energy_curve(base.u, base.C0diag, ωvals)
    ωc0 = cutoff_from_curve_halfpower(ωvals, e0; target=0.5)
    tc0 = (isfinite(ωc0) && ωc0 > 0) ? (1.0/ωc0) : NaN

    # Energetic window [ωL, ωH] from interacting e(ω)
    band = energy_quantile_band(ωvals, eω; qL=qL, qH=qH)
    ωL, ωH = band.ωL, band.ωH

    # Interaction availability metrics at ωc0, ωL, ωH
    m_c0 = Aomega_metrics(base.Abar, base.u, ωL)
    m_L  = Aomega_metrics(base.Abar, base.u, ωL)
    m_H  = Aomega_metrics(base.Abar, base.u, ωH)

    # Actual ΔV/V from perturbed systems (Lyapunov)
    DV_list = Float64[]
    for P in Pdirs
        Abarp = base.Abar + base.eps_struct * P
        Vp = variability_time_domain(Abarp, base.u, base.C0diag)
        isfinite(Vp) || continue
        push!(DV_list, abs(Vp - V_time) / V_time)
    end
    DV_actual = isempty(DV_list) ? NaN : mean(DV_list)

    # Predicted δV/V from g(ω;P):
    # δV/V ≈ ε * | ∫ g(ω;P) e(ω) dω / ∫ e(ω) dω |
    denom = trapz(ωvals[idxE], eω[idxE])
    DV_pred_list = Float64[]
    if isfinite(denom) && denom > 0
        for p in 1:size(gmat,1)
            gω = vec(gmat[p, :])
            good = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω[i]) && eω[i] >= 0 && isfinite(gω[i]), eachindex(eω))
            length(good) < 3 && continue
            num = trapz(ωvals[good], (gω[good] .* eω[good]))
            DVp = base.eps_struct * abs(num / denom)
            isfinite(DVp) && push!(DV_pred_list, DVp)
        end
    end
    DV_pred = isempty(DV_pred_list) ? NaN : mean(DV_pred_list)

    # Summaries of frequency-local fragility
    Gabs = sp.Gabs
    Gabs_band = begin
        idxB = findall(i -> isfinite(ωvals[i]) && isfinite(Gabs[i]) && ωvals[i] >= ωL && ωvals[i] <= ωH, eachindex(Gabs))
        isempty(idxB) ? NaN : mean(Gabs[idxB])
    end

    return (
        nP=length(Pdirs),
        K0=base.K0,

        V_time=V_time,
        V_freq=V_freq,

        DV_actual=DV_actual,
        DV_pred=DV_pred,

        ωc0=ωc0, tc0=tc0,
        ωL=ωL, ωH=ωH,

        rho_c0=m_c0.rho, opn_c0=m_c0.opn,
        rho_L=m_L.rho,   opn_L=m_L.opn,
        rho_H=m_H.rho,   opn_H=m_H.opn,

        Gabs_band=Gabs_band,

        # keep spectra for one example if you want
        eω=eω, Gabs=Gabs, e0=e0
    )
end

# -----------------------------
# Run experiment across many bases
# -----------------------------
function run_experiment(; S::Int=80, base_reps::Int=40, P_reps::Int=12,
    seed::Int=1234,
    ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=70),
    target_alpha::Float64=-0.05,
    eps_rel::Float64=0.15,
    margin::Float64=1e-4,
    nprobe::Int=12,
    qL::Float64=0.05,
    qH::Float64=0.95,
    C0_mode::Symbol=:u2
)
    ωvals = collect(float.(ωvals))

    bases = build_bases(
        S=S, base_reps=base_reps, seed=seed,
        target_alpha=target_alpha,
        eps_rel=eps_rel,
        C0_mode=C0_mode
    )
    @info "Built $(length(bases)) stable bases (attempted $base_reps)."

    # outputs
    K0       = Float64[]
    V_time   = Float64[]
    V_freq   = Float64[]
    DV_act   = Float64[]
    DV_pred  = Float64[]
    ωc0      = Float64[]
    tc0      = Float64[]
    ωL       = Float64[]
    ωH       = Float64[]
    rho_c0   = Float64[]
    rho_L    = Float64[]
    rho_H    = Float64[]
    opn_c0   = Float64[]
    Gband    = Float64[]
    nPacc    = Int[]

    example = nothing

    for (i, base) in enumerate(bases)
        out = eval_base(base, ωvals;
            P_reps=P_reps,
            margin=margin,
            nprobe=nprobe,
            qL=qL, qH=qH,
            seed=seed + 900_000*i
        )
        out === nothing && continue

        push!(K0, out.K0)
        push!(V_time, out.V_time)
        push!(V_freq, out.V_freq)
        push!(DV_act, out.DV_actual)
        push!(DV_pred, out.DV_pred)
        push!(ωc0, out.ωc0)
        push!(tc0, out.tc0)
        push!(ωL, out.ωL)
        push!(ωH, out.ωH)
        push!(rho_c0, out.rho_c0)
        push!(rho_L, out.rho_L)
        push!(rho_H, out.rho_H)
        push!(opn_c0, out.opn_c0)
        push!(Gband, out.Gabs_band)
        push!(nPacc, out.nP)

        if example === nothing
            example = (ωvals=ωvals, out=out)
        end
    end

    return (
        ωvals=ωvals,
        K0=K0, V_time=V_time, V_freq=V_freq,
        DV_act=DV_act, DV_pred=DV_pred,
        ωc0=ωc0, tc0=tc0, ωL=ωL, ωH=ωH,
        rho_c0=rho_c0, rho_L=rho_L, rho_H=rho_H,
        opn_c0=opn_c0, Gband=Gband,
        nPacc=nPacc,
        example=example
    )
end

# -----------------------------
# Plotting summary
# -----------------------------
function logsafe(y; eps=1e-12)
    z = similar(y)
    for i in eachindex(y)
        z[i] = (isfinite(y[i]) && y[i] > eps) ? y[i] : NaN
    end
    return z
end

function summarize_and_plot(res; figsize=(1700, 1100))
    K0     = res.K0
    Vt     = res.V_time
    Vf     = res.V_freq
    DVa    = res.DV_act
    DVp    = res.DV_pred
    ωc0    = res.ωc0
    ωL     = res.ωL
    ωH     = res.ωH
    rho_c0 = res.rho_c0
    opn_c0 = res.opn_c0
    Gband  = res.Gband

    # masks
    mV  = findall(i -> isfinite(Vt[i]) && Vt[i] > 0 && isfinite(Vf[i]) && Vf[i] > 0, eachindex(Vt))
    mDV = findall(i -> isfinite(DVa[i]) && DVa[i] > 0 && isfinite(DVp[i]) && DVp[i] > 0, eachindex(DVa))
    mW  = findall(i -> isfinite(ωc0[i]) && ωc0[i] > 0 && isfinite(ωL[i]) && ωL[i] > 0, eachindex(ωc0))
    mR  = findall(i -> isfinite(DVa[i]) && DVa[i] > 0 && isfinite(rho_c0[i]) && rho_c0[i] > 0, eachindex(DVa))
    mG  = findall(i -> isfinite(DVa[i]) && DVa[i] > 0 && isfinite(Gband[i]) && Gband[i] > 0, eachindex(DVa))
    mO  = findall(i -> isfinite(DVa[i]) && DVa[i] > 0 && isfinite(opn_c0[i]) && opn_c0[i] > 0, eachindex(DVa))
    mK  = findall(i -> isfinite(K0[i]) && K0[i] > 0 && isfinite(rho_c0[i]) && rho_c0[i] > 0, eachindex(K0))

    ρV  = (length(mV)  >= 6) ? cor(log.(Vt[mV]), log.(Vf[mV])) : NaN
    ρDV = (length(mDV) >= 6) ? cor(log.(DVa[mDV]), log.(DVp[mDV])) : NaN
    ρW  = (length(mW)  >= 6) ? cor(log.(ωc0[mW]), log.(ωL[mW])) : NaN
    ρR  = (length(mR)  >= 6) ? cor(log.(DVa[mR]), log.(rho_c0[mR])) : NaN
    ρG  = (length(mG)  >= 6) ? cor(log.(DVa[mG]), log.(Gband[mG])) : NaN
    ρO  = (length(mO)  >= 6) ? cor(log.(DVa[mO]), log.(opn_c0[mO])) : NaN
    ρK  = (length(mK)  >= 6) ? cor(log.(K0[mK]), log.(rho_c0[mK])) : NaN

    @info "Check: V_time vs V_freq:  cor(log,log) = $ρV   N=$(length(mV))"
    @info "Test:  ΔV/V actual vs predicted: cor(log,log) = $ρDV  N=$(length(mDV))"
    @info "Band:  ωc0 (baseline half-power) vs ωL (energy qL): cor(log,log) = $ρW  N=$(length(mW))"
    @info "Link:  ΔV/V vs ρ(A_ωc0): cor(log,log) = $ρR  N=$(length(mR))"
    @info "Link:  ΔV/V vs mean|g| in band: cor(log,log) = $ρG  N=$(length(mG))"
    @info "Link:  ΔV/V vs ||A_ωc0||: cor(log,log) = $ρO  N=$(length(mO))"
    @info "Link:  K0 vs ρ(A_ωc0): cor(log,log) = $ρK  N=$(length(mK))"

    fig = Figure(size=figsize)

    ax1 = Axis(fig[1,1];
        xscale=log10, yscale=log10,
        xlabel="V_freq = (1/2π) ∫ e(ω) dω",
        ylabel="V_time = tr(Σ)/tr(C0)",
        title="A) Variability: frequency integral matches Lyapunov"
    )
    scatter!(ax1, Vf[mV], Vt[mV], markersize=7)
    text!(ax1, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log,log) = $(round(ρV,digits=3))   N=$(length(mV))")

    ax2 = Axis(fig[1,2];
        xscale=log10, yscale=log10,
        xlabel="Predicted ΔV/V  (first-order, resolvent-weighted)",
        ylabel="Actual ΔV/V  (Lyapunov on perturbed systems)",
        title="B) Structural fragility: prediction vs reality"
    )
    scatter!(ax2, DVp[mDV], DVa[mDV], markersize=7)
    text!(ax2, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log,log) = $(round(ρDV,digits=3))   N=$(length(mDV))")

    ax3 = Axis(fig[1,3];
        xscale=log10, yscale=log10,
        xlabel="ωc0 (baseline half-power cutoff)",
        ylabel="ωL (lower energy quantile cutoff)",
        title="C) Non-arbitrary cutoffs: baseline vs energetic band"
    )
    scatter!(ax3, ωc0[mW], ωL[mW], markersize=7)
    text!(ax3, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log,log) = $(round(ρW,digits=3))   N=$(length(mW))")

    ax4 = Axis(fig[2,1];
        xscale=log10, yscale=log10,
        xlabel="ρ(A_ωc0)",
        ylabel="Actual ΔV/V",
        title="D) Indirect-availability proxy vs structural impact"
    )
    scatter!(ax4, rho_c0[mR], DVa[mR], markersize=7)
    text!(ax4, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log,log) = $(round(ρR,digits=3))   N=$(length(mR))")

    ax5 = Axis(fig[2,2];
        xscale=log10, yscale=log10,
        xlabel="mean_ω∈[ωL,ωH] mean_P |g(ω;P)|",
        ylabel="Actual ΔV/V",
        title="E) Frequency-local fragility vs realized fragility"
    )
    scatter!(ax5, Gband[mG], DVa[mG], markersize=7)
    text!(ax5, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log,log) = $(round(ρG,digits=3))   N=$(length(mG))")

    ax6 = Axis(fig[2,3];
        xscale=log10, yscale=log10,
        xlabel="||A_ωL||_2",
        ylabel="Actual ΔV/V",
        title="F) Direct-strength proxy vs structural impact"
    )
    scatter!(ax6, opn_c0[mO], DVa[mO], markersize=7)
    text!(ax6, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log,log) = $(round(ρO,digits=3))   N=$(length(mO))")

    # Example spectra
    ex = res.example
    if ex !== nothing
        ωvals = ex.ωvals
        out = ex.out
        ax7 = Axis(fig[3,1:3];
            xscale=log10, yscale=log10,
            xlabel="ω",
            ylabel="density",
            title="G) Example: energy density e(ω) and fragility spectrum mean|g(ω)|"
        )
        lines!(ax7, ωvals, logsafe(out.eω), linewidth=3)
        lines!(ax7, ωvals, logsafe(out.Gabs), linewidth=3, linestyle=:dash)
        lines!(ax7, ωvals, logsafe(out.e0), linewidth=2, linestyle=:dot)

        # mark ωc0, ωL, ωH if finite
        vlines!(ax7, [out.ωc0, out.ωL, out.ωH], linewidth=2)
    end

    display(fig)
    return nothing
end

# -----------------------------
# MAIN
# -----------------------------
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=70)

res = run_experiment(
    S=120,
    base_reps=80,
    P_reps=20,
    seed=1234,
    ωvals=ωvals,
    target_alpha=-0.05,
    eps_rel=0.2,
    margin=1e-4,
    nprobe=12,
    qL=0.05, qH=0.95,
    C0_mode=:u2    # :u2 or :I
)

summarize_and_plot(res)
################################################################################