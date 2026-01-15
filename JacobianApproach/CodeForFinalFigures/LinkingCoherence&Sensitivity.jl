################################################################################
# COHERENCE → COLLECTIVITY → CUTOFF TIME (t_c) → SENSITIVITY ORGANISATION (t_q)
#
# Goal (single, independent script):
#   Build many PPM communities across a trophic incoherence gradient (q),
#   keeping everything else as constant as possible, then show:
#
#   (A) Coherence (q) organises static collectivity K0 = ρ(A) and dynamic cutoff ωc
#       defined by the first crossing K(ω)=1, where K(ω)=ρ(Aω) and Aω = A*Dω.
#
#   (B) Sensitivity in frequency domain predicts time-domain divergence (relevant window).
#
#   (C) Sensitivity is "organised" by a cutoff time tq derived from the cumulative
#       sensitivity above ω95=1/t95:
#           tq = 1/ωq  where ∫_{ω95}^{ωq} S(ω)dω = qfrac * ∫_{ω95}^{∞} S(ω)dω
#       and we use the normalised τq = tq/t95.
#
#   (D) Link back to collectivity: τc = tc/t95 with tc=1/ωc should covary with τq.
#
# Key modelling choices:
#   - Dynamics: J = diag(u) * Abar,   Abar = -I + A,  A is off-diagonal interactions.
#   - PPM provides adjacency; we build a signed interaction matrix W (prey→predator).
#   - We Frobenius-normalise off-diagonal structure and then choose a scale s so that
#     α(J) ≈ target_alpha (<0), to keep resilience comparable across systems.
#   - rmed(t) is biomass-weighted with C = diag(u^2).
#   - Structural uncertainty: Abar → Abar + eps*P,  with iid Gaussian off-diagonal P,
#     ||P||_F=1. We keep only perturbations that remain stable.
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie
using Base.Threads

# -----------------------------
# Utilities
# -----------------------------
meanfinite(v)  = (x = filter(isfinite, v); isempty(x) ? NaN : mean(x))
medianfinite(v)= (x = filter(isfinite, v); isempty(x) ? NaN : median(x))

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

# integrate y(x) d log x = ∫ y(x) (dx/x)
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

# -----------------------------
# Biomass / time-scale weights u
# -----------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# -----------------------------
# PPM: trophic levels & incoherence
# -----------------------------
function trophic_levels(A::Matrix{Int})
    S = size(A,1)
    kin = sum(A, dims=2)
    v = [max(kin[i],1) for i in 1:S]
    Λ = Diagonal(v) - A
    s = Λ \ v
    return s
end

function trophic_incoherence(A::Matrix{Int}, s)
    xs = Float64[]
    S = length(s)
    for i in 1:S, j in 1:S
        if A[i,j] == 1
            push!(xs, s[i] - s[j])
        end
    end
    return sqrt(mean(xs.^2) - 1)
end

# Preferential Preying Model
function ppm(S::Int, B::Int, L::Int, Tppm::Float64; rng=Random.default_rng())
    A = zeros(Int, S, S)

    β = (S^2 - B^2) / (2L - 1)
    β <= 0 && error("PPM: invalid β. Increase L.")
    beta_dist = Beta(β, β)

    prey_count = zeros(Int, S)
    current = B

    while current < S
        current += 1
        i = current

        existing = 1:(i-1)
        n_i = length(existing)

        # 1) First prey
        j = rand(rng, existing)
        A[i,j] = 1
        prey_count[i] += 1

        # provisional TLs (cheap proxy during construction)
        s_hat = 1 .+ prey_count[1:i]

        # 2) expected prey count
        k_exp = rand(rng, beta_dist) * n_i
        k_total = max(1, round(Int, k_exp))
        k_extra = k_total - 1

        if k_extra > 0
            probs = [exp(-abs(s_hat[j] - s_hat[ℓ]) / Tppm) for ℓ in existing]
            probs ./= sum(probs)
            chosen = rand(rng, Distributions.Categorical(probs), k_extra)
            for idx in unique(chosen)
                prey = existing[idx]
                A[i, prey] = 1
                prey_count[i] += 1
            end
        end
    end

    s = trophic_levels(A)
    q = trophic_incoherence(A, s)
    return (A=A, s=s, q=q)
end

W = ppm(10, 3, Int(round(1.0*10^2)), 0.01).A
# -----------------------------
# From PPM adjacency to signed interaction matrix W
# -----------------------------
function correlated_magnitudes(mag_abs, mag_cv, ρ; rng=Random.default_rng())
    σ = mag_abs * mag_cv
    Σ = [σ^2  ρ*σ^2;
         ρ*σ^2  σ^2]
    d = MvNormal([mag_abs, mag_abs], Σ)
    mag = rand(rng, d)
    return abs.(mag)
end

function build_interaction_matrix(A::Matrix{Int};
        mag_abs=1.0,
        mag_cv=0.5,
        corr_aij_aji=0.99,
        rng=Random.default_rng())

    S = size(A,1)
    W = zeros(Float64, S, S)

    # Convention in your code: A[prey,pred]=1 (prey -> predator)
    for prey in 1:S
        for pred in findall(A[prey, :] .== 1)
            m_preypred, m_predprey = correlated_magnitudes(mag_abs, mag_cv, corr_aij_aji; rng=rng)
            W[prey, pred] = +m_preypred   # prey effect on predator
            W[pred, prey] = -m_predprey   # predator effect on prey
        end
    end
    return W
end

# -----------------------------
# Normalise offdiag structure and standardise stability by scaling
# Abar = -I + s*O   where O has diag 0 and ||O||_F=1
# J = diag(u)*Abar
# Choose s so that α(J) ≈ target_alpha (<0).
# -----------------------------
function offdiag_zero!(M::Matrix{Float64})
    for i in 1:size(M,1)
        M[i,i] = 0.0
    end
    return M
end

function normalize_offdiag!(O::Matrix{Float64})
    n = norm(O)
    n == 0 && return false
    O ./= n
    return true
end

function find_scale_to_target_alpha(
    O::Matrix{Float64}, u::Vector{Float64};
    target_alpha::Float64=-0.05,
    max_grow::Int=40,
    max_iter::Int=60
)
    @assert target_alpha < 0
    Du = Diagonal(u)

    α0 = spectral_abscissa(-Du)     # s=0 => Abar=-I
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

# Your implicit-style t95-from-rmed curve (using exp(-rmed*t))
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

# Time-domain error in normalised time τ=t/t95, relevant window τ<=1
function time_errors_normalised(tvals::Vector{Float64}, rbase::Vector{Float64}, rpert::Vector{Float64})
    t95 = t95_from_rmed_curve(tvals, rbase; target=0.05)
    (isfinite(t95) && t95 > 0) || return nothing

    τ = tvals ./ t95
    Δ = delta_curve(rbase, rpert)

    good_all = findall(i -> isfinite(τ[i]) && τ[i] > 0 && isfinite(Δ[i]), eachindex(τ))
    length(good_all) < 2 && return nothing
    Err_tot = trapz_logx(τ[good_all], Δ[good_all])

    good_rel = findall(i -> isfinite(τ[i]) && τ[i] > 0 && τ[i] <= 1.0 && isfinite(Δ[i]), eachindex(τ))
    Err_rel = (length(good_rel) >= 2) ? trapz_logx(τ[good_rel], Δ[good_rel]) : NaN

    return (t95=t95, τ=τ, Δ=Δ, Err_tot=Err_tot, Err_rel=Err_rel)
end

# -----------------------------
# Uncertainty directions P (noise), ||P||_F=1, diag(P)=0
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
# Frequency domain: typical sensitivity spectrum
# R(ω) = (i ω T - Abar)^(-1),  T=diag(1/u)
# S(ω;P) = eps^2 * || R P R diag(u) ||_F^2 / sum(u^2)
# -----------------------------
function sensitivity_spectrum_typical(Abar::Matrix{Float64}, u::Vector{Float64},
    eps::Float64, ωvals::Vector{Float64}, Pdirs::Vector{Matrix{Float64}}
)
    Tmat = Diagonal(1.0 ./ u)
    U = Matrix{ComplexF64}(Diagonal(u))
    denom = sum(u.^2)

    nω = length(ωvals)
    Smean = fill(NaN, nω)

    for (k, ω0) in enumerate(ωvals)
        ω = float(ω0)
        Mω = Matrix{ComplexF64}(im*ω*Tmat - Abar)
        F = lu(Mω)

        Y = F \ U  # R diag(u)

        vals = Float64[]
        for P in Pdirs
            Z = Matrix{ComplexF64}(P) * Y
            X = F \ Z
            v = (eps^2) * (norm(X)^2) / denom
            (isfinite(v) && v >= 0) && push!(vals, v)
        end
        Smean[k] = isempty(vals) ? NaN : mean(vals)
    end
    return Smean
end

function integrate_S_tot(ωvals::Vector{Float64}, Sω::Vector{Float64})
    idx = findall(i -> isfinite(ωvals[i]) && isfinite(Sω[i]) && ωvals[i] > 0, eachindex(Sω))
    length(idx) < 2 && return NaN
    return trapz(ωvals[idx], Sω[idx])
end

function integrate_S_relevant(ωvals::Vector{Float64}, Sω::Vector{Float64}, ω95::Float64)
    (isfinite(ω95) && ω95 > 0) || return NaN
    idx = findall(i -> isfinite(ωvals[i]) && isfinite(Sω[i]) && ωvals[i] >= ω95, eachindex(Sω))
    length(idx) < 2 && return NaN
    return trapz(ωvals[idx], Sω[idx])
end

# cumulative sensitivity above ω95; ωq at fraction qfrac; tq=1/ωq; τq=tq/t95
function cutoff_time_from_S(ωvals::Vector{Float64}, Sω::Vector{Float64}, ω95::Float64; qfrac::Float64=0.5)
    @assert 0 < qfrac < 1
    (isfinite(ω95) && ω95 > 0) || return (ωq=NaN, tq=NaN)

    idx = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && ωvals[i] >= ω95 &&
                       isfinite(Sω[i]) && Sω[i] >= 0, eachindex(Sω))
    length(idx) < 3 && return (ωq=NaN, tq=NaN)

    ω = ωvals[idx]
    S = Sω[idx]
    Stot = trapz(ω, S)
    (isfinite(Stot) && Stot > 0) || return (ωq=NaN, tq=NaN)

    cum = zeros(Float64, length(ω))
    for i in 2:length(ω)
        cum[i] = cum[i-1] + 0.5*(S[i-1] + S[i])*(ω[i]-ω[i-1])
    end
    target = qfrac * Stot
    j = findfirst(cum .>= target)
    isnothing(j) && return (ωq=NaN, tq=NaN)
    j == 1 && return (ωq=ω[1], tq=1.0/ω[1])

    ω1, ω2 = ω[j-1], ω[j]
    c1, c2 = cum[j-1], cum[j]
    ωq = (c2 == c1) ? ω2 : (ω1 + (target - c1) * (ω2 - ω1) / (c2 - c1))
    tq = (isfinite(ωq) && ωq > 0) ? (1.0/ωq) : NaN
    return (ωq=ωq, tq=tq)
end

# -----------------------------
# Dynamic collectivity K(ω) and cutoff ωc from K(ω)=1 crossing
#
# Abar = -I + A  ⇒  A = Abar + I
# Dω = diag( 1/(1 + i ω T_i) ), where T_i = 1/u_i
# Aω = A * Dω
# K(ω) = ρ(Aω) (spectral radius = max |eig| for complex Aω)
# -----------------------------
function spectral_radius_complex(M::AbstractMatrix{ComplexF64})
    vals = eigvals(Matrix(M))
    return maximum(abs.(vals))
end

function K_spectrum(Abar::Matrix{Float64}, u::Vector{Float64}, ωvals::Vector{Float64})
    S = size(Abar,1)
    A = Abar + Matrix{Float64}(I, S, S)   # remove -I
    Tvec = 1.0 ./ u
    Kω = Vector{Float64}(undef, length(ωvals))

    for (k, ω0) in enumerate(ωvals)
        ω = float(ω0)
        Dω = Diagonal(ComplexF64.(1.0 ./ (1.0 .+ im*ω*Tvec)))
        Aω = Matrix{ComplexF64}(A) * Dω
        Kω[k] = spectral_radius_complex(Aω)
    end
    return Kω
end

# find first ω where K crosses from >thr to <=thr (monotone decreasing typically)
function cutoff_omega_from_K(ωvals::Vector{Float64}, Kω::Vector{Float64}; thr::Float64=1.0)
    idx = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(Kω[i]), eachindex(Kω))
    length(idx) < 3 && return NaN

    ω = ωvals[idx]
    K = Kω[idx]

    # if never above threshold, no crossing
    if K[1] <= thr
        return NaN
    end
    # if always above threshold on grid
    if all(K .> thr)
        return NaN
    end

    for j in 2:length(ω)
        if K[j-1] > thr && K[j] <= thr
            # interpolate in log ω for stability
            x1, x2 = log(ω[j-1]), log(ω[j])
            y1, y2 = K[j-1], K[j]
            if y2 == y1
                return ω[j]
            end
            x = x1 + (thr - y1) * (x2 - x1) / (y2 - y1)
            return exp(x)
        end
    end
    return NaN
end

# -----------------------------
# Base system struct + builder (PPM across coherence range)
# -----------------------------
struct BaseSystem
    Tppm::Float64
    q_incoh::Float64
    u::Vector{Float64}
    Abar::Matrix{Float64}
    rbase::Vector{Float64}
    t95::Float64
    eps::Float64
    K0::Float64
end

function build_base_ppm(; S::Int, B::Int, L::Int, Tppm::Float64,
    u::Vector{Float64},
    mag_abs::Float64=1.0,
    mag_cv::Float64=0.5,
    corr_aij_aji::Float64=0.99,
    target_alpha::Float64=-0.05,
    eps_rel::Float64=0.20,
    tvals::Vector{Float64},
    seed::Int=1
)
    rng = MersenneTwister(seed)

    out_ppm = ppm(S, B, L, float(Tppm); rng=rng)
    Aadj = out_ppm.A
    qobs = out_ppm.q

    W = build_interaction_matrix(Aadj; mag_abs=mag_abs, mag_cv=mag_cv, corr_aij_aji=corr_aij_aji, rng=rng)

    # Off-diagonal structure O from W
    O = copy(W)
    offdiag_zero!(O)
    normalize_offdiag!(O) || return nothing

    s = find_scale_to_target_alpha(O, u; target_alpha=target_alpha)
    isfinite(s) || return nothing

    # Abar = -I + s*O (diag -1)
    Abar = -Matrix{Float64}(I, S, S) + s * O
    J = Diagonal(u) * Abar

    rbase = rmed_curve(J, u, tvals)
    t95 = t95_from_rmed_curve(tvals, rbase; target=0.05)
    (isfinite(t95) && t95 > 0) || return nothing

    offA = Abar + Matrix{Float64}(I, S, S)  # remove -I => offdiag content
    eps = eps_rel * norm(offA)
    (isfinite(eps) && eps > 0) || return nothing

    # static collectivity K0 = ρ(A) where A = Abar + I
    A = offA
    K0 = maximum(abs.(eigvals(Matrix(A))))
    isfinite(K0) || return nothing

    return BaseSystem(float(Tppm), float(qobs), copy(u), Abar, rbase, float(t95), float(eps), float(K0))
end

# -----------------------------
# Evaluate one base: time errors + sensitivity + collectivity cutoff
# -----------------------------
function eval_base(base::BaseSystem, tvals::Vector{Float64}, ωvals::Vector{Float64};
    P_reps::Int=20,
    P_sparsity::Float64=1.0,
    margin::Float64=1e-3,
    qfrac_sens::Float64=0.5,
    seed::Int=1
)
    rng = MersenneTwister(seed)
    S = length(base.u)
    Du = Diagonal(base.u)

    # sample perturbations (keep stable)
    Pdirs = Matrix{Float64}[]
    Err_rel_list = Float64[]
    Err_tot_list = Float64[]

    for k in 1:P_reps
        P = sample_noise_Pdir(S; sparsity_p=P_sparsity, rng=rng)
        P === nothing && continue

        Abarp = base.Abar + base.eps * P
        Jp = Du * Abarp
        αp = spectral_abscissa(Jp)
        (isfinite(αp) && αp < -margin) || continue

        rpert = rmed_curve(Jp, base.u, tvals)
        tm = time_errors_normalised(tvals, base.rbase, rpert)
        tm === nothing && continue

        push!(Pdirs, P)
        push!(Err_rel_list, tm.Err_rel)
        push!(Err_tot_list, tm.Err_tot)
    end

    length(Pdirs) < 8 && return nothing

    # sensitivity spectrum (typical)
    Sω = sensitivity_spectrum_typical(base.Abar, base.u, base.eps, ωvals, Pdirs)
    Sens_tot = integrate_S_tot(ωvals, Sω)
    ω95 = 1.0 / base.t95
    Sens_rel = integrate_S_relevant(ωvals, Sω, ω95)

    ctS = cutoff_time_from_S(ωvals, Sω, ω95; qfrac=qfrac_sens)
    tqS = ctS.tq
    τqS = (isfinite(tqS) && tqS > 0) ? tqS / base.t95 : NaN

    # dynamic collectivity K(ω) and ωc from K=1 crossing
    Kω = K_spectrum(base.Abar, base.u, ωvals)
    ωc = cutoff_omega_from_K(ωvals, Kω; thr=1.0)
    tc = (isfinite(ωc) && ωc > 0) ? (1.0/ωc) : NaN
    τc = (isfinite(tc) && tc > 0) ? (tc / base.t95) : NaN

    return (
        nP=length(Pdirs),
        q_incoh=base.q_incoh,
        Tppm=base.Tppm,
        t95=base.t95,
        K0=base.K0,

        Err_rel=meanfinite(Err_rel_list),
        Err_tot=meanfinite(Err_tot_list),

        Sens_rel=Sens_rel,
        Sens_tot=Sens_tot,

        τqS=τqS,
        ωqS=ctS.ωq,

        ωc=ωc,
        τc=τc,

        # optional for debugging/one example
        Kω=Kω,
        Sω=Sω
    )
end

# -----------------------------
# Run experiment across coherence range
# -----------------------------
function run_coherence_experiment(;
    S::Int=80,
    B::Int=Int(round(0.2S)),
    L::Int=round(Int, 0.15*S^2),     # keep fixed across coherence
    T_grid = collect(10 .^ range(log10(0.02), log10(1.0); length=8)),
    reps_per_T::Int=8,

    # interaction magnitudes
    mag_abs::Float64=1.0,
    mag_cv::Float64=0.5,
    corr_aij_aji::Float64=0.99,

    # stability standardisation
    target_alpha::Float64=-0.05,
    eps_rel::Float64=0.20,
    margin::Float64=1e-3,

    # time/frequency grids
    tvals = 10 .^ range(log10(0.01), log10(200.0); length=50),
    ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=80),

    # uncertainty
    P_reps::Int=20,
    P_sparsity::Float64=1.0,

    # tq from sensitivity mass
    qfrac_sens::Float64=0.5,

    seed::Int=1234,
    u_mean::Float64=1.0,
    u_cv::Float64=0.1
)
    tvals = collect(float.(tvals))
    ωvals = collect(float.(ωvals))
    T_grid = collect(float.(T_grid))

    # Fix u across everything (to keep "everything else equal")
    rng0 = MersenneTwister(seed)
    u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng0))

    # Build bases
    bases = BaseSystem[]
    for (iT, Tppm) in enumerate(T_grid)
        for r in 1:reps_per_T
            bseed = seed + 100_000*iT + 10_007*r
            base = build_base_ppm(
                S=S, B=B, L=L, Tppm=Tppm,
                u=u,
                mag_abs=mag_abs, mag_cv=mag_cv, corr_aij_aji=corr_aij_aji,
                target_alpha=target_alpha,
                eps_rel=eps_rel,
                tvals=tvals,
                seed=bseed
            )
            base === nothing && continue
            push!(bases, base)
        end
    end
    @info "Built $(length(bases)) bases across coherence grid (attempted $(length(T_grid)*reps_per_T))."

    # Evaluate bases (threaded)
    n = length(bases)
    q_incoh = fill(NaN, n)
    Tppm_v  = fill(NaN, n)
    K0      = fill(NaN, n)
    t95     = fill(NaN, n)

    Err_rel = fill(NaN, n)
    Err_tot = fill(NaN, n)
    Sens_rel= fill(NaN, n)
    Sens_tot= fill(NaN, n)

    τqS     = fill(NaN, n)
    ωc      = fill(NaN, n)
    τc      = fill(NaN, n)

    nPacc   = fill(0, n)

    Threads.@threads for i in 1:n
        out = eval_base(bases[i], tvals, ωvals;
            P_reps=P_reps,
            P_sparsity=P_sparsity,
            margin=margin,
            qfrac_sens=qfrac_sens,
            seed=seed + 900_000*i
        )
        out === nothing && continue

        q_incoh[i] = out.q_incoh
        Tppm_v[i]  = out.Tppm
        K0[i]      = out.K0
        t95[i]     = out.t95

        Err_rel[i] = out.Err_rel
        Err_tot[i] = out.Err_tot
        Sens_rel[i]= out.Sens_rel
        Sens_tot[i]= out.Sens_tot

        τqS[i]     = out.τqS
        ωc[i]      = out.ωc
        τc[i]      = out.τc

        nPacc[i]   = out.nP
    end

    return (
        tvals=tvals, ωvals=ωvals, bases=bases,
        q_incoh=q_incoh, Tppm=Tppm_v, K0=K0, t95=t95,
        Err_rel=Err_rel, Err_tot=Err_tot,
        Sens_rel=Sens_rel, Sens_tot=Sens_tot,
        τqS=τqS, ωc=ωc, τc=τc, nPacc=nPacc
    )
end

# -----------------------------
# Plotting
# -----------------------------
function plot_coherence_results(res; figsize=(1700, 1100))
    q   = res.q_incoh
    K0  = res.K0
    ωc  = res.ωc
    τc  = res.τc
    τq  = res.τqS
    Er  = res.Err_rel
    Sr  = res.Sens_rel
    Et  = res.Err_tot
    St  = res.Sens_tot
    Tppm= res.Tppm

    # masks
    mK  = findall(i -> isfinite(q[i]) && isfinite(K0[i]) && K0[i] > 0, eachindex(q))
    mC  = findall(i -> isfinite(K0[i]) && K0[i] > 0 && isfinite(ωc[i]) && ωc[i] > 0, eachindex(K0))
    mTc = findall(i -> isfinite(K0[i]) && K0[i] > 0 && isfinite(τc[i]) && τc[i] > 0, eachindex(K0))
    mTq = findall(i -> isfinite(K0[i]) && K0[i] > 0 && isfinite(τq[i]) && τq[i] > 0, eachindex(K0))

    mE1 = findall(i -> isfinite(Er[i]) && Er[i] > 0 && isfinite(Sr[i]) && Sr[i] > 0, eachindex(Er))
    mE2 = findall(i -> isfinite(Et[i]) && Et[i] > 0 && isfinite(St[i]) && St[i] > 0, eachindex(Et))

    ρ_qK = (length(mK)  >= 6) ? cor(q[mK], log.(K0[mK])) : NaN
    ρ_Kω = (length(mC)  >= 6) ? cor(log.(K0[mC]), log.(ωc[mC])) : NaN
    ρ_Kτc= (length(mTc) >= 6) ? cor(log.(K0[mTc]), log.(τc[mTc])) : NaN
    ρ_Kτq= (length(mTq) >= 6) ? cor(log.(K0[mTq]), log.(τq[mTq])) : NaN

    ρ_rel = (length(mE1) >= 6) ? cor(log.(Er[mE1]), log.(Sr[mE1])) : NaN
    ρ_tot = (length(mE2) >= 6) ? cor(log.(Et[mE2]), log.(St[mE2])) : NaN

    fig = Figure(size=figsize)

    ax1 = Axis(fig[1,1];
        xlabel="PPM incoherence q (observed)",
        ylabel="static collectivity K0 = ρ(A)",
        yscale=log10,
        title="Coherence → collectivity"
    )
    scatter!(ax1, q[mK], K0[mK], markersize=8)
    text!(ax1, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(q, log K0) = $(round(ρ_qK,digits=3))   N=$(length(mK))")

    ax2 = Axis(fig[1,2];
        xscale=log10, yscale=log10,
        xlabel="K0",
        ylabel="cutoff frequency ωc (K(ω)=1 crossing)",
        title="Collectivity → cutoff frequency"
    )
    scatter!(ax2, K0[mC], ωc[mC], markersize=8)
    text!(ax2, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log K0, log ωc) = $(round(ρ_Kω,digits=3))   N=$(length(mC))")

    ax3 = Axis(fig[2,1];
        xscale=log10, yscale=log10,
        xlabel="K0",
        ylabel="τc = (1/ωc) / t95",
        title="Dynamic availability time (normalised)"
    )
    scatter!(ax3, K0[mTc], τc[mTc], markersize=8)
    text!(ax3, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log K0, log τc) = $(round(ρ_Kτc,digits=3))   N=$(length(mTc))")

    ax4 = Axis(fig[2,2];
        xscale=log10, yscale=log10,
        xlabel="K0",
        ylabel="τq = tq/t95 (from relevant sensitivity mass)",
        title="Sensitivity organised by τq"
    )
    scatter!(ax4, K0[mTq], τq[mTq], markersize=8)
    text!(ax4, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log K0, log τq) = $(round(ρ_Kτq,digits=3))   N=$(length(mTq))")

    ax5 = Axis(fig[3,1];
        xscale=log10, yscale=log10,
        xlabel="Sens_rel = ∫_{ω≥1/t95} S(ω) dω",
        ylabel="Err_rel = ∫_{τ≤1} Δ(τ) d log τ",
        title="Relevant-window match (sanity check)"
    )
    scatter!(ax5, Sr[mE1], Er[mE1], markersize=8)
    text!(ax5, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log Err_rel, log Sens_rel) = $(round(ρ_rel,digits=3))   N=$(length(mE1))")

    ax6 = Axis(fig[3,2];
        xscale=log10, yscale=log10,
        xlabel="Sens_tot = ∫ S(ω) dω",
        ylabel="Err_tot = ∫ Δ(τ) d log τ",
        title="Total match (sanity check)"
    )
    scatter!(ax6, St[mE2], Et[mE2], markersize=8)
    text!(ax6, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log Err_tot, log Sens_tot) = $(round(ρ_tot,digits=3))   N=$(length(mE2))")

    display(fig)
    return nothing
end

# Optional: per-T summaries (median trends)
function plot_trends_by_T(res; figsize=(1500, 450))
    Tvals = sort(unique(filter(isfinite, res.Tppm)))
    fig = Figure(size=figsize)

    # helper
    function med_by_T(x)
        out = Float64[]
        for T in Tvals
            idx = findall(i -> isfinite(res.Tppm[i]) && res.Tppm[i] == T && isfinite(x[i]) && x[i] > 0, eachindex(x))
            push!(out, isempty(idx) ? NaN : median(x[idx]))
        end
        out
    end

    Kmed  = med_by_T(res.K0)
    τcmed = med_by_T(res.τc)
    τqmed = med_by_T(res.τqS)

    ax1 = Axis(fig[1,1];
        xscale=log10, yscale=log10,
        xlabel="PPM temperature T (controls coherence)",
        ylabel="median K0",
        title="Trend: coherence knob → collectivity"
    )
    scatter!(ax1, Tvals, Kmed, markersize=10)
    lines!(ax1, Tvals, Kmed, linewidth=3)

    ax2 = Axis(fig[1,2];
        xscale=log10, yscale=log10,
        xlabel="T",
        ylabel="median τc",
        title="Trend: coherence knob → τc"
    )
    scatter!(ax2, Tvals, τcmed, markersize=10)
    lines!(ax2, Tvals, τcmed, linewidth=3)

    ax3 = Axis(fig[1,3];
        xscale=log10, yscale=log10,
        xlabel="T",
        ylabel="median τq",
        title="Trend: coherence knob → τq"
    )
    scatter!(ax3, Tvals, τqmed, markersize=10)
    lines!(ax3, Tvals, τqmed, linewidth=3)

    display(fig)
    return nothing
end

# -----------------------------
# MAIN
# -----------------------------
tvals = 10 .^ range(log10(0.01), log10(200.0); length=45)
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=80)

res = run_coherence_experiment(
    S=120,                      # start moderate; increase later if runtime allows
    B=Int(round(0.25*120)),      # 120*0.25,
    L=round(Int, 0.025*120^2),    # keep fixed across coherence
    T_grid=collect(10 .^ range(log10(0.02), log10(1.0); length=8)),
    reps_per_T=8,

    mag_abs=0.01,
    mag_cv=0.25,
    corr_aij_aji=0.99,

    target_alpha=-0.05,
    eps_rel=0.20,

    tvals=tvals,
    ωvals=ωvals,

    P_reps=18,                 # increase once stable
    P_sparsity=1.0,

    qfrac_sens=0.5,
    seed=1234
)

plot_coherence_results(res)
plot_trends_by_T(res)
################################################################################