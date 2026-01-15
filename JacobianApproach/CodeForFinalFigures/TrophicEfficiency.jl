################################################################################
# CASCADE MODEL → TROPHIC EFFICIENCY (η) GRADIENT
#
# Goal (single, independent script):
#   Build many cascade food webs across a trophic-efficiency gradient η ∈ [0,1],
#   keeping everything else as constant as possible, then show:
#
#   (A) η organises static collectivity K0 = ρ(A) and dynamic cutoff ωc
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
# Network model (Zelnik-style cascade / Pimm et al.):
#   - Species ordered along a niche axis (1..S).
#   - Predator j can feed on any prey i<j with probability p.
#   - For each realized predator–prey pair (i prey, j predator):
#         draw attack rate X ~ Uniform(0, 2a)
#         set interaction pair as:
#             prey → predator  : +X
#             predator → prey  : -η X
#     η ∈ [0,1] interpolates between triangular (η=0) and skew-symmetric pairs (η=1).
#
# Dynamics & standardisation (unchanged):
#   - J = diag(u) * Abar,   Abar = -I + s*O, O is off-diagonal structure ||O||_F=1
#   - Choose s so that spectral abscissa α(J) ≈ target_alpha (<0)
#   - Structural uncertainty: Abar → Abar + eps_struct*P, with iid Gaussian offdiag P,
#     ||P||_F=1. Keep only perturbations that remain stable.
#   - rmed(t) biomass-weighted with C=diag(u^2); define t95 from exp(-rmed*t)=0.05
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
# Cascade food-web generator (Zelnik/Pimm-style)
# Builds signed interaction matrix W directly.
#
# Convention:
#   prey i < predator j:
#      W[i,j] = +X          (prey effect on predator)
#      W[j,i] = -η X        (predator effect on prey)
# -----------------------------
function cascade_interaction_matrix(S::Int; p::Float64, a::Float64, η::Float64,
        ensure_prey_per_pred::Bool=true,
        rng=Random.default_rng())

    @assert 0.0 <= p <= 1.0
    @assert a > 0
    @assert 0.0 <= η <= 1.0

    W = zeros(Float64, S, S)

    # Draw links i<j with prob p
    for pred in 2:S
        for prey in 1:(pred-1)
            if rand(rng) < p
                X = rand(rng) * (2a)  # Uniform(0,2a)
                W[prey, pred] = +X
                W[pred, prey] = -η * X
            end
        end
    end

    # Optional: ensure each predator (2..S) has at least one prey
    if ensure_prey_per_pred
        for pred in 2:S
            # prey links are in column 'pred' above diagonal: W[1:pred-1, pred] > 0
            if all(W[1:(pred-1), pred] .== 0.0)
                prey = rand(rng, 1:(pred-1))
                X = rand(rng) * (2a)
                W[prey, pred] = +X
                W[pred, prey] = -η * X
            end
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

# t95-from-rmed curve (using exp(-rmed*t))
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
    eps_struct::Float64, ωvals::Vector{Float64}, Pdirs::Vector{Matrix{Float64}}
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
            v = (eps_struct^2) * (norm(X)^2) / denom
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

# find first ω where K crosses from >thr to <=thr
function cutoff_omega_from_K(ωvals::Vector{Float64}, Kω::Vector{Float64}; thr::Float64=1.0)
    idx = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(Kω[i]), eachindex(Kω))
    length(idx) < 3 && return NaN

    ω = ωvals[idx]
    K = Kω[idx]

    if K[1] <= thr
        return NaN
    end
    if all(K .> thr)
        return NaN
    end

    for j in 2:length(ω)
        if K[j-1] > thr && K[j] <= thr
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
# Base system struct + builder (cascade across η range)
# -----------------------------
struct BaseSystem
    η::Float64
    u::Vector{Float64}
    Abar::Matrix{Float64}
    rbase::Vector{Float64}
    t95::Float64
    eps_struct::Float64
    K0::Float64
end

function build_base_cascade(; S::Int,
    η::Float64,
    p::Float64,
    a::Float64,
    u::Vector{Float64},
    target_alpha::Float64=-0.05,   # kept for call-compatibility, not used in Fix A
    eps_rel::Float64=0.20,
    tvals::Vector{Float64},
    seed::Int=1,
    ensure_prey_per_pred::Bool=true,
    base_margin::Float64=1e-6       # reject unstable bases (optional but practical)
)
    rng = MersenneTwister(seed)

    # Zelnik/Pimm cascade interaction matrix W (signed)
    W = cascade_interaction_matrix(
        S;
        p=p, a=a, η=η,
        ensure_prey_per_pred=ensure_prey_per_pred,
        rng=rng
    )

    # Fix A: use cascade W directly as A (off-diagonal interactions)
    # Abar = -I + A, with A = W
    Abar = -Matrix{Float64}(I, S, S) + W
    J = Diagonal(u) * Abar

    # If the base system is unstable, skip it (since we no longer retune s to enforce α<0)
    α = spectral_abscissa(J)
    (isfinite(α) && α < -base_margin) || return nothing

    # Baseline rmed curve and t95
    rbase = rmed_curve(J, u, tvals)
    t95 = t95_from_rmed_curve(tvals, rbase; target=0.05)
    (isfinite(t95) && t95 > 0) || return nothing

    # Structural uncertainty scale (relative to interaction strength)
    # Under Fix A: A = W, so use ||W|| as the natural scale
    eps_struct = eps_rel * norm(W)
    (isfinite(eps_struct) && eps_struct > 0) || return nothing

    # Static collectivity K0 = ρ(A) with A = W
    K0 = maximum(abs.(eigvals(Matrix(W))))
    isfinite(K0) || return nothing

    return BaseSystem(float(η), copy(u), Abar, rbase, float(t95), float(eps_struct), float(K0))
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

        Abarp = base.Abar + base.eps_struct * P
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
    Sω = sensitivity_spectrum_typical(base.Abar, base.u, base.eps_struct, ωvals, Pdirs)
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

    # NEW: S(0) and box approximations
    S0 = sensitivity_at_zero_typical(base.Abar, base.u, base.eps_struct, Pdirs)

    Sens_box = (isfinite(S0) && isfinite(ωc) && ωc > 0) ? (S0 * ωc) : NaN
    # Sens_box_rel = (isfinite(S0) && isfinite(ωc) && ωc > 0 && isfinite(ω95) && ω95 > 0) ?
    #                (S0 * max(ωc - ω95, 0.0)) : NaN
    Sens_box_rel = (isfinite(S0) && isfinite(ωc) && ωc > 0) ?
                   (S0 * ωc) : NaN                   

    return (
        nP=length(Pdirs),
        η=base.η,
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

        # NEW outputs
        S0=S0,
        Sens_box=Sens_box,
        Sens_box_rel=Sens_box_rel,

        # optional for debugging
        Kω=Kω,
        Sω=Sω
    )
end

function sensitivity_at_zero_typical(Abar::Matrix{Float64}, u::Vector{Float64},
    eps_struct::Float64, Pdirs::Vector{Matrix{Float64}}
)
    denom = sum(u.^2)
    denom > 0 || return NaN

    # R(0) = (-Abar)^(-1)
    M0 = -Abar
    F = lu(M0)

    U = Matrix{Float64}(Diagonal(u))
    Y = F \ U   # R(0) * diag(u)

    vals = Float64[]
    for P in Pdirs
        Z = P * Y
        X = F \ Z
        v = (eps_struct^2) * (norm(X)^2) / denom
        (isfinite(v) && v >= 0) && push!(vals, v)
    end

    return isempty(vals) ? NaN : mean(vals)
end

# -----------------------------
# Run experiment across trophic efficiency range
# -----------------------------
function run_efficiency_experiment(;
    S::Int=80,
    p::Float64=0.1,
    a::Float64=0.01,
    η_grid = collect(range(0.0, 1.0; length=9)),
    reps_per_η::Int=8,

    target_alpha::Float64=-0.05,   # kept for signature compatibility
    eps_rel::Float64=0.20,
    margin::Float64=1e-3,

    tvals = 10 .^ range(log10(0.01), log10(200.0); length=50),
    ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=80),

    P_reps::Int=20,
    P_sparsity::Float64=1.0,

    qfrac_sens::Float64=0.5,

    seed::Int=1234,
    u_mean::Float64=1.0,
    u_cv::Float64=0.1,

    ensure_prey_per_pred::Bool=true
)
    tvals = collect(float.(tvals))
    ωvals = collect(float.(ωvals))
    η_grid = collect(float.(η_grid))

    rng0 = MersenneTwister(seed)
    u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng0))

    bases = BaseSystem[]
    for (iη, η) in enumerate(η_grid)
        for r in 1:reps_per_η
            bseed = seed + 100_000*iη + 10_007*r
            base = build_base_cascade(
                S=S, η=η, p=p, a=a,
                u=u,
                target_alpha=target_alpha,
                eps_rel=eps_rel,
                tvals=tvals,
                seed=bseed,
                ensure_prey_per_pred=ensure_prey_per_pred,
                base_margin=margin
            )
            base === nothing && continue
            push!(bases, base)
        end
    end
    @info "Built $(length(bases)) bases across η grid (attempted $(length(η_grid)*reps_per_η))."

    n = length(bases)
    η_v     = fill(NaN, n)
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

    # NEW arrays
    S0          = fill(NaN, n)
    Sens_box    = fill(NaN, n)
    Sens_box_rel= fill(NaN, n)

    Threads.@threads for i in 1:n
        out = eval_base(bases[i], tvals, ωvals;
            P_reps=P_reps,
            P_sparsity=P_sparsity,
            margin=margin,
            qfrac_sens=qfrac_sens,
            seed=seed + 900_000*i
        )
        out === nothing && continue

        η_v[i]      = out.η
        K0[i]       = out.K0
        t95[i]      = out.t95

        Err_rel[i]  = out.Err_rel
        Err_tot[i]  = out.Err_tot
        Sens_rel[i] = out.Sens_rel
        Sens_tot[i] = out.Sens_tot

        τqS[i]      = out.τqS
        ωc[i]       = out.ωc
        τc[i]       = out.τc

        nPacc[i]    = out.nP

        # NEW
        S0[i]           = out.S0
        Sens_box[i]     = out.Sens_box
        Sens_box_rel[i] = out.Sens_box_rel
    end

    return (
        tvals=tvals, ωvals=ωvals, bases=bases,
        η=η_v, K0=K0, t95=t95,
        Err_rel=Err_rel, Err_tot=Err_tot,
        Sens_rel=Sens_rel, Sens_tot=Sens_tot,
        τqS=τqS, ωc=ωc, τc=τc, nPacc=nPacc,

        # NEW outputs
        S0=S0, Sens_box=Sens_box, Sens_box_rel=Sens_box_rel
    )
end

# -----------------------------
# Plotting
# -----------------------------
function plot_efficiency_results(res; figsize=(1700, 1400))
    η   = res.η
    K0  = res.K0
    ωc  = res.ωc
    τc  = res.τc
    τq  = res.τqS
    Er  = res.Err_rel
    Sr  = res.Sens_rel
    Et  = res.Err_tot
    St  = res.Sens_tot

    S0      = res.S0
    Sboxrel = res.Sens_box_rel

    mK  = findall(i -> isfinite(η[i]) && isfinite(K0[i]) && K0[i] > 0, eachindex(η))
    mC  = findall(i -> isfinite(K0[i]) && K0[i] > 0 && isfinite(ωc[i]) && ωc[i] > 0, eachindex(K0))
    mTc = findall(i -> isfinite(K0[i]) && K0[i] > 0 && isfinite(τc[i]) && τc[i] > 0, eachindex(K0))
    mTq = findall(i -> isfinite(K0[i]) && K0[i] > 0 && isfinite(τq[i]) && τq[i] > 0, eachindex(K0))

    mE1 = findall(i -> isfinite(Er[i]) && Er[i] > 0 && isfinite(Sr[i]) && Sr[i] > 0, eachindex(Er))
    mE2 = findall(i -> isfinite(Et[i]) && Et[i] > 0 && isfinite(St[i]) && St[i] > 0, eachindex(Et))

    mKS = findall(i -> isfinite(η[i]) && isfinite(K0[i]) && K0[i] > 0 && isfinite(Sr[i]) && Sr[i] > 0, eachindex(η))

    mBox = findall(i -> isfinite(Sboxrel[i]) && Sboxrel[i] > 0 && isfinite(Sr[i]) && Sr[i] > 0, eachindex(Sr))

    ρ_ηK = (length(mK)  >= 6) ? cor(η[mK], K0[mK]) : NaN
    ρ_Kω = (length(mC)  >= 6) ? cor(log.(K0[mC]), log.(ωc[mC])) : NaN
    ρ_Kτc= (length(mTc) >= 6) ? cor(log.(K0[mTc]), log.(τc[mTc])) : NaN
    ρ_Kτq= (length(mTq) >= 6) ? cor(log.(K0[mTq]), log.(τq[mTq])) : NaN

    ρ_rel = (length(mE1) >= 6) ? cor(log.(Er[mE1]), log.(Sr[mE1])) : NaN
    ρ_tot = (length(mE2) >= 6) ? cor(log.(Et[mE2]), log.(St[mE2])) : NaN

    ρ_box = (length(mBox) >= 6) ? cor(log.(Sr[mBox]), log.(Sboxrel[mBox])) : NaN

    ρ_K0_sens_rel = (length(mE1) >= 6) ? cor(log.(K0[mKS]), log.(Sr[mKS])) : NaN

    fig = Figure(size=figsize)

    ax1 = Axis(fig[1,1];
        xlabel="Trophic efficiency η",
        ylabel="Static collectivity K0 = ρ(A)",
        title="η → collectivity"
    )
    scatter!(ax1, η[mK], K0[mK], markersize=8)
    text!(ax1, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(η, K0) = $(round(ρ_ηK,digits=3))   N=$(length(mK))")

    ax2 = Axis(fig[1,2];
        xscale=log10, yscale=log10,
        xlabel="Static collectivity K0 = ρ(A)",
        ylabel="cutoff frequency ωc (K(ω)=1 crossing)",
        title="Collectivity → cutoff frequency"
    )
    scatter!(ax2, K0[mC], ωc[mC], markersize=8)
    text!(ax2, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log K0, log ωc) = $(round(ρ_Kω,digits=3))   N=$(length(mC))")

    ax3 = Axis(fig[1,3];
        xscale=log10, yscale=log10,
        xlabel="Static collectivity K0 = ρ(A)",
        ylabel="τc = (1/ωc) / t95",
        title="Dynamic availability time (normalised)"
    )
    scatter!(ax3, K0[mTc], τc[mTc], markersize=8)
    text!(ax3, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log K0, log τc) = $(round(ρ_Kτc,digits=3))   N=$(length(mTc))")

    ax4 = Axis(fig[2,1];
        xscale=log10, yscale=log10,
        xlabel="Static collectivity K0 = ρ(A)",
        ylabel="τq = tq/t95 (from relevant sensitivity mass)",
        title="Sensitivity organised by τq"
    )
    scatter!(ax4, K0[mTq], τq[mTq], markersize=8)
    text!(ax4, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log K0, log τq) = $(round(ρ_Kτq,digits=3))   N=$(length(mTq))")

    ax5 = Axis(fig[2,2];
        xscale=log10, yscale=log10,
        xlabel="Sens_rel = ∫_{ω≥1/t95} S(ω) dω",
        ylabel="Err_rel = ∫_{τ≤1} Δ(τ) d log τ",
        title="Relevant-window match"
    )
    scatter!(ax5, Sr[mE1], Er[mE1], markersize=8)
    text!(ax5, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log Err_rel, log Sens_rel) = $(round(ρ_rel,digits=3))   N=$(length(mE1))")

    ax6 = Axis(fig[2,3];
        xscale=log10, yscale=log10,
        xlabel="Sens_tot = ∫ S(ω) dω",
        ylabel="Err_tot = ∫ Δ(τ) d log τ",
        title="Total window match"
    )
    scatter!(ax6, St[mE2], Et[mE2], markersize=8)
    text!(ax6, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log Err_tot, log Sens_tot) = $(round(ρ_tot,digits=3))   N=$(length(mE2))")

    # NEW panel: box approximation test
    ax7 = Axis(fig[3,1];
        xscale=log10, yscale=log10,
        xlabel="S(0) * ωc",
        ylabel="Sens_rel",
        title="Sens_rel ≈ S(0) × width"
    )
    scatter!(ax7, Sboxrel[mBox], Sr[mBox], markersize=8)
    text!(ax7, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log Sens_rel, log box) = $(round(ρ_box,digits=3))   N=$(length(mBox))")

    # Optional: show S0 vs K0 (helps interpret failures)
    ax8 = Axis(fig[3,2];
        xscale=log10, yscale=log10,
        xlabel="Static collectivity K0 = ρ(A)",
        ylabel="S0 = S(ω=0)",
        title="Low-frequency height proxy"
    )
    mS0 = findall(i -> isfinite(S0[i]) && S0[i] > 0 && isfinite(K0[i]) && K0[i] > 0, eachindex(S0))
    scatter!(ax8, K0[mS0], S0[mS0], markersize=8)

    ax9 = Axis(fig[3,3];
        xscale=log10, yscale=log10,
        xlabel="Static collectivity K0 = ρ(A)",
        ylabel="Sens_rel = ∫_{ω≥1/t95} S(ω) dω",
        title="Collectivity → relevant-window match"
    )
    scatter!(ax9, K0[mKS], Sr[mKS], markersize=8)
    text!(ax9, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="cor(log Err_rel, log K0) = $(round(ρ_K0_sens_rel,digits=3))   N=$(length(mKS))")
    

    display(fig)
    return nothing
end

# Optional: per-η summaries (median trends)
function plot_trends_by_η(res; figsize=(1500, 450))
    ηvals = sort(unique(filter(isfinite, res.η)))
    fig = Figure(size=figsize)

    function med_by_η(x)
        out = Float64[]
        for η in ηvals
            idx = findall(i -> isfinite(res.η[i]) && res.η[i] == η && isfinite(x[i]) && x[i] > 0, eachindex(x))
            push!(out, isempty(idx) ? NaN : median(x[idx]))
        end
        out
    end

    Kmed  = med_by_η(res.K0)
    τcmed = med_by_η(res.τc)
    τqmed = med_by_η(res.τqS)

    ax1 = Axis(fig[1,1];
        # yscale=log10,
        xlabel="Trophic efficiency η",
        ylabel="median K0",
        title="Trend: η → collectivity"
    )
    scatter!(ax1, ηvals, Kmed, markersize=10)
    lines!(ax1, ηvals, Kmed, linewidth=3)

    ax2 = Axis(fig[1,2];
        yscale=log10,
        xlabel="Trophic efficiency η",
        ylabel="median τc (1/ωc)/t95",
        title="Trend: η → τc"
    )
    scatter!(ax2, ηvals, τcmed, markersize=10)
    lines!(ax2, ηvals, τcmed, linewidth=3)

    ax3 = Axis(fig[1,3];
        yscale=log10,
        xlabel="Trophic efficiency η",
        ylabel="median τq",
        title="Trend: η → τq"
    )
    scatter!(ax3, ηvals, τqmed, markersize=10)
    lines!(ax3, ηvals, τqmed, linewidth=3)

    display(fig)
    return nothing
end

# -----------------------------
# MAIN
# -----------------------------
tvals = 10 .^ range(log10(0.01), log10(200.0); length=45)
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=80)

res = run_efficiency_experiment(
    S=120,

    # cascade parameters
    p=0.2,          # link probability (controls connectance)
    a=0.1,          # attack-rate scale; X ~ U(0,2a)
    η_grid=collect(range(0.01, 1.0; length=9)),
    reps_per_η=8,

    # stability standardisation
    target_alpha=-0.05,
    eps_rel=0.20,

    # grids
    tvals=tvals,
    ωvals=ωvals,

    # uncertainty sampling
    P_reps=18,
    P_sparsity=1.0,

    # sensitivity cutoff fraction
    qfrac_sens=0.5,
    seed=1234
)

plot_efficiency_results(res)
plot_trends_by_η(res)
################################################################################