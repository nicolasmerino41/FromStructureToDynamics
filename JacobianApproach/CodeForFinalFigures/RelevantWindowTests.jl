################################################################################
# STRUCTURAL SENSITIVITY FRAMEWORK (SINGLE, COHERENT SCRIPT)
#
# Tests (across many heterogeneous base systems):
#  1) Total time-domain error matches total frequency-domain sensitivity:
#       log Err_tot  vs  log Sens_tot
#
#  2) Relevant-window (pre-recovery) time-domain error matches relevant-window
#     frequency-domain sensitivity:
#       log Err_rel  vs  log Sens_rel
#
#     Relevant window is defined ONLY by t95 (no arbitrary τ windows):
#       τ = t / t95(base),   relevant window is τ ∈ (0, 1].
#     Relevant frequencies are ω ≥ ω95 = 1/t95(base).
#
#  3) Cutoff time tq (time at which a fraction q of relevant sensitivity mass
#     is accumulated), in normalised time τq = tq / t95, correlates with
#     sensitivity / non-normality.
#
# Notes:
#  - rmed is biomass-weighted only:
#       rmed(t) = -(1/(2t)) * ( log tr(E C E') - log tr(C) ),  C = diag(u^2).
#  - Uncertainty directions P are simple iid Gaussian noise on off-diagonals,
#    Frobenius-normalised (diag(P)=0, ||P||_F=1).
#  - Base networks are "trophic-like" only to ensure heterogeneity:
#    a latent hierarchy biases edge directions, but reciprocity and misalignment
#    are allowed. We randomise generator parameters per base.
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

# -----------------------------
# Biomass weights u
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
# Frequency: generalized resolvent sensitivity spectrum (typical, averaged over P)
# R(ω) = (i ω T - Abar)^(-1),  T = diag(1/u)
# S(ω;P) = eps^2 * || R P R diag(u) ||_F^2 / sum(u^2)
# -----------------------------
function sensitivity_spectrum_typical(Abar::Matrix{Float64}, u::Vector{Float64},
    eps::Float64, ωvals::Vector{Float64}, Pdirs::Vector{Matrix{Float64}}
)
    T = Diagonal(1.0 ./ u)
    U = Matrix{ComplexF64}(Diagonal(u))
    denom = sum(u.^2)

    nω = length(ωvals)
    Smean = fill(NaN, nω)

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

        Smean[k] = isempty(vals) ? NaN : mean(vals)
    end
    return Smean
end

# Total sensitivity mass over ω-grid: ∫ S(ω) dω
function integrate_S_tot(ωvals::Vector{Float64}, Sω::Vector{Float64})
    idx = findall(i -> isfinite(ωvals[i]) && isfinite(Sω[i]) && ωvals[i] > 0, eachindex(Sω))
    length(idx) < 2 && return NaN
    return trapz(ωvals[idx], Sω[idx])
end

# Relevant sensitivity mass: ∫_{ω95}^{∞} S(ω) dω (on finite grid)
function integrate_S_relevant(ωvals::Vector{Float64}, Sω::Vector{Float64}, ω95::Float64)
    if !(isfinite(ω95) && ω95 > 0)
        return NaN
    end
    idx = findall(i -> isfinite(ωvals[i]) && isfinite(Sω[i]) && ωvals[i] >= ω95, eachindex(Sω))
    length(idx) < 2 && return NaN
    return trapz(ωvals[idx], Sω[idx])
end

# Cumulative sensitivity above ω95; find ωq such that fraction q is accumulated
function cutoff_time_from_S(ωvals::Vector{Float64}, Sω::Vector{Float64}, ω95::Float64; q::Float64=0.5)
    @assert 0 < q < 1
    if !(isfinite(ω95) && ω95 > 0)
        return (ωq=NaN, tq=NaN)
    end
    idx = findall(i -> isfinite(ωvals[i]) && isfinite(Sω[i]) && ωvals[i] >= ω95 && ωvals[i] > 0, eachindex(Sω))
    length(idx) < 3 && return (ωq=NaN, tq=NaN)

    ω = ωvals[idx]
    S = Sω[idx]

    Stot = trapz(ω, S)
    (isfinite(Stot) && Stot > 0) || return (ωq=NaN, tq=NaN)

    # cumulative via trapezoids
    cum = zeros(Float64, length(ω))
    for i in 2:length(ω)
        cum[i] = cum[i-1] + 0.5*(S[i-1] + S[i])*(ω[i]-ω[i-1])
    end

    target = q * Stot
    j = findfirst(cum .>= target)
    isnothing(j) && return (ωq=NaN, tq=NaN)
    j == 1 && return (ωq=ω[1], tq=1.0/ω[1])

    # linear interpolation in cum to refine ωq
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
# Time-domain integrals in normalised time τ = t/t95
# Total: ∫ Δ(τ) d log τ (over full available τ range)
# Relevant: ∫_{τ<=1} Δ(τ) d log τ
# -----------------------------
function time_errors_normalised(tvals::Vector{Float64}, rbase::Vector{Float64}, rpert::Vector{Float64};
    q_cutoff::Float64=0.5
)
    @assert length(tvals) == length(rbase) == length(rpert)

    t95 = t95_from_rmed_curve(tvals, rbase; target=0.05)
    (isfinite(t95) && t95 > 0) || return nothing

    τ = tvals ./ t95
    Δ = delta_curve(rbase, rpert)

    good_all = findall(i -> isfinite(τ[i]) && τ[i] > 0 && isfinite(Δ[i]), eachindex(τ))
    length(good_all) < 2 && return nothing
    Err_tot = trapz_logx(τ[good_all], Δ[good_all])

    good_rel = findall(i -> isfinite(τ[i]) && τ[i] > 0 && τ[i] <= 1.0 && isfinite(Δ[i]), eachindex(τ))
    Err_rel = (length(good_rel) >= 2) ? trapz_logx(τ[good_rel], Δ[good_rel]) : NaN

    # NEW: τqΔ from Δ mass in relevant window (τ<=1)
    τqΔ = cutoff_tau_from_delta(τ, Δ; q=q_cutoff, τmax=1.0)

    return (t95=t95, τ=τ, Δ=Δ, Err_tot=Err_tot, Err_rel=Err_rel, τqΔ=τqΔ)
end

function time_errors_normalised(tvals::Vector{Float64}, rbase::Vector{Float64}, rpert::Vector{Float64})
    @assert length(tvals) == length(rbase) == length(rpert)

    t95 = t95_from_rmed_curve(tvals, rbase; target=0.05)
    (isfinite(t95) && t95 > 0) || return nothing

    τ = tvals ./ t95
    Δ = delta_curve(rbase, rpert)

    # Total error: ∫ Δ(τ) dτ   (linear τ)
    good_all = findall(i -> isfinite(τ[i]) && τ[i] > 0 && isfinite(Δ[i]), eachindex(τ))
    length(good_all) < 2 && return nothing
    Err_tot = trapz(τ[good_all], Δ[good_all])

    # Relevant-window error: ∫_{τ<=1} Δ(τ) dτ
    good_rel = findall(i -> isfinite(τ[i]) && τ[i] > 0 && τ[i] <= 1.0 && isfinite(Δ[i]), eachindex(τ))
    Err_rel = (length(good_rel) >= 2) ? trapz(τ[good_rel], Δ[good_rel]) : NaN

    return (t95=t95, τ=τ, Δ=Δ, Err_tot=Err_tot, Err_rel=Err_rel)
end


# -----------------------------
# Non-normality proxy: peak transient gain over time grid (base only)
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
# Build heterogeneous base systems
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
    u_mean::Float64=1.0, u_cv::Float64=0.5,
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

        # eps relative to offdiag(Abar) Fro norm
        offA = Abar + Matrix{Float64}(I, S, S) # remove diag -1 -> offdiag content
        eps = eps_rel * norm(offA)
        (isfinite(eps) && eps > 0) || continue

        Gpk = peak_transient_gain(J, tvals)

        push!(bases, BaseSys(u, Abar, rbase, t95, eps, Gpk))
    end
    return bases
end

# -----------------------------
# Evaluate one base: sample P ensemble, keep stable perturbations,
# compute time errors + sensitivity spectrum
# -----------------------------
function eval_base(base::BaseSys, tvals::Vector{Float64}, ωvals::Vector{Float64};
    P_reps::Int=25,
    P_sparsity::Float64=1.0,
    margin::Float64=1e-3,
    q_cutoff::Float64=0.5,
    seed::Int=1
)
    S = length(base.u)
    rng = MersenneTwister(seed)

    Du = Diagonal(base.u)
    Abar = base.Abar

    Pdirs = Matrix{Float64}[]
    Err_tot_list = Float64[]
    Err_rel_list = Float64[]
    τqΔ_list     = Float64[]   # NEW

    for k in 1:P_reps
        P = sample_noise_Pdir(S; sparsity_p=P_sparsity, rng=rng)
        P === nothing && continue

        Abarp = Abar + base.eps * P
        Jp = Du * Abarp
        αp = spectral_abscissa(Jp)
        (isfinite(αp) && αp < -margin) || continue

        rpert = rmed_curve(Jp, base.u, tvals)
        tm = time_errors_normalised(tvals, base.rbase, rpert; q_cutoff=q_cutoff)  # MOD
        tm === nothing && continue

        push!(Pdirs, P)
        push!(Err_tot_list, tm.Err_tot)
        push!(Err_rel_list, tm.Err_rel)
        push!(τqΔ_list, tm.τqΔ)  # NEW
    end

    length(Pdirs) < 6 && return nothing

    # typical sensitivity spectrum on accepted P ensemble
    Sω = sensitivity_spectrum_typical(Abar, base.u, base.eps, ωvals, Pdirs)

    # total and relevant sensitivity masses
    Sens_tot = integrate_S_tot(ωvals, Sω)
    ω95 = 1.0 / base.t95
    Sens_rel = integrate_S_relevant(ωvals, Sω, ω95)

    # cutoff time tq from sensitivity mass above ω95
    ct = cutoff_time_from_S(ωvals, Sω, ω95; q=q_cutoff)
    tq = ct.tq
    τq = (isfinite(tq) && tq > 0) ? (tq / base.t95) : NaN   # this is τq^S

    return (
        nP=length(Pdirs),
        Err_tot=meanfinite(Err_tot_list),
        Err_rel=meanfinite(Err_rel_list),
        Sens_tot=Sens_tot,
        Sens_rel=Sens_rel,
        ωq=ct.ωq,
        tq=tq,
        τq=τq,                       # τq^S (unchanged name)
        τqΔ=meanfinite(τqΔ_list),    # NEW: τq^Δ
        Gpeak=base.Gpeak,
        t95=base.t95
    )
end

# -----------------------------
# NEW helper: cutoff τq from Δ(τ) mass in relevant window (τ ∈ (0,1])
# Uses d log τ (same measure as Err_rel)
# -----------------------------
function cutoff_tau_from_delta(τ::Vector{Float64}, Δ::Vector{Float64}; q::Float64=0.5, τmax::Float64=1.0)
    @assert 0 < q < 1
    @assert length(τ) == length(Δ)

    idx = findall(i -> isfinite(τ[i]) && τ[i] > 0 && τ[i] <= τmax && isfinite(Δ[i]) && Δ[i] >= 0, eachindex(τ))
    length(idx) < 3 && return NaN

    τs = τ[idx]
    Δs = Δ[idx]

    # ensure increasing τ
    p = sortperm(τs)
    τs = τs[p]; Δs = Δs[p]

    # total mass on d log τ
    tot = trapz_logx(τs, Δs)
    (isfinite(tot) && tot > 0) || return NaN

    # cumulative trapezoids in log τ
    cum = zeros(Float64, length(τs))
    for i in 2:length(τs)
        dlog = log(τs[i]) - log(τs[i-1])
        cum[i] = cum[i-1] + 0.5*(Δs[i-1] + Δs[i]) * dlog
    end

    target = q * tot
    j = findfirst(cum .>= target)
    isnothing(j) && return NaN
    j == 1 && return τs[1]

    τ1, τ2 = τs[j-1], τs[j]
    c1, c2 = cum[j-1], cum[j]
    if c2 == c1
        return τ2
    else
        return τ1 + (target - c1) * (τ2 - τ1) / (c2 - c1)
    end
end

# -----------------------------
# Run experiment across many bases (threaded over bases)
# -----------------------------
function run_experiment(; S::Int=80, base_reps::Int=60, P_reps::Int=20,
    seed::Int=1234,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=50),
    ωvals = 10 .^ range(log10(1e-3), log10(1e3); length=60),
    target_alpha::Float64=-0.05,
    eps_rel::Float64=0.20,
    margin::Float64=1e-3,
    q_cutoff::Float64=0.5
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

    Err_tot  = Vector{Float64}(undef, length(bases))
    Err_rel  = Vector{Float64}(undef, length(bases))
    Sens_tot = Vector{Float64}(undef, length(bases))
    Sens_rel = Vector{Float64}(undef, length(bases))
    τq       = Vector{Float64}(undef, length(bases))   # τq^S (as before)
    τqΔ      = Vector{Float64}(undef, length(bases))   # NEW: τq^Δ
    Gpeak    = Vector{Float64}(undef, length(bases))
    t95      = Vector{Float64}(undef, length(bases))
    nPacc    = Vector{Int}(undef, length(bases))

    fill!(Err_tot, NaN); fill!(Err_rel, NaN)
    fill!(Sens_tot, NaN); fill!(Sens_rel, NaN)
    fill!(τq, NaN); fill!(τqΔ, NaN)
    fill!(Gpeak, NaN); fill!(t95, NaN)
    fill!(nPacc, 0)

    Threads.@threads for i in eachindex(bases)
        out = eval_base(bases[i], tvals, ωvals;
            P_reps=P_reps,
            margin=margin,
            q_cutoff=q_cutoff,
            seed=seed + 900_000*i
        )
        out === nothing && continue

        Err_tot[i]  = out.Err_tot
        Err_rel[i]  = out.Err_rel
        Sens_tot[i] = out.Sens_tot
        Sens_rel[i] = out.Sens_rel
        τq[i]       = out.τq
        τqΔ[i]      = out.τqΔ     # NEW
        Gpeak[i]    = out.Gpeak
        t95[i]      = out.t95
        nPacc[i]    = out.nP
    end

    return (
        tvals=tvals, ωvals=ωvals,
        bases=bases,
        Err_tot=Err_tot, Err_rel=Err_rel,
        Sens_tot=Sens_tot, Sens_rel=Sens_rel,
        τq=τq, τqΔ=τqΔ,                # NEW field
        Gpeak=Gpeak, t95=t95,
        nPacc=nPacc
    )
end

# -----------------------------
# Plot + correlations for the 3 tests
# -----------------------------
function summarize_and_plot(res; figsize=(1200, 700))
    Err_tot  = res.Err_tot
    Err_rel  = res.Err_rel
    Sens_tot = res.Sens_tot
    Sens_rel = res.Sens_rel
    τqS      = res.τq
    τqΔ      = res.τqΔ
    Gpeak    = res.Gpeak

    # 1) Total time error vs total sensitivity
    m1 = findall(i -> isfinite(Err_tot[i]) && Err_tot[i] > 0 && isfinite(Sens_tot[i]) && Sens_tot[i] > 0, eachindex(Err_tot))
    ρ1 = (length(m1) >= 6) ? cor(log.(Err_tot[m1]), log.(Sens_tot[m1])) : NaN
    @info "Test 1: cor(log Err_tot, log Sens_tot) = $ρ1 (N=$(length(m1)))"

    # 2) Relevant-window time error vs relevant sensitivity
    m2 = findall(i -> isfinite(Err_rel[i]) && Err_rel[i] > 0 && isfinite(Sens_rel[i]) && Sens_rel[i] > 0, eachindex(Err_rel))
    ρ2 = (length(m2) >= 6) ? cor(log.(Err_rel[m2]), log.(Sens_rel[m2])) : NaN
    @info "Test 2: cor(log Err_rel, log Sens_rel) = $ρ2 (N=$(length(m2)))"

    m3b = findall(i -> isfinite(τqS[i]) && τqS[i] > 0 && isfinite(Gpeak[i]) && Gpeak[i] > 0, eachindex(τqS))
    ρ3b = (length(m3b) >= 6) ? cor(log.(τqS[m3b]), log.(Gpeak[m3b])) : NaN
    @info "Test 3b: cor(log τq^S, log Gpeak) = $ρ3b (N=$(length(m3b)))"

    # 4a) NEW: τq^Δ vs τq^S (shape alignment, both in normalised time)
    m4a = findall(i -> isfinite(τqΔ[i]) && τqΔ[i] > 0 && isfinite(τqS[i]) && τqS[i] > 0, eachindex(τqΔ))
    ρ4a = (length(m4a) >= 6) ? cor(log.(τqΔ[m4a]), log.(τqS[m4a])) : NaN
    @info "Test 4a: cor(log τq^Δ, log τq^S) = $ρ4a (N=$(length(m4a)))"

    # 3) τq^S vs sensitivity / non-normality (original)
    m3a = findall(i -> isfinite(τqS[i]) && τqS[i] > 0 && isfinite(Sens_rel[i]) && Sens_rel[i] > 0, eachindex(τqS))
    ρ3a = (length(m3a) >= 6) ? cor(log.(τqΔ[m4a]), log.(Sens_rel[m3a])) : NaN
    @info "Test 3a: cor(log τq^S, log Sens_rel) = $ρ3a (N=$(length(m3a)))"

    # 4b) NEW: τq^Δ vs non-normality proxy
    m4b = findall(i -> isfinite(τqΔ[i]) && τqΔ[i] > 0 && isfinite(Gpeak[i]) && Gpeak[i] > 0, eachindex(τqΔ))
    ρ4b = (length(m4b) >= 6) ? cor(log.(τqΔ[m4b]), log.(Gpeak[m4b])) : NaN
    @info "Test 4b: cor(log τq^Δ, log Gpeak) = $ρ4b (N=$(length(m4b)))"

    fig = Figure(size=figsize)

    ax1 = Axis(fig[1,1];
        xscale=log10,
        yscale=log10,
        xlabel="Sens_tot = ∫ S(ω) dω",
        ylabel="Err_tot = ∫ Δ(τ) d log τ (τ=t/t95)",
        title="Test 1: total time error vs total sensitivity"
    )
    scatter!(ax1, Sens_tot[m1], Err_tot[m1], markersize=7)
    text!(ax1, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log) = $(round(ρ1,digits=3))   N=$(length(m1))")

    ax2 = Axis(fig[1,2];
        xscale=log10,
        yscale=log10,
        xlabel="Sens_rel = ∫_{ω≥1/t95} S(ω) dω",
        ylabel="Err_rel = ∫_{τ≤1} Δ(τ) d log τ",
        title="Test 2: relevant-window match"
    )
    scatter!(ax2, Sens_rel[m2], Err_rel[m2], markersize=7)
    text!(ax2, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log) = $(round(ρ2,digits=3))   N=$(length(m2))")

    ax3 = Axis(fig[2,1];
        xscale=log10,
        yscale=log10,
        xlabel="Sens_rel = ∫_{ω≥1/t95} S(ω) dω",
        ylabel="τq = tq/t95 (cuttof time)",
        title="Test 3: cutoff time (S) vs sensitivity"
    )
    scatter!(ax3, Sens_rel[m3a], τqΔ[m4a], markersize=7)
    text!(ax3, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log) = $(round(ρ3a,digits=3))   N=$(length(m3a))")

    # ax4 = Axis(fig[2,2];
    #     xscale=log10, yscale=log10,
    #     xlabel="Gpeak = max_t ||exp(Jt)||_2^2",
    #     ylabel="τq^S",
    #     title="Test 3b: cutoff time (S) vs non-normality proxy"
    # )
    # scatter!(ax4, Gpeak[m3b], τqS[m3b], markersize=7)
    # text!(ax4, 0.05, 0.95, space=:relative, align=(:left,:top),
    #     text="cor(log,log) = $(round(ρ3b,digits=3))   N=$(length(m3b))")

    # ax5 = Axis(fig[3,1];
    #     xscale=log10, yscale=log10,
    #     xlabel="τq^S (from S mass)",
    #     ylabel="τq^Δ (from Δ mass)",
    #     title="Test 4a: shape alignment (time-quantile vs freq-quantile)"
    # )
    # scatter!(ax5, τqS[m4a], τqΔ[m4a], markersize=7)
    # text!(ax5, 0.05, 0.95, space=:relative, align=(:left,:top),
        # text="cor(log,log) = $(round(ρ4a,digits=3))   N=$(length(m4a))")

    ax6 = Axis(fig[2,2];
        xscale=log10, yscale=log10,
        xlabel="Gpeak = max_t ||exp(Jt)||_2^2",
        ylabel="τq",
        title="Test 4: error timing vs non-normality proxy"
    )
    scatter!(ax6, Gpeak[m4b], τqΔ[m4b], markersize=7)
    text!(ax6, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="cor(log,log) = $(round(ρ4b,digits=3))   N=$(length(m4b))")

    display(fig)
    return nothing
end

# -----------------------------
# MAIN
# -----------------------------
tvals = 10 .^ range(log10(0.01), log10(200.0); length=55)
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=80)

res = run_experiment(
    S=120,              # start moderate; increase later (e.g. 120)
    base_reps=70,      # how many heterogeneous bases to try to build
    P_reps=50,         # uncertainty draws per base
    seed=1234,
    tvals=tvals,
    ωvals=ωvals,
    target_alpha=-0.05, # standardise resilience-ish level across bases
    eps_rel=0.20,       # uncertainty magnitude relative to offdiag(Abar)
    margin=1e-3,
    q_cutoff=0.1        # q for cutoff time tq
)

summarize_and_plot(res)
################################################################################