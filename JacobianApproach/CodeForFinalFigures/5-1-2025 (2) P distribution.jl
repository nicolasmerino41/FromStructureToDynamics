################################################################################
# OPTION A (JEFF) — FREQUENCY-RESOLVED STRUCTURAL SENSITIVITY
# Fully self-contained Julia script + figures
#
# Core idea:
#   Fix ONE stable baseline system, model interaction uncertainty as A -> A + ε P,
#   and quantify how sensitive the biomass-consistent input→state mapping is,
#   decomposed by Fourier frequency ω via the generalized resolvent
#
#     R(ω) = (i ω T - A)^(-1),  with T = diag(1/u),  A_ii = -1
#
# Biomass-consistent channel:
#   We adopt the forced formulation  T xdot = A x + U w,  U = diag(u).
#   This matches the biomass-weighted setting C = diag(u^2) because C = U^2.
#
# Sensitivity spectrum (for a perturbation direction P, ||P||_F = 1):
#
#   S(ω;P) = ε^2 * || R(ω) P R(ω) U ||_F^2 / tr(C),   C = U^2
#
# Typical (ensemble) sensitivity:
#   pick an uncertainty ensemble for P (here: simple rewiring-based directions),
#   then compute mean/quantiles of S(ω;P) over P.
#
# Worst-case (capacity) sensitivity (unconstrained except ||P||_F=1):
#   sup_P || R P (R U) ||_F = ||R||_2 * ||R U||_2
#   so
#     S_max(ω) = ε^2 * (||R||_2^2 * ||R U||_2^2) / tr(C)
#
# What we show in figures:
#   1) Example spectra at a chosen η: mean S(ω) vs worst-case S_max(ω)
#   2) Typical band ratio  ρ = S_mid / S_low  vs η (plus worst-case ratio)
#   3) Total sensitivity (integrated over ω) vs η (typical + worst-case)
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# If you see oversubscription (BLAS + Julia threads), uncomment:
# BLAS.set_num_threads(1)

# ============================
# Utilities
# ============================
spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigen(J).values))

function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# Extract off-diagonal part (diag -> 0)
function offdiag_part(M::AbstractMatrix)
    S = size(M,1)
    O = copy(Matrix(M))
    @inbounds for i in 1:S
        O[i,i] = 0.0
    end
    return O
end

# Permute off-diagonal entries (including zeros), keeping diagonal fixed
function reshuffle_offdiagonal(M::AbstractMatrix; rng=Random.default_rng())
    S = size(M, 1)
    M2 = copy(Matrix(M))

    vals = Float64[]
    idxs = Tuple{Int,Int}[]
    @inbounds for i in 1:S, j in 1:S
        if i != j
            push!(vals, float(M2[i,j]))
            push!(idxs, (i,j))
        end
    end

    perm = randperm(rng, length(vals))
    @inbounds for k in 1:length(vals)
        (i,j) = idxs[k]
        M2[i,j] = vals[perm[k]]
    end
    return M2
end

# Simple trapezoidal integration on (ω, y)
function trapz(ω::AbstractVector, y::AbstractVector)
    @assert length(ω) == length(y)
    s = 0.0
    for k in 1:(length(ω)-1)
        w1 = float(ω[k]); w2 = float(ω[k+1])
        y1 = float(y[k]); y2 = float(y[k+1])
        (isfinite(y1) && isfinite(y2)) || continue
        s += 0.5 * (y1 + y2) * (w2 - w1)
    end
    return s
end

# ============================
# One knob η: directionality / non-normality (simple, controllable)
# ============================
"""
Build sparse random M, then
  Oη = U + (1-η)*L   (U upper, L lower)
η=0 ~ more bidirectional; η=1 ~ feedforward-ish.
"""
function make_O_eta(S::Int, η::Real; p::Real=0.05, σ::Real=1.0, rng=Random.default_rng())
    @assert 0.0 <= η <= 1.0
    M = zeros(Float64, S, S)
    @inbounds for i in 1:S, j in 1:S
        i == j && continue
        rand(rng) < p && (M[i,j] = randn(rng) * σ)
    end
    U = triu(M, 1)
    L = tril(M, -1)
    O = U + (1.0 - float(η)) * L

    # normalize structure so η is not secretly a strength knob
    n = norm(O)
    n > 0 || return O
    return O / n
end

# ============================
# Base construction with controlled stability (reduce confounds)
# ============================
"""
Choose scale s so that the baseline has spectral abscissa approximately -R_target:
  J = diag(u) * Abar,   Abar = -I + s*O,   diag(Abar)=-1

We bracket and binary search in s.
"""
function scale_to_target_resilience(O::Matrix{Float64}, u::Vector{Float64};
                                    R_target::Real=0.5,
                                    s_init::Real=1.0,
                                    margin::Real=1e-3,
                                    max_expand::Int=60,
                                    max_bisect::Int=60)
    S = length(u)
    I_S = Matrix{Float64}(I, S, S)

    # We want α(J) ≈ -R_target < 0
    target = -float(R_target)

    # helper: α(s)
    function α_of_s(s)
        Abar = -I_S + s * O
        J = Diagonal(u) * Abar
        return spectral_abscissa(J)
    end

    # Start at s=0 (pure diagonal) => α = max(-u) < 0
    s_lo = 0.0
    α_lo = α_of_s(s_lo)

    # Expand s_hi until α crosses target (becomes > target), or we give up
    s_hi = float(s_init)
    α_hi = α_of_s(s_hi)
    k = 0
    while (isfinite(α_hi) && α_hi < target) && k < max_expand
        s_hi *= 2.0
        α_hi = α_of_s(s_hi)
        k += 1
    end

    # If never crossed target, return NaN (too stable even at large s)
    if !(isfinite(α_hi) && α_hi >= target)
        return (s=NaN, α=NaN)
    end

    # Bisection for α(s)=target
    for _ in 1:max_bisect
        s_mid = 0.5*(s_lo + s_hi)
        α_mid = α_of_s(s_mid)
        if !isfinite(α_mid)
            s_hi = s_mid
            continue
        end
        if α_mid < target
            s_lo = s_mid
            α_lo = α_mid
        else
            s_hi = s_mid
            α_hi = α_mid
        end
    end

    s = 0.5*(s_lo + s_hi)
    α = α_of_s(s)
    # Ensure still stable by margin
    if !(isfinite(α) && α < -float(margin))
        return (s=NaN, α=NaN)
    end
    return (s=s, α=α)
end

# ============================
# Resolvent + sensitivity spectra
# ============================
"""
Compute R(ω) = (i ω T - Abar)^(-1) via solve, not explicit inverse.
Inputs:
  Abar: real S×S with diag -1
  u: positive vector
Returns:
  R: Complex matrix
"""
function resolvent(Abar::Matrix{Float64}, u::Vector{Float64}, ω::Float64)
    T = Diagonal(1.0 ./ u)
    M = Matrix{ComplexF64}(im*ω*T - Abar)
    # Return linear operator matrix by inverting via factorization solve against I
    # but we only need R acting on things; we will solve twice instead.
    return M
end

"""
Ensemble sensitivity:
  S(ω;P) = ε^2 * || R P R U ||_F^2 / tr(C)
Compute for a given ω by two solves (as in your earlier code).
"""
function S_ensemble_oneω(Abar::Matrix{Float64}, u::Vector{Float64}, P::Matrix{Float64},
                        ω::Float64, ε::Float64)
    S = length(u)
    U = Matrix{ComplexF64}(Diagonal(u))
    trC = sum(u.^2)

    M = resolvent(Abar, u, ω)             # M = (i ω T - Abar)
    Y = M \ U                              # Y = R U
    Z = Matrix{ComplexF64}(P) * Y          # Z = P R U
    X = M \ Z                              # X = R P R U

    val = (ε^2) * (norm(X)^2) / trC
    return (isfinite(val) && val > 0) ? val : NaN
end

"""
Worst-case (unconstrained except ||P||_F=1):
  sup_P || R P (R U) ||_F = ||R||_2 * ||R U||_2
So
  S_max(ω) = ε^2 * (||R||_2^2 * ||R U||_2^2) / tr(C)
"""
function S_worstcase_oneω(Abar::Matrix{Float64}, u::Vector{Float64},
                          ω::Float64, ε::Float64)
    U = Matrix{ComplexF64}(Diagonal(u))
    trC = sum(u.^2)

    M = resolvent(Abar, u, ω)
    # We need opnorm(R) and opnorm(RU).
    # Compute RU via one solve; compute R via solving against I (heavier).
    Y = M \ U                       # RU

    # opnorm(RU)
    nRU = opnorm(Y)

    # opnorm(R): solve M \ I
    S = length(u)
    I_S = Matrix{ComplexF64}(I, S, S)
    Rmat = M \ I_S
    nR = opnorm(Rmat)

    val = (ε^2) * (nR^2) * (nRU^2) / trC
    return (isfinite(val) && val > 0) ? val : NaN
end

# ============================
# Perturbation ensemble for P (simple)
# ============================
"""
Simple uncertainty ensemble:
  sample P by rewiring (reshuffling) off-diagonal entries of Abar (diag fixed -1),
  then normalize Δ = (Abar_rew - Abar) to Frobenius norm 1.
"""
function sample_P_rewire(Abar::Matrix{Float64}; rng=Random.default_rng())
    S = size(Abar,1)
    off = Abar + Matrix{Float64}(I, S, S)            # remove diag -1 -> diag 0
    off_rew = reshuffle_offdiagonal(off; rng=rng)    # preserve offdiag multiset
    Abar_rew = -Matrix{Float64}(I, S, S) + off_rew

    Δ = Abar_rew - Abar
    nΔ = norm(Δ)
    nΔ == 0 && return nothing
    return Δ / nΔ
end

# ============================
# Experiment runner
# ============================
struct BaseSystem3
    η::Float64
    u::Vector{Float64}
    Abar::Matrix{Float64}   # diag -1
    eps::Float64
    alphaJ::Float64
end

function build_bases(;
    S::Int,
    η_grid::Vector{Float64},
    base_reps::Int,
    seed::Int,
    u_mean::Float64,
    u_cv::Float64,
    p::Float64,
    σ::Float64,
    R_target::Float64,
    eps_rel::Float64,
    margin::Float64
)
    bases = BaseSystem3[]
    for (iη, η0) in enumerate(η_grid)
        η = float(η0)
        for b in 1:base_reps
            rng = MersenneTwister(seed + 1_000_000*iη + 10_007*b)

            u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))
            O = make_O_eta(S, η; p=p, σ=σ, rng=rng)

            sc = scale_to_target_resilience(O, u; R_target=R_target, margin=margin)
            isfinite(sc.s) || continue

            Abar = -Matrix{Float64}(I, S, S) + sc.s * O
            J = Diagonal(u) * Abar
            αJ = spectral_abscissa(J)

            # ε fixed per base, proportional to offdiag(Abar)
            eps = eps_rel * norm(offdiag_part(Abar))

            push!(bases, BaseSystem3(η, u, Abar, eps, αJ))
        end
    end
    return bases
end

"""
Compute spectra for each base:
  - typical: mean and q90 over P samples at each ω
  - worst-case: Smax at each ω
Then pool across bases within each η by averaging (mean across bases).
"""
function run_sensitivity_experiment(;
    S::Int=80,
    η_grid = collect(range(0.0, 1.0; length=7)),
    base_reps::Int=4,
    P_samples::Int=60,
    seed::Int=1234,
    u_mean::Float64=1.0,
    u_cv::Float64=0.5,
    p::Float64=0.05,
    σ::Float64=1.0,
    R_target::Float64=0.5,
    eps_rel::Float64=0.20,
    margin::Float64=1e-3,
    ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=80),
    ωL::Float64=1e-1,
    ωH::Float64=1e+1
)
    η_grid = collect(float.(η_grid))
    ωvals = collect(float.(ωvals))

    bases = build_bases(
        S=S, η_grid=η_grid, base_reps=base_reps, seed=seed,
        u_mean=u_mean, u_cv=u_cv, p=p, σ=σ,
        R_target=R_target, eps_rel=eps_rel, margin=margin
    )
    @info "Built $(length(bases)) bases (η×rep after filtering), target α(J)≈-$R_target."

    ηs = [b.η for b in bases]
    uniqη = sort(unique(ηs))
    nb = length(bases)
    nω = length(ωvals)

    # Store per-base spectra
    Smean = fill(NaN, nb, nω)
    Sq90  = fill(NaN, nb, nω)
    Smax  = fill(NaN, nb, nω)

    # For reproducibility: each (base, sample) has its own rng
    for bi in 1:nb
        base = bases[bi]

        # sample P directions once per base
        Ps = Matrix{Float64}[]
        for k in 1:P_samples
            rng = MersenneTwister(seed + 9_000_000*bi + 17_021*k)
            P = sample_P_rewire(base.Abar; rng=rng)
            P === nothing && continue
            push!(Ps, P)
        end

        for (wi, ω) in enumerate(ωvals)
            # typical: compute S over ensemble
            vals = Float64[]
            for P in Ps
                s = S_ensemble_oneω(base.Abar, base.u, P, ω, base.eps)
                isfinite(s) && push!(vals, s)
            end
            if !isempty(vals)
                Smean[bi, wi] = mean(vals)
                Sq90[bi, wi]  = quantile(vals, 0.90)
            end

            # worst-case: compute upper envelope
            Smax[bi, wi] = S_worstcase_oneω(base.Abar, base.u, ω, base.eps)
        end
    end

    # Pool across bases within each η (mean over bases)
    pooled = Dict{Float64, NamedTuple}()
    for η in uniqη
        idx = findall(x -> x == η, ηs)
        # average spectra across bases (ignoring NaNs)
        function nanmean_col(mat)
            out = Vector{Float64}(undef, nω)
            for wi in 1:nω
                col = [mat[bi, wi] for bi in idx if isfinite(mat[bi, wi]) && mat[bi, wi] > 0]
                out[wi] = isempty(col) ? NaN : mean(col)
            end
            out
        end
        pooled[η] = (
            Smean = nanmean_col(Smean),
            Sq90  = nanmean_col(Sq90),
            Smax  = nanmean_col(Smax),
        )
    end

    # Band integrals + ratios for each η
    band = Dict{Float64, NamedTuple}()
    for η in uniqη
        Sm = pooled[η].Smean
        Sx = pooled[η].Smax

        low_idx = findall(w -> w > 0 && w <= ωL, ωvals)
        mid_idx = findall(w -> w > ωL && w <= ωH, ωvals)

        # integrate on ω with trapz
        S_low_typ = trapz(ωvals[low_idx], Sm[low_idx])
        S_mid_typ = trapz(ωvals[mid_idx], Sm[mid_idx])

        S_low_wst = trapz(ωvals[low_idx], Sx[low_idx])
        S_mid_wst = trapz(ωvals[mid_idx], Sx[mid_idx])

        ρ_typ = (isfinite(S_low_typ) && S_low_typ > 0 && isfinite(S_mid_typ)) ? (S_mid_typ / S_low_typ) : NaN
        ρ_wst = (isfinite(S_low_wst) && S_low_wst > 0 && isfinite(S_mid_wst)) ? (S_mid_wst / S_low_wst) : NaN

        Tot_typ = trapz(ωvals, Sm)
        Tot_wst = trapz(ωvals, Sx)

        band[η] = (
            Slow_typ=S_low_typ, Smid_typ=S_mid_typ, rho_typ=ρ_typ, Tot_typ=Tot_typ,
            Slow_wst=S_low_wst, Smid_wst=S_mid_wst, rho_wst=ρ_wst, Tot_wst=Tot_wst
        )
    end

    return (bases=bases, ηs=ηs, uniqη=uniqη, ωvals=ωvals,
            pooled=pooled, band=band, ωL=ωL, ωH=ωH, params=(S=S, base_reps=base_reps, P_samples=P_samples,
            u_mean=u_mean, u_cv=u_cv, p=p, σ=σ, R_target=R_target, eps_rel=eps_rel, margin=margin))
end

# ============================
# Plotting
# ============================
function plot_example_spectra(res; ηpick=nothing, figsize=(1100, 650))
    ω = res.ωvals
    uniqη = res.uniqη
    ηpick === nothing && (ηpick = uniqη[clamp(round(Int, length(uniqη)/2), 1, length(uniqη))])

    spec = res.pooled[ηpick]
    Sm = spec.Smean
    Sq = spec.Sq90
    Sx = spec.Smax

    fig = Figure(size=figsize)
    ax = Axis(fig[1,1];
        xscale=log10, yscale=log10,
        xlabel="ω",
        ylabel="sensitivity spectrum",
        title="Example spectra at η=$(round(ηpick,digits=2)) (typical vs worst-case)"
    )

    lines!(ax, ω, Sm; linewidth=3, label="typical mean S(ω)")
    lines!(ax, ω, Sq; linewidth=3, linestyle=:dash, label="typical q90 S(ω)")
    lines!(ax, ω, Sx; linewidth=3, label="worst-case S_max(ω)")

    vlines!(ax, [res.ωL, res.ωH]; linestyle=:dot)
    axislegend(ax; position=:rb, framevisible=false)
    display(fig)
    
end

function plot_band_ratios(res; figsize=(1100, 650))
    xs = res.uniqη
    ρt = [res.band[η].rho_typ for η in xs]
    ρw = [res.band[η].rho_wst for η in xs]

    fig = Figure(size=figsize)
    ax = Axis(fig[1,1];
        xlabel="η",
        ylabel="ρ = S_mid / S_low",
        title="Regime diagnostic: intermediate vs low frequency sensitivity"
    )
    lines!(ax, xs, ρt; linewidth=3, label="typical ρ")
    scatter!(ax, xs, ρt, markersize=10)
    lines!(ax, xs, ρw; linewidth=3, linestyle=:dash, label="worst-case ρ")
    scatter!(ax, xs, ρw, markersize=10)

    axislegend(ax; position=:rb, framevisible=false)
    display(fig)
    
end

function plot_total_sensitivity(res; figsize=(1100, 650))
    xs = res.uniqη
    Tt = [res.band[η].Tot_typ for η in xs]
    Tw = [res.band[η].Tot_wst for η in xs]

    fig = Figure(size=figsize)
    ax = Axis(fig[1,1];
        xlabel="η",
        ylabel="∫ S(ω) dω",
        yscale=log10,
        title="Overall structural sensitivity (integrated over ω)"
    )
    lines!(ax, xs, Tt; linewidth=3, label="typical total")
    scatter!(ax, xs, Tt, markersize=10)
    lines!(ax, xs, Tw; linewidth=3, linestyle=:dash, label="worst-case total")
    scatter!(ax, xs, Tw, markersize=10)
    axislegend(ax; position=:rb, framevisible=false)
    display(fig)
end

# ============================
# MAIN
# ============================
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=90)

res = run_sensitivity_experiment(
    S=80,
    η_grid=collect(range(0.0, 1.0; length=7)),
    base_reps=4,
    P_samples=60,
    seed=1234,
    u_mean=1.0,
    u_cv=0.5,
    p=0.05,
    σ=1.0,
    R_target=0.5,     # controls baseline stability across η (reduces confounds)
    eps_rel=0.20,     # magnitude of interaction uncertainty
    margin=1e-3,
    ωvals=ωvals,
    ωL=1e-1,          # low/mid cutoff (can change later)
    ωH=1e+1           # mid/high cutoff
)

# Figures
plot_example_spectra(res; ηpick=0.0)
plot_example_spectra(res; ηpick=1.0)
plot_band_ratios(res)
plot_total_sensitivity(res)
