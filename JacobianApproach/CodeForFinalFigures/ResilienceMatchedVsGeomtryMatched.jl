using Random
using LinearAlgebra
using Statistics
using Distributions
using CairoMakie
using Arpack

# ============================================================
# Helpers: community generation
# ============================================================

function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

jacobian(A, u) = Diagonal(u) * (A - I)

function random_interaction_matrix(S::Int, connectance::Real; σ::Real=1.0, rng=Random.default_rng())
    A = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        if i != j && rand(rng) < connectance
            A[i, j] = rand(rng, Normal(0, σ))
        end
    end
    return A
end

"""
Reshuffle off-diagonal entries of a matrix while keeping diagonal intact.
Permutes all off-diagonal values (including zeros), preserving the multiset.
Works for A or J.
"""
function reshuffle_offdiagonal(M::AbstractMatrix; rng=Random.default_rng())
    S = size(M, 1)
    M2 = copy(Matrix(M))

    vals = Float64[]
    idxs = Tuple{Int,Int}[]
    for i in 1:S, j in 1:S
        if i != j
            push!(vals, M2[i, j])
            push!(idxs, (i, j))
        end
    end

    perm = randperm(rng, length(vals))
    for k in 1:length(vals)
        (i, j) = idxs[k]
        M2[i, j] = vals[perm[k]]
    end
    return M2
end

"""
Recover interaction matrix A from Jacobian J and biomass vector u,
assuming J = diag(u) * (A - I).
"""
function extract_interaction_matrix(J::AbstractMatrix, u::AbstractVector)
    @assert size(J, 1) == size(J, 2) == length(u)
    @assert all(u .> 0)
    Dinv = Diagonal(1.0 ./ u)
    return Dinv * J + I
end

# ============================================================
# Median return rate + thresholds
# ============================================================

"""
Compute t-crossing from an rmed(t) curve using the implicit definition:
exp(-rmed(t)*t) = target

Inputs:
- tvals  : vector of times (increasing)
- rmed   : vector rmed(t) at those times (same length)
Returns:
- t_cross (Float64) or Inf if never crosses target.
"""
function t95_from_rmed(tvals::AbstractVector, rmed::AbstractVector; target::Real=0.05)
    @assert length(tvals) == length(rmed)
    y = @. exp(-rmed * tvals)
    idx = findfirst(y .<= target)
    isnothing(idx) && return Inf
    idx == 1 && return float(tvals[1])

    t1, t2 = float(tvals[idx-1]), float(tvals[idx])
    y1, y2 = float(y[idx-1]), float(y[idx])
    y2 == y1 && return t2

    return t1 + (target - y1) * (t2 - t1) / (y2 - y1)
end

"""
Return BOTH:
- rmed(t) = -(logA)/(2t)
- logA(t) = log( Tr(E*C*E') / Tr(C) )
using the SAME matrix exponential E = exp(t*J).
"""
function median_return_rate_and_logA(J::AbstractMatrix, u::AbstractVector;
    t::Real=0.01, perturbation::Symbol=:biomass
)
    S = size(J, 1)
    if S == 0 || any(!isfinite, J)
        return (NaN, NaN)
    end

    E = exp(t * J)
    if any(!isfinite, E)
        return (NaN, NaN)
    end

    if perturbation === :uniform
        T = tr(E * transpose(E))
        if !isfinite(T) || T <= 0
            return (NaN, NaN)
        end
        num = log(T)
        den = log(S)
    elseif perturbation === :biomass
        w = u .^ 2
        C = Diagonal(w)
        T = tr(E * C * transpose(E))
        if !isfinite(T) || T <= 0
            return (NaN, NaN)
        end
        num = log(T)
        den = log(sum(w))
    else
        error("Unknown perturbation model: $perturbation")
    end

    logA = num - den            # aligned, C-weighted log amplification
    rmed = -(logA) / (2 * t)    # your definition

    if !isfinite(rmed) || !isfinite(logA)
        return (NaN, NaN)
    end
    return (rmed, logA)
end

"""
Compute rmed(t) curve AND logA(t) curve in one pass.
No extra exponentials beyond what rmed already needs.
"""
function rmed_logA_curve(J, u, tvals; perturbation=:biomass)
    r = fill(NaN, length(tvals))
    logA = fill(NaN, length(tvals))
    for (ti, t) in enumerate(tvals)
        ri, li = median_return_rate_and_logA(J, u; t=t, perturbation=perturbation)
        r[ti] = ri
        logA[ti] = li
    end
    return r, logA
end

# ============================================================
# Intermediate window + peak metrics
# ============================================================

"""
Define an intermediate-time window using your recovery thresholds:
- early_target = 0.95 corresponds to 5% recovered (remaining fraction 0.95)
- late_target  = 0.05 corresponds to 95% recovered (remaining fraction 0.05)

Pairwise window:
t_lo = max(t_early(J), t_early(J'))
t_hi = min(t_late(J),  t_late(J'))
"""
function intermediate_window(tvals, r1, r2; early_target=0.95, late_target=0.05)
    t05_1 = t95_from_rmed(tvals, r1; target=early_target)
    t05_2 = t95_from_rmed(tvals, r2; target=early_target)
    t95_1 = t95_from_rmed(tvals, r1; target=late_target)
    t95_2 = t95_from_rmed(tvals, r2; target=late_target)

    t_lo = max(t05_1, t05_2)
    t_hi = min(t95_1, t95_2)
    return t_lo, t_hi
end

"""
Peak-excess for a delta curve Δ(t) on [t_lo, t_hi].

P    = max_{t in window} |Δ(t)|
Pex  = P - max(|Δ(t_lo)|, |Δ(t_hi)|)

If Pex > 0, there is a genuine intermediate bump not explained by endpoints.
"""
function peak_excess(tvals, delta, t_lo, t_hi)
    idx = findall(t -> (t >= t_lo) && (t <= t_hi), tvals)
    if length(idx) < 3
        return (P=NaN, Pex=NaN)
    end
    P = maximum(abs.(delta[idx]))
    baseline = max(abs(delta[first(idx)]), abs(delta[last(idx)]))
    return (P=P, Pex=P - baseline)
end

"""
Aligned transient-amplification scalar on a window:
Γ(J) = max_{t in window} logA(t), but include 0 as t=0 baseline.
"""
function gamma_window(tvals, logA, t_lo, t_hi)
    idx = findall(t -> (t >= t_lo) && (t <= t_hi), tvals)
    isempty(idx) && return NaN
    return max(0.0, maximum(logA[idx]))
end

# ============================================================
# Long-term / (old) geometry proxies (optional, keep for comparison)
# ============================================================

# spectral abscissa α(J) = max Re(λ)
function spectral_abscissa(J; tol=1e-6, maxiter=3000)
    vals, _ = eigs(J; nev=1, which=:LR, tol=tol, maxiter=maxiter)
    return real(vals[1])
end

# numerical abscissa ω(J) = λmax((J+J')/2)
function numerical_abscissa(J)
    H = Symmetric((J + J') / 2)
    return eigmax(H)
end

non_normal_gap(J) = numerical_abscissa(J) - spectral_abscissa(J)

# ============================================================
# Candidate selection: reshuffle A, rebuild J
# ============================================================

"""
Given baseline J and u:
- Recover A from J and u
- Generate m reshuffled A' by permuting off-diagonal entries of A
- Rebuild J' = diag(u) * (A' - I)

Pick best candidate according to mode:

:match_resilience -> minimize L = |α(J) - α(J')|
:match_geometry   -> minimize G = |(ω-α)(J) - (ω-α)(J')|   (old geometry proxy)

(We do NOT select using Γ by default; we compute it later for the aligned test.)
"""
function pick_reshuffled(J, u; m::Int=30, rng=Random.default_rng(), mode::Symbol=:match_resilience)
    αJ = spectral_abscissa(J)
    gJ = non_normal_gap(J)

    A = extract_interaction_matrix(J, u)

    bestJp = nothing
    bestScore = Inf
    best = (L=NaN, G=NaN, αJ=αJ, αp=NaN, gJ=gJ, gp=NaN)

    for _ in 1:m
        A_sh = reshuffle_offdiagonal(A; rng=rng)
        Jp = jacobian(A_sh, u)

        αp = try
            spectral_abscissa(Jp)
        catch
            continue
        end
        gp = try
            non_normal_gap(Jp)
        catch
            continue
        end

        L = abs(αJ - αp)
        G = abs(gJ - gp)

        score = mode == :match_resilience ? L :
                mode == :match_geometry   ? G :
                error("Unknown mode: $mode")

        if score < bestScore
            bestScore = score
            bestJp = Jp
            best = (L=L, G=G, αJ=αJ, αp=αp, gJ=gJ, gp=gp)
        end
    end

    return bestJp, best
end

# ============================================================
# Pipeline
# ============================================================

function run_pipeline(;
    S::Int=120,
    connectance::Real=0.1,
    n::Int=50,
    m_candidates::Int=200,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    σA::Real=0.5,
    seed::Int=1234,
    perturbation::Symbol=:biomass,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=30),
    early_target::Real=0.95,
    late_target::Real=0.05
)
    rng = MersenneTwister(seed)
    nt = length(tvals)

    # Per-replicate scalars
    L_R  = fill(NaN, n); G_R  = fill(NaN, n)
    L_G  = fill(NaN, n); G_G  = fill(NaN, n)
    P_R  = fill(NaN, n); Pex_R = fill(NaN, n)
    P_G  = fill(NaN, n); Pex_G = fill(NaN, n)

    # NEW aligned amplification change scalars (what you asked for)
    Gc_R = fill(NaN, n)
    Gc_G = fill(NaN, n)

    # Store curves for plotting (optional)
    r_base_all = fill(NaN, n, nt)
    r_R_all    = fill(NaN, n, nt)
    r_G_all    = fill(NaN, n, nt)

    kept = 0

    for k in 1:n
        A = random_interaction_matrix(S, connectance; σ=σA, rng=rng)
        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
        J = jacobian(A, u)

        # baseline curves: rmed + logA (same exponentials)
        r_base, logA_base = rmed_logA_curve(J, u, tvals; perturbation=perturbation)
        if any(isnan, r_base) || any(isnan, logA_base)
            continue
        end

        # pick two reshuffles using different selection criteria
        Jp_R, met_R = pick_reshuffled(J, u; m=m_candidates, rng=rng, mode=:match_resilience)
        Jp_G, met_G = pick_reshuffled(J, u; m=m_candidates, rng=rng, mode=:match_geometry)
        if Jp_R === nothing || Jp_G === nothing
            continue
        end

        r_R, logA_R = rmed_logA_curve(Jp_R, u, tvals; perturbation=perturbation)
        r_G, logA_G = rmed_logA_curve(Jp_G, u, tvals; perturbation=perturbation)
        if any(isnan, r_R) || any(isnan, logA_R) || any(isnan, r_G) || any(isnan, logA_G)
            continue
        end

        # Resilience-matched: peak-excess
        ΔR = r_base .- r_R
        t_lo, t_hi = intermediate_window(tvals, r_base, r_R; early_target=early_target, late_target=late_target)
        pk = peak_excess(tvals, ΔR, t_lo, t_hi)

        # Resilience-matched: aligned amplification change Gc_R = |ΔΓ|
        Γ_base_R = gamma_window(tvals, logA_base, t_lo, t_hi)
        Γ_R      = gamma_window(tvals, logA_R,    t_lo, t_hi)
        Gc_pair_R = abs(Γ_base_R - Γ_R)

        # Geometry-matched: peak-excess
        ΔGcurve = r_base .- r_G
        t_lo2, t_hi2 = intermediate_window(tvals, r_base, r_G; early_target=early_target, late_target=late_target)
        pk2 = peak_excess(tvals, ΔGcurve, t_lo2, t_hi2)

        # Geometry-matched: aligned amplification change
        Γ_base_G = gamma_window(tvals, logA_base, t_lo2, t_hi2)
        Γ_G      = gamma_window(tvals, logA_G,    t_lo2, t_hi2)
        Gc_pair_G = abs(Γ_base_G - Γ_G)

        # store
        kept += 1
        r_base_all[kept, :] .= r_base
        r_R_all[kept, :]    .= r_R
        r_G_all[kept, :]    .= r_G

        L_R[kept]   = met_R.L
        G_R[kept]   = met_R.G
        P_R[kept]   = pk.P
        Pex_R[kept] = pk.Pex
        Gc_R[kept]  = Gc_pair_R

        L_G[kept]   = met_G.L
        G_G[kept]   = met_G.G
        P_G[kept]   = pk2.P
        Pex_G[kept] = pk2.Pex
        Gc_G[kept]  = Gc_pair_G
    end

    # trim to kept
    r_base_all = r_base_all[1:kept, :]
    r_R_all    = r_R_all[1:kept, :]
    r_G_all    = r_G_all[1:kept, :]

    L_R   = L_R[1:kept];   G_R   = G_R[1:kept];   P_R   = P_R[1:kept];   Pex_R = Pex_R[1:kept];  Gc_R = Gc_R[1:kept]
    L_G   = L_G[1:kept];   G_G   = G_G[1:kept];   P_G   = P_G[1:kept];   Pex_G = Pex_G[1:kept];  Gc_G = Gc_G[1:kept]

    mean_base = vec(mean(r_base_all; dims=1))
    mean_R    = vec(mean(r_R_all; dims=1))
    mean_G    = vec(mean(r_G_all; dims=1))

    return (
        kept=kept,
        tvals=tvals,
        r_base_all=r_base_all, r_R_all=r_R_all, r_G_all=r_G_all,
        mean_base=mean_base, mean_R=mean_R, mean_G=mean_G,
        L_R=L_R, G_R=G_R, P_R=P_R, Pex_R=Pex_R, Gc_R=Gc_R,
        L_G=L_G, G_G=G_G, P_G=P_G, Pex_G=Pex_G, Gc_G=Gc_G
    )
end

# ============================================================
# Plotting
# ============================================================

function make_plots(res; save_prefix::Union{Nothing,String}=nothing)
    t = res.tvals

    # ---- Plot 1: mean curves ----
    fig1 = Figure(size=(1050, 650))
    ax1 = Axis(fig1[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="mean rmed(t)",
        title="Mean rmed(t): baseline vs reshuffle-selected rewires"
    )
    lines!(ax1, t, res.mean_base, linewidth=3, label="baseline J")
    lines!(ax1, t, res.mean_R,    linewidth=3, label="J' (min |Δα|)")
    lines!(ax1, t, res.mean_G,    linewidth=3, label="J' (min |Δ(ω-α)|)")
    axislegend(ax1; position=:rt)

    # ---- Plot 2: mean |Δrmed(t)| ----
    fig2 = Figure(size=(1050, 650))
    ax2 = Axis(fig2[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="mean |Δrmed(t)|",
        title="Mean |Δrmed(t)| for each selection regime"
    )
    mean_absΔ_R = vec(mean(abs.(res.r_base_all .- res.r_R_all); dims=1))
    mean_absΔ_G = vec(mean(abs.(res.r_base_all .- res.r_G_all); dims=1))
    lines!(ax2, t, mean_absΔ_R, linewidth=3, label="min |Δα|")
    lines!(ax2, t, mean_absΔ_G, linewidth=3, label="min |Δ(ω-α)|")
    axislegend(ax2; position=:rt)

    # ---- Plot 3: scatter diagnostics (OLD proxy and NEW aligned proxy) ----
    fig3 = Figure(size=(1400, 900))

    # OLD proxy: |Δ(ω-α)|
    ax3a = Axis(fig3[1,1];
        xlabel="G = |Δ(ω-α)|",
        ylabel="Pex",
        title="Resilience-matched: Pex vs old geometry proxy"
    )
    scatter!(ax3a, res.G_R, res.Pex_R)

    ax3b = Axis(fig3[1,2];
        xlabel="L = |Δα|",
        ylabel="Pex",
        title="Resilience-matched: Pex vs long-term divergence"
    )
    scatter!(ax3b, res.L_R, res.Pex_R)

    ax3c = Axis(fig3[2,1];
        xlabel="G = |Δ(ω-α)|",
        ylabel="Pex",
        title="Geometry-matched: Pex vs old geometry proxy"
    )
    scatter!(ax3c, res.G_G, res.Pex_G)

    ax3d = Axis(fig3[2,2];
        xlabel="L = |Δα|",
        ylabel="Pex",
        title="Geometry-matched: Pex vs long-term divergence"
    )
    scatter!(ax3d, res.L_G, res.Pex_G)

    # NEW aligned proxy: |ΔΓ| where Γ = max logA(t) on the pairwise intermediate window
    ax3e = Axis(fig3[1,3];
        xlabel="Gc = |ΔΓ|  (Γ = max logA(t) on window)",
        ylabel="Pex",
        title="Resilience-matched: Pex vs aligned amplification-change"
    )
    scatter!(ax3e, res.Gc_R, res.Pex_R)

    ax3f = Axis(fig3[2,3];
        xlabel="Gc = |ΔΓ|  (Γ = max logA(t) on window)",
        ylabel="Pex",
        title="Geometry-matched: Pex vs aligned amplification-change"
    )
    scatter!(ax3f, res.Gc_G, res.Pex_G)

    if save_prefix !== nothing
        save("$(save_prefix)_mean_curves.png", fig1)
        save("$(save_prefix)_mean_abs_delta.png", fig2)
        save("$(save_prefix)_scatter_diagnostics.png", fig3)
    end

    display(fig1)
    display(fig2)
    display(fig3)

    # Quick numeric summaries
    @info "kept replicates = $(res.kept)"

    # Old proxy correlations
    @info "Resilience-matched: corr(Pex, old G=|Δ(ω-α)|) = $(cor(res.Pex_R, res.G_R))"
    @info "Resilience-matched: corr(Pex, L=|Δα|)         = $(cor(res.Pex_R, res.L_R))"
    @info "Geometry-matched:   corr(Pex, old G=|Δ(ω-α)|) = $(cor(res.Pex_G, res.G_G))"
    @info "Geometry-matched:   corr(Pex, L=|Δα|)         = $(cor(res.Pex_G, res.L_G))"

    # NEW aligned proxy correlations
    @info "Resilience-matched: corr(Pex, NEW Gc=|ΔΓ|)     = $(cor(res.Pex_R, res.Gc_R))"
    @info "Geometry-matched:   corr(Pex, NEW Gc=|ΔΓ|)     = $(cor(res.Pex_G, res.Gc_G))"
end

# ============================================================
# Main
# ============================================================

tvals = 10 .^ range(log10(0.01), log10(100.0); length=30)

res = run_pipeline(
    S=120,
    connectance=0.1,
    n=50,
    m_candidates=200,
    u_mean=1.0,
    u_cv=0.5,
    σA=0.5,
    seed=1234,
    perturbation=:biomass,
    tvals=tvals,
    early_target=0.95,
    late_target=0.05
)

make_plots(res; save_prefix=nothing)
