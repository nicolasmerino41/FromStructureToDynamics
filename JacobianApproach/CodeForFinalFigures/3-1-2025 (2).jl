################################################################################
# BUMP-REGIME (RELATIVE TIME τ) — FULL SELF-CONTAINED SCRIPT
#
# Goal:
#   Characterise an INTERMEDIATE-TIME "bump regime" of Δrmed(t) WITHOUT relying on
#   "large t is late". We define windows in relative time τ = t * R∞, where
#   R∞ = -max(real eig(J)) > 0 for the BASE system.
#
# Key design choices (fixing the issues you saw):
#   1) η is a PURE structure knob:
#        - build a sparse Oη, then NORMALISE Oη by ||Oη||_F.
#        - choose a scale s so that the BASE has a FIXED target resilience R∞ ≈ R_target
#          across η (so η does not secretly change strength / distance to instability).
#   2) perturbation size ε is CONSTANT across bases (since Pdir is Fro-normalised).
#   3) bump metrics are regime-based:
#        - Late level L = median Δrmed over τ >= τ_late_min
#        - B_excess = max_mid Δrmed - L          (SIGNED, in rmed units)
#        - A_excess = ∫_{τ_mid} max(Δrmed-L,0) dlog(τ)   (area, regime measure)
#        - bump_flag = 1 if max_mid > (1+δ)*L
#   4) Links to previous predictors:
#        - old_peakΔG(ω): peak change in biomass-weighted resolvent gain between base & perturbed
#        - jeff_peakS(ω): peak sensitivity spectrum for a SINGLE base (RPR idea)
#
# Output:
#   - summary plots vs η
#   - scatterplots linking bump regime metrics to predictors
#
# Notes:
#   - Uses biomass-weighted rmed ONLY.
#   - Uses exp(tJ) so can be heavy; tune t-grid, P_reps, base_reps if needed.
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie
using Base.Threads

# Recommended to avoid BLAS oversubscription when threading this script:
try
    BLAS.set_num_threads(1)
catch
end

# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------
spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigen(J).values))

"""
Asymptotic resilience R∞ = -max(real eig(J)).
Returns NaN if not stable.
"""
function asymptotic_resilience(J::AbstractMatrix)
    α = spectral_abscissa(J)
    Rinf = -α
    return (isfinite(Rinf) && Rinf > 0) ? Rinf : NaN
end

"""
Biomass-weighted rmed(t) ONLY:
  rmed(t) = - ( log(tr(E C E')) - log(tr(C)) ) / (2t)
  C = diag(u^2), E = exp(J t)
"""
function rmed_biomass(J::AbstractMatrix, u::AbstractVector; t::Real)
    t = float(t)
    t <= 0 && return NaN
    E = exp(t * J)
    any(!isfinite, E) && return NaN
    w = u .^ 2
    Ttr = tr(E * Diagonal(w) * transpose(E))
    (!isfinite(Ttr) || Ttr <= 0) && return NaN
    r = -(log(Ttr) - log(sum(w))) / (2t)
    return isfinite(r) ? r : NaN
end

function rmed_curve(J::AbstractMatrix, u::AbstractVector, tvals::Vector{Float64})
    r = Vector{Float64}(undef, length(tvals))
    for (i,t) in enumerate(tvals)
        r[i] = rmed_biomass(J, u; t=t)
    end
    return r
end

function delta_curve(rbase::AbstractVector, rpert::AbstractVector)
    n = length(rbase)
    @assert length(rpert) == n
    Δ = Vector{Float64}(undef, n)
    for i in 1:n
        a = rbase[i]; b = rpert[i]
        Δ[i] = (isfinite(a) && isfinite(b)) ? abs(a - b) : NaN
    end
    return Δ
end

# ------------------------------------------------------------------------------
# Timescales u
# ------------------------------------------------------------------------------
function random_u(S::Int; mean::Real=1.0, cv::Real=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + float(cv)^2))
    mu = log(float(mean)) - sigma^2/2
    return rand(rng, LogNormal(mu, sigma), S)
end

# ------------------------------------------------------------------------------
# One structure knob η: directionality / non-normality proxy
# ------------------------------------------------------------------------------
"""
Build sparse random M, then
  Oη = U + (1-η)*L
η=0 -> more bidirectional-ish
η=1 -> more feedforward-ish
"""
function make_O_eta(S::Int, η::Real; p::Real=0.05, σ::Real=1.0, rng=Random.default_rng())
    η = float(η)
    @assert 0.0 <= η <= 1.0
    M = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        i == j && continue
        rand(rng) < p && (M[i,j] = randn(rng) * float(σ))
    end
    U = triu(M, 1)
    L = tril(M, -1)
    return U + (1.0 - η) * L
end

# ------------------------------------------------------------------------------
# Strength control: choose scale s so base has target R∞ (same across η)
# ------------------------------------------------------------------------------
"""
Return scale s such that J(s) = -diag(u) + s*diag(u)*O has spectral abscissa = -R_target.
O should have diag=0 and already be normalised if you want η as "structure-only".
"""
function scale_to_target_resilience(O::AbstractMatrix, u::AbstractVector;
                                    R_target::Real=1.0,
                                    s_lo::Real=0.0,
                                    s_hi::Real=1.0,
                                    max_expand::Int=40,
                                    max_iter::Int=70,
                                    tol::Real=1e-6)

    Rt = float(R_target)
    Rt <= 0 && return NaN

    α_of_s(s) = spectral_abscissa(-Diagonal(u) + float(s) * (Diagonal(u) * O))
    f(s) = α_of_s(s) + Rt   # target f(s*) = 0

    flo = f(float(s_lo))
    isfinite(flo) || return NaN
    # At s=0, α = -min(u) < 0 so flo should be < 0 if Rt isn't absurdly huge
    flo < 0 || return NaN

    hi = float(s_hi)
    k = 0
    while k < max_expand
        fhi = f(hi)
        if isfinite(fhi) && fhi > 0
            break
        end
        hi *= 2.0
        k += 1
    end
    fhi = f(hi)
    (isfinite(fhi) && fhi > 0) || return NaN

    lo = float(s_lo)

    for _ in 1:max_iter
        mid = 0.5*(lo + hi)
        fmid = f(mid)
        isfinite(fmid) || return NaN
        if abs(fmid) < tol
            return mid
        elseif fmid > 0
            hi = mid
        else
            lo = mid
        end
    end
    return 0.5*(lo + hi)
end

"""
Build base Abar with diag=-1 and controlled R∞ ≈ R_target.
Abar = -I + s*O, J = diag(u)*Abar.
Returns NamedTuple or nothing.
"""
function build_base_controlled(S::Int, η::Real, u::Vector{Float64};
                               p::Real=0.05, σ::Real=1.0,
                               R_target::Real=1.0,
                               rng=Random.default_rng())

    O = make_O_eta(S, η; p=p, σ=σ, rng=rng)

    nO = norm(O)
    nO == 0 && return nothing
    O ./= nO                    # remove strength confound

    s = scale_to_target_resilience(O, u; R_target=R_target)
    isfinite(s) || return nothing

    Abar = -Matrix{Float64}(I, S, S) + s * O
    J = Diagonal(u) * Abar
    Rinf = asymptotic_resilience(J)
    isfinite(Rinf) || return nothing

    return (Abar=Abar, J=Matrix(J), Rinf=Rinf, s=s)
end

# ------------------------------------------------------------------------------
# Perturbation directions P via rewiring (diag stays 0)
# ------------------------------------------------------------------------------
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

"""
Sample a perturbation direction P by rewiring off-diagonals of Abar (diag -1 fixed),
and return Pdir = (Abar_rew - Abar)/||...||_F (diag 0, Fro-norm 1).
"""
function sample_Pdir_from_rewire(Abar::Matrix{Float64}; rng=Random.default_rng())
    S = size(Abar,1)
    off = Abar + Matrix{Float64}(I, S, S)         # diag becomes 0
    off_rew = reshuffle_offdiagonal(off; rng=rng)
    Abar_rew = -Matrix{Float64}(I, S, S) + off_rew
    Δ = Abar_rew - Abar
    nΔ = norm(Δ)
    nΔ == 0 && return nothing
    return Δ / nΔ
end

# ------------------------------------------------------------------------------
# Frequency-domain predictors (old vs Jeff)
# ------------------------------------------------------------------------------
"""
Old biomass resolvent gain:
  G(ω) = || (iω*T - Abar)^(-1) * diag(u) ||_F^2 / sum(u^2)
T = diag(1/u).
"""
function G_old_biomass(Abar::Matrix{Float64}, u::Vector{Float64}, ω::Float64)
    T = Diagonal(1.0 ./ u)
    Mω = Matrix{ComplexF64}(im*ω*T - Abar)
    U = Matrix{ComplexF64}(Diagonal(u))
    Y = Mω \ U
    val = (norm(Y)^2) / sum(u.^2)
    return (isfinite(val) && val > 0) ? val : NaN
end

"""
Jeff sensitivity spectrum (single base + direction P):
  S(ω) = ε^2 * || R(ω) * P * R(ω) * diag(u) ||_F^2 / sum(u^2)
computed via:
  Y = R*diag(u)
  X = R*(P*Y)
"""
function S_jeff_biomass(Abar::Matrix{Float64}, P::Matrix{Float64},
                        u::Vector{Float64}, ω::Float64, ε::Float64)
    T = Diagonal(1.0 ./ u)
    Mω = Matrix{ComplexF64}(im*ω*T - Abar)
    U = Matrix{ComplexF64}(Diagonal(u))
    Y = Mω \ U
    Z = Matrix{ComplexF64}(P) * Y
    X = Mω \ Z
    val = (ε^2) * (norm(X)^2) / sum(u.^2)
    return (isfinite(val) && val > 0) ? val : NaN
end

function peak_on_grid(xvals::Vector{Float64}, g::Vector{Float64})
    idx = findall(i -> isfinite(g[i]) && g[i] > 0, eachindex(g))
    isempty(idx) && return (peak=NaN, xstar=NaN)
    gsub = g[idx]; xsub = xvals[idx]
    imax = argmax(gsub)
    return (peak=gsub[imax], xstar=xsub[imax])
end

# ------------------------------------------------------------------------------
# Bump-regime metrics in relative time τ = t*R∞
# ------------------------------------------------------------------------------
"""
Compute bump regime metrics from Δrmed(t) with relative time τ=t*Rinf.

Windows:
  - Mid: τ in [τ_mid[1], τ_mid[2]]
  - Late: τ >= τ_late_min

Outputs:
  L: median late level
  max_mid: max mid Δ
  B_excess: max_mid - L     (SIGNED)
  A_excess: ∫_{mid} max(Δ-L,0) dlog(τ)
  bump_flag: 1 if max_mid > (1+δ)*L
"""
function bump_metrics_relative_time(tvals::Vector{Float64}, Δt::Vector{Float64}, Rinf::Float64;
                                    τ_mid::Tuple{Float64,Float64}=(0.3, 3.0),
                                    τ_late_min::Float64=10.0,
                                    δ::Float64=0.2)

    n = length(tvals)
    @assert length(Δt) == n
    (!isfinite(Rinf) || Rinf <= 0) && return (L=NaN, max_mid=NaN, B_excess=NaN, A_excess=NaN, bump_flag=0)

    τ = @. tvals * Rinf
    τ1, τ2 = τ_mid

    mid_idx  = findall(i -> isfinite(Δt[i]) && isfinite(τ[i]) && τ[i] > 0 && τ[i] >= τ1 && τ[i] <= τ2, 1:n)
    late_idx = findall(i -> isfinite(Δt[i]) && isfinite(τ[i]) && τ[i] > 0 && τ[i] >= τ_late_min, 1:n)

    (isempty(mid_idx) || isempty(late_idx)) && return (L=NaN, max_mid=NaN, B_excess=NaN, A_excess=NaN, bump_flag=0)

    L = median(Δt[late_idx])
    (!isfinite(L) || L <= 0) && return (L=NaN, max_mid=NaN, B_excess=NaN, A_excess=NaN, bump_flag=0)

    max_mid = maximum(Δt[mid_idx])
    B_excess = max_mid - L

    # area over log(τ) in mid window (sorted τ)
    τm = τ[mid_idx]
    Δm = Δt[mid_idx]
    ord = sortperm(τm)
    τm = τm[ord]; Δm = Δm[ord]

    A = 0.0
    for k in 1:(length(τm)-1)
        x1 = log(τm[k]); x2 = log(τm[k+1])
        y1 = max(Δm[k]   - L, 0.0)
        y2 = max(Δm[k+1] - L, 0.0)
        A += 0.5 * (y1 + y2) * (x2 - x1)
    end
    A_excess = (isfinite(A) && A >= 0) ? A : NaN
    bump_flag = (isfinite(max_mid) && max_mid > (1.0 + δ) * L) ? 1 : 0

    return (L=L, max_mid=max_mid, B_excess=B_excess, A_excess=A_excess, bump_flag=bump_flag)
end

# ------------------------------------------------------------------------------
# Base system container
# ------------------------------------------------------------------------------
struct BaseSystem2
    η::Float64
    u::Vector{Float64}
    Abar::Matrix{Float64}          # diag -1
    J::Matrix{Float64}             # diag(u)*Abar
    Rinf::Float64                  # asymptotic resilience
    rmed_base::Vector{Float64}     # rmed(t) base
    eps::Float64                   # constant ε
    Gbase::Vector{Float64}         # old G(ω) base, precomputed
end

# ------------------------------------------------------------------------------
# Evaluate one perturbation P for a fixed base
# ------------------------------------------------------------------------------
function eval_one_P(base::BaseSystem2, Pdir::Matrix{Float64};
                    tvals::Vector{Float64}, ωvals::Vector{Float64},
                    margin::Float64,
                    τ_mid::Tuple{Float64,Float64}, τ_late_min::Float64, δ::Float64)

    u = base.u
    Abar = base.Abar
    eps = base.eps

    Abarp = Abar + eps * Pdir
    Jp = Diagonal(u) * Abarp

    αp = spectral_abscissa(Jp)
    if !(isfinite(αp) && αp < -float(margin))
        return nothing
    end

    rpert = rmed_curve(Jp, u, tvals)
    Δt = delta_curve(base.rmed_base, rpert)

    bm = bump_metrics_relative_time(tvals, Δt, base.Rinf; τ_mid=τ_mid, τ_late_min=τ_late_min, δ=δ)

    # OLD predictor: peak ΔG(ω)
    Gp = Vector{Float64}(undef, length(ωvals))
    for (k, ω) in enumerate(ωvals)
        Gp[k] = G_old_biomass(Abarp, u, ω)
    end
    ΔG = abs.(base.Gbase .- Gp)
    old_peak = peak_on_grid(ωvals, ΔG).peak

    # JEFF predictor: peak S(ω) using base only
    Sω = Vector{Float64}(undef, length(ωvals))
    for (k, ω) in enumerate(ωvals)
        Sω[k] = S_jeff_biomass(Abar, Pdir, u, ω, eps)
    end
    jeff_peak = peak_on_grid(ωvals, Sω).peak

    return (
        η=base.η,
        Rinf=base.Rinf,
        L=bm.L,
        max_mid=bm.max_mid,
        B_excess=bm.B_excess,
        A_excess=bm.A_excess,
        bump_flag=bm.bump_flag,
        old_peakΔG=old_peak,
        jeff_peakS=jeff_peak
    )
end

# ------------------------------------------------------------------------------
# Main pipeline (threaded)
# ------------------------------------------------------------------------------
function run_bump_regime_pipeline(;
    S::Int=120,
    η_grid = collect(range(0.0, 1.0; length=7)),
    base_reps::Int=6,
    P_reps::Int=60,
    seed::Int=1234,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    p::Real=0.05,
    σ::Real=1.0,
    margin::Real=1e-3,
    R_target::Real=1.0,          # strength control: fix R∞ across η
    ε0::Real=0.05,               # constant perturbation amplitude
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=60),
    ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=40),
    τ_mid::Tuple{Float64,Float64}=(0.3, 3.0),
    τ_late_min::Float64=10.0,
    bump_delta::Float64=0.2
)
    tvals = collect(float.(tvals))
    ωvals = collect(float.(ωvals))
    η_grid = collect(float.(η_grid))

    # ---- build controlled bases
    bases = BaseSystem2[]
    accepted_bases = Dict{Float64,Int}(η => 0 for η in η_grid)
    tried_bases    = Dict{Float64,Int}(η => 0 for η in η_grid)

    for (iη, η0) in enumerate(η_grid)
        η = float(η0)
        for b in 1:base_reps
            tried_bases[η] += 1
            rng = MersenneTwister(seed + 1_000_000*iη + 10_003*b)
            u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))

            built = build_base_controlled(S, η, u; p=p, σ=σ, R_target=R_target, rng=rng)
            built === nothing && continue

            Abar = built.Abar
            J    = built.J
            Rinf = built.Rinf

            α = spectral_abscissa(J)
            (isfinite(α) && α < -float(margin)) || continue

            rbase = rmed_curve(J, u, tvals)

            # precompute base G(ω)
            Gbase = Vector{Float64}(undef, length(ωvals))
            for (k, ω) in enumerate(ωvals)
                Gbase[k] = G_old_biomass(Abar, u, ω)
            end

            push!(bases, BaseSystem2(η, u, Abar, J, Rinf, rbase, float(ε0), Gbase))
            accepted_bases[η] += 1
        end
    end

    @info "Built $(length(bases)) controlled bases total."
    for η in η_grid
        @info "η=$(η): bases accepted=$(accepted_bases[η]) / tried=$(tried_bases[η])"
    end

    nb = length(bases)
    N = nb * P_reps

    # outputs
    η_out      = fill(NaN, N)
    Rinf_out   = fill(NaN, N)
    L_out      = fill(NaN, N)
    maxmid_out = fill(NaN, N)
    Bex_out    = fill(NaN, N)
    Aex_out    = fill(NaN, N)
    flag_out   = fill(0,   N)
    old_out    = fill(NaN, N)
    jeff_out   = fill(NaN, N)

    accepted = Base.Threads.Atomic{Int}(0)
    rejected = Base.Threads.Atomic{Int}(0)

    Threads.@threads for job in 1:N
        bi = (job - 1) ÷ P_reps + 1
        pr = (job - 1) % P_reps + 1
        base = bases[bi]

        rng = MersenneTwister(seed + 9_000_000*bi + 13_007*pr)
        Pdir = sample_Pdir_from_rewire(base.Abar; rng=rng)
        Pdir === nothing && continue

        out = eval_one_P(base, Pdir;
                         tvals=tvals, ωvals=ωvals,
                         margin=float(margin),
                         τ_mid=τ_mid, τ_late_min=τ_late_min, δ=bump_delta)
        if out === nothing
            atomic_add!(rejected, 1)
            continue
        end
        atomic_add!(accepted, 1)

        η_out[job]      = out.η
        Rinf_out[job]   = out.Rinf
        L_out[job]      = out.L
        maxmid_out[job] = out.max_mid
        Bex_out[job]    = out.B_excess
        Aex_out[job]    = out.A_excess
        flag_out[job]   = out.bump_flag
        old_out[job]    = out.old_peakΔG
        jeff_out[job]   = out.jeff_peakS
    end

    @info "Perturbations: accepted=$(accepted[]) rejected=$(rejected[]) (acc rate=$(accepted[] / max(1, accepted[]+rejected[])))"

    return (
        params=(S=S, η_grid=η_grid, base_reps=base_reps, P_reps=P_reps,
                u_mean=u_mean, u_cv=u_cv, p=p, σ=σ, margin=margin,
                R_target=R_target, ε0=ε0,
                τ_mid=τ_mid, τ_late_min=τ_late_min, bump_delta=bump_delta),
        tvals=tvals, ωvals=ωvals,
        η=η_out, Rinf=Rinf_out,
        L=L_out, max_mid=maxmid_out,
        B_excess=Bex_out, A_excess=Aex_out, bump_flag=flag_out,
        old_peakΔG=old_out, jeff_peakS=jeff_out
    )
end

# ------------------------------------------------------------------------------
# Analysis & plotting
# ------------------------------------------------------------------------------
function analyze_and_plot_bump(res; figsize=(1800, 1000))
    η = res.η
    B = res.B_excess
    A = res.A_excess
    flag = res.bump_flag
    old = res.old_peakΔG
    jeff = res.jeff_peakS

    # masks
    mB = map(i -> isfinite(η[i]) && isfinite(B[i]) && isfinite(old[i]) && old[i] > 0, eachindex(B))
    mJ = map(i -> isfinite(η[i]) && isfinite(B[i]) && isfinite(jeff[i]) && jeff[i] > 0, eachindex(B))
    mAold = map(i -> isfinite(A[i]) && A[i] > 0 && isfinite(old[i]) && old[i] > 0, eachindex(A))
    mAjeff = map(i -> isfinite(A[i]) && A[i] > 0 && isfinite(jeff[i]) && jeff[i] > 0, eachindex(A))

    # correlations (only for positive quantities in logs)
    # For B_excess we correlate using B_plus = max(B,0) to avoid nonsense.
    Bplus_old = max.(B[mB], 0.0)
    Bok = Bplus_old .> 0
    ρ_old_B = (count(Bok) >= 10) ? cor(log.(Bplus_old[Bok]), log.(old[mB][Bok])) : NaN

    Bplus_jeff = max.(B[mJ], 0.0)
    Bok2 = Bplus_jeff .> 0
    ρ_jeff_B = (count(Bok2) >= 10) ? cor(log.(Bplus_jeff[Bok2]), log.(jeff[mJ][Bok2])) : NaN

    ρ_old_A = (count(mAold) >= 10) ? cor(log.(A[mAold]), log.(old[mAold])) : NaN
    ρ_jeff_A = (count(mAjeff) >= 10) ? cor(log.(A[mAjeff]), log.(jeff[mAjeff])) : NaN

    @info "cor(log B_plus, log old_peakΔG)  = $ρ_old_B"
    @info "cor(log B_plus, log jeff_peakS)  = $ρ_jeff_B"
    @info "cor(log A_excess, log old_peakΔG)= $ρ_old_A"
    @info "cor(log A_excess, log jeff_peakS)= $ρ_jeff_A"

    # summaries per η
    η_grid = sort(unique(res.params.η_grid))

    function iqr_stats(v)
        v = filter(isfinite, v)
        isempty(v) && return (med=NaN, q25=NaN, q75=NaN)
        return (med=median(v), q25=quantile(v, 0.25), q75=quantile(v, 0.75))
    end

    Bmed = Float64[]; Bq25 = Float64[]; Bq75 = Float64[]
    Amed = Float64[]; Aq25 = Float64[]; Aq75 = Float64[]
    Pb   = Float64[]

    for η0 in η_grid
        idx = findall(i -> isfinite(η[i]) && η[i] == η0, eachindex(η))

        sb = iqr_stats(B[idx])
        push!(Bmed, sb.med); push!(Bq25, sb.q25); push!(Bq75, sb.q75)

        sa = iqr_stats(A[idx])
        push!(Amed, sa.med); push!(Aq25, sa.q25); push!(Aq75, sa.q75)

        f = flag[idx]
        f2 = filter(x -> x == 0 || x == 1, f)
        push!(Pb, isempty(f2) ? NaN : mean(f2))
    end

    fig = Figure(size=figsize)

    # (1) SIGNED B_excess vs η (linear)
    ax1 = Axis(fig[1,1], xlabel="η", ylabel="B_excess (signed, rmed units)",
               title="B_excess = max_mid Δrmed - late level (relative time τ)")
    lines!(ax1, η_grid, Bmed, linewidth=3)
    scatter!(ax1, η_grid, Bmed, markersize=10)
    for i in eachindex(η_grid)
        if isfinite(Bq25[i]) && isfinite(Bq75[i])
            lines!(ax1, [η_grid[i], η_grid[i]], [Bq25[i], Bq75[i]], linewidth=4)
        end
    end
    hlines!(ax1, [0.0], linestyle=:dash)

    # (2) A_excess vs η (log)
    ax2 = Axis(fig[1,2], xlabel="η", ylabel="A_excess", title="A_excess (area over log τ)", yscale=log10)
    lines!(ax2, η_grid, Amed, linewidth=3)
    scatter!(ax2, η_grid, Amed, markersize=10)
    for i in eachindex(η_grid)
        if isfinite(Aq25[i]) && isfinite(Aq75[i]) && Aq25[i] > 0 && Aq75[i] > 0
            lines!(ax2, [η_grid[i], η_grid[i]], [Aq25[i], Aq75[i]], linewidth=4)
        end
    end

    # (3) bump probability vs η
    ax3 = Axis(
        fig[1,3], xlabel="η", ylabel="P(bump)",
        title="P(max_mid > (1+δ)*L)",
        # ylimits=(0,1)
    )
    lines!(ax3, η_grid, Pb, linewidth=3)
    scatter!(ax3, η_grid, Pb, markersize=10)
    ylims!(ax3, (0,1))

    # (4) old predictor vs A_excess (more stable than B_excess)
    ax4 = Axis(fig[2,1], xlabel="A_excess", ylabel="old_peakΔG", title="Old predictor vs A_excess",
               xscale=log10, yscale=log10)
    scatter!(ax4, A[mAold], old[mAold], markersize=4)

    # (5) jeff predictor vs A_excess
    ax5 = Axis(fig[2,2], xlabel="A_excess", ylabel="jeff_peakS", title="Jeff predictor vs A_excess",
               xscale=log10, yscale=log10)
    scatter!(ax5, A[mAjeff], jeff[mAjeff], markersize=4)

    # (6) old vs jeff predictors
    mBoth = map(i -> isfinite(old[i]) && old[i] > 0 && isfinite(jeff[i]) && jeff[i] > 0, eachindex(old))
    ax6 = Axis(fig[2,3], xlabel="old_peakΔG", ylabel="jeff_peakS", title="Old vs Jeff predictors",
               xscale=log10, yscale=log10)
    scatter!(ax6, old[mBoth], jeff[mBoth], markersize=4)

    display(fig)
    return fig
end

# ------------------------------------------------------------------------------
# MAIN (edit parameters here)
# ------------------------------------------------------------------------------
tvals = collect(10 .^ range(log10(0.01), log10(100.0); length=60))
ωvals = collect(10 .^ range(log10(1e-4), log10(1e4); length=40))

res = run_bump_regime_pipeline(
    S=120,
    η_grid=collect(range(0.0, 1.0; length=7)),
    base_reps=6,
    P_reps=60,
    seed=1234,
    u_mean=1.0,
    u_cv=0.5,
    p=0.05,
    σ=1.0,
    margin=1e-3,
    R_target=1.0,          # try 0.5, 1.0, 2.0 to see robustness
    ε0=0.05,               # try 0.02–0.08; too large increases rejection
    tvals=tvals,
    ωvals=ωvals,
    τ_mid=(0.3, 3.0),
    τ_late_min=10.0,
    bump_delta=0.2
)

analyze_and_plot_bump(res)