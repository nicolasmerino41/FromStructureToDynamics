using Random
using LinearAlgebra
using Statistics
using Distributions
using CairoMakie

# -----------------------------
# Core: median return rate rmed(t)
# -----------------------------
function median_return_rate(J::AbstractMatrix, u::AbstractVector; t::Real=0.01, perturbation::Symbol=:biomass)
    S = size(J,1)
    if S == 0 || any(!isfinite, J)
        return NaN
    end

    E = exp(t*J)
    if any(!isfinite, E)
        return NaN
    end

    if perturbation === :uniform
        T = tr(E * transpose(E))
        if !isfinite(T) || T <= 0
            return NaN
        end
        num = log(T)
        den = log(S)
    elseif perturbation === :biomass
        @assert u !== nothing
        w = u .^ 2
        C = Diagonal(w)
        T = tr(E * C * transpose(E))
        if !isfinite(T) || T <= 0
            return NaN
        end
        num = log(T)
        den = log(sum(w))
    else
        error("Unknown perturbation model: $perturbation")
    end

    r = -(num - den) / (2*t)
    return isfinite(r) ? r : NaN
end

# -----------------------------
# t95 from rmed(t)
# -----------------------------
function t95_from_rmed(tvals::AbstractVector, rmed::AbstractVector; target::Real=0.05)
    @assert length(tvals) == length(rmed)
    @assert target > 0 && target < 1

    n = length(tvals)
    idx = 0
    y_prev = NaN
    t_prev = NaN

    for i in 1:n
        ti = float(tvals[i])
        ri = float(rmed[i])
        if !isfinite(ti) || !isfinite(ri) || ti <= 0
            continue
        end
        yi = exp(-ri * ti)
        if !isfinite(yi)
            continue
        end
        if yi <= target
            idx = i
            break
        end
        y_prev = yi
        t_prev = ti
    end

    idx == 0 && return Inf

    ti = float(tvals[idx])
    ri = float(rmed[idx])
    yi = exp(-ri * ti)

    if !isfinite(y_prev) || !isfinite(t_prev)
        return ti
    end
    if yi == y_prev || yi <= 0 || y_prev <= 0
        return ti
    end

    ℓ1 = log(y_prev)
    ℓ2 = log(yi)
    ℓt = log(float(target))
    if ℓ2 == ℓ1
        return ti
    end
    return t_prev + (ℓt - ℓ1) * (ti - t_prev) / (ℓ2 - ℓ1)
end

# -----------------------------
# Community generation
# -----------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

function random_interaction_matrix(S::Int, connectance::Real; σ::Real=1.0, rng=Random.default_rng())
    A = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        if i != j && rand(rng) < connectance
            A[i,j] = rand(rng, Normal(0, σ))
        end
    end
    return A
end

jacobian(A,u) = Diagonal(u) * (A - I)

"""
Permute off-diagonal entries of J (including zeros), keeping diagonal fixed.
This is your "rewire off-diagonals only" move.
"""
function reshuffle_offdiagonal(J::AbstractMatrix; rng=Random.default_rng())
    S = size(J, 1)
    J2 = copy(Matrix(J))

    vals = Float64[]
    idxs = Tuple{Int,Int}[]
    for i in 1:S, j in 1:S
        if i != j
            push!(vals, J2[i,j])
            push!(idxs, (i,j))
        end
    end

    perm = randperm(rng, length(vals))
    for k in 1:length(vals)
        (i,j) = idxs[k]
        J2[i,j] = vals[perm[k]]
    end
    return J2
end

# -----------------------------
# κ metrics (per-mode) via left/right eigenvectors
# κ_i = ||x_i|| * ||y_i|| with yᵢ' xᵢ = 1 (enforced by y from inv(V)')
# -----------------------------
function mode_kappas(J::AbstractMatrix)
    S = size(J,1)
    try
        F = eigen(J)          # right eigenvectors
        V = F.vectors
        Vinv = inv(V)
        Y = Vinv'             # columns are left eigenvectors with Y'V = I
        κ = [norm(V[:,i]) * norm(Y[:,i]) for i in 1:S]
        return (κ, S)
    catch
        return fill(NaN, S)
    end
end

function kappa_mean_max_sum(κ::AbstractVector)
    vals = filter(x -> isfinite(x) && x > 0, κ)
    isempty(vals) && return (NaN, NaN)
    return (mean(vals), maximum(vals), sum(vals))
end

# -----------------------------
# Transient energy to infinity:
# T∞ = E[ ∫0∞ ||x(t)||^2 dt ] / E[||x0||^2]
# with x(t)=exp(J t) x0 and x0 ~ N(0, I) isotropic.
#
# Identity: X = ∫0∞ exp(J t) I exp(J' t) dt solves:
#   JX + XJ' = -I
# so T∞ = tr(X)/S.
# Requires J Hurwitz (stable); otherwise lyap may fail or return nonsense.
# -----------------------------
function transient_T_infty(J::AbstractMatrix)
    S = size(J,1)
    Q = Matrix{Float64}(I, S, S)
    try
        X = lyap(J, Q)            # solves JX + XJ' + Q = 0  => JX + XJ' = -Q
        T = tr(X) / S
        return (isfinite(T) && T > 0) ? T : NaN
    catch
        return NaN
    end
end

# -----------------------------
# Pipeline: compute rmed curves, t95, κ metrics, T∞, and compare original vs rewired
# -----------------------------
meanfinite(x) = (vals = filter(isfinite, x); isempty(vals) ? NaN : mean(vals))

function run_pipeline(;
    S::Int=120,
    connectance::Real=0.05,
    n::Int=50,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    σA::Real=0.25,
    seed::Int=1234,
    perturbation::Symbol=:biomass,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=30),
    check_for_resilience::Bool=true
)
    rng = MersenneTwister(seed)
    nt = length(tvals)

    rmed_orig = fill(NaN, n, nt)
    rmed_rew  = fill(NaN, n, nt)

    T_orig = fill(NaN, n)
    T_rew  = fill(NaN, n)

    kmean_orig = fill(NaN, n)
    kmax_orig  = fill(NaN, n)
    kmean_rew  = fill(NaN, n)
    kmax_rew   = fill(NaN, n)

    ksum_orig = fill(NaN, n)
    ksum_rew  = fill(NaN, n)

    t95_orig = fill(Inf, n)
    t95_rew  = fill(Inf, n)

    α_orig = fill(NaN, n)
    α_rew  = fill(NaN, n)

    G2_orig = fill(NaN, n)
    G2_rew  = fill(NaN, n)
    tG_orig = fill(NaN, n)
    tG_rew  = fill(NaN, n)

    R_orig = fill(NaN, n)
    R_rew  = fill(NaN, n)

    for k in 1:n
        # Base community
        A = random_interaction_matrix(S, connectance; σ=σA, rng=rng)
        
        # PPM
        # b = PPMBuilder()
        # set!(b; S=S, B=0.25S, L=connectance*S^2, T=0.01, η=0.2)
        # net = build(b)
        # A = net.A
        # # --- turn topology into signed interaction matrix W ---
        # A = build_interaction_matrix(A;
        #     mag_abs=σA,
        #     mag_cv=σA,
        #     corr_aij_aji=0.0,
        #     rng=rng
        # )

        # Strongly non-normal
        # A = A_feedforward(S; p = 0.05)
        # A = A_jordan(S; β=0.15)
        # A = A_illconditioned(S)
        
        # A = random_trophic_matrix_degcv(S, connectance; σ=σA, rng=rng)
        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
        # A, _ = build_A_diverse(u; rng=rng)

        J = jacobian(A, u)
        # J, c, α  = shift_to_margin(J)
        α_o = resilience(J)
        if check_for_resilience
            if α_o > 0
                @info "Resilience = $(resilience(J)) > 0, skipping..."
                continue
            else
                @info "Resilience = $(resilience(J))"
            end
        end
        # Rewire off-diagonals only (u fixed)
        A_rew = reshuffle_offdiagonal(A; rng=rng)
        # A_rew = random_interaction_matrix(S, connectance; σ=σA, rng=rng)
        # A_rew = partial_reshuffle_offdiagonal(A; frac=0.1)
        # A_rew = reshuffle_trophic_pairs(A; rng=rng)
        Jrew = jacobian(A_rew, u)
        # Jrew, c_rew, α_rews = shift_to_margin(Jrew)
        α_r = resilience(Jrew)
        if check_for_resilience
            if α_r > 0
                @info "Resilience rew = $(resilience(Jrew)) > 0, skipping..."
                continue
            else
                @info "Resilience rew = $(resilience(Jrew))"
            end
        end

        # κ metrics
        κo, _ = mode_kappas(J)
        κr, _ = mode_kappas(Jrew)
        km_o, kmax_o, ksum_o  = kappa_mean_max_sum(κo)
        kmean_orig[k], kmax_orig[k], ksum_orig[k] = km_o, kmax_o, ksum_o
        
        km_r, kmax_r, ksum_r  = kappa_mean_max_sum(κr)
        kmean_rew[k], kmax_rew[k], ksum_rew[k] = km_r, kmax_r, ksum_r

        # T∞ metrics
        T_orig[k] = transient_T_infty(J)
        T_rew[k]  = transient_T_infty(Jrew)

        # α metrics
        α_orig[k] = α_o
        α_rew[k]  = α_r

        G2_orig[k], tG_orig[k] = peak_singular_gain(J, tvals)
        G2_rew[k],  tG_rew[k]  = peak_singular_gain(Jrew, tvals)

        R_orig[k] = numerical_abscissa(J)
        R_rew[k]  = numerical_abscissa(Jrew)

        # rmed curves
        for (ti, t) in enumerate(tvals)
            rmed_orig[k, ti] = median_return_rate(J, u; t=t, perturbation=perturbation)
            rmed_rew[k,  ti] = median_return_rate(Jrew, u; t=t, perturbation=perturbation)
        end
        t0 = tvals[round(Int, length(tvals) ÷ 2)]
        ru = median_return_rate(J, u; t=t0, perturbation=:uniform)
        rb = median_return_rate(J, u; t=t0, perturbation=:biomass)
        @info "check rmed at t=$t0: uniform=$ru biomass=$rb diff=$(ru-rb)"

        # t95 per community
        t95_orig[k] = t95_from_rmed(tvals, vec(rmed_orig[k, :]))
        t95_rew[k]  = t95_from_rmed(tvals, vec(rmed_rew[k,  :]))
    end

    # Differences you want to relate
    ΔT    = abs.(T_orig .- T_rew)./T_orig
    Δkmean = kmean_orig # abs.(kmean_orig .- kmean_rew)
    Δkmax  = kmax_orig # abs.(kmax_orig  .- kmax_rew)
    Δksum  = ksum_orig # abs.(ksum_orig  .- ksum_rew)
    Δα     = abs.(α_orig .- α_rew)
    ΔG2 = abs.(G2_orig .- G2_rew) ./ G2_orig
    ΔR = abs.(R_orig .- R_rew)

    # Summary curves
    mean_orig = [meanfinite(view(rmed_orig, :, ti)) for ti in 1:nt]
    mean_rew  = [meanfinite(view(rmed_rew,  :, ti)) for ti in 1:nt]
    mean_delta = [meanfinite(view(rmed_orig .- rmed_rew, :, ti)) for ti in 1:nt]

    return (
        tvals=tvals,
        rmed_orig=rmed_orig, rmed_rew=rmed_rew,
        mean_orig=mean_orig, mean_rew=mean_rew, mean_delta=mean_delta,
        T_orig=T_orig, T_rew=T_rew, ΔT=ΔT,
        kmean_orig=kmean_orig, kmean_rew=kmean_rew, Δkmean=Δkmean,
        kmax_orig=kmax_orig,   kmax_rew=kmax_rew,   Δkmax=Δkmax,
        ksum_orig=ksum_orig,   ksum_rew=ksum_rew,   Δksum=Δksum,
        Δα=Δα, α_orig=α_orig, α_rew=α_rew,
        G2_orig=G2_orig, G2_rew=G2_rew, ΔG2=ΔG2,
        R_orig=R_orig, R_rew=R_rew, ΔR=ΔR,
        t95_orig=t95_orig, t95_rew=t95_rew
    )
end

# -----------------------------
# Plotting & quick association checks
# -----------------------------
function analyze_and_plot(results)
    ΔT = results.ΔT
    Δkmean = results.Δkmean
    Δkmax  = results.Δkmax

    # Finite-only vectors for correlations
    mask_mean = map(i -> isfinite(ΔT[i]) && isfinite(Δkmean[i]) && ΔT[i] > 0 && Δkmean[i] > 0, eachindex(ΔT))
    mask_max  = map(i -> isfinite(ΔT[i]) && isfinite(Δkmax[i])  && ΔT[i] > 0 && Δkmax[i]  > 0, eachindex(ΔT))

    x1 = Δkmean[mask_mean]; y1 = ΔT[mask_mean]
    x2 = Δkmax[mask_max];   y2 = ΔT[mask_max]

    ρ_mean = (length(x1) >= 3) ? cor(log.(x1), log.(y1)) : NaN
    ρ_max  = (length(x2) >= 3) ? cor(log.(x2), log.(y2)) : NaN

    @info "log-log correlation cor(log Δkmean, log ΔT) = $ρ_mean  (N=$(length(x1)))"
    @info "log-log correlation cor(log Δkmax,  log ΔT) = $ρ_max   (N=$(length(x2)))"

    # Scatter plots (log scales)
    fig = Figure(size=(1200, 500))

    ax1 = Axis(
        fig[1,1],
        xscale=log10,
        yscale=log10,
        xlabel="mean_kappa_of_the_original_community", ylabel="ΔT∞ / T∞",
        title="kmean vs ΔT∞"
    )
    scatter!(ax1, x1, y1, markersize=10)

    ax2 = Axis(
        fig[1,2],
        xscale=log10,
        yscale=log10,
        xlabel="max_kappa_of_the_original_community", ylabel="ΔT∞ / T∞",
        title="kmax vs ΔT∞"
    )
    scatter!(ax2, x2, y2, markersize=10)

    # ax3 = Axis(
    #     fig[1,3],
    #     xscale=log10, yscale=log10,
    #     xlabel="Δksum", ylabel="ΔT∞",
    #     title="Association: Δksum vs ΔT∞"
    # )
    # scatter!(ax3, results.Δksum[mask_mean], y1, markersize=10)

    ax4 = Axis(
        fig[1,3],
        # xscale=log10,
        yscale=log10,
        xlabel="ΔResilience", ylabel="ΔT∞ / T∞",
        title="Δresilience vs ΔT∞"
    )
    scatter!(ax4, results.Δα[mask_mean], y1, markersize=10)

    ax5 = Axis(
        fig[2, 1],
        # xscale=log10,
        yscale=log10,
        xlabel="Δ max_t ||exp(J t)||_2^2", ylabel="ΔT∞ / T∞",
        title="Δpeak propagator vs ΔT∞"
    )
    scatter!(ax5, results.ΔG2[mask_mean], y1, markersize=10)

    ax6 = Axis(
        fig[2, 2],
        # xscale=log10,
        yscale=log10,
        xlabel="ΔReactivity", ylabel="ΔT∞ / T∞",
        title="ΔReactivity vs ΔT∞"
    )
    scatter!(ax6, results.ΔR[mask_mean], y1, markersize=10)

    display(fig)
end

# -----------------------------
# Main
# -----------------------------
tvals = 10 .^ range(log10(0.01), log10(100.0); length=30)

results = run_pipeline(
    S=120,
    connectance=0.1,
    n=100,
    u_mean=1.0,
    u_cv=0.5,
    σA=0.25,
    seed=1234,
    perturbation=:biomass,
    tvals=tvals,
    check_for_resilience=true
)

analyze_and_plot(results)