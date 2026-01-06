################################################################################
# STRUCTURAL SENSITIVITY FRAMEWORK (NO "BUMP" REQUIRED)
#
# Goal:
#   Show that systems with relatively larger *mid-frequency* sensitivity (ρ)
#   tend to have relatively larger *intermediate-time* structural error (MI),
#   with time standardized by τ = t / t95 (t95 computed from rmed curve).
#
# Key outputs per base system:
#   - MI      = Err_mid / Err_late   (time-domain mid dominance)
#   - Err_tot = ∫ Δ(τ) d log τ       (time-domain total error)
#   - ρ_typ   = S_mid / S_low from typical spectrum S_typ(ω)
#   - ρ_wc    = S_mid / S_low from worst-case spectrum S_wc(ω)
#   - S_tot   = ∫ S(ω) dω            (frequency "energy-like" total)
#
# Networks:
#   "Trophicness" enforced by a latent hierarchy h_i; edges are biased to align
#   with increasing h, but reciprocity + misaligned edges are allowed.
#   We also randomize several generator parameters per base to ensure a
#   heterogeneous set of networks.
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# If you see oversubscription (BLAS + Julia threads), you can do:
# BLAS.set_num_threads(1)

# ---------------------------
# Small utilities
# ---------------------------
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

function trapz_logx(x::AbstractVector, y::AbstractVector)
    # integrates y(x) d log x  = ∫ y(x) (dx/x)
    @assert length(x) == length(y)
    n = length(x)
    n < 2 && return NaN
    s = 0.0
    for i in 1:(n-1)
        x1, x2 = float(x[i]), float(x[i+1])
        y1, y2 = float(y[i]), float(y[i+1])
        if isfinite(x1) && isfinite(x2) && x1 > 0 && x2 > 0 && isfinite(y1) && isfinite(y2)
            dx_over_x = log(x2) - log(x1)
            s += 0.5 * (y1 + y2) * dx_over_x
        end
    end
    return s
end

spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigvals(Matrix(J))))

# ---------------------------
# Biomasses u
# ---------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# ---------------------------
# Trophic network generator (heterogeneous)
# ---------------------------
"""
Generate an off-diagonal interaction matrix O (diag=0) with "trophicness":
a latent hierarchy h_i ~ Uniform(0,1) biases edge direction to go from low h to high h.

Parameters:
- connectance: probability an unordered pair has at least one directed link
- trophic_align: probability a single directed link aligns with hierarchy (in [0,1])
- reciprocity: probability an edge is bidirectional when a pair is selected
- σ: weight scale for iid Normal weights on each realized directed edge

Return:
- O::Matrix{Float64} with O[i,i]=0
"""
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

    h = rand(rng, S)                 # latent trophic "height"
    O = zeros(Float64, S, S)

    for i in 1:S-1, j in i+1:S
        rand(rng) < connectance || continue

        if rand(rng) < reciprocity
            # bidirectional
            O[i,j] = randn(rng) * σ
            O[j,i] = randn(rng) * σ
        else
            # single directed edge, aligned with hierarchy with prob trophic_align
            low, high = (h[i] <= h[j]) ? (i, j) : (j, i)
            aligned = rand(rng) < trophic_align
            if aligned
                O[low, high] = randn(rng) * σ
            else
                O[high, low] = randn(rng) * σ
            end
        end
    end

    # ensure diag = 0
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

# ---------------------------
# Base matrix construction and stability control
# Abar = -I + s*O
# J = diag(u)*Abar
# We choose s by bisection to hit α(J) ≈ target_alpha (<0).
# ---------------------------
function find_scale_to_target_alpha(O::Matrix{Float64}, u::Vector{Float64};
    target_alpha::Float64 = -0.05,
    max_grow::Int=40,
    max_iter::Int=60
)
    @assert target_alpha < 0

    Du = Diagonal(u)

    # α(s) at s=0
    J0 = -Du
    α0 = spectral_abscissa(J0)
    # If already less stable than target (α0 > target), can't fix by scaling O reliably.
    # In practice α0 is usually more negative than target.
    if !(isfinite(α0))
        return NaN
    end

    # Find s_hi such that α(s_hi) >= target_alpha (i.e., less negative / closer to 0)
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

    # s_lo=0 always gives α_lo = α0 (likely < target_alpha)
    s_lo = 0.0
    α_lo = α0
    if α_lo > target_alpha
        # already too close to 0 at s=0; just return 0
        return 0.0
    end

    # Bisection for α(s)=target_alpha
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

# ---------------------------
# Biomass-weighted rmed(t)
# ---------------------------
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

# User-provided style: t95 from rmed curve (implicit)
function t95_from_rmed_curve(t_vals::AbstractVector, rmed::AbstractVector; target::Real=0.05)
    @assert length(t_vals) == length(rmed)
    y = @. exp(-rmed * t_vals)                 # predicted remaining fraction
    idx = findfirst(y .<= target)
    isnothing(idx) && return Inf
    idx == 1 && return float(t_vals[1])

    # linear interpolation between grid points
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

# ---------------------------
# Uncertainty directions P (noise uncertainty)
# ---------------------------
"""
Gaussian noise on off-diagonals (optionally sparse), then Frobenius-normalize.
diag(P)=0 and ||P||_F=1.
"""
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

# ---------------------------
# Generalized resolvent and sensitivity spectra
# R(ω) = (i ω T - Abar)^(-1), T=diag(1/u)
# Typical: mean_P S(ω;P)
# Worst-case: S_wc(ω) = eps^2/sum(u^2) * ||R||_2^2 * ||R diag(u)||_2^2
# ---------------------------
function sensitivity_spectra_typ_and_wc(Abar::Matrix{Float64}, u::Vector{Float64},
    eps::Float64, ωvals::Vector{Float64}, Pdirs::Vector{Matrix{Float64}}
)
    S = size(Abar,1)
    T = Diagonal(1.0 ./ u)
    U = Matrix{ComplexF64}(Diagonal(u))
    denom = sum(u.^2)

    nω = length(ωvals)
    nP = length(Pdirs)

    Smean = fill(NaN, nω)
    Sq90  = fill(NaN, nω)
    Swc   = fill(NaN, nω)

    for (k, ω) in enumerate(ωvals)
        ω = float(ω)
        Mω = Matrix{ComplexF64}(im*ω*T - Abar)

        # factor once
        F = lu(Mω)

        # Y = R diag(u)
        Y = F \ U

        # typical S over P
        sval = Vector{Float64}(undef, nP)
        for (pidx, P) in enumerate(Pdirs)
            Z = Matrix{ComplexF64}(P) * Y
            X = F \ Z                       # X = R P R diag(u)
            v = (eps^2) * (norm(X)^2) / denom
            sval[pidx] = (isfinite(v) && v >= 0) ? v : NaN
        end
        good = filter(isfinite, sval)
        if !isempty(good)
            Smean[k] = mean(good)
            Sq90[k]  = quantile(good, 0.90)
        end

        # worst-case spectrum
        # ||R||_2 = 1 / σ_min(Mω)
        svalsM = svdvals(Mω)
        σmin = svalsM[end]
        Rop = (isfinite(σmin) && σmin > 0) ? 1.0/σmin : NaN

        Yop = opnorm(Y)     # ||R diag(u)||_2

        vwc = (eps^2) * (Rop^2) * (Yop^2) / denom
        Swc[k] = (isfinite(vwc) && vwc >= 0) ? vwc : NaN
    end

    return (Smean=Smean, Sq90=Sq90, Swc=Swc)
end

# Rule 1 bands: ωL=cL/t95, ωH=cH/t95
function band_integrals(ωvals::Vector{Float64}, Sω::Vector{Float64}, ωL::Float64, ωH::Float64)
    @assert length(ωvals) == length(Sω)
    if !(isfinite(ωL) && isfinite(ωH) && ωL > 0 && ωH > ωL)
        return (Slow=NaN, Smid=NaN, ρ=NaN, Stot=NaN)
    end

    # total integral
    good_tot = findall(i -> isfinite(Sω[i]) && isfinite(ωvals[i]) && ωvals[i] > 0, eachindex(Sω))
    if length(good_tot) < 2
        return (Slow=NaN, Smid=NaN, ρ=NaN, Stot=NaN)
    end
    ωt = ωvals[good_tot]; St = Sω[good_tot]
    Stot = trapz(ωt, St)

    # low band
    idx_low = findall(i -> isfinite(Sω[i]) && ωvals[i] > 0 && ωvals[i] <= ωL, eachindex(Sω))
    Slow = (length(idx_low) >= 2) ? trapz(ωvals[idx_low], Sω[idx_low]) : NaN

    # mid band
    idx_mid = findall(i -> isfinite(Sω[i]) && ωvals[i] > ωL && ωvals[i] <= ωH, eachindex(Sω))
    Smid = (length(idx_mid) >= 2) ? trapz(ωvals[idx_mid], Sω[idx_mid]) : NaN

    ρ = (isfinite(Slow) && Slow > 0 && isfinite(Smid)) ? (Smid / Slow) : NaN
    return (Slow=Slow, Smid=Smid, ρ=ρ, Stot=Stot)
end

# ---------------------------
# Time-domain regime metrics on τ grid
# MI = Err_mid / Err_late; Err_tot = ∫ Δ(τ) d log τ
# ---------------------------
function time_regime_metrics(tvals::Vector{Float64}, rbase::Vector{Float64}, rpert::Vector{Float64};
    τ_mid=(0.3, 3.0),
    τ_late=(3.0, 30.0),
    t95_target=0.05
)
    Δt = delta_curve(rbase, rpert)
    t95 = t95_from_rmed_curve(tvals, rbase; target=t95_target)
    if !(isfinite(t95) && t95 > 0)
        return nothing
    end
    τ = tvals ./ t95

    # window means
    mid_idx  = findall(i -> isfinite(τ[i]) && τ[i] >= τ_mid[1] && τ[i] <= τ_mid[2] && isfinite(Δt[i]), eachindex(τ))
    late_idx = findall(i -> isfinite(τ[i]) && τ[i] >= τ_late[1] && τ[i] <= τ_late[2] && isfinite(Δt[i]), eachindex(τ))

    Err_mid  = (length(mid_idx)  >= 2) ? mean(Δt[mid_idx])  : NaN
    Err_late = (length(late_idx) >= 2) ? mean(Δt[late_idx]) : NaN

    MI = (isfinite(Err_mid) && isfinite(Err_late) && Err_late > 0) ? (Err_mid / Err_late) : NaN

    # total area on log τ
    good = findall(i -> isfinite(τ[i]) && τ[i] > 0 && isfinite(Δt[i]), eachindex(τ))
    Err_tot = (length(good) >= 2) ? trapz_logx(τ[good], Δt[good]) : NaN

    return (t95=t95, τ=τ, Δ=Δt, Err_mid=Err_mid, Err_late=Err_late, MI=MI, Err_tot=Err_tot)
end

# ---------------------------
# One base system container
# ---------------------------
struct BaseSys
    u::Vector{Float64}
    Abar::Matrix{Float64}
    rbase::Vector{Float64}
    t95::Float64
    eps::Float64
end

# ---------------------------
# Build heterogeneous trophic bases
# ---------------------------
function build_trophic_bases(; S::Int, base_reps::Int, seed::Int,
    u_mean::Float64=1.0, u_cv::Float64=0.5,
    # randomize generator params to ensure heterogeneity
    connectance_rng=(0.03, 0.12),
    trophic_align_rng=(0.55, 0.98),
    reciprocity_rng=(0.00, 0.15),
    σ_rng=(0.3, 1.5),
    target_alpha::Float64=-0.05,
    eps_rel::Float64=0.20,
    tvals::Vector{Float64}=10 .^ range(log10(0.01), log10(100.0); length=60)
)
    tvals = collect(float.(tvals))
    bases = BaseSys[]
    for b in 1:base_reps
        rng = MersenneTwister(seed + 10007*b)

        u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))

        # randomize trophic generator parameters
        c = rand(rng, Uniform(connectance_rng[1], connectance_rng[2]))
        γ = rand(rng, Uniform(trophic_align_rng[1], trophic_align_rng[2]))
        r = rand(rng, Uniform(reciprocity_rng[1], reciprocity_rng[2]))
        σ = rand(rng, Uniform(σ_rng[1], σ_rng[2]))

        O = trophic_O(S; connectance=c, trophic_align=γ, reciprocity=r, σ=σ, rng=rng)
        normalize_offdiag!(O) || continue

        s = find_scale_to_target_alpha(O, u; target_alpha=target_alpha)
        isfinite(s) || continue

        Abar = -Matrix{Float64}(I, S, S) + s * O
        J = Diagonal(u) * Abar

        rbase = rmed_curve(J, u, tvals)
        t95 = t95_from_rmed_curve(tvals, rbase; target=0.05)
        (isfinite(t95) && t95 > 0) || continue

        # eps relative to offdiag(Abar) Fro norm; with O Fro-normalized, offdiag norm ≈ s
        eps = eps_rel * norm(Abar + Matrix{Float64}(I, S, S))  # remove diag -1 => offdiag
        (isfinite(eps) && eps > 0) || continue

        push!(bases, BaseSys(u, Abar, rbase, t95, eps))
    end
    return (bases=bases, tvals=tvals)
end

# ---------------------------
# Evaluate one base: sample P ensemble, compute time metrics + spectra
# ---------------------------
function eval_base_system(base::BaseSys, tvals::Vector{Float64}, ωvals::Vector{Float64};
    P_reps::Int=30,
    P_sparsity::Float64=1.0,
    margin::Float64=1e-3,
    τ_mid=(0.3, 3.0),
    τ_late=(3.0, 30.0),
    cL::Float64=0.3,
    cH::Float64=3.0,
    seed::Int=1
)
    S = length(base.u)
    rng = MersenneTwister(seed)

    # sample Pdirs and keep only those that preserve stability
    Pdirs = Matrix{Float64}[]
    MI_list = Float64[]
    Errtot_list = Float64[]

    # for optional example curves
    τ_ref = nothing
    Δ_curves = Vector{Vector{Float64}}()

    Du = Diagonal(base.u)
    Jbase = Du * base.Abar

    for k in 1:P_reps
        P = sample_noise_Pdir(S; sparsity_p=P_sparsity, rng=rng)
        P === nothing && continue

        Abarp = base.Abar + base.eps * P
        Jp = Du * Abarp

        αp = spectral_abscissa(Jp)
        (isfinite(αp) && αp < -margin) || continue

        rpert = rmed_curve(Jp, base.u, tvals)
        tm = time_regime_metrics(tvals, base.rbase, rpert; τ_mid=τ_mid, τ_late=τ_late)
        tm === nothing && continue

        push!(Pdirs, P)
        push!(MI_list, tm.MI)
        push!(Errtot_list, tm.Err_tot)

        if τ_ref === nothing
            τ_ref = tm.τ
        end
        push!(Δ_curves, tm.Δ)
    end

    if length(Pdirs) < 5
        return nothing
    end

    # frequency sensitivity spectra on the accepted P ensemble
    sp = sensitivity_spectra_typ_and_wc(base.Abar, base.u, base.eps, ωvals, Pdirs)

    # Rule 1 frequency bands from t95
    ωL = cL / base.t95
    ωH = cH / base.t95

    b_typ = band_integrals(ωvals, sp.Smean, ωL, ωH)
    b_wc  = band_integrals(ωvals, sp.Swc,   ωL, ωH)

    return (
        nP=length(Pdirs),
        MI=meanfinite(MI_list),
        MI_med=medianfinite(MI_list),
        Err_tot=meanfinite(Errtot_list),

        ρ_typ=b_typ.ρ,
        ρ_wc=b_wc.ρ,
        Stot_typ=b_typ.Stot,
        Stot_wc=b_wc.Stot,

        ωL=ωL, ωH=ωH,
        spectra=sp,

        τ=τ_ref,
        Δcurves=Δ_curves
    )
end

# ---------------------------
# Run full experiment across many heterogeneous trophic bases
# ---------------------------
function run_experiment(; S::Int=60, base_reps::Int=50, P_reps::Int=25,
    seed::Int=1234,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=60),
    ωvals = 10 .^ range(log10(1e-3), log10(1e3); length=60),
    # stability targeting and uncertainty scale
    target_alpha::Float64=-0.05,
    eps_rel::Float64=0.20,
    # time windows (in τ)
    τ_mid=(0.3, 3.0),
    τ_late=(3.0, 30.0),
    # rule-1 constants
    cL::Float64=0.3,
    cH::Float64=3.0
)
    ωvals = collect(float.(ωvals))

    built = build_trophic_bases(
        S=S, base_reps=base_reps, seed=seed,
        target_alpha=target_alpha, eps_rel=eps_rel,
        tvals=collect(float.(tvals))
    )
    bases = built.bases
    tvals = built.tvals

    @info "Built $(length(bases)) stable trophic bases (out of $base_reps attempts)."

    # store per-base outputs
    MI      = Float64[]
    MI_med  = Float64[]
    Err_tot = Float64[]

    ρ_typ   = Float64[]
    ρ_wc    = Float64[]
    St_typ  = Float64[]
    St_wc   = Float64[]

    ωL_vec  = Float64[]
    ωH_vec  = Float64[]

    # store spectra for one example base later
    example = nothing

    for (i, base) in enumerate(bases)
        out = eval_base_system(base, tvals, ωvals;
            P_reps=P_reps,
            τ_mid=τ_mid, τ_late=τ_late,
            cL=cL, cH=cH,
            seed=seed + 900_000*i
        )
        out === nothing && continue

        push!(MI, out.MI)
        push!(MI_med, out.MI_med)
        push!(Err_tot, out.Err_tot)

        push!(ρ_typ, out.ρ_typ)
        push!(ρ_wc, out.ρ_wc)
        push!(St_typ, out.Stot_typ)
        push!(St_wc, out.Stot_wc)

        push!(ωL_vec, out.ωL)
        push!(ωH_vec, out.ωH)

        # pick an example with finite ρ_typ
        if example === nothing && isfinite(out.ρ_typ) && isfinite(out.ρ_wc)
            example = (out=out, ωvals=ωvals)
        end
    end

    return (
        tvals=tvals, ωvals=ωvals,
        MI=MI, MI_med=MI_med, Err_tot=Err_tot,
        ρ_typ=ρ_typ, ρ_wc=ρ_wc,
        St_typ=St_typ, St_wc=St_wc,
        ωL=ωL_vec, ωH=ωH_vec,
        example=example
    )
end

# ---------------------------
# Plotting
# ---------------------------
function plot_results(res; figsize=(1400, 900))
    fig = Figure(size=figsize)

    # --- Example spectra ---
    ex = res.example
    if ex !== nothing
        out = ex.out
        ωvals = ex.ωvals
        ax0 = Axis(fig[1,1];
            xscale=log10, yscale=log10,
            xlabel="ω", ylabel="S(ω)",
            title="Example sensitivity spectra (typical mean + q90, worst-case)"
        )
        lines!(ax0, ωvals, out.spectra.Smean, linewidth=3)
        lines!(ax0, ωvals, out.spectra.Sq90,  linewidth=3, linestyle=:dash)
        lines!(ax0, ωvals, out.spectra.Swc,   linewidth=4)

        vlines!(ax0, [out.ωL, out.ωH], linestyle=:dot, linewidth=2)
    else
        Axis(fig[1,1]; title="No example spectra available (too many rejects?)")
    end

    # --- MI vs ρ (typical & worst-case) ---
    ax1 = Axis(fig[1,2];
        xlabel="ρ_typ = S_mid/S_low",
        ylabel="MI = Err_mid/Err_late",
        title="Time regime vs frequency regime (typical)",
    )
    mask1 = findall(i -> isfinite(res.ρ_typ[i]) && isfinite(res.MI[i]) && res.ρ_typ[i] > 0 && res.MI[i] > 0, eachindex(res.MI))
    scatter!(ax1, res.ρ_typ[mask1], res.MI[mask1], markersize=7)
    if length(mask1) >= 6
        ρ = cor(log.(res.ρ_typ[mask1]), log.(res.MI[mask1]))
        text!(ax1, 0.05, 0.95, text="cor(log ρ, log MI) = $(round(ρ,digits=3))",
              space=:relative, align=(:left,:top))
    end

    ax2 = Axis(fig[2,2];
        xlabel="ρ_wc = S_mid/S_low",
        ylabel="MI = Err_mid/Err_late",
        title="Time regime vs frequency regime (worst-case)",
    )
    mask2 = findall(i -> isfinite(res.ρ_wc[i]) && isfinite(res.MI[i]) && res.ρ_wc[i] > 0 && res.MI[i] > 0, eachindex(res.MI))
    scatter!(ax2, res.ρ_wc[mask2], res.MI[mask2], markersize=7)
    if length(mask2) >= 6
        ρ = cor(log.(res.ρ_wc[mask2]), log.(res.MI[mask2]))
        text!(ax2, 0.05, 0.95, text="cor(log ρ, log MI) = $(round(ρ,digits=3))",
              space=:relative, align=(:left,:top))
    end

    # --- Total time error vs total sensitivity ---
    ax3 = Axis(fig[2,1];
        xscale=log10, yscale=log10,
        xlabel="∫ S_typ(ω) dω",
        ylabel="Err_tot = ∫ Δ(τ) d log τ",
        title="Energy-like link: total sensitivity vs total time error (typical)",
    )
    mask3 = findall(i -> isfinite(res.St_typ[i]) && res.St_typ[i] > 0 && isfinite(res.Err_tot[i]) && res.Err_tot[i] > 0, eachindex(res.Err_tot))
    scatter!(ax3, res.St_typ[mask3], res.Err_tot[mask3], markersize=7)
    if length(mask3) >= 6
        ρ = cor(log.(res.St_typ[mask3]), log.(res.Err_tot[mask3]))
        text!(ax3, 0.05, 0.95, text="cor(log Stot, log Err_tot) = $(round(ρ,digits=3))",
              space=:relative, align=(:left,:top))
    end

    display(fig)
end

# Optional: plot example Δ(τ) curves for low-ρ vs high-ρ bases
function plot_example_delta_curves(res; ncurves=20, figsize=(1200, 500))
    # We only stored Δ-curves for a single example base (to keep memory light).
    ex = res.example
    ex === nothing && return nothing
    out = ex.out
    τ = out.τ
    Δcurves = out.Δcurves

    fig = Figure(size=figsize)
    ax = Axis(fig[1,1];
        xscale=log10,
        xlabel="τ = t/t95",
        ylabel="Δ(τ) = |r_base - r_pert|",
        title="Example Δ(τ) curves (single base, multiple uncertainty draws)"
    )

    # plot a subset
    m = min(ncurves, length(Δcurves))
    for k in 1:m
        Δ = Δcurves[k]
        lines!(ax, τ, Δ, linewidth=2, transparency=true)
    end

    vlines!(ax, [0.3, 3.0, 30.0], linestyle=:dot, linewidth=2)
    display(fig)
end

# ---------------------------
# MAIN
# ---------------------------
tvals = 10 .^ range(log10(0.01), log10(100.0); length=60)
ωvals = 10 .^ range(log10(1e-3), log10(1e3); length=60)

res = run_experiment(
    S=120,                 # increase later (e.g., 120) once satisfied
    base_reps=60,          # heterogeneity comes from randomized trophic generator params
    P_reps=25,             # uncertainty ensemble per base
    seed=1234,
    tvals=tvals,
    ωvals=ωvals,
    target_alpha=-0.05,    # bring systems close to instability but stable
    eps_rel=0.20,          # uncertainty magnitude
    τ_mid=(0.3, 3.0),
    τ_late=(3.0, 30.0),
    cL=0.3, cH=3.0          # Rule-1 frequency bands: ωL=cL/t95, ωH=cH/t95
)

plot_results(res)
plot_example_delta_curves(res)
################################################################################
