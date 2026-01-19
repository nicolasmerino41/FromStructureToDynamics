################################################################################
################################################################################
# PHASE 4: ENERGY vs LEVERAGE DECOMPOSITION
#
# Goal:
#   For each community, decompose "where structural sensitivity lives" into:
#     - Energy availability across frequency: e(ω)
#     - Structural leverage across frequency: L(ω) = E_P |g(ω;P)|   (your Gabs)
#     - Fragility mass kernel:              m(ω) = L(ω) * e(ω)
#
#   Then quantify "early vs late" via frequency quantiles / centroids of:
#     e(ω), L(ω), m(ω), and interaction-induced energy redistribution:
#     Δe(ω) = e(ω) - e0(ω)  (baseline no-interaction reference)
#
#   Finally link these to structural knobs / metrics:
#     - generator knobs: connectance c, trophic_align gamma, reciprocity rr, sigma σ
#     - basic matrix metrics: K0 = ρ(A), ||A||F, etc.
#
# Output:
#   - Summary correlations/regressions (log space)
#   - Plots:
#       (1) omega_m50 vs omega_e50 and omega_L50
#       (2) energy-shift vs structure knobs
#       (3) leverage-location vs structure knobs
#       (4) classification scatter: energy-shift vs leverage-location colored by omega_m50
#       (5) example spectra: e,e0,Δe,L,m for representative systems
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# -----------------------------
# Utilities
# -----------------------------
isposfinite(x) = isfinite(x) && x > 0

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

function cumtrapz(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    n = length(x)
    c = zeros(Float64, n)
    n < 2 && return c
    for i in 2:n
        x1, x2 = float(x[i-1]), float(x[i])
        y1, y2 = float(y[i-1]), float(y[i])
        if isfinite(x1) && isfinite(x2) && isfinite(y1) && isfinite(y2)
            c[i] = c[i-1] + 0.5*(y1+y2)*(x2-x1)
        else
            c[i] = c[i-1]
        end
    end
    return c
end

# quantile location of a nonnegative "density" y over x (integral-based)
function ω_quantile(x::Vector{Float64}, y::Vector{Float64}, q::Float64)
    @assert 0.0 < q < 1.0
    idx = findall(i -> isposfinite(x[i]) && isfinite(y[i]) && y[i] >= 0, eachindex(y))
    length(idx) < 3 && return NaN
    xx = x[idx]; yy = y[idx]
    total = trapz(xx, yy)
    (isfinite(total) && total > 0) || return NaN
    c = cumtrapz(xx, yy)
    target = q * total
    j = findfirst(c .>= target)
    isnothing(j) && return NaN
    j == 1 && return xx[1]
    x1, x2 = xx[j-1], xx[j]
    c1, c2 = c[j-1], c[j]
    if c2 == c1
        return x2
    end
    return x1 + (target - c1) * (x2 - x1) / (c2 - c1)
end

# log-frequency centroid: exp( ∫ log(ω) y dω / ∫ y dω )
function ω_logcentroid(x::Vector{Float64}, y::Vector{Float64})
    idx = findall(i -> isposfinite(x[i]) && isfinite(y[i]) && y[i] > 0, eachindex(y))
    length(idx) < 3 && return NaN
    xx = x[idx]; yy = y[idx]
    total = trapz(xx, yy)
    (isfinite(total) && total > 0) || return NaN
    num = trapz(xx, log.(xx) .* yy)
    return exp(num / total)
end

# simple average-rank (ties get averaged)
function rankdata(v::Vector{Float64})
    n = length(v)
    p = sortperm(v)
    r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[p[j]] == v[p[j+1]]
            j += 1
        end
        # average rank for ties, ranks are 1..n
        avg = 0.5*(i + j)
        for k in i:j
            r[p[k]] = avg
        end
        i = j + 1
    end
    return r
end

function spearman(x::Vector{Float64}, y::Vector{Float64})
    idx = findall(i -> isfinite(x[i]) && isfinite(y[i]), eachindex(x))
    length(idx) < 6 && return NaN
    rx = rankdata(x[idx])
    ry = rankdata(y[idx])
    return cor(rx, ry)
end

function pearson_log(x::Vector{Float64}, y::Vector{Float64})
    idx = findall(i -> isposfinite(x[i]) && isposfinite(y[i]), eachindex(x))
    length(idx) < 6 && return NaN
    return cor(log.(x[idx]), log.(y[idx]))
end

spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigvals(Matrix(J))))

function spectral_radius_complex(M::AbstractMatrix{ComplexF64})
    vals = eigvals(Matrix(M))
    return maximum(abs.(vals))
end

# -----------------------------
# Model pieces (same as your pipeline, minimal)
# -----------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

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

    h = rand(rng, S)  # latent ordering only used to bias directionality
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

# Hutchinson probe estimator at single omega:
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

    probes = Vector{Vector{Float64}}(undef, nprobe)
    for k in 1:nprobe
        probes[k] = rand(rng, (-1.0, 1.0), S)
    end

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
        Pc = ComplexF64.(P)
        for k in 1:nprobe
            x = x_list[k]
            y = Pc * x
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
        good = filter(isfinite, gvec)
        Gabs[k] = isempty(good) ? NaN : mean(abs.(good))
    end

    return (eω=eω, Gabs=Gabs, gmat=gmat)
end

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

# energy-quantile band [ωL,ωH] of e(ω)
function energy_quantile_band(ωvals::Vector{Float64}, eω::Vector{Float64}; qL::Float64=0.05, qH::Float64=0.95)
    @assert 0.0 < qL < qH < 1.0
    ωL = ω_quantile(ωvals, eω, qL)
    ωH = ω_quantile(ωvals, eω, qH)
    return (ωL=ωL, ωH=ωH)
end

# -----------------------------
# Base system with structural knobs stored
# -----------------------------
struct BaseSys
    u::Vector{Float64}
    Abar::Matrix{Float64}       # -I + A
    A::Matrix{Float64}          # A (off-diagonal interactions) for structural metrics
    eps_struct::Float64
    C0diag::Vector{Float64}
    K0::Float64                 # rho(A)
    normA::Float64              # ||A||F
    connectance::Float64
    trophic_align::Float64
    reciprocity::Float64
    sigma::Float64
end

function build_bases(; S::Int=120, base_reps::Int=60, seed::Int=1234,
    u_mean::Float64=1.0, u_cv::Float64=0.5,
    connectance_rng=(0.03, 0.12),
    trophic_align_rng=(0.55, 0.98),
    reciprocity_rng=(0.00, 0.20),
    σ_rng=(0.3, 1.5),
    target_alpha::Float64=-0.05,
    eps_rel::Float64=0.15,
    C0_mode::Symbol=:u2
)
    bases = BaseSys[]
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

        normA = norm(A)
        eps_struct = eps_rel * normA
        (isfinite(eps_struct) && eps_struct > 0) || continue

        C0diag = if C0_mode == :u2
            u.^2
        elseif C0_mode == :I
            ones(Float64, S)
        else
            error("Unknown C0_mode. Use :u2 or :I.")
        end

        # K0 = rho(A)
        K0 = maximum(abs.(eigvals(Matrix(A))))
        isfinite(K0) || continue

        push!(bases, BaseSys(u, Abar, A, eps_struct, C0diag, K0, normA, c, γ, rr, σ))
    end
    return bases
end

# -----------------------------
# Evaluate one base: compute decomposition scalars
# -----------------------------
function eval_base_decomp(base::BaseSys, ωvals::Vector{Float64};
    P_reps::Int=14,
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

    # P ensemble (keep stable at eps_struct)
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

    # base V (time)
    V_time = variability_time_domain(base.Abar, base.u, base.C0diag)
    isfinite(V_time) || return nothing

    # spectra
    sp = estimate_energy_and_fragility_spectra(base.Abar, base.u, base.C0diag, ωvals, Pdirs; nprobe=nprobe, rng=rng)
    eω = sp.eω
    Lω = sp.Gabs  # leverage spectrum = E|g|
    e0 = baseline_energy_curve(base.u, base.C0diag, ωvals)
    Δe = eω .- e0
    Δe_pos = map(x -> (isfinite(x) && x > 0) ? x : 0.0, Δe)

    # m(ω) = L(ω)*e(ω)
    mω = similar(eω)
    for i in eachindex(mω)
        if isfinite(eω[i]) && eω[i] >= 0 && isfinite(Lω[i]) && Lω[i] >= 0
            mω[i] = eω[i] * Lω[i]
        else
            mω[i] = NaN
        end
    end

    # frequency-domain V (for completeness)
    idxE = findall(i -> isposfinite(ωvals[i]) && isfinite(eω[i]) && eω[i] >= 0, eachindex(eω))
    length(idxE) < 3 && return nothing
    V_freq = (1.0/(2π)) * trapz(ωvals[idxE], eω[idxE])
    V0_freq = (1.0/(2π)) * trapz(ωvals[idxE], e0[idxE])

    # energy band [ωL,ωH]
    band = energy_quantile_band(ωvals, eω; qL=qL, qH=qH)
    ωL, ωH = band.ωL, band.ωH

    # quantiles + centroids (locations)
    ωe50 = ω_quantile(ωvals, eω, 0.50)
    ωe95 = ω_quantile(ωvals, eω, 0.95)
    ωL50 = ω_quantile(ωvals, Lω, 0.50)       # leverage-only location (integral-weighted)
    ωm50 = ω_quantile(ωvals, mω, 0.50)
    ωm95 = ω_quantile(ωvals, mω, 0.95)
    ωd50 = ω_quantile(ωvals, Δe_pos, 0.50)   # where interactions add energy above baseline

    ωe_ctr  = ω_logcentroid(ωvals, eω)
    ωe0_ctr = ω_logcentroid(ωvals, e0)
    ωL_ctr  = ω_logcentroid(ωvals, Lω)
    ωm_ctr  = ω_logcentroid(ωvals, mω)
    ωd_ctr  = ω_logcentroid(ωvals, Δe_pos)

    # energy redistribution indices
    totalE = trapz(ωvals[idxE], eω[idxE])
    posΔ  = trapz(ωvals[idxE], Δe_pos[idxE])
    absΔ  = trapz(ωvals[idxE], abs.(Δe[idxE]))
    redist_pos_frac = (isfinite(totalE) && totalE > 0) ? (posΔ / totalE) : NaN
    redist_abs_frac = (isfinite(totalE) && totalE > 0) ? (absΔ / totalE) : NaN

    # leverage in the energy band
    Gband = begin
        idxB = findall(i -> isposfinite(ωvals[i]) && isfinite(Lω[i]) && isfinite(ωL) && isfinite(ωH) &&
                           (ωvals[i] >= ωL) && (ωvals[i] <= ωH), eachindex(Lω))
        isempty(idxB) ? NaN : mean(Lω[idxB])
    end

    return (
        # structural
        K0=base.K0,
        normA=base.normA,
        c=base.connectance,
        gamma=base.trophic_align,
        rr=base.reciprocity,
        sigma=base.sigma,

        # stability
        V_time=V_time,
        V_freq=V_freq,
        V0_freq=V0_freq,
        V_shift=(isfinite(V0_freq) && isfinite(V_freq) && V0_freq > 0) ? (V_freq - V0_freq)/V0_freq : NaN,

        # spectra summaries
        ωL=ωL, ωH=ωH,
        ωe50=ωe50, ωe95=ωe95,
        ωL50=ωL50,
        ωm50=ωm50, ωm95=ωm95,
        ωd50=ωd50,

        ωe_ctr=ωe_ctr, ωe0_ctr=ωe0_ctr,
        ωL_ctr=ωL_ctr, ωm_ctr=ωm_ctr, ωd_ctr=ωd_ctr,

        redist_pos_frac=redist_pos_frac,
        redist_abs_frac=redist_abs_frac,
        Gband=Gband,

        # keep spectra for later examples
        eω=eω, e0=e0, Δe=Δe, Lω=Lω, mω=mω
    )
end

# -----------------------------
# Run decomp experiment
# -----------------------------
function run_decomp_experiment(; S::Int=120, base_reps::Int=60, P_reps::Int=14,
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

    outs = Any[]
    for (i, base) in enumerate(bases)
        out = eval_base_decomp(base, ωvals;
            P_reps=P_reps,
            margin=margin,
            nprobe=nprobe,
            qL=qL, qH=qH,
            seed=seed + 900_000*i
        )
        out === nothing && continue
        push!(outs, out)
    end

    return (ωvals=ωvals, outs=outs)
end

# -----------------------------
# Plot + tests
# -----------------------------
function pull(res, key::Symbol)
    v = Vector{Float64}(undef, length(res.outs))
    for i in eachindex(v)
        v[i] = res.outs[i][key]
    end
    return v
end

function summarize_decomp(res; figsize=(1700, 1100))
    outs = res.outs
    n = length(outs)
    n < 10 && (@warn "Too few systems ($n) to summarize."; return nothing)

    # main location metrics
    ωm50 = pull(res, :ωm50)
    ωe50 = pull(res, :ωe50)
    ωL50 = pull(res, :ωL50)
    ωd50 = pull(res, :ωd50)

    ωm_ctr = pull(res, :ωm_ctr)
    ωe_ctr = pull(res, :ωe_ctr)
    ωe0_ctr = pull(res, :ωe0_ctr)
    ωL_ctr = pull(res, :ωL_ctr)
    ωd_ctr = pull(res, :ωd_ctr)

    red_pos = pull(res, :redist_pos_frac)
    red_abs = pull(res, :redist_abs_frac)
    Gband  = pull(res, :Gband)

    # structural knobs / metrics
    K0    = pull(res, :K0)
    normA = pull(res, :normA)
    c     = pull(res, :c)
    gamma = pull(res, :gamma)
    rr    = pull(res, :rr)
    sigma = pull(res, :sigma)

    # key tests: what predicts omega_m50 (energy vs leverage vs both)
    ρ_me = pearson_log(ωm50, ωe50)
    ρ_mL = pearson_log(ωm50, ωL50)
    ρ_md = pearson_log(ωm50, ωd50)
    @info "Decomposition: cor_log(omega_m50, omega_e50) = $(round(ρ_me, digits=3))"
    @info "Decomposition: cor_log(omega_m50, omega_L50) = $(round(ρ_mL, digits=3))"
    @info "Decomposition: cor_log(omega_m50, omega_d50) = $(round(ρ_md, digits=3))"

    # multi-regression in log space: log ωm50 ~ a log ωe50 + b log ωL50
    idx = findall(i -> isposfinite(ωm50[i]) && isposfinite(ωe50[i]) && isposfinite(ωL50[i]), eachindex(ωm50))
    if length(idx) >= 10
        Y = log.(ωm50[idx])
        X = hcat(ones(length(idx)), log.(ωe50[idx]), log.(ωL50[idx]))
        β = X \ Y
        @info "log ωm50 ~ β0 + βe log ωe50 + βL log ωL50"
        @info "  β0=$(round(β[1],digits=3))  βe=$(round(β[2],digits=3))  βL=$(round(β[3],digits=3))  (N=$(length(idx)))"
    end

    # correlations with structure (Spearman is safer)
    function showcorr(name, x)
        s1 = spearman(x, ωd_ctr)  # energy-redistribution location
        s2 = spearman(x, ωL_ctr)  # leverage location
        s3 = spearman(x, ωm_ctr)  # fragility-mass location
        s4 = spearman(x, red_pos) # amount of positive redistribution
        s5 = spearman(x, Gband)   # leverage magnitude in energy band
        @info "$(name): Spearman with ωd_ctr=$(round(s1,digits=3))  ωL_ctr=$(round(s2,digits=3))  ωm_ctr=$(round(s3,digits=3))  red_pos=$(round(s4,digits=3))  Gband=$(round(s5,digits=3))"
    end

    showcorr("connectance c", c)
    showcorr("trophic_align gamma", gamma)
    showcorr("reciprocity rr", rr)
    showcorr("sigma", sigma)
    showcorr("K0=rho(A)", K0)
    showcorr("||A||F", normA)

    # -----------------------------
    # PLOTS
    # -----------------------------
    fig = Figure(size=figsize)

    # A) omega_m50 vs omega_e50
    ax1 = Axis(fig[1,1]; xscale=log10, yscale=log10,
        xlabel="omega_e50 (energy median)",
        ylabel="omega_m50 (fragility-mass median)",
        title="A) Where fragility sits vs where energy sits"
    )
    idx1 = findall(i -> isposfinite(ωe50[i]) && isposfinite(ωm50[i]), eachindex(ωm50))
    scatter!(ax1, ωe50[idx1], ωm50[idx1], markersize=7)
    text!(ax1, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="log-Pearson = $(round(ρ_me,digits=3))  N=$(length(idx1))")

    # B) omega_m50 vs omega_L50
    ax2 = Axis(fig[1,2]; xscale=log10, yscale=log10,
        xlabel="omega_L50 (leverage-only median)",
        ylabel="omega_m50 (fragility-mass median)",
        title="B) Where fragility sits vs where leverage sits"
    )
    idx2 = findall(i -> isposfinite(ωL50[i]) && isposfinite(ωm50[i]), eachindex(ωm50))
    scatter!(ax2, ωL50[idx2], ωm50[idx2], markersize=7)
    text!(ax2, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="log-Pearson = $(round(ρ_mL,digits=3))  N=$(length(idx2))")

    # C) Energy redistribution location vs leverage location (centroids)
    ax3 = Axis(fig[1,3]; xscale=log10, yscale=log10,
        xlabel="omega_d_ctr  (centroid of positive Delta e)",
        ylabel="omega_L_ctr  (centroid of leverage)",
        title="C) Energy-redistribution location vs leverage location"
    )
    idx3 = findall(i -> isposfinite(ωd_ctr[i]) && isposfinite(ωL_ctr[i]), eachindex(ωd_ctr))
    scatter!(ax3, ωd_ctr[idx3], ωL_ctr[idx3], markersize=7)

    # D) Energy-shift (omega_e_ctr / omega_e0_ctr) vs structure knob(s)
    # define energy shift as ratio of centroids (interacting vs baseline)
    ωeshift = similar(ωe_ctr)
    for i in eachindex(ωeshift)
        ωeshift[i] = (isposfinite(ωe_ctr[i]) && isposfinite(ωe0_ctr[i])) ? (ωe_ctr[i]/ωe0_ctr[i]) : NaN
    end

    ax4 = Axis(fig[2,1]; xscale=log10,
        xlabel="energy centroid shift  (omega_e_ctr / omega_e0_ctr)",
        ylabel="trophic_align gamma",
        title="D) Does structure shift energy location?"
    )
    idx4 = findall(i -> isposfinite(ωeshift[i]) && isfinite(gamma[i]), eachindex(ωeshift))
    scatter!(ax4, ωeshift[idx4], gamma[idx4], markersize=7)
    text!(ax4, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="Spearman = $(round(spearman(ωeshift, gamma),digits=3))  N=$(length(idx4))")

    ax5 = Axis(fig[2,2]; xscale=log10,
        xlabel="omega_L_ctr (leverage centroid)",
        ylabel="reciprocity rr",
        title="E) Does structure shift leverage location?"
    )
    idx5 = findall(i -> isposfinite(ωL_ctr[i]) && isfinite(rr[i]), eachindex(ωL_ctr))
    scatter!(ax5, ωL_ctr[idx5], rr[idx5], markersize=7)
    text!(ax5, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="Spearman = $(round(spearman(ωL_ctr, rr),digits=3))  N=$(length(idx5))")

    # F) Classification plane: energy shift vs leverage location, color by omega_m50
    ax6 = Axis(fig[2,3]; xscale=log10, yscale=log10,
        xlabel="energy centroid shift (omega_e_ctr / omega_e0_ctr)",
        ylabel="omega_L_ctr (leverage centroid)",
        title="F) Organize communities by energy vs leverage"
    )
    idx6 = findall(i -> isposfinite(ωeshift[i]) && isposfinite(ωL_ctr[i]) && isposfinite(ωm50[i]), eachindex(ωm50))
    sc = scatter!(ax6, ωeshift[idx6], ωL_ctr[idx6], markersize=8, color=log10.(ωm50[idx6]))
    Colorbar(fig[2,4], sc; label="log10(omega_m50)")

    # G) Example spectra: pick three systems (low/median/high omega_m50) if possible
    ax7 = Axis(fig[3,1:4]; xscale=log10, yscale=log10,
        xlabel="omega",
        ylabel="value",
        title="G) Example spectra: e(ω), e0(ω), Delta e(ω), L(ω), m(ω)"
    )

    # choose example indices
    good = findall(i -> isposfinite(ωm50[i]), eachindex(ωm50))
    if length(good) >= 6
        ord = sortperm(ωm50[good])
        picks = [good[ord[1]], good[ord[clamp(Int(round(length(ord)/2)),1,length(ord))]], good[ord[end]]]
        ω = res.ωvals
        for pid in picks
            out = res.outs[pid]
            e  = out[:eω]
            e0 = out[:e0]
            L  = out[:Lω]
            m  = out[:mω]
            Δ  = out[:Δe]
            # plot e, L, m as solid/dashed; plot e0 dotted; plot positive Δe thin
            lines!(ax7, ω, map(x -> (isposfinite(x)) ? x : NaN, e), linewidth=3)
            lines!(ax7, ω, map(x -> (isposfinite(x)) ? x : NaN, L), linewidth=3, linestyle=:dash)
            lines!(ax7, ω, map(x -> (isposfinite(x)) ? x : NaN, m), linewidth=3, linestyle=:dashdot)
            lines!(ax7, ω, map(x -> (isposfinite(x)) ? x : NaN, e0), linewidth=2, linestyle=:dot)
            lines!(ax7, ω, map(x -> (isfinite(x) && x > 0) ? x : NaN, Δ), linewidth=1)
        end
    end

    display(fig)
    return nothing
end

# -----------------------------
# MAIN
# -----------------------------
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=70)

res = run_decomp_experiment(
    S=120,
    base_reps=70,
    P_reps=14,
    seed=1234,
    ωvals=ωvals,
    target_alpha=-0.05,
    eps_rel=0.15,
    margin=1e-4,
    nprobe=12,
    qL=0.05, qH=0.95,
    C0_mode=:u2
)

summarize_decomp(res)
################################################################################
