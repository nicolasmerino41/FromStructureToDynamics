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

################################################################################
# PATCH: Realized structural metrics + strength–timescale alignment + feedback
#        + non-normality + magnitude-vs-location diagnostics (Point C)
#
# Drop-in changes only: paste these blocks into your Phase 4 script.
# You do NOT need to rewrite the untouched parts.
################################################################################

isposfinite(x) = isfinite(x) && x > 0

function spearman(x::AbstractVector, y::AbstractVector)
    idx = findall(i -> isfinite(x[i]) && isfinite(y[i]), eachindex(x))
    length(idx) < 6 && return NaN
    xv = x[idx]; yv = y[idx]

    # ranks with average ties
    function ranks(v)
        p = sortperm(v)
        r = similar(v, Float64)
        i = 1
        while i <= length(v)
            j = i
            while j < length(v) && v[p[j+1]] == v[p[i]]
                j += 1
            end
            avg = (i + j) / 2
            for k in i:j
                r[p[k]] = avg
            end
            i = j + 1
        end
        return r
    end

    rx = ranks(xv)
    ry = ranks(yv)
    return cor(rx, ry)
end

function realized_connectance(A::AbstractMatrix{<:Real}; tol::Float64=0.0)
    S = size(A,1)
    off = 0
    nnz = 0
    for i in 1:S, j in 1:S
        i == j && continue
        off += 1
        nnz += (abs(A[i,j]) > tol) ? 1 : 0
    end
    return nnz / off
end

function sign_balance(A::AbstractMatrix{<:Real}; tol::Float64=0.0)
    S = size(A,1)
    pos = 0
    neg = 0
    for i in 1:S, j in 1:S
        i == j && continue
        a = A[i,j]
        abs(a) > tol || continue
        if a > 0
            pos += 1
        elseif a < 0
            neg += 1
        end
    end
    tot = pos + neg
    return (fpos = tot>0 ? pos/tot : NaN,
            fneg = tot>0 ? neg/tot : NaN,
            tot  = tot)
end

function realized_reciprocity(A::AbstractMatrix{<:Real}; tol::Float64=0.0)
    S = size(A,1)
    edges = 0
    mutual_edges = 0
    for i in 1:S, j in 1:S
        i == j && continue
        if abs(A[i,j]) > tol
            edges += 1
            mutual_edges += (abs(A[j,i]) > tol) ? 1 : 0
        end
    end
    return edges > 0 ? (mutual_edges / edges) : NaN
end

function weighted_mutuality(A::AbstractMatrix{<:Real})
    S = size(A,1)
    num = 0.0
    den = 0.0
    for i in 1:S, j in 1:S
        i == j && continue
        num += abs(A[i,j] * A[j,i])
        den += abs(A[i,j])^2
    end
    return den > 0 ? (num / den) : NaN
end

function feedback_trace_power(A::AbstractMatrix{<:Real}, k::Int)
    k < 1 && return NaN
    Ak = Matrix(A)
    if k == 1
        return tr(Ak)
    end
    for _ in 2:k
        Ak = Ak * A
    end
    return tr(Ak)
end

function feedback_trace_power_abs(A::AbstractMatrix{<:Real}, k::Int)
    B = abs.(A)
    return feedback_trace_power(B, k)
end

function feedback_metrics(A::AbstractMatrix{<:Real})
    nA = norm(A)
    nA == 0 && return (trA2=NaN, trA3=NaN, trAbs2=NaN, trAbs3=NaN)
    trA2   = feedback_trace_power(A, 2) / (nA^2)
    trA3   = feedback_trace_power(A, 3) / (nA^3)
    trAbs2 = feedback_trace_power_abs(A, 2) / (nA^2)
    trAbs3 = feedback_trace_power_abs(A, 3) / (nA^3)
    return (trA2=trA2, trA3=trA3, trAbs2=trAbs2, trAbs3=trAbs3)
end

function henrici_non_normality(A::AbstractMatrix{<:Real})
    nA = norm(A)
    nA == 0 && return NaN
    AtA = transpose(A) * A
    AAt = A * transpose(A)
    return norm(AtA - AAt) / (nA^2)
end

function node_strength(A::AbstractMatrix{<:Real}; mode::Symbol=:both)
    out = vec(sum(abs.(A), dims=2))
    inn = vec(sum(abs.(A), dims=1))
    if mode == :out
        return out
    elseif mode == :in
        return inn
    else
        return out .+ inn
    end
end

function weighted_logcentroid_of_invT(u::Vector{Float64}, w::Vector{Float64})
    idx = findall(i -> isfinite(u[i]) && u[i] > 0 && isfinite(w[i]) && w[i] > 0, eachindex(u))
    isempty(idx) && return NaN
    W = sum(w[idx])
    W <= 0 && return NaN
    return exp(sum(w[idx] .* log.(u[idx])) / W)
end

function strength_timescale_alignment(u::Vector{Float64}, A::AbstractMatrix{<:Real})
    S = node_strength(A; mode=:both)
    T = 1.0 ./ u
    idx = findall(i -> isfinite(T[i]) && T[i] > 0 && isfinite(S[i]) && S[i] > 0, eachindex(u))
    ρ = length(idx) >= 6 ? spearman(log.(T[idx]), log.(S[idx])) : NaN
    ω_str = weighted_logcentroid_of_invT(u, S)  # frequency-scale where strong edges live
    return (rho_logT_logS = ρ,
            omega_strength = ω_str,
            meanS = mean(S),
            cvS = std(S) / (mean(S) + 1e-12))
end

function ω_quantile(ω::Vector{Float64}, f::Vector{Float64}; q::Float64=0.5)
    @assert 0.0 < q < 1.0
    idx = findall(i -> isposfinite(ω[i]) && isfinite(f[i]) && f[i] >= 0, eachindex(f))
    length(idx) < 3 && return NaN
    ww = ω[idx]; ff = f[idx]
    tot = trapz(ww, ff)
    (isfinite(tot) && tot > 0) || return NaN

    cum = zeros(Float64, length(ww))
    for i in 2:length(ww)
        cum[i] = cum[i-1] + 0.5*(ff[i-1]+ff[i])*(ww[i]-ww[i-1])
    end
    target = q * tot
    j = findfirst(cum .>= target)
    isnothing(j) && return NaN
    j == 1 && return ww[1]
    w1, w2 = ww[j-1], ww[j]
    c1, c2 = cum[j-1], cum[j]
    c2 == c1 && return w2
    return w1 + (target - c1) * (w2 - w1) / (c2 - c1)
end

function ω_logcentroid(ω::Vector{Float64}, f::Vector{Float64})
    idx = findall(i -> isposfinite(ω[i]) && isfinite(f[i]) && f[i] >= 0, eachindex(f))
    length(idx) < 3 && return NaN
    ww = ω[idx]; ff = f[idx]
    tot = trapz(ww, ff)
    (isfinite(tot) && tot > 0) || return NaN
    num = trapz(ww, ff .* log.(ww))
    return exp(num / tot)
end

# -----------------------------
# Base system with structural knobs stored
# -----------------------------
struct BaseSys1
    u::Vector{Float64}
    Abar::Matrix{Float64}           # -I + A
    A::Matrix{Float64}              # off-diagonal interaction matrix (diag=0)
    eps_struct::Float64
    C0diag::Vector{Float64}
    K0::Float64
    normA::Float64

    # generator knobs (keep if you want reporting; not for explanation)
    connectance::Float64
    trophic_align::Float64
    reciprocity::Float64
    sigma::Float64

    # realized metrics on A
    creal::Float64
    rreal::Float64
    wmut::Float64
    fpos::Float64
    henrici::Float64
    trA2::Float64
    trA3::Float64
    trAbs2::Float64
    trAbs3::Float64

    # strength–timescale alignment
    align_rho::Float64
    omega_strength::Float64
    cv_strength::Float64
end

function build_bases(; S::Int=80, base_reps::Int=40, seed::Int=1234,
    # u distribution
    u_mean::Float64=1.0, u_cv::Float64=0.0,
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
    C0_mode::Symbol=:u2
)
    bases = BaseSys1[]
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

        A = s * O                         # diag=0
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

        K0 = maximum(abs.(eigvals(Matrix(A))))
        isfinite(K0) || continue

        # realized metrics on A
        creal = realized_connectance(A)
        rreal = realized_reciprocity(A)
        wmut  = weighted_mutuality(A)
        sb    = sign_balance(A)
        hen   = henrici_non_normality(A)
        fb    = feedback_metrics(A)
        al    = strength_timescale_alignment(u, A)

        push!(bases, BaseSys1(
            u, Abar, A, eps_struct, C0diag, K0, normA,
            c, γ, rr, σ,
            creal, rreal, wmut, sb.fpos, hen,
            fb.trA2, fb.trA3, fb.trAbs2, fb.trAbs3,
            al.rho_logT_logS, al.omega_strength, al.cvS
        ))
    end
    return bases
end

# -----------------------------
# Evaluate one base: compute decomposition scalars
# -----------------------------
function eval_base_decomp(base::BaseSys1, ωvals::Vector{Float64};
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

    # --- P ensemble (keep only stable perturbations) ---
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

    # --- time-domain variability (Lyapunov) ---
    V_time = variability_time_domain(base.Abar, base.u, base.C0diag)
    isfinite(V_time) || return nothing

    # --- interacting spectra: e(ω) and leverage L(ω)=mean_P|g| ---
    sp = estimate_energy_and_fragility_spectra(base.Abar, base.u, base.C0diag, ωvals, Pdirs;
                                               nprobe=nprobe, rng=rng)
    eω = sp.eω
    Lω = sp.Gabs                    # leverage spectrum
    gmat = sp.gmat

    # --- baseline spectrum e0(ω) and redistribution Δe(ω) ---
    e0 = baseline_energy_curve(base.u, base.C0diag, ωvals)
    Δe = similar(eω)
    for i in eachindex(Δe)
        Δe[i] = (isfinite(eω[i]) && isfinite(e0[i])) ? (eω[i] - e0[i]) : NaN
    end
    Δe_pos = map(x -> (isfinite(x) && x > 0) ? x : 0.0, Δe)

    # --- energetic band from e(ω) ---
    band = energy_quantile_band(ωvals, eω; qL=qL, qH=qH)
    ωL, ωH = band.ωL, band.ωH

    # --- normalized variability via frequency integral ---
    idxE = findall(i -> isposfinite(ωvals[i]) && isfinite(eω[i]) && eω[i] >= 0, eachindex(eω))
    length(idxE) < 3 && return nothing
    V_freq  = (1.0/(2π)) * trapz(ωvals[idxE], eω[idxE])

    idxE0 = findall(i -> isposfinite(ωvals[i]) && isfinite(e0[i]) && e0[i] >= 0, eachindex(e0))
    V0_freq = (1.0/(2π)) * trapz(ωvals[idxE0], e0[idxE0])

    # --- m(ω)=L(ω) e(ω) ---
    mω = similar(eω)
    for i in eachindex(mω)
        if isfinite(eω[i]) && eω[i] >= 0 && isfinite(Lω[i]) && Lω[i] >= 0
            mω[i] = eω[i] * Lω[i]
        else
            mω[i] = NaN
        end
    end

    # --- quantile locations ---
    ωe50 = ω_quantile(ωvals, eω; q=0.5)
    ωe95 = ω_quantile(ωvals, eω; q=0.95)
    ωL50 = ω_quantile(ωvals, Lω; q=0.5)
    ωm50 = ω_quantile(ωvals, mω; q=0.5)
    ωm95 = ω_quantile(ωvals, mω; q=0.95)
    ωd50 = ω_quantile(ωvals, Δe_pos; q=0.5)

    # --- centroids (log-space) ---
    ωe_ctr  = ω_logcentroid(ωvals, eω)
    ωe0_ctr = ω_logcentroid(ωvals, e0)
    ωL_ctr  = ω_logcentroid(ωvals, Lω)
    ωm_ctr  = ω_logcentroid(ωvals, mω)
    ωd_ctr  = ω_logcentroid(ωvals, Δe_pos)

    # --- redistribution fractions ---
    E_mass = trapz(ωvals[idxE], eω[idxE])
    posΔ_mass = trapz(ωvals[idxE], Δe_pos[idxE])
    absΔ_mass = trapz(ωvals[idxE], abs.(map(x -> isfinite(x) ? x : 0.0, Δe))[idxE])

    redist_pos_frac = (isfinite(E_mass) && E_mass > 0) ? (posΔ_mass / E_mass) : NaN
    redist_abs_frac = (isfinite(E_mass) && E_mass > 0) ? (absΔ_mass / E_mass) : NaN

    # --- band-averaged leverage level (for continuity with your earlier metric) ---
    Gband = begin
        idxB = findall(i -> isposfinite(ωvals[i]) && isfinite(Lω[i]) &&
                           isfinite(ωL) && isfinite(ωH) &&
                           (ωvals[i] >= ωL) && (ωvals[i] <= ωH), eachindex(Lω))
        isempty(idxB) ? NaN : mean(Lω[idxB])
    end

    # --- Point C: magnitudes (separate magnitude vs location) ---
    L_mass = begin
        idx = findall(i -> isposfinite(ωvals[i]) && isfinite(Lω[i]) && Lω[i] >= 0, eachindex(Lω))
        isempty(idx) ? NaN : trapz(ωvals[idx], Lω[idx])
    end
    m_mass = begin
        idx = findall(i -> isposfinite(ωvals[i]) && isfinite(mω[i]) && mω[i] >= 0, eachindex(mω))
        isempty(idx) ? NaN : trapz(ωvals[idx], mω[idx])
    end
    Lband_mass = begin
        idxB = findall(i -> isposfinite(ωvals[i]) && isfinite(Lω[i]) && Lω[i] >= 0 &&
                           isfinite(ωL) && isfinite(ωH) &&
                           (ωvals[i] >= ωL) && (ωvals[i] <= ωH), eachindex(Lω))
        isempty(idxB) ? NaN : trapz(ωvals[idxB], Lω[idxB])
    end
    mband_mass = begin
        idxB = findall(i -> isposfinite(ωvals[i]) && isfinite(mω[i]) && mω[i] >= 0 &&
                           isfinite(ωL) && isfinite(ωH) &&
                           (ωvals[i] >= ωL) && (ωvals[i] <= ωH), eachindex(mω))
        isempty(idxB) ? NaN : trapz(ωvals[idxB], mω[idxB])
    end

    return (
        # realized structure (use these as predictors)
        K0=base.K0,
        normA=base.normA,
        creal=base.creal,
        rreal=base.rreal,
        wmut=base.wmut,
        fpos=base.fpos,
        henrici=base.henrici,
        trA2=base.trA2, trA3=base.trA3, trAbs2=base.trAbs2, trAbs3=base.trAbs3,
        align_rho=base.align_rho,
        omega_strength=base.omega_strength,
        cv_strength=base.cv_strength,

        # knobs (optional reporting)
        c=base.connectance,
        gamma=base.trophic_align,
        rr=base.reciprocity,
        sigma=base.sigma,

        # stability + spectra integrals
        V_time=V_time,
        V_freq=V_freq,
        V0_freq=V0_freq,
        V_shift=(isfinite(V0_freq) && isfinite(V_freq) && V0_freq > 0) ? (V_freq - V0_freq)/V0_freq : NaN,

        # band and locations
        ωL=ωL, ωH=ωH,
        ωe50=ωe50, ωe95=ωe95,
        ωL50=ωL50,
        ωm50=ωm50, ωm95=ωm95,
        ωd50=ωd50,
        ωe_ctr=ωe_ctr, ωe0_ctr=ωe0_ctr,
        ωL_ctr=ωL_ctr, ωm_ctr=ωm_ctr, ωd_ctr=ωd_ctr,

        # redistribution
        redist_pos_frac=redist_pos_frac,
        redist_abs_frac=redist_abs_frac,
        Gband=Gband,

        # Point C magnitudes
        posΔ_mass=posΔ_mass,
        E_mass=E_mass,
        L_mass=L_mass,
        m_mass=m_mass,
        Lband_mass=Lband_mass,
        mband_mass=mband_mass,

        # keep spectra for examples/diagnostics
        eω=eω, e0=e0, Δe=Δe, Lω=Lω, mω=mω, gmat=gmat
    )
end

# -----------------------------
# Run decomp experiment
# -----------------------------
function run_experiment_decomp(; S::Int=80, base_reps::Int=40, P_reps::Int=12,
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

    # realized metrics
    K0 = Float64[]
    normA = Float64[]
    creal = Float64[]
    rreal = Float64[]
    wmut = Float64[]
    fpos = Float64[]
    henrici = Float64[]
    trA2 = Float64[]; trA3 = Float64[]; trAbs2 = Float64[]; trAbs3 = Float64[]
    align_rho = Float64[]
    omega_strength = Float64[]
    cv_strength = Float64[]

    # dynamics summaries
    V_time = Float64[]
    V_freq = Float64[]
    V0_freq = Float64[]
    V_shift = Float64[]

    ωL = Float64[]; ωH = Float64[]
    ωe50 = Float64[]; ωe95 = Float64[]
    ωL50 = Float64[]
    ωm50 = Float64[]; ωm95 = Float64[]
    ωd50 = Float64[]
    ωe_ctr = Float64[]; ωe0_ctr = Float64[]
    ωL_ctr = Float64[]; ωm_ctr = Float64[]; ωd_ctr = Float64[]

    redist_pos_frac = Float64[]
    redist_abs_frac = Float64[]
    Gband = Float64[]

    # Point C magnitudes
    posΔ_mass = Float64[]
    E_mass = Float64[]
    L_mass = Float64[]
    m_mass = Float64[]
    Lband_mass = Float64[]
    mband_mass = Float64[]

    example = nothing

    for (i, base) in enumerate(bases)
        out = eval_base_decomp(base, ωvals;
            P_reps=P_reps,
            margin=margin,
            nprobe=nprobe,
            qL=qL, qH=qH,
            seed=seed + 900_000*i
        )
        out === nothing && continue

        push!(K0, out.K0)
        push!(normA, out.normA)
        push!(creal, out.creal)
        push!(rreal, out.rreal)
        push!(wmut, out.wmut)
        push!(fpos, out.fpos)
        push!(henrici, out.henrici)
        push!(trA2, out.trA2); push!(trA3, out.trA3); push!(trAbs2, out.trAbs2); push!(trAbs3, out.trAbs3)
        push!(align_rho, out.align_rho)
        push!(omega_strength, out.omega_strength)
        push!(cv_strength, out.cv_strength)

        push!(V_time, out.V_time)
        push!(V_freq, out.V_freq)
        push!(V0_freq, out.V0_freq)
        push!(V_shift, out.V_shift)

        push!(ωL, out.ωL); push!(ωH, out.ωH)
        push!(ωe50, out.ωe50); push!(ωe95, out.ωe95)
        push!(ωL50, out.ωL50)
        push!(ωm50, out.ωm50); push!(ωm95, out.ωm95)
        push!(ωd50, out.ωd50)
        push!(ωe_ctr, out.ωe_ctr); push!(ωe0_ctr, out.ωe0_ctr)
        push!(ωL_ctr, out.ωL_ctr); push!(ωm_ctr, out.ωm_ctr); push!(ωd_ctr, out.ωd_ctr)

        push!(redist_pos_frac, out.redist_pos_frac)
        push!(redist_abs_frac, out.redist_abs_frac)
        push!(Gband, out.Gband)

        push!(posΔ_mass, out.posΔ_mass)
        push!(E_mass, out.E_mass)
        push!(L_mass, out.L_mass)
        push!(m_mass, out.m_mass)
        push!(Lband_mass, out.Lband_mass)
        push!(mband_mass, out.mband_mass)

        if example === nothing
            example = (ωvals=ωvals, out=out)
        end
    end

    return (
        ωvals=ωvals,
        K0=K0, normA=normA, creal=creal, rreal=rreal, wmut=wmut, fpos=fpos,
        henrici=henrici, trA2=trA2, trA3=trA3, trAbs2=trAbs2, trAbs3=trAbs3,
        align_rho=align_rho, omega_strength=omega_strength, cv_strength=cv_strength,

        V_time=V_time, V_freq=V_freq, V0_freq=V0_freq, V_shift=V_shift,
        ωL=ωL, ωH=ωH,
        ωe50=ωe50, ωe95=ωe95,
        ωL50=ωL50,
        ωm50=ωm50, ωm95=ωm95,
        ωd50=ωd50,
        ωe_ctr=ωe_ctr, ωe0_ctr=ωe0_ctr,
        ωL_ctr=ωL_ctr, ωm_ctr=ωm_ctr, ωd_ctr=ωd_ctr,
        redist_pos_frac=redist_pos_frac, redist_abs_frac=redist_abs_frac,
        Gband=Gband,

        posΔ_mass=posΔ_mass, E_mass=E_mass, L_mass=L_mass, m_mass=m_mass,
        Lband_mass=Lband_mass, mband_mass=mband_mass,

        example=example
    )
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

function summarize_decomp(res; figsize=(1700, 1400))
    pull(sym) = getproperty(res, sym)

    ωd_ctr = pull(:ωd_ctr)
    ωL_ctr = pull(:ωL_ctr)
    ωm50   = pull(:ωm50)
    red_pos = pull(:redist_pos_frac)
    Gband   = pull(:Gband)

    posΔ_mass = pull(:posΔ_mass)
    Lband_mass = pull(:Lband_mass)

    creal   = pull(:creal)
    rreal   = pull(:rreal)
    wmut    = pull(:wmut)
    fpos    = pull(:fpos)
    hen     = pull(:henrici)
    trAbs2  = pull(:trAbs2)
    trAbs3  = pull(:trAbs3)
    trA2    = pull(:trA2)
    trA3    = pull(:trA3)
    align_r = pull(:align_rho)
    ω_str   = pull(:omega_strength)
    cvS     = pull(:cv_strength)

    function showcorr(name, x)
        s1 = spearman(x, ωd_ctr)
        s2 = spearman(x, ωL_ctr)
        s3 = spearman(x, ωm50)
        s4 = spearman(x, red_pos)
        s5 = spearman(x, posΔ_mass)
        s6 = spearman(x, Gband)
        s7 = spearman(x, Lband_mass)
        @info "$(name): ωd_ctr=$(round(s1,digits=3))  ωL_ctr=$(round(s2,digits=3))  ωm50=$(round(s3,digits=3))  red_pos=$(round(s4,digits=3))  posΔ_mass=$(round(s5,digits=3))  Gband=$(round(s6,digits=3))  Lband_mass=$(round(s7,digits=3))"
    end

    # print the realized-structure correlations
    showcorr("align_rho (logT vs strength)", align_r)
    showcorr("omega_strength (strength-weighted u)", ω_str)
    showcorr("cv_strength", cvS)
    showcorr("realized connectance", creal)
    showcorr("realized reciprocity", rreal)
    showcorr("weighted mutuality", wmut)
    showcorr("fpos (sign balance)", fpos)
    showcorr("henrici non-normality", hen)
    showcorr("feedback tr(|A|^2)", trAbs2)
    showcorr("feedback tr(|A|^3)", trAbs3)
    showcorr("feedback tr(A^2) signed", trA2)
    showcorr("feedback tr(A^3) signed", trA3)

    fig = Figure(size=figsize)

    # Point C plots (location vs magnitude separation)
    ax1 = Axis(fig[1,1]; xscale=log10, yscale=log10,
        xlabel="omega_d_ctr (location of +Delta e)",
        ylabel="posDelta_mass = ∫ Delta e_+(ω) dω",
        title="Point C1: +Delta e location vs magnitude"
    )
    idx1 = findall(i -> isposfinite(ωd_ctr[i]) && isposfinite(posΔ_mass[i]), eachindex(ωd_ctr))
    scatter!(ax1, ωd_ctr[idx1], posΔ_mass[idx1], markersize=7)
    text!(ax1, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="Spearman=$(round(spearman(ωd_ctr, posΔ_mass),digits=3))  N=$(length(idx1))")

    ax2 = Axis(fig[1,2]; xscale=log10, yscale=log10,
        xlabel="omega_L_ctr (leverage location)",
        ylabel="Lband_mass = ∫_{ωL}^{ωH} L(ω) dω",
        title="Point C2: leverage location vs magnitude in band"
    )
    idx2 = findall(i -> isposfinite(ωL_ctr[i]) && isposfinite(Lband_mass[i]), eachindex(ωL_ctr))
    scatter!(ax2, ωL_ctr[idx2], Lband_mass[idx2], markersize=7)
    text!(ax2, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="Spearman=$(round(spearman(ωL_ctr, Lband_mass),digits=3))  N=$(length(idx2))")

    ax3 = Axis(fig[1,3]; xscale=log10, yscale=log10,
        xlabel="henrici non-normality",
        ylabel="posDelta_mass",
        title="Does non-normality control +Delta e magnitude?"
    )
    idx3 = findall(i -> isposfinite(hen[i]) && isposfinite(posΔ_mass[i]), eachindex(hen))
    scatter!(ax3, hen[idx3], posΔ_mass[idx3], markersize=7)
    text!(ax3, 0.05, 0.95, space=:relative, align=(:left,:top),
        text="Spearman=$(round(spearman(hen, posΔ_mass),digits=3))  N=$(length(idx3))")

    # quick organization view: energy-redistribution location vs leverage location, colored by ωm50
    ax4 = Axis(fig[2,1]; xscale=log10, yscale=log10,
        xlabel="omega_d_ctr (energy redistribution centroid)",
        ylabel="omega_L_ctr (leverage centroid)",
        title="Organization: where energy moves vs where leverage sits"
    )
    idx4 = findall(i -> isposfinite(ωd_ctr[i]) && isposfinite(ωL_ctr[i]) && isposfinite(ωm50[i]), eachindex(ωd_ctr))
    sc = scatter!(ax4, ωd_ctr[idx4], ωL_ctr[idx4], color=log10.(ωm50[idx4]), markersize=8)
    # Colorbar(fig[2,2], sc, label="log10(omega_m50)")

    # example spectra row (if available)
    ex = getproperty(res, :example)
    if ex !== nothing
        ωvals = ex.ωvals
        out = ex.out
        ax5 = Axis(fig[3,1:3]; xscale=log10, yscale=log10,
            xlabel="ω",
            ylabel="value",
            title="Example spectra: e(ω), e0(ω), L(ω), m(ω)=L e"
        )
        lines!(ax5, ωvals, map(x -> (isfinite(x) && x>0) ? x : NaN, out.eω), linewidth=3)
        lines!(ax5, ωvals, map(x -> (isfinite(x) && x>0) ? x : NaN, out.e0), linewidth=2, linestyle=:dot)
        lines!(ax5, ωvals, map(x -> (isfinite(x) && x>0) ? x : NaN, out.Lω), linewidth=3, linestyle=:dash)
        lines!(ax5, ωvals, map(x -> (isfinite(x) && x>0) ? x : NaN, out.mω), linewidth=3, linestyle=:dashdot)
    end

    display(fig)
    return nothing
end

# -----------------------------
# MAIN
# -----------------------------
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=70)

res = run_experiment_decomp(
    S=120,
    base_reps=150,
    P_reps=15,
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