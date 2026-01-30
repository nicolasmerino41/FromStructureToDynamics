################################################################################
# CONTROLLED ALIGNMENT SWEEP (WIDER Aidx RANGE): "ALIGNMENT ORGANIZES WHEN"
#
# This version forces a *much wider* spread of Aidx by doing THREE things:
#
# (i)  Make node strengths heterogeneous (so the strength-weights w_i are not ~uniform)
#      via a node-fitness scaling applied to the off-diagonal template O.
#
# (ii) Make timescales more heterogeneous (bigger u_cv by default).
#
# (iii) Generate MANY candidate u-assignments (permutations) spanning the full
#       alignment continuum (anti-aligned -> random -> aligned), then STRATIFY-
#       SUBSAMPLE them to keep ~uniform coverage across the Aidx range.
#
# Also: stability filtering for P is now done with a *Gershgorin sufficient condition*
# that guarantees stability for ANY u assignment (positive diagonal scaling).
# This avoids losing extreme-alignment cases just because P filtering fails.
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# -----------------------------
# Small utilities
# -----------------------------
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

spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigvals(Matrix(J))))

function logsafe_vec(y::AbstractVector; eps=1e-18)
    z = similar(y, Float64)
    for i in eachindex(y)
        yi = y[i]
        z[i] = (isfinite(yi) && yi > eps) ? yi : NaN
    end
    return z
end

# max row sum of absolute off-diagonal entries for a matrix with diag possibly nonzero
function max_offdiag_row_sum_abs(M::AbstractMatrix{<:Real})
    S = size(M,1)
    mx = 0.0
    @inbounds for i in 1:S
        s = 0.0
        for j in 1:S
            i == j && continue
            s += abs(M[i,j])
        end
        mx = max(mx, s)
    end
    return mx
end

# -----------------------------
# Random u (rates)
# -----------------------------
function random_u(S; mean=1.0, cv=1.2, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# -----------------------------
# Node fitness scaling to create heterogeneous strengths
# (keeps connectance pattern, changes weights)
# -----------------------------
function lognormal_fitness(S; mean=1.0, cv=1.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

function apply_node_fitness!(O::Matrix{Float64}, fout::Vector{Float64}, fin::Vector{Float64})
    S = size(O,1)
    @inbounds for i in 1:S, j in 1:S
        i == j && continue
        O[i,j] *= fout[i] * fin[j]
    end
    return O
end

# -----------------------------
# Trophic-ish off-diagonal generator O (diag=0)
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
# Find scale s so α(J) ≈ target_alpha (<0) for a *reference u*,
# but ALSO enforce a robust Gershgorin condition:
#   max_i ∑_{j≠i} |A_ij| ≤ row_sum_cap < 1
# so stability holds for ANY permutation of u (positive row scaling).
#
# J = diag(u) * (-I + s O) = -diag(u) + diag(u) * (s O)
# Gershgorin sufficient condition: ∑_{j≠i} |(sO)_ij| < 1  ∀i.
# -----------------------------
function find_scale_to_target_alpha(O::Matrix{Float64}, u::Vector{Float64};
    target_alpha::Float64=-0.20,
    row_sum_cap::Float64=0.60,
    max_grow::Int=40,
    max_iter::Int=60
)
    @assert target_alpha < 0
    @assert 0.0 < row_sum_cap < 1.0

    # alpha-tuning (bisection) on reference u
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
        s_alpha = 0.0
    else
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
        s_alpha = 0.5*(s_lo + s_hi)
    end

    # Gershgorin row-sum cap (on A = sO)
    mxrowO = max_offdiag_row_sum_abs(O)  # diag is 0 already
    if !(isfinite(mxrowO) && mxrowO > 0)
        return NaN
    end
    s_row = row_sum_cap / mxrowO

    # take the more conservative (ensures row_sum cap, still stable)
    return min(s_alpha, s_row)
end

# -----------------------------
# Structural perturbation directions P (diag=0, ||P||_F = 1)
# We will filter P using Gershgorin on A + eps P:
#   max_i ∑_{j≠i} |A_ij + eps P_ij| ≤ row_sum_cap_pert < 1
# so ALL P dirs are stable for ANY u.
# -----------------------------
function sample_Pdir_edgesonly(A::Matrix{Float64}; rng=Random.default_rng())
    S = size(A,1)
    P = zeros(Float64, S, S)
    @inbounds for i in 1:S, j in 1:S
        i == j && continue
        A[i,j] == 0.0 && continue
        P[i,j] = randn(rng)
    end
    nP = norm(P)
    nP == 0 && return nothing
    P ./= nP
    return P
end

# -----------------------------
# Frequency-domain estimators
# -----------------------------
function estimate_energy_and_g_at_ω!(
    F::LU{ComplexF64, Matrix{ComplexF64}},
    sqrtc::Vector{Float64},
    trC0::Float64,
    Pdirs::Vector{Matrix{Float64}};
    nprobe::Int=10,
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

    trChat = mean(xnorm2)
    eω = (isfinite(trChat) && trC0 > 0) ? (trChat / trC0) : NaN

    gvec = fill(NaN, nP)
    for (pidx, P) in enumerate(Pdirs)
        inn = zeros(Float64, nprobe)
        Pc = ComplexF64.(P)
        for k in 1:nprobe
            x = x_list[k]
            y = Pc * x
            z = F \ y
            inn[k] = real(dot(conj.(x), z))
        end
        num = mean(inn)
        if isfinite(num) && isfinite(trChat) && trChat > 0
            gvec[pidx] = 2.0 * (num / trChat)
        end
    end

    return eω, gvec
end

function estimate_energy_and_fragility_spectra(
    Abar::Matrix{Float64},
    u::Vector{Float64},
    C0diag::Vector{Float64},
    ωvals::Vector{Float64},
    Pdirs::Vector{Matrix{Float64}};
    nprobe::Int=10,
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

        goodg = filter(isfinite, gvec)
        Gabs[k] = isempty(goodg) ? NaN : mean(abs.(goodg))
    end

    return (eω=eω, Gabs=Gabs, gmat=gmat)
end

# -----------------------------
# Variability via Lyapunov
# -----------------------------
function variability_time_domain(Abar::Matrix{Float64}, u::Vector{Float64}, C0diag::Vector{Float64})
    S = length(u)
    Du = Diagonal(u)
    J = Du * Abar
    Q = Du * Diagonal(C0diag) * Du

    # keep check (cheap but exact)
    α = spectral_abscissa(J)
    (isfinite(α) && α < 0) || return NaN

    Σ = lyap(Matrix(J), Matrix(Q))
    trC0 = sum(C0diag)
    return (isfinite(tr(Σ)) && trC0 > 0) ? (tr(Σ) / trC0) : NaN
end

# -----------------------------
# WHERE statistics for nonnegative spectrum y(ω)
# -----------------------------
function spectrum_location_stats(ω::Vector{Float64}, y::Vector{Float64})
    idx = findall(i -> isfinite(ω[i]) && ω[i] > 0 && isfinite(y[i]) && y[i] >= 0, eachindex(y))
    length(idx) < 4 && return (m50=NaN, q10=NaN, q90=NaN, lctr=NaN, mass=NaN)

    w = ω[idx]
    v = y[idx]
    total = trapz(w, v)
    (isfinite(total) && total > 0) || return (m50=NaN, q10=NaN, q90=NaN, lctr=NaN, mass=NaN)

    cum = zeros(Float64, length(w))
    for i in 2:length(w)
        cum[i] = cum[i-1] + 0.5*(v[i-1] + v[i]) * (w[i] - w[i-1])
    end

    function interp_at(frac)
        tgt = frac * total
        j = findfirst(cum .>= tgt)
        isnothing(j) && return NaN
        j == 1 && return w[1]
        w1, w2 = w[j-1], w[j]
        c1, c2 = cum[j-1], cum[j]
        if c2 == c1
            return w2
        else
            return w1 + (tgt - c1) * (w2 - w1) / (c2 - c1)
        end
    end

    # log-centroid
    num = 0.0
    den = 0.0
    for i in 1:(length(w)-1)
        w1, w2 = w[i], w[i+1]
        v1, v2 = v[i], v[i+1]
        dw = (w2 - w1)
        if !(isfinite(w1) && isfinite(w2) && isfinite(v1) && isfinite(v2) && w1 > 0 && w2 > 0)
            continue
        end
        den += 0.5*(v1 + v2) * dw
        num += 0.5*(v1*log(w1) + v2*log(w2)) * dw
    end
    lctr = (den > 0) ? exp(num / den) : NaN

    return (m50=interp_at(0.50), q10=interp_at(0.10), q90=interp_at(0.90), lctr=lctr, mass=total)
end

# -----------------------------
# Alignment metrics
# -----------------------------
function node_strength_abs(A::Matrix{Float64})
    out = vec(sum(abs.(A), dims=2))
    inn = vec(sum(abs.(A), dims=1))
    return out .+ inn
end

function alignment_metrics(A::Matrix{Float64}, u::Vector{Float64})
    s = node_strength_abs(A)
    stot = sum(s)
    w = (stot > 0) ? (s ./ stot) : fill(1/length(s), length(s))

    T = 1.0 ./ u
    logT = log.(T)

    Aidx = dot(w, logT) - mean(logT)
    u_str = exp(dot(w, log.(u)))

    Tbar = mean(T)
    wtilde_pred = u_str * Tbar

    cidx = (std(s) > 0 && std(logT) > 0) ? cor(s, logT) : NaN
    return (Aidx=Aidx, Aidx_corr=cidx, u_str=u_str, Tbar=Tbar, wtilde_pred=wtilde_pred)
end

# -----------------------------
# Evaluate one (A,u), with a PRE-BUILT P ensemble (stable for all u)
# -----------------------------
function eval_system!(
    Abar::Matrix{Float64},
    A::Matrix{Float64},
    u::Vector{Float64},
    Pdirs::Vector{Matrix{Float64}};
    ωtilde_vals::Vector{Float64},
    C0_mode::Symbol=:I,
    nprobe::Int=10,
    eps_rel::Float64=0.02,
    seed::Int=1
)
    rng = MersenneTwister(seed)
    S = length(u)

    am = alignment_metrics(A, u)
    Tbar = am.Tbar
    ωvals = ωtilde_vals ./ Tbar

    C0diag = if C0_mode == :I
        ones(Float64, S)
    elseif C0_mode == :u2
        u.^2
    else
        error("Unknown C0_mode. Use :I or :u2.")
    end

    eps = eps_rel * norm(A)
    isfinite(eps) && eps > 0 || return nothing

    # base variability
    V0 = variability_time_domain(Abar, u, C0diag)
    isfinite(V0) && V0 > 0 || return nothing

    # spectra
    sp = estimate_energy_and_fragility_spectra(Abar, u, C0diag, ωvals, Pdirs; nprobe=nprobe, rng=rng)
    eω = sp.eω
    Gω = sp.Gabs
    mω = eω .* Gω

    mstats = spectrum_location_stats(ωtilde_vals, mω)
    wtilde_m50  = mstats.m50
    wtilde_q10  = mstats.q10
    wtilde_q90  = mstats.q90
    wtilde_lctr = mstats.lctr

    # predicted slopes
    idxE = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω[i]) && eω[i] >= 0, eachindex(eω))
    length(idxE) >= 4 || return nothing
    denom = trapz(ωvals[idxE], eω[idxE])
    isfinite(denom) && denom > 0 || return nothing

    gmat = sp.gmat
    slopes = Float64[]
    slopes_abs = Float64[]
    for p in 1:size(gmat,1)
        gω = vec(gmat[p, :])
        good = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω[i]) && eω[i] >= 0 && isfinite(gω[i]),
                       eachindex(eω))
        length(good) >= 4 || continue
        num = trapz(ωvals[good], gω[good] .* eω[good])
        sgn = num / denom
        isfinite(sgn) || continue
        push!(slopes, sgn)
        push!(slopes_abs, abs(sgn))
    end
    isempty(slopes_abs) && return nothing
    H_real = mean(slopes_abs)
    slope_mean = mean(slopes)

    # realized per-unit magnitude
    realized_abs = Float64[]
    realized_sgn = Float64[]
    for P in Pdirs
        Vp = variability_time_domain(Abar + eps*P, u, C0diag)
        isfinite(Vp) && Vp > 0 || continue
        push!(realized_abs, abs(Vp - V0) / V0 / eps)
        push!(realized_sgn, (Vp - V0) / V0 / eps)
    end
    isempty(realized_abs) && return nothing
    R_abs = mean(realized_abs)
    R_sgn = mean(realized_sgn)

    # Homogeneous-timescale facet
    u_hom = fill(1.0 / Tbar, S)
    C0_hom = (C0_mode == :I) ? ones(Float64, S) : u_hom.^2
    sp_h = estimate_energy_and_fragility_spectra(Abar, u_hom, C0_hom, ωvals, Pdirs; nprobe=nprobe, rng=rng)
    eω_h = sp_h.eω
    gmat_h = sp_h.gmat
    idxEh = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω_h[i]) && eω_h[i] >= 0, eachindex(eω_h))
    denom_h = trapz(ωvals[idxEh], eω_h[idxEh])
    H_hom = NaN
    if isfinite(denom_h) && denom_h > 0
        slopes_abs_h = Float64[]
        for p in 1:size(gmat_h,1)
            gωh = vec(gmat_h[p, :])
            good = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(eω_h[i]) && eω_h[i] >= 0 && isfinite(gωh[i]),
                           eachindex(eω_h))
            length(good) >= 4 || continue
            num = trapz(ωvals[good], gωh[good] .* eω_h[good])
            push!(slopes_abs_h, abs(num / denom_h))
        end
        H_hom = isempty(slopes_abs_h) ? NaN : mean(slopes_abs_h)
    end

    return (
        Aidx=am.Aidx,
        Aidx_corr=am.Aidx_corr,
        u_str=am.u_str,
        Tbar=Tbar,
        wtilde_pred=am.wtilde_pred,

        wtilde_m50=wtilde_m50,
        wtilde_q10=wtilde_q10,
        wtilde_q90=wtilde_q90,
        wtilde_lctr=wtilde_lctr,

        H_real=H_real,
        H_hom=H_hom,
        R_abs=R_abs,
        R_sgn=R_sgn,
        slope_mean=slope_mean,

        ωtilde=ωtilde_vals,
        ω=ωvals,
        eω=eω,
        Gω=Gω,
        mω=mω,
        eω_h=eω_h,
        Gω_h=sp_h.Gabs,
        mω_h=(sp_h.eω .* sp_h.Gabs),

        eps=eps,
        nP=length(Pdirs),
        V0=V0
    )
end

# -----------------------------
# BasePackk: build ONE A, Abar, and ONE u multiset
# -----------------------------
struct BasePackk
    A::Matrix{Float64}
    Abar::Matrix{Float64}
    uvals::Vector{Float64}
    svec::Vector{Float64}
end

function build_base_pack(; S::Int=120, seed::Int=1234,
    u_mean::Float64=1.0, u_cv::Float64=1.2,
    connectance::Float64=0.06,
    trophic_align::Float64=0.85,
    reciprocity::Float64=0.10,
    σ::Float64=1.0,
    target_alpha::Float64=-0.20,
    # NEW: induce strength heterogeneity
    fitness_cv_out::Float64=1.5,
    fitness_cv_in::Float64=1.5,
    # NEW: enforce robust stability for any permutation of u
    row_sum_cap::Float64=0.60,
    max_tries::Int=40
)
    rng = MersenneTwister(seed)

    for attempt in 1:max_tries
        u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))

        O = trophic_O(S; connectance=connectance, trophic_align=trophic_align,
                     reciprocity=reciprocity, σ=σ, rng=rng)

        # NEW: node fitness scaling -> heterogenous strengths
        fout = lognormal_fitness(S; cv=fitness_cv_out, rng=rng)
        fin  = lognormal_fitness(S; cv=fitness_cv_in,  rng=rng)
        apply_node_fitness!(O, fout, fin)

        normalize_offdiag!(O) || continue

        s = find_scale_to_target_alpha(O, u; target_alpha=target_alpha, row_sum_cap=row_sum_cap)
        isfinite(s) || continue

        A = s * O
        Abar = -Matrix{Float64}(I, S, S) + A

        # Robust Gershgorin check (ensures stability for any u permutation)
        mxA = max_offdiag_row_sum_abs(A)
        (isfinite(mxA) && mxA < 1.0) || continue

        # Check on reference u (should be stable)
        α = spectral_abscissa(Diagonal(u) * Abar)
        (isfinite(α) && α < 0) || continue

        svec = node_strength_abs(A)
        return BasePackk(A, Abar, u, svec)
    end

    error("Failed to build a suitable base pack after $max_tries attempts.")
end

# -----------------------------
# Build a P ensemble that is stable for ANY u (Gershgorin sufficient)
# -----------------------------
function build_Pensemble_gershgorin(A::Matrix{Float64}; eps::Float64,
    P_reps::Int=14,
    row_sum_cap_pert::Float64=0.80,
    seed::Int=1
)
    rng = MersenneTwister(seed)
    S = size(A,1)
    Pdirs = Matrix{Float64}[]

    for k in 1:(5P_reps)  # oversample then keep those that satisfy cap
        P = sample_Pdir_edgesonly(A; rng=rng)
        P === nothing && continue
        # require perturbed row sums remain < 1 (cap < 1 for margin)
        mx = max_offdiag_row_sum_abs(A + eps*P)
        if isfinite(mx) && mx <= row_sum_cap_pert
            push!(Pdirs, P)
        end
        length(Pdirs) >= P_reps && break
    end

    length(Pdirs) >= max(6, Int(floor(P_reps/2))) || error("Could not build enough stable P directions.")
    return Pdirs
end

# -----------------------------
# Generate u assignments that span Aidx widely, then stratified subsample
# -----------------------------
function extreme_u_assignments(uvals::Vector{Float64}, svec::Vector{Float64})
    S = length(uvals)
    ord_str = sortperm(svec; rev=true)          # strong -> weak
    ord_u_slow = sortperm(uvals; rev=false)     # small u = slow
    ord_u_fast = sortperm(uvals; rev=true)      # large u = fast

    u_aligned = similar(uvals)
    u_antial  = similar(uvals)
    for k in 1:S
        u_aligned[ord_str[k]] = uvals[ord_u_slow[k]]
        u_antial[ord_str[k]]  = uvals[ord_u_fast[k]]
    end
    return u_aligned, u_antial
end

function swap_k!(u::Vector{Float64}, k::Int, rng)
    S = length(u)
    for _ in 1:k
        i = rand(rng, 1:S)
        j = rand(rng, 1:S)
        u[i], u[j] = u[j], u[i]
    end
    return u
end

function stratified_subsample_by_Aidx(A::Matrix{Float64}, candidates::Vector{Vector{Float64}};
    nbins::Int=18,
    nkeep::Int=120,
    seed::Int=1
)
    rng = MersenneTwister(seed)
    Aidxs = Float64[]
    for u in candidates
        push!(Aidxs, alignment_metrics(A, u).Aidx)
    end

    ok = findall(i -> isfinite(Aidxs[i]), eachindex(Aidxs))
    isempty(ok) && error("No finite Aidx candidates.")

    amin = minimum(Aidxs[ok])
    amax = maximum(Aidxs[ok])
    if amin == amax
        @warn "All Aidx identical; stratification cannot widen range."
        return candidates[ok[1:min(nkeep, length(ok))]]
    end

    edges = range(amin, amax; length=nbins+1) |> collect
    bins = [Int[] for _ in 1:nbins]

    for i in ok
        a = Aidxs[i]
        b = searchsortedlast(edges, a) - 1
        b = clamp(b, 1, nbins)
        push!(bins[b], i)
    end

    # pick round-robin across bins to cover range
    selected = Int[]
    while length(selected) < min(nkeep, length(ok))
        progressed = false
        for b in 1:nbins
            isempty(bins[b]) && continue
            i = rand(rng, bins[b])
            push!(selected, i)
            filter!(x -> x != i, bins[b])
            progressed = true
            length(selected) >= min(nkeep, length(ok)) && break
        end
        progressed || break
    end

    # if still short, fill randomly
    if length(selected) < min(nkeep, length(ok))
        remaining = setdiff(ok, selected)
        shuffle!(rng, remaining)
        append!(selected, remaining[1:(min(nkeep, length(ok)) - length(selected))])
    end

    return [candidates[i] for i in selected]
end

# -----------------------------
# Main alignment sweep
# -----------------------------
function run_alignment_sweep(; S::Int=120, seed::Int=1234,
    nperm::Int=120,
    ωtilde_vals = 10 .^ range(log10(1e-4), log10(2e-1); length=70),
    C0_mode::Symbol=:I,
    P_reps::Int=14,
    nprobe::Int=10,
    eps_rel::Float64=0.02,
    # NEW knobs for wider Aidx
    u_cv::Float64=1.2,
    fitness_cv_out::Float64=1.5,
    fitness_cv_in::Float64=1.5,
    row_sum_cap::Float64=0.60,
    row_sum_cap_pert::Float64=0.80,
    n_candidates::Int=900,
    nbins::Int=18
)
    ωtilde_vals = collect(float.(ωtilde_vals))

    base = build_base_pack(
        S=S, seed=seed,
        u_cv=u_cv,
        fitness_cv_out=fitness_cv_out,
        fitness_cv_in=fitness_cv_in,
        row_sum_cap=row_sum_cap
    )
    A, Abar, uvals, svec = base.A, base.Abar, base.uvals, base.svec

    # fixed eps for P ensemble construction (P must be stable for ANY u)
    eps = eps_rel * norm(A)
    Pdirs = build_Pensemble_gershgorin(A; eps=eps, P_reps=P_reps, row_sum_cap_pert=row_sum_cap_pert, seed=seed + 777)

    # build MANY candidate u assignments spanning the alignment continuum
    rng = MersenneTwister(seed + 99)
    u_aligned, u_antial = extreme_u_assignments(uvals, svec)

    candidates = Vector{Vector{Float64}}()
    push!(candidates, copy(u_aligned))
    push!(candidates, copy(u_antial))

    # generate swap-ladder around extremes + random perms
    max_swaps = 4S
    for _ in 1:(n_candidates-2)
        u0 = (rand(rng) < 0.5) ? copy(u_aligned) : copy(u_antial)
        # mixture: with some prob do a small swap (near extremes), else larger (near random)
        if rand(rng) < 0.65
            k = rand(rng, 0:round(Int, 0.6S))
        else
            k = rand(rng, 0:max_swaps)
        end
        swap_k!(u0, k, rng)
        push!(candidates, u0)
    end

    # stratified subsample to keep uniform coverage across Aidx
    u_list = stratified_subsample_by_Aidx(A, candidates; nbins=nbins, nkeep=nperm, seed=seed + 2024)

    # evaluate
    results = NamedTuple[]
    kept = 0
    for (k, ucase) in enumerate(u_list)
        out = eval_system!(Abar, A, ucase, Pdirs;
            ωtilde_vals=ωtilde_vals,
            C0_mode=C0_mode,
            nprobe=nprobe,
            eps_rel=eps_rel,
            seed=seed + 50_000 + k
        )
        out === nothing && continue
        push!(results, out)
        kept += 1
    end

    @info "Alignment sweep: kept $kept / $(length(u_list)) evaluable u-assignments."
    Aidx_vals = [r.Aidx for r in results if isfinite(r.Aidx)]
    if !isempty(Aidx_vals)
        @info "Aidx range (kept): ($(minimum(Aidx_vals)), $(maximum(Aidx_vals)))"
    end

    return (base=base, ωtilde=ωtilde_vals, Pdirs=Pdirs, res=results)
end

# -----------------------------
# Plot and interpret diagnostics
# -----------------------------
function plot_alignment_sweep(run; figsize=(1650, 950))
    res = run.res
    N = length(res)
    N == 0 && error("No results to plot.")

    Aidx        = [r.Aidx for r in res]
    w_m50       = [r.wtilde_m50 for r in res]
    w_lctr      = [r.wtilde_lctr for r in res]
    w_pred      = [r.wtilde_pred for r in res]
    H_real      = [r.H_real for r in res]
    H_hom       = [r.H_hom for r in res]
    R_abs       = [r.R_abs for r in res]

    m_when50 = findall(i -> isfinite(Aidx[i]) && isfinite(w_m50[i]) && w_m50[i] > 0, 1:N)
    m_whenc  = findall(i -> isfinite(Aidx[i]) && isfinite(w_lctr[i]) && w_lctr[i] > 0, 1:N)
    m_predw  = findall(i -> isfinite(w_pred[i]) && w_pred[i] > 0 && isfinite(w_m50[i]) && w_m50[i] > 0, 1:N)
    m_mag    = findall(i -> isfinite(H_real[i]) && H_real[i] > 0 && isfinite(R_abs[i]) && R_abs[i] > 0, 1:N)
    m_mag_h  = findall(i -> isfinite(H_hom[i]) && H_hom[i] > 0 && isfinite(R_abs[i]) && R_abs[i] > 0, 1:N)

    ρ_when50 = (length(m_when50) >= 6) ? cor(Aidx[m_when50], log.(w_m50[m_when50])) : NaN
    ρ_whenc  = (length(m_whenc)  >= 6) ? cor(Aidx[m_whenc],  log.(w_lctr[m_whenc])) : NaN
    ρ_predw  = (length(m_predw)  >= 6) ? cor(log.(w_pred[m_predw]), log.(w_m50[m_predw])) : NaN
    ρ_mag    = (length(m_mag)    >= 6) ? cor(log.(H_real[m_mag]), log.(R_abs[m_mag])) : NaN
    ρ_mag_h  = (length(m_mag_h)  >= 6) ? cor(log.(H_hom[m_mag_h]),  log.(R_abs[m_mag_h])) : NaN

    @info "WHEN: corr(Aidx, log ωtilde_m50)  = $(round(ρ_when50,digits=3))   N=$(length(m_when50))"
    @info "WHEN: corr(Aidx, log ωtilde_ctr)  = $(round(ρ_whenc,digits=3))   N=$(length(m_whenc))"
    @info "WHEN: corr(log ωtilde_pred, log ωtilde_m50) = $(round(ρ_predw,digits=3))   N=$(length(m_predw))"
    @info "HOW MUCH: corr(log H_real, log realized per-unit) = $(round(ρ_mag,digits=3))   N=$(length(m_mag))"
    @info "HOW MUCH: corr(log H_hom,  log realized per-unit) = $(round(ρ_mag_h,digits=3))  N=$(length(m_mag_h))"

    Aidx_fin = filter(isfinite, Aidx)
    @info "Range Aidx (plotted) = ($(minimum(Aidx_fin)), $(maximum(Aidx_fin)))"

    fig = Figure(size=figsize)

    ax1 = Axis(fig[1,1];
        xlabel="alignment Aidx (strength-weighted logT minus mean)",
        ylabel="log10(ωtilde_m50)",
        title="A) Alignment vs WHEN (median location)"
    )
    scatter!(ax1, Aidx[m_when50], log10.(w_m50[m_when50]), markersize=7)
    text!(ax1, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(Aidx, log ωtilde_m50) = $(round(ρ_when50,digits=3))   N=$(length(m_when50))")

    ax2 = Axis(fig[1,2];
        xlabel="alignment Aidx",
        ylabel="log10(ωtilde_ctr)",
        title="B) Alignment vs WHEN (log-centroid)"
    )
    scatter!(ax2, Aidx[m_whenc], log10.(w_lctr[m_whenc]), markersize=7)
    text!(ax2, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(Aidx, log ωtilde_ctr) = $(round(ρ_whenc,digits=3))   N=$(length(m_whenc))")

    ax3 = Axis(fig[1,3];
        xscale=log10, yscale=log10,
        xlabel="ωtilde_pred = u_str * Tbar",
        ylabel="ωtilde_m50",
        title="C) Strength-weighted predictor vs WHEN"
    )
    scatter!(ax3, w_pred[m_predw], w_m50[m_predw], markersize=7)
    text!(ax3, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log) = $(round(ρ_predw,digits=3))   N=$(length(m_predw))")

    ax4 = Axis(fig[2,1];
        xscale=log10, yscale=log10,
        xlabel="predicted per-unit magnitude H_real",
        ylabel="realized per-unit magnitude mean(|ΔV|/V)/eps",
        title="D) HOW MUCH per unit: prediction vs realized"
    )
    scatter!(ax4, H_real[m_mag], R_abs[m_mag], markersize=7)
    text!(ax4, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log) = $(round(ρ_mag,digits=3))   N=$(length(m_mag))")

    ax5 = Axis(fig[2,2];
        xscale=log10, yscale=log10,
        xlabel="H_hom (timescales homogenized, Tbar preserved)",
        ylabel="realized per-unit magnitude",
        title="E) HOW MUCH: dynamic facet (timescales removed)"
    )
    scatter!(ax5, H_hom[m_mag_h], R_abs[m_mag_h], markersize=7)
    text!(ax5, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log) = $(round(ρ_mag_h,digits=3))   N=$(length(m_mag_h))")

    ax6 = Axis(fig[2,3];
        xlabel="Aidx",
        ylabel="count",
        title="F) Did we excite the alignment axis?"
    )
    hist!(ax6, Aidx_fin, bins=25)

    # Example spectra at extreme Aidx
    idx_ok = findall(i -> isfinite(res[i].Aidx) && isfinite(res[i].wtilde_m50), 1:N)
    if length(idx_ok) >= 4
        amin = idx_ok[argmin(Aidx[idx_ok])]
        amax = idx_ok[argmax(Aidx[idx_ok])]

        ax7 = Axis(fig[3,1:3];
            xscale=log10, yscale=log10,
            xlabel="ωtilde = ω * Tbar",
            ylabel="value",
            title="G) Example spectra at extreme alignment: e(ω), G(ω), m(ω)=eG (real vs hom)"
        )

        function plot_one(r; lab="")
            w = r.ωtilde
            lines!(ax7, w, logsafe_vec(r.eω), linewidth=2, label="e "*lab)
            lines!(ax7, w, logsafe_vec(r.Gω), linewidth=2, linestyle=:dash, label="G "*lab)
            lines!(ax7, w, logsafe_vec(r.mω), linewidth=3, linestyle=:dot, label="m "*lab)
            lines!(ax7, w, logsafe_vec(r.eω_h), linewidth=2, linestyle=:dashdot, label="e_hom "*lab)
            lines!(ax7, w, logsafe_vec(r.mω_h), linewidth=2, linestyle=:dash, label="m_hom "*lab)
        end

        plot_one(res[amin]; lab="(low Aidx)")
        plot_one(res[amax]; lab="(high Aidx)")
        axislegend(ax7; position=:rb, nbanks=2)
    end

    display(fig)
    return nothing
end

# -----------------------------
# MAIN
# -----------------------------
ωtilde_vals = 10 .^ range(log10(1e-4), log10(2e-1); length=70)

run = run_alignment_sweep(
    S=120,
    seed=1234,
    nperm=120,
    ωtilde_vals=ωtilde_vals,
    C0_mode=:I,
    P_reps=14,
    nprobe=10,
    eps_rel=0.02,

    # widen Aidx by default
    u_cv=1.2,
    fitness_cv_out=1.5,
    fitness_cv_in=1.5,
    row_sum_cap=0.60,
    row_sum_cap_pert=0.80,
    n_candidates=900,
    nbins=18
)

plot_alignment_sweep(run)
################################################################################
