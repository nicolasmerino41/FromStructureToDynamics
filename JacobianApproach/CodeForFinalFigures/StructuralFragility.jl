################################################################################
# STRUCTURAL FRAGILITY PIPELINE (FULL SCRIPT)
#
# Implements:
#  1) Epsilon-sweep linearity tests (signed + absolute, per system and pooled)
#  2) Repeatability across P ensembles (dense/sparse/edge-only/sign-preserving)
#  3) Repeatability across C0 ensembles (I, u, u^2, inv_u^2)
#  4) Cumulative contributions of m(omega) = E_P |g(omega;P)| * e(omega)
#  5) Phase 3: interpretable candidate axes and tests against mechanism summaries
#     (omega_m50, omega_m95, omega_m_mean) derived from m(omega)
#
# Notes:
#  - Uses stochastic probe vectors (Hutchinson) for traces.
#  - Uses Lyapunov equation for exact stationary variability V_time.
#  - g(omega;P) defined as fractional sensitivity of tr(C_hat_omega).
#  - All code is self-contained; only standard packages + CairoMakie required.
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# -----------------------------
# Basic utilities
# -----------------------------
meanfinite(v)   = (x = filter(isfinite, v); isempty(x) ? NaN : mean(x))
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
            s += 0.5*(y1+y2)*(x2-x1)
        end
    end
    return s
end

function cumulative_trapz(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    n = length(x)
    cum = zeros(Float64, n)
    n < 2 && return cum
    for i in 2:n
        x1, x2 = float(x[i-1]), float(x[i])
        y1, y2 = float(y[i-1]), float(y[i])
        if isfinite(x1) && isfinite(x2) && isfinite(y1) && isfinite(y2)
            cum[i] = cum[i-1] + 0.5*(y1+y2)*(x2-x1)
        else
            cum[i] = cum[i-1]
        end
    end
    return cum
end

function interp_quantile_from_cum(x::Vector{Float64}, cum::Vector{Float64}, q::Float64)
    @assert 0.0 <= q <= 1.0
    total = cum[end]
    (!isfinite(total) || total <= 0) && return NaN
    target = q * total
    j = findfirst(cum .>= target)
    j === nothing && return NaN
    j == 1 && return x[1]
    x1, x2 = x[j-1], x[j]
    c1, c2 = cum[j-1], cum[j]
    if c2 == c1
        return x2
    end
    return x1 + (target - c1) * (x2 - x1) / (c2 - c1)
end

spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigvals(Matrix(J))))

# numeric abscissa (reactivity-like): max eig of (J + J^T)/2
function numeric_abscissa(J::AbstractMatrix{Float64})
    H = 0.5 * (J + J')
    return maximum(eigvals(Symmetric(H)))
end

# nonnormality index: 1 - sum(|lambda|^2)/||J||_F^2   (0 if normal, >0 if non-normal)
function nonnormality_index(J::AbstractMatrix{Float64})
    fn2 = norm(J)^2
    fn2 <= 0 && return NaN
    lam = eigvals(Matrix(J))
    s = sum(abs2.(lam))
    return max(0.0, (fn2 - s) / fn2)
end

# Spearman rank correlation with average ranks (handles ties)
function spearman(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    idx = findall(i -> isfinite(x[i]) && isfinite(y[i]), eachindex(x))
    length(idx) < 6 && return NaN
    xr = rank_average(x[idx])
    yr = rank_average(y[idx])
    return cor(xr, yr)
end

function rank_average(v::AbstractVector{<:Real})
    n = length(v)
    p = sortperm(v)
    r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && v[p[j+1]] == v[p[i]]
            j += 1
        end
        avg = 0.5 * (i + j)
        for k in i:j
            r[p[k]] = avg
        end
        i = j + 1
    end
    return r
end

# safe log transform for correlation
function logmask(x::Vector{Float64}; eps=1e-18)
    y = similar(x)
    for i in eachindex(x)
        y[i] = (isfinite(x[i]) && x[i] > eps) ? log(x[i]) : NaN
    end
    return y
end

# -----------------------------
# Random time-scales u
# -----------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# -----------------------------
# Trophic-ish heterogeneous off-diagonal generator O (diag=0)
# -----------------------------
function trophic_O(S::Int;
    connectance::Float64,
    trophic_align::Float64,
    reciprocity::Float64,
    sigmaA::Float64,
    rng=Random.default_rng()
)
    @assert 0.0 <= connectance <= 1.0
    @assert 0.0 <= trophic_align <= 1.0
    @assert 0.0 <= reciprocity <= 1.0

    h = rand(rng, S)           # latent ordering
    O = zeros(Float64, S, S)

    for i in 1:S-1, j in i+1:S
        rand(rng) < connectance || continue

        if rand(rng) < reciprocity
            O[i,j] = randn(rng) * sigmaA
            O[j,i] = randn(rng) * sigmaA
        else
            low, high = (h[i] <= h[j]) ? (i, j) : (j, i)
            aligned = rand(rng) < trophic_align
            if aligned
                O[low, high] = randn(rng) * sigmaA
            else
                O[high, low] = randn(rng) * sigmaA
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

# Choose scale s so alpha(J) ~ target_alpha (<0)
# Base: Abar = -I + s*O
# J = diag(u) * Abar
function find_scale_to_target_alpha(O::Matrix{Float64}, u::Vector{Float64};
    target_alpha::Float64 = -0.05,
    max_grow::Int=40,
    max_iter::Int=60
)
    @assert target_alpha < 0
    Du = Diagonal(u)

    alpha0 = spectral_abscissa(-Du)
    isfinite(alpha0) || return NaN

    s_hi = 1.0
    alpha_hi = spectral_abscissa(-Du + s_hi*(Du*O))
    k = 0
    while (isfinite(alpha_hi) && alpha_hi < target_alpha) && k < max_grow
        s_hi *= 2.0
        alpha_hi = spectral_abscissa(-Du + s_hi*(Du*O))
        k += 1
    end
    if !(isfinite(alpha_hi)) || alpha_hi < target_alpha
        return NaN
    end

    s_lo = 0.0
    alpha_lo = alpha0
    if alpha_lo > target_alpha
        return 0.0
    end

    for _ in 1:max_iter
        s_mid = 0.5*(s_lo + s_hi)
        alpha_mid = spectral_abscissa(-Du + s_mid*(Du*O))
        if !isfinite(alpha_mid)
            s_hi = s_mid
            continue
        end
        if alpha_mid < target_alpha
            s_lo = s_mid
        else
            s_hi = s_mid
        end
    end
    return 0.5*(s_lo + s_hi)
end

# -----------------------------
# Base system
# -----------------------------
struct BaseSystem
    u::Vector{Float64}
    Abar::Matrix{Float64}     # includes -I on diag
    A::Matrix{Float64}        # interaction-only, A = Abar + I
    alphaJ::Float64
    K0::Float64               # spectral radius proxy of A (static)
    connectance::Float64      # realized connectance of A (offdiag)
end

function build_bases(; S::Int=120, base_reps::Int=60, seed::Int=1234,
    u_mean::Float64=1.0, u_cv::Float64=0.5,
    connectance_rng=(0.03, 0.12),
    trophic_align_rng=(0.55, 0.98),
    reciprocity_rng=(0.00, 0.20),
    sigmaA_rng=(0.3, 1.5),
    target_alpha::Float64=-0.05
)
    bases = BaseSystem[]
    for b in 1:base_reps
        rng = MersenneTwister(seed + 10007*b)

        u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))

        c  = rand(rng, Uniform(connectance_rng[1], connectance_rng[2]))
        ta = rand(rng, Uniform(trophic_align_rng[1], trophic_align_rng[2]))
        rr = rand(rng, Uniform(reciprocity_rng[1], reciprocity_rng[2]))
        sig = rand(rng, Uniform(sigmaA_rng[1], sigmaA_rng[2]))

        O = trophic_O(S; connectance=c, trophic_align=ta, reciprocity=rr, sigmaA=sig, rng=rng)
        normalize_offdiag!(O) || continue

        s = find_scale_to_target_alpha(O, u; target_alpha=target_alpha)
        isfinite(s) || continue

        A = s * O
        Abar = -Matrix{Float64}(I, S, S) + A
        J = Diagonal(u) * Abar
        alphaJ = spectral_abscissa(J)
        (isfinite(alphaJ) && alphaJ < 0) || continue

        # static K0 proxy: spectral radius of A (real)
        K0 = maximum(abs.(eigvals(Matrix(A))))
        isfinite(K0) || continue

        # realized connectance offdiag
        off = 0
        tot = S*(S-1)
        for i in 1:S, j in 1:S
            i == j && continue
            off += (A[i,j] != 0.0) ? 1 : 0
        end
        conn_real = off / tot

        push!(bases, BaseSystem(u, Abar, A, alphaJ, K0, conn_real))
    end
    return bases
end

# -----------------------------
# C0 ensembles (noise models)
# -----------------------------
function C0diag_from_mode(mode::Symbol, u::Vector{Float64})
    S = length(u)
    if mode == :I
        return ones(Float64, S)
    elseif mode == :u
        return copy(u)
    elseif mode == :u2
        return u.^2
    elseif mode == :inv_u2
        return (1.0 ./ (u.^2 .+ 1e-18))
    else
        error("Unknown C0 mode: $mode. Use :I, :u, :u2, :inv_u2.")
    end
end

# -----------------------------
# P ensembles (structural uncertainty classes)
# -----------------------------
struct PEnsemble
    name::String
    sparsity::Float64           # probability of perturbing an allowed entry
    mask_mode::Symbol           # :all, :edges, :nonedges
    sign_mode::Symbol           # :free, :sign_preserve
end

function allowed_mask(base::BaseSystem, mode::Symbol)
    S = size(base.A, 1)
    M = falses(S, S)
    if mode == :all
        for i in 1:S, j in 1:S
            i == j && continue
            M[i,j] = true
        end
    elseif mode == :edges
        for i in 1:S, j in 1:S
            i == j && continue
            M[i,j] = (base.A[i,j] != 0.0)
        end
    elseif mode == :nonedges
        for i in 1:S, j in 1:S
            i == j && continue
            M[i,j] = (base.A[i,j] == 0.0)
        end
    else
        error("Unknown mask_mode: $mode")
    end
    return M
end

function sample_Pdir(base::BaseSystem, cfg::PEnsemble; rng=Random.default_rng())
    S = size(base.Abar, 1)
    mask = allowed_mask(base, cfg.mask_mode)
    P = zeros(Float64, S, S)

    for i in 1:S, j in 1:S
        i == j && continue
        mask[i,j] || continue
        rand(rng) < cfg.sparsity || continue
        if cfg.sign_mode == :free
            P[i,j] = randn(rng)
        elseif cfg.sign_mode == :sign_preserve
            sgn = (base.A[i,j] == 0.0) ? 1.0 : sign(base.A[i,j])
            P[i,j] = abs(randn(rng)) * sgn
        else
            error("Unknown sign_mode: $(cfg.sign_mode)")
        end
    end

    nP = norm(P)
    nP == 0 && return nothing
    P ./= nP
    return P
end

# -----------------------------
# Variability via Lyapunov
# T xdot = Abar x + xi,   E[xi xi^T] = C0 delta
# Convert to xdot = J x + eta, J = diag(u) Abar, eta = diag(u) xi
# => Q = diag(u) C0 diag(u)
# Solve J Sigma + Sigma J^T + Q = 0
# Return V = tr(Sigma)/tr(C0)
# -----------------------------
function variability_time_domain(Abar::Matrix{Float64}, u::Vector{Float64}, C0diag::Vector{Float64})
    S = length(u)
    Du = Diagonal(u)
    J = Du * Abar
    Q = Du * Diagonal(C0diag) * Du

    alpha = spectral_abscissa(J)
    (isfinite(alpha) && alpha < 0) || return NaN

    Sigma = lyap(Matrix(J), Matrix(Q))
    trC0 = sum(C0diag)
    (isfinite(tr(Sigma)) && trC0 > 0) ? (tr(Sigma) / trC0) : NaN
end

# -----------------------------
# Resolvent-based estimation of e(omega) and g(omega;P)
#
# For fixed omega, M = i omega T - Abar, R = M^{-1}.
# Let sqrtC = sqrt.(C0diag).
#
# Hutchinson:
#   tr(Chat) = E_v || R (sqrtC .* v) ||^2 , v Rademacher
# and with x = R (sqrtC .* v), E[x x^†] = Chat
#   tr(R P Chat) = E_v x^† (R P x)
#
# Then:
#   g(omega;P) = 2 Re tr(R P Chat) / tr(Chat)
# -----------------------------
function estimate_e_and_g_at_omega(
    F::LU{ComplexF64, Matrix{ComplexF64}},
    sqrtC::Vector{Float64},
    trC0::Float64,
    Pdirs::Vector{Matrix{Float64}};
    nprobe::Int=12,
    rng=Random.default_rng()
)
    S = length(sqrtC)
    nP = length(Pdirs)

    # probes
    probes = Vector{Vector{Float64}}(undef, nprobe)
    for k in 1:nprobe
        probes[k] = rand(rng, (-1.0, 1.0), S)
    end

    # compute x_k = R (sqrtC .* v_k)
    x_list = Vector{Vector{ComplexF64}}(undef, nprobe)
    xnorm2 = zeros(Float64, nprobe)
    for k in 1:nprobe
        rhs = ComplexF64.(sqrtC .* probes[k])
        x = F \ rhs
        x_list[k] = x
        xnorm2[k] = real(dot(conj.(x), x))
    end

    trChat = mean(xnorm2)               # ~ tr(R C0 R^†)
    e = (isfinite(trChat) && trC0 > 0) ? (trChat / trC0) : NaN

    gvec = fill(NaN, nP)
    if !(isfinite(trChat) && trChat > 0)
        return e, gvec
    end

    # for each P: estimate Re tr(R P Chat) via mean_k Re[ x_k^† (R P x_k) ]
    for (pidx, P) in enumerate(Pdirs)
        inners = zeros(Float64, nprobe)
        PC = ComplexF64.(P)
        for k in 1:nprobe
            x = x_list[k]
            y = PC * x
            z = F \ y
            inners[k] = real(dot(conj.(x), z))
        end
        num = mean(inners)              # ~ Re tr(R P Chat)
        gvec[pidx] = 2.0 * (num / trChat)
    end

    return e, gvec
end

function estimate_energy_and_g_spectra(
    Abar::Matrix{Float64}, u::Vector{Float64}, C0diag::Vector{Float64},
    omega_vals::Vector{Float64}, Pdirs::Vector{Matrix{Float64}};
    nprobe::Int=12,
    rng=Random.default_rng()
)
    S = length(u)
    sqrtC = sqrt.(C0diag)
    trC0 = sum(C0diag)

    Tmat = Diagonal(1.0 ./ u)
    n_om = length(omega_vals)
    nP = length(Pdirs)

    e = fill(NaN, n_om)
    gmat = fill(NaN, nP, n_om)
    meanabs_g = fill(NaN, n_om)

    for (k, om0) in enumerate(omega_vals)
        om = float(om0)
        M = Matrix{ComplexF64}(im * om * Tmat - Abar)
        F = lu(M)
        ek, gvec = estimate_e_and_g_at_omega(F, sqrtC, trC0, Pdirs; nprobe=nprobe, rng=rng)
        e[k] = ek
        for p in 1:nP
            gmat[p, k] = gvec[p]
        end
        good = filter(isfinite, gvec)
        meanabs_g[k] = isempty(good) ? NaN : mean(abs.(good))
    end

    return (e=e, gmat=gmat, meanabs_g=meanabs_g)
end

# -----------------------------
# Choose P directions stable at eps_max
# -----------------------------
function sample_stable_Pdirs(
    base::BaseSystem,
    cfg::PEnsemble;
    nP_target::Int=16,
    eps_max::Float64=0.15,
    margin::Float64=1e-4,
    max_tries::Int=5000,
    rng=Random.default_rng()
)
    S = length(base.u)
    Du = Diagonal(base.u)

    Pdirs = Matrix{Float64}[]
    tries = 0
    while length(Pdirs) < nP_target && tries < max_tries
        tries += 1
        P = sample_Pdir(base, cfg; rng=rng)
        P === nothing && continue

        Abarp = base.Abar + eps_max * P
        Jp = Du * Abarp
        ap = spectral_abscissa(Jp)
        (isfinite(ap) && ap < -margin) || continue

        push!(Pdirs, P)
    end

    return Pdirs
end

# -----------------------------
# Predicted slopes from g(omega;P)
# If e_eps(omega) ~ e(omega) * (1 + eps*g(omega;P)),
# then d/d eps [V] / V = (int g e) / (int e)  (signed).
# -----------------------------
function predicted_slopes_from_spectra(omega_vals::Vector{Float64}, e::Vector{Float64}, gmat::Matrix{Float64})
    idx = findall(i -> isfinite(omega_vals[i]) && omega_vals[i] > 0 && isfinite(e[i]) && e[i] >= 0, eachindex(e))
    length(idx) < 3 && return (slope_signed=Float64[], slope_abs=Float64[], denom=NaN)

    denom = trapz(omega_vals[idx], e[idx])
    (!isfinite(denom) || denom <= 0) && return (slope_signed=Float64[], slope_abs=Float64[], denom=denom)

    nP = size(gmat, 1)
    slope_signed = Float64[]
    slope_abs = Float64[]
    for p in 1:nP
        g = vec(gmat[p, :])
        good = findall(i -> (i in idx) && isfinite(g[i]), eachindex(g))
        length(good) < 3 && continue
        num = trapz(omega_vals[good], (g[good] .* e[good]))
        s = num / denom
        isfinite(s) || continue
        push!(slope_signed, s)
        push!(slope_abs, abs(s))
    end
    return (slope_signed=slope_signed, slope_abs=slope_abs, denom=denom)
end

# -----------------------------
# Epsilon sweep: compare actual DV/V (Lyapunov) vs eps * predicted slope
# We compute:
#  - per-P signed DV: (V_eps - V0)/V0
#  - per-P abs DV: abs(V_eps - V0)/V0
#  - mean across P for each eps
#  - pooled linearity diagnostics: slope_hat (fit through origin), R2
# -----------------------------
function epsilon_sweep_linearity(
    base::BaseSystem,
    C0diag::Vector{Float64},
    Pdirs::Vector{Matrix{Float64}},
    eps_list::Vector{Float64};
    omega_vals::Vector{Float64},
    nprobe::Int=12,
    rng=Random.default_rng()
)
    S = length(base.u)

    # base variability
    V0 = variability_time_domain(base.Abar, base.u, C0diag)
    isfinite(V0) || return nothing

    # spectra at eps = 0 for predicted slopes
    sp = estimate_energy_and_g_spectra(base.Abar, base.u, C0diag, omega_vals, Pdirs; nprobe=nprobe, rng=rng)
    e = sp.e
    gmat = sp.gmat

    slopes = predicted_slopes_from_spectra(omega_vals, e, gmat)
    length(slopes.slope_signed) < 3 && return nothing

    slope_signed_mean = mean(slopes.slope_signed)
    slope_abs_mean = mean(slopes.slope_abs)

    nP = length(Pdirs)
    Du = Diagonal(base.u)

    # actual DV for each eps
    dv_signed_mean = fill(NaN, length(eps_list))
    dv_abs_mean    = fill(NaN, length(eps_list))

    # pooled per-P per-eps for linearity
    X = Float64[]
    Y = Float64[]   # signed
    YA = Float64[]  # abs

    for (k, eps) in enumerate(eps_list)
        dvs = Float64[]
        dva = Float64[]
        for (pidx, P) in enumerate(Pdirs)
            Abarp = base.Abar + eps * P
            Vp = variability_time_domain(Abarp, base.u, C0diag)
            isfinite(Vp) || continue
            dv = (Vp - V0) / V0
            push!(dvs, dv)
            push!(dva, abs(dv))

            push!(X, eps)
            push!(Y, dv)
            push!(YA, abs(dv))
        end
        dv_signed_mean[k] = isempty(dvs) ? NaN : mean(dvs)
        dv_abs_mean[k]    = isempty(dva) ? NaN : mean(dva)
    end

    # fit through origin: y = b x
    function fit_origin(x::Vector{Float64}, y::Vector{Float64})
        idx = findall(i -> isfinite(x[i]) && isfinite(y[i]), eachindex(x))
        length(idx) < 10 && return (b=NaN, r2=NaN)
        xx = x[idx]; yy = y[idx]
        b = sum(xx .* yy) / sum(xx .* xx)
        yhat = b .* xx
        ssr = sum((yy .- yhat).^2)
        sst = sum((yy .- mean(yy)).^2)
        r2 = (sst > 0) ? (1.0 - ssr/sst) : NaN
        return (b=b, r2=r2)
    end

    fitS = fit_origin(X, Y)
    fitA = fit_origin(X, YA)

    return (
        V0=V0,
        e=e,
        meanabs_g=sp.meanabs_g,
        gmat=gmat,
        slope_signed_mean=slope_signed_mean,
        slope_abs_mean=slope_abs_mean,
        dv_signed_mean=dv_signed_mean,
        dv_abs_mean=dv_abs_mean,
        fit_signed=fitS,
        fit_abs=fitA
    )
end

# -----------------------------
# Mechanism spectrum m(omega) = E_P |g(omega;P)| * e(omega)
# and cumulative distribution in omega.
# Also returns omega_m50, omega_m95, omega_m_mean (log-mean).
# -----------------------------
function m_spectrum_and_cumulative(omega_vals::Vector{Float64}, e::Vector{Float64}, gmat::Matrix{Float64})
    idx = findall(i -> isfinite(omega_vals[i]) && omega_vals[i] > 0 && isfinite(e[i]) && e[i] >= 0, eachindex(e))
    length(idx) < 3 && return nothing

    om = omega_vals[idx]
    ee = e[idx]

    # mean abs g at each omega
    nP = size(gmat, 1)
    meanabs = zeros(Float64, length(idx))
    for (k, ii) in enumerate(idx)
        gv = gmat[:, ii]
        good = filter(isfinite, gv)
        meanabs[k] = isempty(good) ? NaN : mean(abs.(good))
    end

    m = meanabs .* ee
    # clean
    for i in eachindex(m)
        if !(isfinite(m[i]) && m[i] >= 0)
            m[i] = 0.0
        end
    end

    cum = cumulative_trapz(om, m)
    total = cum[end]
    if !(isfinite(total) && total > 0)
        return nothing
    end

    omega_m50 = interp_quantile_from_cum(om, cum, 0.50)
    omega_m95 = interp_quantile_from_cum(om, cum, 0.95)

    # log-mean omega under m (more stable than raw mean for wide ranges)
    wlog = log.(om)
    num = trapz(om, (wlog .* m))
    omega_m_mean = exp(num / total)

    return (omega=om, m=m, cum=cum, total=total, omega_m50=omega_m50, omega_m95=omega_m95, omega_m_mean=omega_m_mean)
end

# -----------------------------
# Aomega proxies at a given omega (direct/indirect)
# Aomega = A * D(omega), D = diag(1/(1+i omega T_i))
# return rho(Aomega) and opnorm(Aomega)
# -----------------------------
function spectral_radius_complex(M::AbstractMatrix{ComplexF64})
    vals = eigvals(Matrix(M))
    return maximum(abs.(vals))
end

function Aomega_metrics(A::Matrix{Float64}, u::Vector{Float64}, omega::Float64)
    S = size(A, 1)
    (isfinite(omega) && omega > 0) || return (rho=NaN, opn=NaN)

    Tvec = 1.0 ./ u
    D = Diagonal(ComplexF64.(1.0 ./ (1.0 .+ im*omega*Tvec)))
    Aom = Matrix{ComplexF64}(A) * D
    rho = spectral_radius_complex(Aom)
    opn = opnorm(Aom)
    return (rho=rho, opn=opn)
end

# -----------------------------
# Phase 3 candidates (interpretable axes)
# We compute a small set of candidate descriptors per base:
#  - timescale heterogeneity: std(log T), CV(T)
#  - nonnormality index of J
#  - numeric abscissa (reactivity-like) of J
#  - alphaJ (stability margin already controlled but record)
#  - K0, connectance
# and test against mechanism summaries: omega_m50, omega_m95, omega_m_mean
# and fragility slope (signed/abs)
# -----------------------------
function candidate_metrics(base::BaseSystem)
    u = base.u
    T = 1.0 ./ u
    logT = log.(T)
    cvT = std(T) / mean(T)
    sdlogT = std(logT)

    J = Diagonal(u) * base.Abar
    nn = nonnormality_index(J)
    nu = numeric_abscissa(J)

    return (
        cvT=cvT,
        sdlogT=sdlogT,
        nonnormality=nn,
        num_abscissa=nu,
        alphaJ=base.alphaJ,
        K0=base.K0,
        connectance=base.connectance
    )
end

# -----------------------------
# One scenario evaluation on one base:
#  - sample stable P dirs
#  - epsilon sweep linearity
#  - compute m-spectrum and omega_m quantiles
#  - compute Aomega proxies at omega_m50 and omega_m95
# -----------------------------
function eval_base_scenario(
    base::BaseSystem,
    cfgP::PEnsemble,
    C0mode::Symbol;
    omega_vals::Vector{Float64},
    eps_max::Float64=0.15,
    eps_list::Vector{Float64}=[0.03, 0.06, 0.10, 0.15],
    nP_target::Int=16,
    margin::Float64=1e-4,
    nprobe::Int=12,
    seed::Int=1
)
    rng = MersenneTwister(seed)

    C0diag = C0diag_from_mode(C0mode, base.u)

    Pdirs = sample_stable_Pdirs(base, cfgP; nP_target=nP_target, eps_max=eps_max, margin=margin, rng=rng)
    length(Pdirs) < max(6, Int(floor(nP_target/2))) && return nothing

    sweep = epsilon_sweep_linearity(base, C0diag, Pdirs, eps_list; omega_vals=omega_vals, nprobe=nprobe, rng=rng)
    sweep === nothing && return nothing

    ms = m_spectrum_and_cumulative(omega_vals, sweep.e, sweep.gmat)
    ms === nothing && return nothing

    # Aomega proxies at omega_m50 and omega_m95
    A = base.A
    m50 = Aomega_metrics(A, base.u, ms.omega_m50)
    m95 = Aomega_metrics(A, base.u, ms.omega_m95)

    cand = candidate_metrics(base)

    return (
        nP=length(Pdirs),
        C0mode=C0mode,
        Pname=cfgP.name,
        eps_list=eps_list,

        V0=sweep.V0,
        slope_signed_mean=sweep.slope_signed_mean,
        slope_abs_mean=sweep.slope_abs_mean,
        dv_signed_mean=sweep.dv_signed_mean,
        dv_abs_mean=sweep.dv_abs_mean,
        fit_signed=sweep.fit_signed,
        fit_abs=sweep.fit_abs,

        omega_m50=ms.omega_m50,
        omega_m95=ms.omega_m95,
        omega_m_mean=ms.omega_m_mean,

        rho_m50=m50.rho, opn_m50=m50.opn,
        rho_m95=m95.rho, opn_m95=m95.opn,

        cand=cand,

        # keep for plotting if needed
        omega=ms.omega, m=ms.m, cum=ms.cum,
        e=sweep.e, meanabs_g=sweep.meanabs_g
    )
end

# -----------------------------
# Run experiment over many bases and scenarios
# -----------------------------
using Base.Threads

function run_full_experiment(;
    S::Int=120,
    base_reps::Int=50,
    seed::Int=1234,
    target_alpha::Float64=-0.05,
    omega_vals = 10 .^ range(log10(1e-4), log10(1e4); length=70),
    eps_max::Float64=0.15,
    eps_list::Vector{Float64}=[0.03, 0.06, 0.10, 0.15],
    nP_target::Int=16,
    nprobe::Int=12,
    margin::Float64=1e-4
)
    omega_vals = collect(float.(omega_vals))

    bases = build_bases(
        S=S,
        base_reps=base_reps,
        seed=seed,
        target_alpha=target_alpha
    )
    @info "Built $(length(bases)) bases (attempted $base_reps)."

    P_ensembles = [
        PEnsemble("dense_all_free",       1.00, :all,     :free),
        PEnsemble("sparse_all_free",      0.10, :all,     :free),
        PEnsemble("very_sparse_all_free", 0.02, :all,     :free),
        PEnsemble("edges_only_free",      1.00, :edges,   :free),
        PEnsemble("edges_only_sign",      1.00, :edges,   :sign_preserve),
        PEnsemble("nonedges_only_free",   0.20, :nonedges,:free)
    ]

    C0_modes = [:I, :u, :u2, :inv_u2]

    res = Dict{String, Vector{Any}}()

    for C0mode in C0_modes
        for cfgP in P_ensembles
            key = "C0=$(String(C0mode))__P=$(cfgP.name)"
            outvec = Vector{Any}(undef, length(bases))

            h = hash(key)
            h_int = Int(h % UInt64(typemax(Int)))

            @threads for i in eachindex(bases)
                base = bases[i]
                outvec[i] = eval_base_scenario(
                    base, cfgP, C0mode;
                    omega_vals=omega_vals,
                    eps_max=eps_max,
                    eps_list=eps_list,
                    nP_target=nP_target,
                    margin=margin,
                    nprobe=nprobe,
                    seed = seed + 900_000*i + 1000*h_int
                )
            end

            res[key] = outvec
            @info "Finished scenario: $key"
        end
    end

    return (
        bases=bases,
        omega_vals=omega_vals,
        eps_list=eps_list,
        P_ensembles=P_ensembles,
        C0_modes=C0_modes,
        res=res
    )
end

# -----------------------------
# Helpers to extract arrays from results
# -----------------------------
function extract_metric(exp, key::String, field::Symbol)
    outs = exp.res[key]
    v = fill(NaN, length(outs))
    for i in eachindex(outs)
        out = outs[i]
        out === nothing && continue
        val = getproperty(out, field)
        v[i] = (val isa Real) ? float(val) : NaN
    end
    return v
end

function extract_candidate(exp, key::String, cand_field::Symbol)
    outs = exp.res[key]
    v = fill(NaN, length(outs))
    for i in eachindex(outs)
        out = outs[i]
        out === nothing && continue
        v[i] = getproperty(out.cand, cand_field)
    end
    return v
end

# -----------------------------
# Repeatability matrices across scenarios
# Compute Spearman correlations of:
#  - slope_abs_mean
#  - omega_m50, omega_m95
# -----------------------------
function repeatability_matrix(exp; metric::Symbol=:slope_abs_mean)
    keyss = collect(keys(exp.res))
    n = length(keyss)
    M = fill(NaN, n, n)
    for i in 1:n
        xi = extract_metric(exp, keyss[i], metric)
        for j in i:n
            yj = extract_metric(exp, keyss[j], metric)
            r = spearman(xi, yj)
            M[i,j] = r
            M[j,i] = r
        end
    end
    return (keys=keyss, M=M)
end

# -----------------------------
# Phase 3 analysis:
# Correlate candidate axes with mechanism summaries omega_m50/m95/mean and slope.
# Uses Spearman and log-Pearson (optional).
# -----------------------------
function phase3_correlations(exp, key::String)
    # mechanism summaries
    om50 = extract_metric(exp, key, :omega_m50)
    om95 = extract_metric(exp, key, :omega_m95)
    omm  = extract_metric(exp, key, :omega_m_mean)
    slopeA = extract_metric(exp, key, :slope_abs_mean)
    slopeS = extract_metric(exp, key, :slope_signed_mean)

    # candidates
    cand_names = [:cvT, :sdlogT, :nonnormality, :num_abscissa, :alphaJ, :K0, :connectance]
    cand = Dict{Symbol,Vector{Float64}}()
    for nm in cand_names
        cand[nm] = extract_candidate(exp, key, nm)
    end

    # compute correlations
    out = Dict{String,Any}()
    for target in [(:omega_m50, om50), (:omega_m95, om95), (:omega_m_mean, omm), (:slope_abs_mean, slopeA), (:slope_signed_mean, slopeS)]
        tname, tv = target
        rows = []
        for nm in cand_names
            rv = cand[nm]
            rS = spearman(rv, tv)
            # log-Pearson where positive
            rLP = cor(logmask(rv), logmask(tv))
            push!(rows, (cand=nm, spearman=rS, logpearson=rLP))
        end
        out[string(tname)] = rows
    end
    return out
end

# -----------------------------
# Plotting
# -----------------------------
function plot_eps_sweep_examples(exp, key::String; nshow::Int=6)
    outs = exp.res[key]
    eps_list = exp.eps_list

    # pick some valid systems
    good = findall(i -> outs[i] !== nothing, eachindex(outs))
    isempty(good) && error("No valid systems for key=$key")
    rng = MersenneTwister(202)
    sel = good[1:min(nshow, length(good))]
    shuffle!(rng, sel)

    fig = Figure(size=(1400, 800))
    ax1 = Axis(fig[1,1], xscale=log10, yscale=log10, xlabel="epsilon", ylabel="mean |(V_eps - V0)| / V0",
               title="Epsilon sweep (abs change) with predicted slope")
    ax2 = Axis(fig[1,2], xscale=log10, yscale=log10, xlabel="epsilon", ylabel="mean (V_eps - V0)/V0",
               title="Epsilon sweep (signed change) with predicted slope")

    for i in sel
        out = outs[i]
        eps = eps_list
        dvA = out.dv_abs_mean
        dvS = out.dv_signed_mean
        predA = out.slope_abs_mean .* eps
        predS = out.slope_signed_mean .* eps

        lines!(ax1, eps, dvA)
        lines!(ax1, eps, predA, linestyle=:dash)

        lines!(ax2, eps, abs.(dvS) .+ 1e-18)  # visualize magnitude on log axis
        lines!(ax2, eps, abs.(predS) .+ 1e-18, linestyle=:dash)
    end

    display(fig)
end

function plot_m_cumulative_examples(exp, key::String; nshow::Int=8)
    outs = exp.res[key]
    good = findall(i -> outs[i] !== nothing, eachindex(outs))
    isempty(good) && error("No valid systems for key=$key")

    rng = MersenneTwister(303)
    sel = good[1:min(nshow, length(good))]
    shuffle!(rng, sel)

    fig = Figure(size=(1400, 700))
    ax = Axis(fig[1,1], xscale=log10, xlabel="omega", ylabel="cumulative mass",
              title="Cumulative mass of m(omega) = E|g| e(omega)")

    # also plot median curve (by interpolating onto a common grid)
    omega_grid = exp.omega_vals[findall(w -> w > 0, exp.omega_vals)]
    Ccurves = Float64[]
    # plot individual curves
    for i in sel
        out = outs[i]
        om = out.omega
        cum = out.cum
        tot = cum[end]
        c = tot > 0 ? cum ./ tot : cum
        lines!(ax, om, c, linewidth=2)
    end

    display(fig)
end

function plot_repeatability_heatmap(rep; title::String="Repeatability")
    keyss = rep.keys
    M = rep.M
    n = length(keyss)

    fig = Figure(size=(1200, 1000))
    ax = Axis(fig[1,1], title=title, xlabel="Scenario", ylabel="Scenario")
    hm = heatmap!(ax, 1:n, 1:n, M)
    Colorbar(fig[1,2], hm)

    ax.xticks = (1:n, keyss)
    ax.yticks = (1:n, keyss)
    ax.xticklabelrotation = pi/2
    display(fig)
end

function plot_phase3_scatter(exp, key::String; target::Symbol=:omega_m50, cand::Symbol=:sdlogT)
    y = extract_metric(exp, key, target)
    x = extract_candidate(exp, key, cand)

    idx = findall(i -> isfinite(x[i]) && isfinite(y[i]) && y[i] > 0, eachindex(x))
    fig = Figure(size=(900, 700))
    ax = Axis(fig[1,1], xscale=identity, yscale=log10,
              xlabel=String(cand), ylabel=String(target),
              title="Phase 3: $(String(cand)) vs $(String(target))")
    scatter!(ax, x[idx], y[idx], markersize=7)

    rS = spearman(x, y)
    rLP = cor(logmask(x), logmask(y))
    text!(ax, 0.02, 0.98, space=:relative, align=(:left,:top),
          text="Spearman = $(round(rS, digits=3))   log-Pearson = $(round(rLP, digits=3))")

    display(fig)
end

# -----------------------------
# MAIN
# -----------------------------
omega_vals = 10 .^ range(log10(1e-4), log10(1e4); length=70)

experiment = run_full_experiment(
    S=120,
    base_reps=45,
    seed=1234,
    target_alpha=-0.05,
    omega_vals=omega_vals,
    eps_max=0.15,
    eps_list=[0.03, 0.06, 0.15, 0.3],
    nP_target=14,
    nprobe=10,
    margin=1e-4
)

# Pick a default scenario key for example plots
keys_all = collect(keys(experiment.res))
sort!(keys_all)
example_key = keys_all[1]
@info "Example scenario key: $example_key"

# 1) Epsilon sweep examples
fig_eps = plot_eps_sweep_examples(experiment, example_key; nshow=6)
display(fig_eps)

# 2) Cumulative m(omega) examples
fig_m = plot_m_cumulative_examples(experiment, example_key; nshow=8)
display(fig_m)

# 3) Repeatability across all scenarios (can be big)
rep_slope = repeatability_matrix(experiment; metric=:slope_abs_mean)
fig_rep_slope = plot_repeatability_heatmap(rep_slope; title="Repeatability: slope_abs_mean (Spearman)")

rep_om50 = repeatability_matrix(experiment; metric=:omega_m50)
fig_rep_om50 = plot_repeatability_heatmap(rep_om50; title="Repeatability: omega_m50 (Spearman)")


# 4) Phase 3 correlations for one scenario (print)
ph3 = phase3_correlations(experiment, example_key)
@info "Phase 3 correlations for key=$example_key"
for (tgt, rows) in ph3
    println("\nTarget: ", tgt)
    for r in rows
        println("  ", r.cand, "  Spearman=", round(r.spearman, digits=3), "  logPearson=", round(r.logpearson, digits=3))
    end
end

# 5) Example phase 3 scatter: timescale heterogeneity vs where fragility lives
fig_sc = plot_phase3_scatter(experiment, example_key; target=:omega_m50, cand=:sdlogT)
################################################################################