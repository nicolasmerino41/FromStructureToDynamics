################################################################################
# THREE-FACET DECOMPOSITION TEST (Jeff's idea)
#
# Goal:
#   Test whether the realized response (|ΔV|/V) to a structural perturbation
#   can be explained by three facets:
#
#     (1) Raw structural change magnitude:      d = ||A' - A||_F
#     (2) Dynamical/resolvent component:        S_hom  (computed with homogeneous T)
#     (3) Timescale (filtering/alignment) part: M_T = S_real / S_hom
#
#   so that predicted sensitivity is:
#       |ΔV|/V  ≈  d * S_hom * M_T  = d * S_real
#
# We then quantify:
#   - how well prediction matches realized |ΔV|/V (Lyapunov, perturbed system)
#   - how much variance in log(|ΔV|/V) is explained by (log d, log S_hom, log M_T)
#   - Shapley-style R^2 contributions of the three facets
#
# Notes:
# - Systems are (T,A) pairs. Here T = diag(1/u). Abar = -I + A (diag fixed -1).
# - Perturbation direction P is obtained from a REWIRING move on A (topology change),
#   then applied in a linear regime: Abar' = Abar + d * P  with ||P||_F = 1.
# - Variability: V = tr(Sigma)/tr(C0) where Sigma solves Lyapunov:
#       J Sigma + Sigma J' + Q = 0,  J = diag(u)*Abar,  Q = diag(u)C0diag diag(u)
# - Sensitivity S(T,A,P) comes from frequency-domain first order theory:
#       S = | ∫ g(ω;P) e(ω) dω / ∫ e(ω) dω |
#
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# -----------------------------
# Helpers
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
            s += 0.5*(y1+y2)*(x2-x1)
        end
    end
    return s
end

logsafe(x; eps=1e-300) = (isfinite(x) && x > eps) ? log(x) : NaN

function spectral_abscissa(J::AbstractMatrix)
    maximum(real.(eigvals(Matrix(J))))
end

# Henrici departure from normality (Frobenius)
# H = sqrt( ||J||_F^2 - sum |eig|^2 ) / ||J||_F
function henrici_departure(J::AbstractMatrix)
    nJ = norm(J)
    nJ == 0 && return 0.0
    lam = eigvals(Matrix(J))
    s2 = sum(abs2, lam)
    val = max(nJ^2 - s2, 0.0)
    return sqrt(val) / nJ
end

# -----------------------------
# Random timescale vector u
# -----------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# Homogeneous u with same mean timescale as u:
# mean(T) = mean(1/u). Set u_hom = 1/mean(T).
function homogeneous_u_like(u::Vector{Float64})
    Tm = mean(1.0 ./ u)
    uh = (Tm > 0) ? (1.0 / Tm) : 1.0
    fill(uh, length(u))
end

# -----------------------------
# Build a sparse "ecological-like" interaction template O (diag=0)
# (simple trophic-ish generator as before)
# -----------------------------
function trophic_O(S::Int;
    connectance::Float64,
    trophic_align::Float64,
    reciprocity::Float64,
    σ::Float64,
    rng=Random.default_rng()
)
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

# Choose scale s so that α(J)=target_alpha (<0), where:
# Abar = -I + s*O,  J = diag(u)*Abar
function find_scale_to_target_alpha(O::Matrix{Float64}, u::Vector{Float64};
    target_alpha::Float64 = -0.05,
    max_grow::Int=40,
    max_iter::Int=60
)
    @assert target_alpha < 0
    S = length(u)
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
# Variability in time domain (Lyapunov)
# T xdot = Abar x + ξ,  E[ξξ']=C0 δ
# Equivalent: xdot = J x + η with J=diag(u)Abar and Q=diag(u)C0 diag(u)
# Lyapunov: JΣ + ΣJ' + Q = 0
# V = tr(Σ)/tr(C0)
# -----------------------------
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

# -----------------------------
# Rewiring direction P from A (diag=0)
# 1) pick m existing edges
# 2) move them to m currently empty off-diagonal positions
# This creates A_rew with same #edges and same weights moved.
# Then direction is D = A_rew - A, normalized: P = D/||D||_F.
# -----------------------------
function rewire_direction(A::Matrix{Float64};
    frac_move::Float64=0.20,
    rng=Random.default_rng()
)
    S = size(A,1)
    @assert size(A,2) == S

    edges = Tuple{Int,Int}[]
    nonedges = Tuple{Int,Int}[]
    for i in 1:S, j in 1:S
        i == j && continue
        if A[i,j] != 0.0
            push!(edges, (i,j))
        else
            push!(nonedges, (i,j))
        end
    end
    length(edges) < 5 && return nothing
    length(nonedges) < 5 && return nothing

    m = max(1, Int(round(frac_move * length(edges))))
    m = min(m, length(edges), length(nonedges))

    sel_edges = rand(rng, edges, m)
    sel_new   = rand(rng, nonedges, m)

    Anew = copy(A)
    w = [A[i,j] for (i,j) in sel_edges]
    for (i,j) in sel_edges
        Anew[i,j] = 0.0
    end
    # move weights (keep multiset; random assignment)
    wperm = shuffle(rng, w)
    for k in 1:m
        i,j = sel_new[k]
        Anew[i,j] = wperm[k]
    end

    D = Anew - A
    nD = norm(D)
    nD == 0 && return nothing
    P = D ./ nD
    return P
end

# -----------------------------
# Frequency-domain sensitivity S(T,A,P)
# R(ω) = (i ω T - Abar)^(-1),  T=diag(1/u)
# e(ω) = tr(R C0 R†)/tr(C0)
# g(ω;P) ≈ 2 Re tr(R P Ĉ)/tr(Ĉ),   Ĉ = R C0 R†
# S = | ∫ g e dω / ∫ e dω |
#
# We estimate traces by Hutchinson probes:
# tr(R C0 R†) = E_v || R sqrt(C0) v ||^2
# Re tr(R P Ĉ) = E_v Re[ x† (R P x) ], where x = R sqrt(C0) v
# -----------------------------
function estimate_e_and_g_single!(
    F::LU{ComplexF64, Matrix{ComplexF64}},
    sqrtc::Vector{Float64},
    trC0::Float64,
    P::Matrix{Float64};
    nprobe::Int=12,
    rng=Random.default_rng()
)
    S = length(sqrtc)

    # probes Rademacher
    xnorm2 = zeros(Float64, nprobe)
    inners = zeros(Float64, nprobe)

    for k in 1:nprobe
        v = rand(rng, (-1.0, 1.0), S)
        rhs = ComplexF64.(sqrtc .* v)
        x = F \ rhs
        xnorm2[k] = real(dot(conj.(x), x))

        y = ComplexF64.(P) * x
        z = F \ y
        inners[k] = real(dot(conj.(x), z))
    end

    trChat = mean(xnorm2)
    eω = (isfinite(trChat) && trC0 > 0) ? (trChat / trC0) : NaN

    if isfinite(trChat) && trChat > 0 && isfinite(mean(inners))
        gω = 2.0 * (mean(inners) / trChat)
    else
        gω = NaN
    end
    return eω, gω
end

function sensitivity_S(Abar::Matrix{Float64}, u::Vector{Float64}, C0diag::Vector{Float64},
                       P::Matrix{Float64}, ωvals::Vector{Float64};
                       nprobe::Int=12, rng=Random.default_rng())
    Sdim = length(u)
    @assert length(C0diag) == Sdim
    @assert size(P,1) == Sdim && size(P,2) == Sdim

    sqrtc = sqrt.(C0diag)
    trC0  = sum(C0diag)
    Tmat  = Diagonal(1.0 ./ u)

    nω = length(ωvals)
    e = fill(NaN, nω)
    g = fill(NaN, nω)

    for (k, ω0) in enumerate(ωvals)
        ω = float(ω0)
        Mω = Matrix{ComplexF64}(im*ω*Tmat - Abar)
        F = lu(Mω)
        ek, gk = estimate_e_and_g_single!(F, sqrtc, trC0, P; nprobe=nprobe, rng=rng)
        e[k] = ek
        g[k] = gk
    end

    idx = findall(i -> isfinite(ωvals[i]) && ωvals[i] > 0 && isfinite(e[i]) && e[i] >= 0 && isfinite(g[i]), eachindex(e))
    length(idx) < 5 && return (S=NaN, denom=NaN, e=e, g=g)

    denom = trapz(ωvals[idx], e[idx])
    (isfinite(denom) && denom > 0) || return (S=NaN, denom=denom, e=e, g=g)

    num = trapz(ωvals[idx], (g[idx] .* e[idx]))
    Sval = abs(num / denom)

    return (S=Sval, denom=denom, e=e, g=g)
end

# -----------------------------
# OLS + R2 + Shapley (3 predictors)
# -----------------------------
function ols_fit(X::Matrix{Float64}, y::Vector{Float64})
    # X includes intercept column if desired
    β = (X'X) \ (X'y)
    yhat = X * β
    ssr = sum((y .- yhat).^2)
    sst = sum((y .- mean(y)).^2)
    R2  = (sst > 0) ? (1.0 - ssr/sst) : NaN
    return (β=β, yhat=yhat, R2=R2)
end

function r2_subset(xraw, xdyn, xT, y; use_raw::Bool, use_dyn::Bool, use_T::Bool)
    n = length(y)
    cols = Float64[]
    # build X with intercept
    X = ones(Float64, n, 1)
    if use_raw
        X = hcat(X, xraw)
    end
    if use_dyn
        X = hcat(X, xdyn)
    end
    if use_T
        X = hcat(X, xT)
    end
    fit = ols_fit(X, y)
    return fit.R2
end

function shapley_R2(xraw, xdyn, xT, y)
    # enumerate all permutations of 3 predictors
    perms = [
        (:raw, :dyn, :T),
        (:raw, :T, :dyn),
        (:dyn, :raw, :T),
        (:dyn, :T, :raw),
        (:T, :raw, :dyn),
        (:T, :dyn, :raw),
    ]
    contrib = Dict(:raw=>0.0, :dyn=>0.0, :T=>0.0)

    for p in perms
        used_raw = false
        used_dyn = false
        used_T   = false
        Rprev = r2_subset(xraw, xdyn, xT, y; use_raw=false, use_dyn=false, use_T=false)
        for name in p
            if name == :raw
                used_raw = true
            elseif name == :dyn
                used_dyn = true
            else
                used_T = true
            end
            Rnew = r2_subset(xraw, xdyn, xT, y; use_raw=used_raw, use_dyn=used_dyn, use_T=used_T)
            contrib[name] += (Rnew - Rprev)
            Rprev = Rnew
        end
    end

    for k in keys(contrib)
        contrib[k] /= length(perms)
    end
    return contrib
end

# -----------------------------
# System container
# -----------------------------
struct BaseSys
    u::Vector{Float64}
    Abar::Matrix{Float64}       # -I + A
    C0diag::Vector{Float64}
    henrici::Float64
end

function build_systems(; S::Int=80, nsys::Int=25, seed::Int=1234,
    target_alpha::Float64=-0.05,
    u_mean::Float64=1.0, u_cv::Float64=0.5,
    connectance_rng=(0.03, 0.12),
    trophic_align_rng=(0.55, 0.98),
    reciprocity_rng=(0.00, 0.20),
    σ_rng=(0.3, 1.5),
    C0_mode::Symbol=:u2
)
    rng0 = MersenneTwister(seed)
    out = BaseSys[]
    attempts = 0
    while length(out) < nsys && attempts < 10_000
        attempts += 1
        rng = MersenneTwister(rand(rng0, 1:10^9))

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

        # stability check
        α = spectral_abscissa(Diagonal(u) * Abar)
        (isfinite(α) && α < 0) || continue

        C0diag = if C0_mode == :u2
            u.^2
        elseif C0_mode == :I
            ones(Float64, S)
        else
            error("C0_mode must be :u2 or :I")
        end

        # "gain" proxy
        Joff = Diagonal(u) * (Abar + Matrix{Float64}(I, S, S))  # offdiag dynamics (diag removed)
        H = henrici_departure(Joff)

        push!(out, BaseSys(u, Abar, C0diag, H))
    end
    @info "Built $(length(out)) systems after $attempts attempts."
    return out
end

# -----------------------------
# Main experiment: generate samples (system, rewiring-direction, d)
# and compute:
#   y_act = |ΔV|/V  (Lyapunov)
#   facets: d, S_hom, M_T
#   y_pred = d * S_hom * M_T
# -----------------------------
function run_threefacet_test(; S::Int=80, nsys::Int=25, ndir::Int=3, neps::Int=3,
    seed::Int=1234,
    ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=70),
    d_range = (1e-2, 3e-1),         # raw magnitudes to sample (log-uniform)
    frac_move = 0.20,               # rewiring intensity used to define P direction
    stability_margin = 1e-4,
    nprobe::Int=10,
    target_alpha::Float64=-0.05,
    C0_mode::Symbol=:u2
)
    ωvals = collect(float.(ωvals))
    rng0  = MersenneTwister(seed)

    systems = build_systems(S=S, nsys=nsys, seed=seed, target_alpha=target_alpha, C0_mode=C0_mode)

    # dataset arrays
    y_act = Float64[]
    y_pred = Float64[]

    d_raw = Float64[]
    S_hom = Float64[]
    M_T   = Float64[]

    Hgain = Float64[]   # optional diagnostic: henrici
    sys_id = Int[]

    example = nothing

    for (sid, sys) in enumerate(systems)
        rng = MersenneTwister(rand(rng0, 1:10^9))

        V0 = variability_time_domain(sys.Abar, sys.u, sys.C0diag)
        isfinite(V0) || continue
        V0 > 0 || continue

        uh = homogeneous_u_like(sys.u)

        for kdir in 1:ndir
            P = rewire_direction(sys.Abar + Matrix{Float64}(I, S, S); frac_move=frac_move, rng=rng)
            P === nothing && continue

            # enforce diag(P)=0 and ||P||=1 (rewire_direction already does, but be safe)
            for i in 1:S
                P[i,i] = 0.0
            end
            nP = norm(P)
            nP == 0 && continue
            P ./= nP

            # compute sensitivity components once per (sys,P)
            s_real = sensitivity_S(sys.Abar, sys.u, sys.C0diag, P, ωvals; nprobe=nprobe, rng=rng)
            s_hom  = sensitivity_S(sys.Abar, uh,   sys.C0diag, P, ωvals; nprobe=nprobe, rng=rng)

            (isfinite(s_real.S) && s_real.S > 0) || continue
            (isfinite(s_hom.S)  && s_hom.S  > 0) || continue

            MT = s_real.S / s_hom.S
            (isfinite(MT) && MT > 0) || continue

            # sample raw magnitudes d (log-uniform)
            dmin, dmax = d_range
            for ke in 1:neps
                d = exp(rand(rng, Uniform(log(dmin), log(dmax))))

                # perturbed system
                Abarp = sys.Abar + d .* P
                αp = spectral_abscissa(Diagonal(sys.u) * Abarp)
                (isfinite(αp) && αp < -stability_margin) || continue

                Vp = variability_time_domain(Abarp, sys.u, sys.C0diag)
                (isfinite(Vp) && Vp > 0) || continue

                y = abs(Vp - V0) / V0
                yhat = d * s_hom.S * MT   # = d * s_real.S

                (isfinite(y) && y > 0) || continue
                (isfinite(yhat) && yhat > 0) || continue

                push!(y_act, y)
                push!(y_pred, yhat)

                push!(d_raw, d)
                push!(S_hom, s_hom.S)
                push!(M_T, MT)

                push!(Hgain, sys.henrici)
                push!(sys_id, sid)

                if example === nothing
                    example = (ωvals=ωvals, e=s_real.e, g=s_real.g, e_hom=s_hom.e, g_hom=s_hom.g, d=d)
                end
            end
        end
    end

    return (y_act=y_act, y_pred=y_pred,
            d_raw=d_raw, S_hom=S_hom, M_T=M_T,
            Hgain=Hgain, sys_id=sys_id,
            example=example)
end

# -----------------------------
# Analysis + plotting
# -----------------------------
function summarize_threefacet(res; figsize=(1600, 900))
    y  = res.y_act
    yp = res.y_pred

    d  = res.d_raw
    Sh = res.S_hom
    MT = res.M_T

    # log space vectors (filter finite)
    idx = findall(i ->
        isfinite(y[i])  && y[i]  > 0 &&
        isfinite(yp[i]) && yp[i] > 0 &&
        isfinite(d[i])  && d[i]  > 0 &&
        isfinite(Sh[i]) && Sh[i] > 0 &&
        isfinite(MT[i]) && MT[i] > 0, eachindex(y))

    @info "Samples kept: N=$(length(idx))"
    length(idx) < 20 && @warn "Few samples; increase nsys/ndir/neps or relax constraints."

    ly  = log.(y[idx])
    lyp = log.(yp[idx])

    xraw = log.(d[idx])
    xdyn = log.(Sh[idx])
    xT   = log.(MT[idx])

    # accuracy of theoretical prediction
    ρ_pred = (length(idx) >= 6) ? cor(ly, lyp) : NaN
    R2_pred = (length(idx) >= 6) ? (cor(ly, lyp)^2) : NaN

    # OLS fit: ly ~ 1 + xraw + xdyn + xT
    X = hcat(ones(Float64, length(idx)), xraw, xdyn, xT)
    fit = ols_fit(X, ly)

    # Shapley R2 contributions
    contrib = shapley_R2(xraw, xdyn, xT, ly)

    @info "log(y) vs log(y_pred): corr=$(round(ρ_pred, digits=3))  R2≈$(round(R2_pred, digits=3))"
    @info "OLS R2 (three facets): $(round(fit.R2, digits=3))"
    @info "OLS betas: intercept=$(round(fit.β[1],digits=3))  raw=$(round(fit.β[2],digits=3))  dyn=$(round(fit.β[3],digits=3))  T=$(round(fit.β[4],digits=3))"
    @info "Shapley R2 contributions: raw=$(round(contrib[:raw],digits=3))  dyn=$(round(contrib[:dyn],digits=3))  T=$(round(contrib[:T],digits=3))"

    fig = Figure(size=figsize)

    # correlations for individual facets
    ρ_raw = length(xraw) >= 6 ? cor(xraw, ly) : NaN
    ρ_dyn = length(xdyn) >= 6 ? cor(xdyn, ly) : NaN
    ρ_T   = length(xT)   >= 6 ? cor(xT,   ly) : NaN

    fig = Figure(size=figsize)

    # Panel A: predicted vs realized
    ax1 = Axis(fig[1,1]; xscale=log10, yscale=log10,
        xlabel="predicted |ΔV|/V  = d * S_hom * M_T",
        ylabel="realized |ΔV|/V  (Lyapunov)",
        title="A) Prediction check (frequency theory vs Lyapunov)"
    )
    scatter!(ax1, yp[idx], y[idx], markersize=6)
    text!(ax1, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr(log,log)=$(round(ρ_pred,digits=3))  R²≈$(round(R2_pred,digits=3))  N=$(length(idx))")

    # Panel B: raw facet
    ax2 = Axis(fig[1,2];
        xlabel="log d = log ||A'−A||_F", ylabel="log |ΔV|/V",
        title="B) Raw structural change vs response"
    )
    scatter!(ax2, xraw, ly, markersize=6)
    text!(ax2, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr=$(round(ρ_raw, digits=3))")

    # Panel C: dynamical facet
    ax3 = Axis(fig[1,3];
        xlabel="log S_hom (homogeneous timescales)", ylabel="log |ΔV|/V",
        title="C) Dynamical / resolvent facet vs response"
    )
    scatter!(ax3, xdyn, ly, markersize=6)
    text!(ax3, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr=$(round(ρ_dyn, digits=3))")

    # Panel D: timescale facet
    ax4 = Axis(fig[2,1];
        xlabel="log M_T = log(S_real / S_hom)", ylabel="log |ΔV|/V",
        title="D) Timescale modulation vs response"
    )
    scatter!(ax4, xT, ly, markersize=6)
    text!(ax4, 0.05, 0.95, space=:relative, align=(:left,:top),
          text="corr=$(round(ρ_T, digits=3))")

    # Panel E: Shapley barplot
    ax5 = Axis(fig[2,2];
        xlabel="facet", ylabel="Shapley contribution to R2",
        title="E) Variance explained contributions (Shapley R2)"
    )
    xs = [1,2,3]
    ys = [contrib[:raw], contrib[:dyn], contrib[:T]]
    barplot!(ax5, xs, ys)
    ax5.xticks = (xs, ["raw (d)", "dyn (S_hom)", "timescale (M_T)"])

    # Panel F: example spectra
    ex = res.example
    if ex !== nothing
        ω = ex.ωvals
        ax6 = Axis(fig[2,3]; xscale=log10, yscale=log10,
            xlabel="omega", ylabel="value",
            title="F) Example spectra: e(ω), g(ω) (real vs hom)"
        )
        # e(ω)
        lines!(ax6, ω, [x>0 && isfinite(x) ? x : NaN for x in ex.e], linewidth=3)
        lines!(ax6, ω, [x>0 && isfinite(x) ? x : NaN for x in ex.e_hom], linewidth=2, linestyle=:dash)
        # |g(ω)|
        lines!(ax6, ω, [abs(x)>0 && isfinite(x) ? abs(x) : NaN for x in ex.g], linewidth=3, linestyle=:dot)
        lines!(ax6, ω, [abs(x)>0 && isfinite(x) ? abs(x) : NaN for x in ex.g_hom], linewidth=2, linestyle=:dashdot)
    end

    display(fig)
    return nothing
end

# -----------------------------
# RUN
# -----------------------------
ωvals = 10 .^ range(log10(1e-4), log10(1e4); length=70)

res = run_threefacet_test(
    S=120,
    nsys=60,
    ndir=4,
    neps=3,
    seed=1234,
    ωvals=ωvals,
    d_range=(1e-2, 2e-1),
    frac_move=0.25,
    stability_margin=1e-4,
    nprobe=10,
    target_alpha=-0.05,
    C0_mode=:u2
)

summarize_threefacet(res)
################################################################################