################################################################################
# ALIGNMENT (WHEN) vs AMPLIFICATION (HOW MUCH)
#
# Goal (simple, publishable logic):
#   1) Show that timescale–strength alignment organizes WHEN structural fragility lives
#      (location of m(w) = e(w)*L(w)).
#   2) Show that amplification / non-normality organizes HOW MUCH fragility you get
#      (magnitude of energy-weighted leverage), while holding alignment ~fixed.
#
# Standardization across communities:
#   - Same S, same u vector (so same diagonal of J = -u_i)
#   - Same diagonal of Abar = -I
#   - Same stability margin: scale A so spectral abscissa alpha(J) = target_alpha
#   - Same frequency grid wvals and same noise C0
#
# Outputs:
#   Experiment A (alignment sweep):
#     vary only the pairing between timescales and network position via permutations
#     -> alignment index changes strongly -> location changes strongly
#
#   Experiment B (non-normality sweep):
#     vary matrix non-normality (symmetric -> feedforward mix)
#     while forcing max alignment each time
#     -> magnitude changes strongly with non-normality, location weakly
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie

# -----------------------------
# helpers
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
            s += 0.5*(y1+y2)*(x2-x1)
        end
    end
    return s
end

spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigvals(Matrix(J))))

function normalize_offdiag!(A::Matrix{Float64})
    for i in 1:size(A,1)
        A[i,i] = 0.0
    end
    n = norm(A)
    n == 0 && return false
    A ./= n
    return true
end

function random_u(S; mean=1.0, cv=0.6, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

function rademacher(rng, S::Int)
    v = Vector{Float64}(undef, S)
    for i in 1:S
        v[i] = (rand(rng) < 0.5) ? -1.0 : 1.0
    end
    return v
end

function sample_Pdirs(S::Int, nP::Int; rng=Random.default_rng())
    Pdirs = Matrix{Float64}[]
    for _ in 1:nP
        P = randn(rng, S, S)
        for i in 1:S
            P[i,i] = 0.0
        end
        n = norm(P)
        n == 0 && continue
        P ./= n
        push!(Pdirs, P)
    end
    return Pdirs
end

# -----------------------------
# frequency machinery
#   R(w) = (i*w*T - Abar)^(-1),  T = diag(1/u)
#   e(w) = tr(R C0 R^dag) / tr(C0)
#   g(w;P) ≈ 2 Re tr(R P Chat) / tr(Chat), Chat = R C0 R^dag
#   L(w) = mean_P |g(w;P)|
# -----------------------------
function estimate_energy_and_g_at_w!(
    F::LU{ComplexF64, Matrix{ComplexF64}},
    sqrtc::Vector{Float64},
    trC0::Float64,
    Pdirs::Vector{Matrix{Float64}};
    nprobe::Int=10,
    rng=Random.default_rng()
)
    S = length(sqrtc)
    nP = length(Pdirs)

    # probes: x_k = R * sqrt(C0) * v_k
    x_list = Vector{Vector{ComplexF64}}(undef, nprobe)
    xnorm2 = zeros(Float64, nprobe)

    for k in 1:nprobe
        v = rademacher(rng, S)
        rhs = ComplexF64.(sqrtc .* v)
        x = F \ rhs
        x_list[k] = x
        xnorm2[k] = real(dot(conj.(x), x))
    end

    trChat = mean(xnorm2)  # ≈ tr(R C0 R^dag)
    e = (isfinite(trChat) && trC0 > 0) ? (trChat / trC0) : NaN

    g = fill(NaN, nP)
    for (pidx, P) in enumerate(Pdirs)
        inn = 0.0
        good = 0
        Pc = ComplexF64.(P)
        for k in 1:nprobe
            x = x_list[k]
            y = Pc * x
            z = F \ y              # z = R P x
            inner = dot(conj.(x), z)
            val = real(inner)
            if isfinite(val)
                inn += val
                good += 1
            end
        end
        if good > 0 && isfinite(trChat) && trChat > 0
            num = inn / good
            g[pidx] = 2.0 * (num / trChat)
        end
    end

    return e, g
end

function spectra_e_L_m(Abar::Matrix{Float64}, u::Vector{Float64}, C0diag::Vector{Float64},
                      wvals::Vector{Float64}, Pdirs::Vector{Matrix{Float64}};
                      nprobe::Int=10,
                      rng=Random.default_rng())
    S = length(u)
    @assert length(C0diag) == S

    sqrtc = sqrt.(C0diag)
    trC0 = sum(C0diag)
    Tmat = Diagonal(1.0 ./ u)

    nw = length(wvals)
    e = fill(NaN, nw)
    L = fill(NaN, nw)

    for (k, w) in enumerate(wvals)
        M = Matrix{ComplexF64}(im*w*Tmat - Abar)
        F = lu(M)
        ek, gk = estimate_energy_and_g_at_w!(F, sqrtc, trC0, Pdirs; nprobe=nprobe, rng=rng)
        e[k] = ek
        gg = filter(isfinite, gk)
        L[k] = isempty(gg) ? NaN : mean(abs.(gg))
    end

    m = similar(e)
    for i in eachindex(m)
        if isfinite(e[i]) && e[i] >= 0 && isfinite(L[i]) && L[i] >= 0
            m[i] = e[i] * L[i]
        else
            m[i] = NaN
        end
    end
    return (e=e, L=L, m=m)
end

function w_logcentroid(w::Vector{Float64}, f::Vector{Float64})
    idx = findall(i -> isposfinite(w[i]) && isfinite(f[i]) && f[i] >= 0, eachindex(f))
    length(idx) < 3 && return NaN
    ww = w[idx]; ff = f[idx]
    den = trapz(ww, ff)
    (isfinite(den) && den > 0) || return NaN
    num = trapz(ww, ff .* log.(ww))
    return exp(num / den)
end

function w_quantile(w::Vector{Float64}, f::Vector{Float64}; q::Float64=0.5)
    @assert 0.0 < q < 1.0
    idx = findall(i -> isposfinite(w[i]) && isfinite(f[i]) && f[i] >= 0, eachindex(f))
    length(idx) < 3 && return NaN
    ww = w[idx]; ff = f[idx]
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

# energy-weighted mean leverage (a clean "HOW MUCH" proxy)
#   Lbar_e = (∫ L(w)e(w) dw) / (∫ e(w) dw) = (∫ m(w) dw)/(∫ e(w) dw)
function energy_weighted_leverage(w::Vector{Float64}, e::Vector{Float64}, m::Vector{Float64})
    idx = findall(i -> isposfinite(w[i]) && isfinite(e[i]) && e[i] >= 0 &&
                       isfinite(m[i]) && m[i] >= 0, eachindex(e))
    length(idx) < 3 && return NaN
    den = trapz(w[idx], e[idx])
    num = trapz(w[idx], m[idx])
    (isfinite(den) && den > 0) || return NaN
    return num / den
end

# -----------------------------
# alignment + amplification descriptors
# -----------------------------
function col_strength_sq(M::AbstractMatrix{<:Real})
    S = size(M,1)
    s2 = zeros(Float64, S)
    for j in 1:S
        s2[j] = dot(M[:, j], M[:, j])
    end
    return s2
end

# alignment index: strength-weighted log(T) minus mean log(T)
# weights from column strengths of J_off = diag(u)*A (off-diagonal dynamics)
function alignment_index(u::Vector{Float64}, A::Matrix{Float64})
    S = length(u)
    T = 1.0 ./ u
    Joff = Diagonal(u) * A
    s2 = col_strength_sq(Joff)
    tot = sum(s2)
    tot <= 0 && return (Aidx=NaN, wstr=NaN)
    w = s2 ./ tot
    mu_w = sum(w .* log.(T))
    mu_0 = mean(log.(T))
    Aidx = mu_w - mu_0
    # predicted characteristic frequency where strong columns live: exp(sum w log(1/T)) = exp(sum w log(u))
    wstr = exp(sum(w .* log.(u)))
    return (Aidx=Aidx, wstr=wstr)
end

# Henrici non-normality on off-diagonal dynamical coupling (J_off)
function henrici_non_normality(M::AbstractMatrix{<:Real})
    nM = norm(M)
    nM == 0 && return NaN
    MtM = transpose(M) * M
    MMt = M * transpose(M)
    return norm(MtM - MMt) / (nM^2)
end

# -----------------------------
# build standardized system: Abar = -I + s*Operm, choose s so alpha(J)=target_alpha
# with J = diag(u)*Abar and diag(Abar) fixed (-1)
# -----------------------------
function find_scale_to_target_alpha(O::Matrix{Float64}, u::Vector{Float64};
    target_alpha::Float64=-0.05,
    s_hi0::Float64=1.0,
    max_grow::Int=40,
    max_iter::Int=60
)
    @assert target_alpha < 0
    S = length(u)
    Du = Diagonal(u)
    Iden = Matrix{Float64}(I, S, S)

    # alpha at s=0: J = -Du
    alpha0 = spectral_abscissa(-Du)
    isfinite(alpha0) || return NaN

    # bracket
    s_lo = 0.0
    s_hi = s_hi0
    function alpha_at(s)
        Abar = -Iden + s*O
        J = Du * Abar
        return spectral_abscissa(J)
    end

    a_hi = alpha_at(s_hi)
    k = 0
    while (isfinite(a_hi) && a_hi < target_alpha) && k < max_grow
        s_hi *= 2.0
        a_hi = alpha_at(s_hi)
        k += 1
    end
    if !(isfinite(a_hi)) || a_hi < target_alpha
        return NaN
    end

    # bisection
    for _ in 1:max_iter
        s_mid = 0.5*(s_lo + s_hi)
        a_mid = alpha_at(s_mid)
        if !isfinite(a_mid)
            s_hi = s_mid
            continue
        end
        if a_mid < target_alpha
            s_lo = s_mid
        else
            s_hi = s_mid
        end
    end
    return 0.5*(s_lo + s_hi)
end

# permute rows+cols consistently
permute_rc(A::Matrix{Float64}, perm::Vector{Int}) = A[perm, perm]

# make a permutation that maximizes or minimizes alignment between column strengths and T
# "max": strong columns -> slow nodes (large T)
# "min": strong columns -> fast nodes (small T)
function perm_for_alignment(O::Matrix{Float64}, u::Vector{Float64}; mode::Symbol=:max)
    S = length(u)
    T = 1.0 ./ u
    # use dynamical column strengths of Joff = diag(u)*O (scale cancels)
    Joff = Diagonal(u) * O
    s2 = col_strength_sq(Joff)
    ord_s = sortperm(s2; rev=true)                 # strongest first
    ord_T = sortperm(T; rev=(mode==:max))          # max: slow first; min: fast first

    perm = Vector{Int}(undef, S)
    # new index ord_T[k] gets old node ord_s[k]
    for k in 1:S
        perm[ord_T[k]] = ord_s[k]
    end
    return perm
end

# -----------------------------
# matrix families
# -----------------------------
function random_sparse_W(S::Int; connectance::Float64=0.06, rng=Random.default_rng())
    W = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        i == j && continue
        rand(rng) < connectance || continue
        W[i,j] = randn(rng)
    end
    return W
end

function make_O_from_W(W::Matrix{Float64}; kind::Symbol=:raw)
    S = size(W,1)
    O = zeros(Float64, S, S)
    if kind == :raw
        O .= W
    elseif kind == :sym
        O .= 0.5*(W + transpose(W))
    elseif kind == :ff
        # feed-forward: strictly upper triangular
        for i in 1:S-1, j in (i+1):S
            O[i,j] = W[i,j]
        end
    else
        error("unknown kind")
    end
    for i in 1:S
        O[i,i] = 0.0
    end
    normalize_offdiag!(O) || error("cannot normalize O")
    return O
end

function mix_O(Oa::Matrix{Float64}, Ob::Matrix{Float64}, eta::Float64)
    @assert 0.0 <= eta <= 1.0
    O = (1.0-eta).*Oa .+ eta.*Ob
    for i in 1:size(O,1)
        O[i,i] = 0.0
    end
    normalize_offdiag!(O) || error("cannot normalize mixed O")
    return O
end

# -----------------------------
# one standardized evaluation
# -----------------------------
function eval_system(O::Matrix{Float64}, u::Vector{Float64}, wvals::Vector{Float64}, Pdirs;
    target_alpha::Float64=-0.05,
    C0_mode::Symbol=:u2,
    nprobe::Int=10,
    rng=Random.default_rng()
)
    S = length(u)
    Iden = Matrix{Float64}(I, S, S)

    # scale to target alpha(J)
    s = find_scale_to_target_alpha(O, u; target_alpha=target_alpha)
    isfinite(s) || return nothing

    A = s .* O
    Abar = -Iden + A
    J = Diagonal(u) * Abar
    alphaJ = spectral_abscissa(J)
    (isfinite(alphaJ) && alphaJ < 0) || return nothing

    C0diag = (C0_mode == :u2) ? (u.^2) : ones(Float64, S)

    sp = spectra_e_L_m(Abar, u, C0diag, wvals, Pdirs; nprobe=nprobe, rng=rng)
    e, L, m = sp.e, sp.L, sp.m

    # "when" summaries
    w_m50  = w_quantile(wvals, m; q=0.5)
    w_mctr = w_logcentroid(wvals, m)
    w_Lctr = w_logcentroid(wvals, L)
    w_ectr = w_logcentroid(wvals, e)

    # "how much" summary
    Lbar_e = energy_weighted_leverage(wvals, e, m)

    # alignment + predicted location from weights
    al = alignment_index(u, A)

    # amplification descriptor
    Joff = Diagonal(u) * A
    hen = henrici_non_normality(Joff)

    return (
        s=s, alpha=alphaJ,
        Aidx=al.Aidx, wstr=al.wstr,
        hen=hen,
        w_m50=w_m50, w_mctr=w_mctr,
        w_Lctr=w_Lctr, w_ectr=w_ectr,
        Lbar_e=Lbar_e
    )
end

# -----------------------------
# correlation helper (Spearman)
# -----------------------------
function spearman(x::AbstractVector, y::AbstractVector)
    idx = findall(i -> isfinite(x[i]) && isfinite(y[i]), eachindex(x))
    length(idx) < 6 && return NaN
    xv = x[idx]; yv = y[idx]

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

# -----------------------------
# EXPERIMENT A: alignment sweep (vary pairing only)
#   - fixed O (same interaction pattern family)
#   - vary permutation of nodes relative to fixed u
#   - each case scaled to same alpha(J)
# -----------------------------
function experiment_alignment_sweep(; S::Int=90, nperm::Int=70, seed::Int=1,
    connectance::Float64=0.06,
    target_alpha::Float64=-0.05,
    nP::Int=12,
    nprobe::Int=10,
    wvals = 10 .^ range(log10(1e-4), log10(1e4); length=70)
)
    rng = MersenneTwister(seed)
    u = collect(random_u(S; mean=1.0, cv=0.6, rng=rng))

    W = random_sparse_W(S; connectance=connectance, rng=rng)
    O_sym = make_O_from_W(W; kind=:sym)
    O_ff  = make_O_from_W(W; kind=:ff)
    # pick a moderate non-normal mix and keep it fixed
    O0 = mix_O(O_sym, O_ff, 0.55)

    Pdirs = sample_Pdirs(S, nP; rng=rng)
    wvals = collect(float.(wvals))

    perms = Vector{Vector{Int}}()
    push!(perms, perm_for_alignment(O0, u; mode=:max))
    push!(perms, perm_for_alignment(O0, u; mode=:min))
    for _ in 1:(nperm-2)
        push!(perms, randperm(rng, S))
    end

    Aidx = Float64[]
    w_m50 = Float64[]
    w_mctr = Float64[]
    wstr = Float64[]
    Lbar_e = Float64[]
    hen = Float64[]

    for (k, perm) in enumerate(perms)
        O = permute_rc(copy(O0), perm)
        normalize_offdiag!(O) || continue

        out = eval_system(O, u, wvals, Pdirs;
                          target_alpha=target_alpha,
                          C0_mode=:u2,
                          nprobe=nprobe,
                          rng=MersenneTwister(seed + 10000*k))
        out === nothing && continue

        push!(Aidx, out.Aidx)
        push!(w_m50, out.w_m50)
        push!(w_mctr, out.w_mctr)
        push!(wstr, out.wstr)
        push!(Lbar_e, out.Lbar_e)
        push!(hen, out.hen)
    end

    return (u=u, wvals=wvals,
            Aidx=Aidx, w_m50=w_m50, w_mctr=w_mctr, wstr=wstr,
            Lbar_e=Lbar_e, hen=hen)
end

# -----------------------------
# EXPERIMENT B: non-normality sweep (vary amplification only)
#   - vary eta from symmetric (normal) to feedforward (non-normal)
#   - force max alignment each time (so alignment ~constant)
#   - scale to same alpha(J)
# -----------------------------
function experiment_nonnormal_sweep(; S::Int=90, seed::Int=2,
    connectance::Float64=0.06,
    etas = range(0.0, 1.0; length=13),
    target_alpha::Float64=-0.05,
    nP::Int=12,
    nprobe::Int=10,
    wvals = 10 .^ range(log10(1e-4), log10(1e4); length=70)
)
    rng = MersenneTwister(seed)
    u = collect(random_u(S; mean=1.0, cv=0.6, rng=rng))

    W = random_sparse_W(S; connectance=connectance, rng=rng)
    O_sym = make_O_from_W(W; kind=:sym)
    O_ff  = make_O_from_W(W; kind=:ff)

    Pdirs = sample_Pdirs(S, nP; rng=rng)
    wvals = collect(float.(wvals))

    eta_v = Float64[]
    Aidx = Float64[]
    w_m50 = Float64[]
    w_mctr = Float64[]
    wstr = Float64[]
    Lbar_e = Float64[]
    hen = Float64[]

    for (k, eta) in enumerate(etas)
        Oeta = mix_O(O_sym, O_ff, float(eta))

        # enforce maximal alignment by permuting nodes
        perm = perm_for_alignment(Oeta, u; mode=:max)
        Oeta = permute_rc(copy(Oeta), perm)
        normalize_offdiag!(Oeta) || continue

        out = eval_system(Oeta, u, wvals, Pdirs;
                          target_alpha=target_alpha,
                          C0_mode=:u2,
                          nprobe=nprobe,
                          rng=MersenneTwister(seed + 20000*k))
        out === nothing && continue

        push!(eta_v, float(eta))
        push!(Aidx, out.Aidx)
        push!(w_m50, out.w_m50)
        push!(w_mctr, out.w_mctr)
        push!(wstr, out.wstr)
        push!(Lbar_e, out.Lbar_e)
        push!(hen, out.hen)
    end

    return (u=u, wvals=wvals, eta=eta_v,
            Aidx=Aidx, w_m50=w_m50, w_mctr=w_mctr, wstr=wstr,
            Lbar_e=Lbar_e, hen=hen)
end

# -----------------------------
# plotting + printed diagnostics
# -----------------------------
function summarize_and_plot(al, nn; figsize=(1750, 1100))
    # alignment sweep: expect Aidx -> location (log w_m50), weak -> magnitude (log Lbar_e)
    xA = al.Aidx
    y_when = log10.(al.w_m50)
    y_how  = log10.(al.Lbar_e)

    # non-normal sweep: expect hen -> magnitude, weak -> location; alignment ~flat
    xH = log10.(nn.hen)
    y2_how  = log10.(nn.Lbar_e)
    y2_when = log10.(nn.w_m50)

    @info "ALIGN sweep: Spearman(Aidx, log10(w_m50)) = $(round(spearman(xA, y_when), digits=3))"
    @info "ALIGN sweep: Spearman(Aidx, log10(Lbar_e)) = $(round(spearman(xA, y_how), digits=3))"
    @info "ALIGN sweep: Spearman(log10(wstr), log10(w_m50)) = $(round(spearman(log10.(al.wstr), log10.(al.w_m50)), digits=3))"
    @info "NONNORM sweep: Spearman(log10(hen), log10(Lbar_e)) = $(round(spearman(xH, y2_how), digits=3))"
    @info "NONNORM sweep: Spearman(log10(hen), log10(w_m50)) = $(round(spearman(xH, y2_when), digits=3))"
    @info "NONNORM sweep: alignment range (Aidx) = [$(round(minimum(nn.Aidx),digits=3)), $(round(maximum(nn.Aidx),digits=3))]"

    fig = Figure(size=figsize)

    # A: alignment -> WHEN
    ax1 = Axis(fig[1,1];
        xlabel="alignment index Aidx (strength on slow minus mean)",
        ylabel="log10(w_m50)  (where m(w) sits)",
        title="Experiment A: alignment organizes WHEN"
    )
    scatter!(ax1, xA, y_when, markersize=7)
    text!(ax1, 0.03, 0.97, space=:relative, align=(:left,:top),
          text="Spearman = $(round(spearman(xA, y_when),digits=3))\nN=$(length(xA))")

    # B: alignment -> HOW MUCH (should be weaker)
    ax2 = Axis(fig[1,2];
        xlabel="alignment index Aidx",
        ylabel="log10(Lbar_e)  (energy-weighted mean leverage)",
        title="Experiment A: alignment weakly affects HOW MUCH"
    )
    scatter!(ax2, xA, y_how, markersize=7)
    text!(ax2, 0.03, 0.97, space=:relative, align=(:left,:top),
          text="Spearman = $(round(spearman(xA, y_how),digits=3))")

    # C: predicted location from weights (wstr) vs observed (w_m50)
    ax3 = Axis(fig[1,3];
        xscale=log10, yscale=log10,
        xlabel="w_str = exp(sum w_j log(u_j))  (strength-weighted timescale)",
        ylabel="w_m50  (median location of m(w))",
        title="Experiment A: alignment predictor matches location"
    )
    idxC = findall(i -> isposfinite(al.wstr[i]) && isposfinite(al.w_m50[i]), eachindex(al.wstr))
    scatter!(ax3, al.wstr[idxC], al.w_m50[idxC], markersize=7)
    text!(ax3, 0.03, 0.97, space=:relative, align=(:left,:top),
          text="Spearman(log,log) = $(round(spearman(log10.(al.wstr), log10.(al.w_m50)),digits=3))")

    # D: non-normality -> HOW MUCH
    ax4 = Axis(fig[2,1];
        xlabel="log10(henrici non-normality of J_off)",
        ylabel="log10(Lbar_e)",
        title="Experiment B: amplification organizes HOW MUCH"
    )
    scatter!(ax4, xH, y2_how, markersize=7)
    text!(ax4, 0.03, 0.97, space=:relative, align=(:left,:top),
          text="Spearman = $(round(spearman(xH, y2_how),digits=3))\nN=$(length(xH))")

    # E: non-normality -> WHEN (should be weaker)
    ax5 = Axis(fig[2,2];
        xlabel="log10(henrici non-normality of J_off)",
        ylabel="log10(w_m50)",
        title="Experiment B: amplification weakly affects WHEN"
    )
    scatter!(ax5, xH, y2_when, markersize=7)
    text!(ax5, 0.03, 0.97, space=:relative, align=(:left,:top),
          text="Spearman = $(round(spearman(xH, y2_when),digits=3))")

    # F: alignment stays ~fixed across eta (design check)
    ax6 = Axis(fig[2,3];
        xlabel="eta (symmetric -> feedforward)",
        ylabel="alignment index Aidx",
        title="Experiment B: alignment held (approximately) constant"
    )
    scatter!(ax6, nn.eta, nn.Aidx, markersize=7)

    display(fig)
end

# -----------------------------
# MAIN
# -----------------------------
S = 120
wvals = 10 .^ range(log10(1e-4), log10(1e4); length=70)

al = experiment_alignment_sweep(
    S=S,
    nperm=80,
    seed=11,
    connectance=0.06,
    target_alpha=-0.05,
    nP=14,
    nprobe=10,
    wvals=wvals
)

nn = experiment_nonnormal_sweep(
    S=S,
    seed=22,
    connectance=0.06,
    etas=range(0.0, 1.0; length=13),
    target_alpha=-0.05,
    nP=14,
    nprobe=10,
    wvals=wvals
)

summarize_and_plot(al, nn)
################################################################################
