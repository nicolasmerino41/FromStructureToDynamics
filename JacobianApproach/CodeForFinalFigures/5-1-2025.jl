################################################################################
# SIMPLE TIME-DOMAIN STRUCTURE EFFECT PIPELINE (WEIGHTED rmed)
#
# What this script does (time-domain only):
#   For each η (directionality knob), build several stable base communities.
#   For each base, generate many "rewire-like" perturbation directions Pdir by
#   reshuffling the off-diagonal of Abar (diag fixed at -1), then normalise Pdir.
#   Apply a fixed ε per base:  Abar_pert = Abar + ε * Pdir
#   Reject if the perturbed system becomes unstable (α >= -margin).
#
# Compute:
#   - biomass-weighted rmed(t) curves for base and perturbed
#   - Δ(t) = |rmed_base(t) - rmed_pert(t)|
#   - max_delta = max_t Δ(t)
#   - delta_end = Δ(t_max) (last time point)
#   - bump_strength = max_delta / delta_end
#   - bump_excess   = max_delta - delta_end
#
# Plot functions requested:
#   1) plot_rmed_lines_by_eta(res): one panel per η, base rmed in black, rewired in red
#   2) plot_delta_profiles_by_eta(res): one axis, Δ(t) mean profile per η (different colors)
#   3) plot_bump_summaries(res): bump_strength vs η, bump_excess vs η,
#      max_delta vs η, delta_end vs η, and bump_strength distribution vs η
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie
using Base.Threads

# If you see BLAS oversubscription (Julia threads + BLAS threads), uncomment:
# BLAS.set_num_threads(1)

# ---------------------------
# Helpers
# ---------------------------
meanfinite(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
medfinite(x)  = (v = filter(isfinite, x); isempty(v) ? NaN : median(v))
q25(x) = (v = filter(isfinite, x); isempty(v) ? NaN : quantile(v, 0.25))
q75(x) = (v = filter(isfinite, x); isempty(v) ? NaN : quantile(v, 0.75))

spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigen(J).values))

function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# Extract off-diagonal part (diag -> 0)
function offdiag_part(M::AbstractMatrix)
    S = size(M,1)
    O = copy(Matrix(M))
    @inbounds for i in 1:S
        O[i,i] = 0.0
    end
    return O
end

# Permute off-diagonal entries (including zeros), keeping diagonal fixed
function reshuffle_offdiagonal(M::AbstractMatrix; rng=Random.default_rng())
    S = size(M, 1)
    M2 = copy(Matrix(M))
    vals = Float64[]
    idxs = Tuple{Int,Int}[]
    @inbounds for i in 1:S, j in 1:S
        if i != j
            push!(vals, float(M2[i,j]))
            push!(idxs, (i,j))
        end
    end
    perm = randperm(rng, length(vals))
    @inbounds for k in 1:length(vals)
        (i,j) = idxs[k]
        M2[i,j] = vals[perm[k]]
    end
    return M2
end

# ---------------------------
# One knob η: directionality / non-normality
# ---------------------------
"""
Build sparse random M, then
  Oη = U + (1-η)*L   (U upper, L lower)
η=0 ~ more bidirectional; η=1 ~ feedforward-ish.
"""
function make_O_eta(S::Int, η::Real; p::Real=0.05, σ::Real=1.0, rng=Random.default_rng())
    @assert 0.0 <= η <= 1.0
    M = zeros(Float64, S, S)
    @inbounds for i in 1:S, j in 1:S
        i == j && continue
        rand(rng) < p && (M[i,j] = randn(rng) * σ)
    end
    U = triu(M, 1)
    L = tril(M, -1)
    return U + (1.0 - float(η)) * L
end

"""
Build a strictly trophic interaction matrix B.

For every unordered pair (i,j), allowed patterns only:
  (+,-), (-,+), (+,0), (0,-), or (0,0)

Forbidden:
  (++), (--), (-,0), (0,+)

Arguments:
- S        : number of species
- L        : number of interacting unordered pairs
- σ        : interaction strength scale
- rng      : RNG

Returns:
- B :: Matrix{Float64} with diag = 0
"""
function make_O_eta_trophic(
    S::Int,
    η::Real;
    p::Real = 0.05,
    σ::Real = 1.0,
    rng = Random.default_rng()
)
    @assert 0.0 ≤ η ≤ 1.0

    M = zeros(Float64, S, S)

    @inbounds for i in 1:S-1, j in i+1:S
        rand(rng) ≥ p && continue

        # choose interaction magnitude
        w = abs(randn(rng) * σ)

        # choose trophic direction
        if rand(rng) < 0.5
            # i consumes j
            M[i,j] = +w
            M[j,i] = -w
        else
            # j consumes i
            M[i,j] = -w
            M[j,i] = +w
        end
    end

    # η-directionality control (EXACT same idea as before)
    U = triu(M, 1)
    L = tril(M, -1)

    return U + (1.0 - float(η)) * L
end

# ---------------------------
# Stabilize base by scaling interaction strength s
# Base: Abar = -I + s*O  (diag -1)
# J = diag(u)*Abar = -diag(u) + s*diag(u)*O
# ---------------------------
function find_stable_scale(O::AbstractMatrix, u::AbstractVector;
                           s0::Real=1.0, margin::Real=1e-3, max_shrinks::Int=80)
    s = float(s0)
    J = -Diagonal(u) + s * (Diagonal(u) * O)
    α = spectral_abscissa(J)
    k = 0
    while !(isfinite(α) && α < -margin) && k < max_shrinks
        s *= 0.5
        J = -Diagonal(u) + s * (Diagonal(u) * O)
        α = spectral_abscissa(J)
        k += 1
    end
    return (isfinite(α) && α < -margin) ? s : NaN
end

# ---------------------------
# Biomass-weighted rmed(t)
# ---------------------------
function rmed_biomass(J::AbstractMatrix, u::AbstractVector; t::Real)
    E = exp(float(t) * J)
    any(!isfinite, E) && return NaN
    w = u .^ 2
    C = Diagonal(w)
    Ttr = tr(E * C * transpose(E))
    (!isfinite(Ttr) || Ttr <= 0) && return NaN
    r = -(log(Ttr) - log(sum(w))) / (2*float(t))
    return isfinite(r) ? r : NaN
end

function rmed_curve(J::AbstractMatrix, u::AbstractVector, tvals::AbstractVector)
    r = Vector{Float64}(undef, length(tvals))
    @inbounds for (i,t) in enumerate(tvals)
        r[i] = rmed_biomass(J, u; t=t)
    end
    return r
end

function delta_curve(r_base::AbstractVector, r_pert::AbstractVector)
    @assert length(r_base) == length(r_pert)
    Δ = Vector{Float64}(undef, length(r_base))
    @inbounds for i in eachindex(r_base)
        Δ[i] = (isfinite(r_base[i]) && isfinite(r_pert[i])) ? abs(r_base[i] - r_pert[i]) : NaN
    end
    return Δ
end

function max_finite(x::AbstractVector)
    vals = filter(isfinite, x)
    isempty(vals) && return NaN
    return maximum(vals)
end

# ---------------------------
# Perturbation direction from a "rewire" of Abar offdiagonal
# Abar has diag -1. We rewiring only the off-diagonal part and keep diag -1.
# Pdir = (Abar_rew - Abar) / ||Abar_rew - Abar||_F
# ---------------------------
function sample_Pdir_from_rewire(Abar::Matrix{Float64}; rng=Random.default_rng())
    S = size(Abar,1)
    off = Abar + Matrix{Float64}(I, S, S)          # remove diag -1 => diag 0
    off_rew = reshuffle_offdiagonal(off; rng=rng)  # diag remains 0
    Abar_rew = -Matrix{Float64}(I, S, S) + off_rew

    Δ = Abar_rew - Abar
    nΔ = norm(Δ)
    nΔ == 0 && return nothing
    return Δ / nΔ
end

# ---------------------------
# Data containers
# ---------------------------
struct BaseSystem1
    η::Float64
    u::Vector{Float64}
    Abar::Matrix{Float64}       # diag -1
    rbase::Vector{Float64}      # rmed_base(t)
    eps::Float64                # fixed ε for this base
end

"""
Build stable base systems for each η (base_reps each, after stability).
eps = eps_rel * ||offdiag(Abar)||_F  (fixed per base)
"""
function build_bases(; S::Int,
    η_grid::Vector{Float64},
    base_reps::Int,
    seed::Int,
    u_mean::Float64,
    u_cv::Float64,
    p::Float64,
    σ::Float64,
    margin::Float64,
    eps_rel::Float64,
    tvals::Vector{Float64}
)
    bases = BaseSystem1[]
    for (iη, η0) in enumerate(η_grid)
        η = float(η0)
        for b in 1:base_reps
            rng = MersenneTwister(seed + 1_000_000*iη + 10_003*b)

            u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))
            O = make_O_eta_trophic(S, η; p=p, σ=σ, rng=rng)

            s = find_stable_scale(O, u; s0=1.0, margin=margin)
            isfinite(s) || continue

            Abar = -Matrix{Float64}(I, S, S) + s * O
            J = Diagonal(u) * Abar

            rbase = rmed_curve(J, u, tvals)

            eps = eps_rel * norm(offdiag_part(Abar))
            push!(bases, BaseSystem1(η, u, Abar, rbase, eps))
        end
    end
    return bases
end

# ---------------------------
# Main pipeline (threaded)
# Stores:
#   - base curves (nb × nt)
#   - pert curves (nb × P_reps × nt) for accepted samples, NaN otherwise
#   - metrics (nb × P_reps)
# ---------------------------
function run_time_domain_pipeline(;
    S::Int=120,
    η_grid = collect(range(0.0, 1.0; length=11)),
    base_reps::Int=6,
    P_reps::Int=40,
    seed::Int=1234,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    p::Real=0.05,
    σ::Real=1.0,
    margin::Real=1e-3,
    eps_rel::Real=0.20,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=100)
)
    η_grid = collect(float.(η_grid))
    tvals = collect(float.(tvals))
    nt = length(tvals)

    bases = build_bases(
        S=S, η_grid=η_grid, base_reps=base_reps, seed=seed,
        u_mean=float(u_mean), u_cv=float(u_cv),
        p=float(p), σ=float(σ), margin=float(margin),
        eps_rel=float(eps_rel), tvals=tvals
    )
    nb = length(bases)
    @info "Built $nb base systems (after stability filtering)."

    # store base curves
    rbase_mat = fill(NaN, nb, nt)
    @inbounds for bi in 1:nb
        rbase_mat[bi, :] .= bases[bi].rbase
    end

    # store pert curves & metrics per (base, P)
    rpert_3d = fill(NaN, nb, P_reps, nt)

    max_delta    = fill(NaN, nb, P_reps)
    delta_end    = fill(NaN, nb, P_reps)
    bump_strength = fill(NaN, nb, P_reps)
    bump_excess   = fill(NaN, nb, P_reps)

    accepted = fill(0, nb)
    rejected = fill(0, nb)

    njobs = nb * P_reps
    Threads.@threads for job in 1:njobs
        bi = (job - 1) ÷ P_reps + 1
        pr = (job - 1) % P_reps + 1

        base = bases[bi]
        rng = MersenneTwister(seed + 9_000_000*bi + 13_007*pr)

        Pdir = sample_Pdir_from_rewire(base.Abar; rng=rng)
        if Pdir === nothing
            @inbounds rejected[bi] += 1
            continue
        end

        # fixed ε per base
        Abarp = base.Abar + base.eps * Pdir
        Jp = Diagonal(base.u) * Abarp

        αp = spectral_abscissa(Jp)
        if !(isfinite(αp) && αp < -float(margin))
            @inbounds rejected[bi] += 1
            continue
        end

        # rmed curve
        rpert = rmed_curve(Jp, base.u, tvals)
        @inbounds rpert_3d[bi, pr, :] .= rpert

        # Δ(t) + metrics
        Δt = delta_curve(base.rbase, rpert)

        md = max_finite(Δt)
        de = (isfinite(Δt[end]) ? Δt[end] : NaN)

        max_delta[bi, pr] = md
        delta_end[bi, pr] = de

        if isfinite(md) && isfinite(de) && de > 0
            bump_strength[bi, pr] = md / de
            bump_excess[bi, pr]   = md - de
        else
            bump_strength[bi, pr] = NaN
            bump_excess[bi, pr]   = NaN
        end

        @inbounds accepted[bi] += 1
    end

    return (
        S=S,
        η_grid=η_grid,
        tvals=tvals,
        bases=bases,
        rbase=rbase_mat,
        rpert=rpert_3d,
        max_delta=max_delta,
        delta_end=delta_end,
        bump_strength=bump_strength,
        bump_excess=bump_excess,
        accepted=accepted,
        rejected=rejected
    )
end

# ---------------------------
# Plot 1: rmed lines, one panel per η
#   - base curves in black
#   - rewired curves in red
# ---------------------------
function plot_rmed_lines_by_eta(res; figsize=(1800, 1200), alpha_base=0.20, alpha_rew=0.12, lw=1.5)
    bases = res.bases
    tvals = res.tvals
    ηs = [b.η for b in bases]
    uniqη = sort(unique(ηs))

    nη = length(uniqη)
    ncols = min(4, nη)
    nrows = ceil(Int, nη / ncols)

    fig = Figure(size=figsize)
    k = 0
    for η in uniqη
        k += 1
        r = (k-1) ÷ ncols + 1
        c = (k-1) % ncols + 1

        ax = Axis(fig[r, c];
            xscale=log10,
            xlabel="t",
            ylabel="rmed(t) (biomass-weighted)",
            title="η = $(round(η,digits=2))"
        )

        idx = findall(x -> x == η, ηs)

        # base curves (black)
        for bi in idx
            lines!(ax, tvals, res.rbase[bi, :]; color=(:black, alpha_base), linewidth=lw)
        end

        # rewired curves (red)
        for bi in idx
            for pr in 1:size(res.rpert, 2)
                rp = view(res.rpert, bi, pr, :)
                # skip fully-NaN
                any(isfinite, rp) || continue
                lines!(ax, tvals, rp; color=(:red, alpha_rew), linewidth=lw)
            end
        end
    end

    display(fig)
end

# ---------------------------
# Plot 2: Δrmed profiles, one line per η (mean across all accepted pairs)
# --------------------------
function plot_delta_profiles_by_eta(res; figsize=(1100, 650), cmap=:viridis)
    bases = res.bases
    tvals = res.tvals
    ηs = [b.η for b in bases]
    uniqη = sort(unique(ηs))

    nb = length(bases)
    P = size(res.rpert, 2)
    nt = length(tvals)

    # normalize η to [0,1] for colormap
    ηmin, ηmax = minimum(uniqη), maximum(uniqη)
    ηnorm(η) = (η - ηmin) / (ηmax - ηmin + eps())

    fig = Figure(size=figsize)
    ax = Axis(fig[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="mean Δ(t) = mean |rmed_base(t) - rmed_rewired(t)|",
        title="Mean structure-effect profile by η (biomass rmed)"
    )

    for η in uniqη
        idx = findall(x -> x == η, ηs)

        sumΔ = zeros(Float64, nt)
        cntΔ = zeros(Int, nt)

        for bi in idx
            rbase = view(res.rbase, bi, :)
            for pr in 1:P
                rpert = view(res.rpert, bi, pr, :)
                any(isfinite, rpert) || continue
                @inbounds for ti in 1:nt
                    rb = rbase[ti]
                    rp = rpert[ti]
                    if isfinite(rb) && isfinite(rp)
                        sumΔ[ti] += abs(rb - rp)
                        cntΔ[ti] += 1
                    end
                end
            end
        end

        meanΔ = Vector{Float64}(undef, nt)
        @inbounds for ti in 1:nt
            meanΔ[ti] = cntΔ[ti] > 0 ? sumΔ[ti] / cntΔ[ti] : NaN
        end

        cs = getproperty(ColorSchemes, cmap)
        color = get(cs, ηnorm(η))

        lines!(ax, tvals, meanΔ; linewidth=3, color=color)
    end

    # colorbar instead of legend
    Colorbar(
        fig[1,2],
        colormap=cmap,
        limits=(ηmin, ηmax),
        label="η"
    )

    display(fig)
end


# ---------------------------
# Plot 3: bump summaries vs η
#   - bump_strength vs η (median + IQR)
#   - bump_excess vs η (median + IQR)
#   - max_delta vs η (median + IQR)
#   - delta_end vs η (median + IQR)
#   - bump_strength distribution vs η (scatter)
# ---------------------------
function plot_bump_summaries(res; figsize=(1800, 1100))
    bases = res.bases
    ηs = [b.η for b in bases]
    uniqη = sort(unique(ηs))

    P = size(res.max_delta, 2)

    # gather per-η pooled samples (across bases and P)
    function pooled_vals(mat, η)
        idx = findall(x -> x == η, ηs)
        vals = Float64[]
        for bi in idx
            append!(vals, vec(mat[bi, :]))
        end
        return filter(isfinite, vals)
    end

    stats = Dict{Float64, NamedTuple}()
    for η in uniqη
        bs = pooled_vals(res.bump_strength, η)
        be = pooled_vals(res.bump_excess, η)
        md = pooled_vals(res.max_delta, η)
        de = pooled_vals(res.delta_end, η)

        stats[η] = (
            bs_med = isempty(bs) ? NaN : median(bs),
            bs_q25 = q25(bs),
            bs_q75 = q75(bs),

            be_med = isempty(be) ? NaN : median(be),
            be_q25 = q25(be),
            be_q75 = q75(be),

            md_med = isempty(md) ? NaN : median(md),
            md_q25 = q25(md),
            md_q75 = q75(md),

            de_med = isempty(de) ? NaN : median(de),
            de_q25 = q25(de),
            de_q75 = q75(de),

            bs_samples = bs
        )
    end

    xs = uniqη

    fig = Figure(size=figsize)

    # (1) bump_strength vs η
    ax1 = Axis(fig[1,1];
        xlabel="η",
        ylabel="bump_strength = maxΔ / Δ_end",
        title="Does intermediate dominate late?"
    )
    y = [stats[η].bs_med for η in xs]
    ylo = [stats[η].bs_q25 for η in xs]
    yhi = [stats[η].bs_q75 for η in xs]
    lines!(ax1, xs, y, linewidth=3)
    scatter!(ax1, xs, y, markersize=10)
    for (x, lo, hi) in zip(xs, ylo, yhi)
        if isfinite(lo) && isfinite(hi)
            lines!(ax1, [x,x], [lo,hi], linewidth=3)
        end
    end
    hlines!(ax1, [1.0]; linestyle=:dash)

    # (2) bump_excess vs η
    ax2 = Axis(fig[1,2];
        xlabel="η",
        ylabel="bump_excess = maxΔ - Δ_end",
        title="Absolute intermediate excess over end"
    )
    y = [stats[η].be_med for η in xs]
    ylo = [stats[η].be_q25 for η in xs]
    yhi = [stats[η].be_q75 for η in xs]
    lines!(ax2, xs, y, linewidth=3)
    scatter!(ax2, xs, y, markersize=10)
    for (x, lo, hi) in zip(xs, ylo, yhi)
        if isfinite(lo) && isfinite(hi)
            lines!(ax2, [x,x], [lo,hi], linewidth=3)
        end
    end
    hlines!(ax2, [0.0]; linestyle=:dash)

    # (3) max_delta vs η
    ax3 = Axis(fig[2,1];
        xlabel="η",
        ylabel="maxΔ = max_t Δ(t)",
        title="Median maxΔ vs η"
    )
    y = [stats[η].md_med for η in xs]
    ylo = [stats[η].md_q25 for η in xs]
    yhi = [stats[η].md_q75 for η in xs]
    lines!(ax3, xs, y, linewidth=3)
    scatter!(ax3, xs, y, markersize=10)
    for (x, lo, hi) in zip(xs, ylo, yhi)
        if isfinite(lo) && isfinite(hi)
            lines!(ax3, [x,x], [lo,hi], linewidth=3)
        end
    end

    # (4) delta_end vs η
    ax4 = Axis(fig[2,2];
        xlabel="η",
        ylabel="Δ_end = Δ(t_max)",
        title="Median endΔ vs η"
    )
    y = [stats[η].de_med for η in xs]
    ylo = [stats[η].de_q25 for η in xs]
    yhi = [stats[η].de_q75 for η in xs]
    lines!(ax4, xs, y, linewidth=3)
    scatter!(ax4, xs, y, markersize=10)
    for (x, lo, hi) in zip(xs, ylo, yhi)
        if isfinite(lo) && isfinite(hi)
            lines!(ax4, [x,x], [lo,hi], linewidth=3)
        end
    end

    # (5) bump_strength distribution vs η
    ax5 = Axis(fig[3,1:2];
        xlabel="η",
        ylabel="bump_strength",
        title="Bump strength distribution (each dot = one base/rewire pair)"
    )
    for η in xs
        bs = stats[η].bs_samples
        isempty(bs) && continue
        # jitter a little so you can see density
        xjit = η .+ 0.015 .* (rand(length(bs)) .- 0.5)
        scatter!(ax5, xjit, bs, markersize=4)
    end
    hlines!(ax5, [1.0]; linestyle=:dash)

    display(fig)
end

# ---------------------------
# MAIN RUN
# ---------------------------
tvals = 10 .^ range(log10(0.01), log10(100.0); length=120)

res_domain_trophic = run_time_domain_pipeline(
    S=120,
    η_grid=collect(range(0.0, 1.0; length=12)),
    base_reps=6,
    P_reps=50,
    seed=1234,
    u_mean=1.0,
    u_cv=0.5,
    p=0.05,
    σ=1.0,
    margin=1e-3,
    eps_rel=0.20,    # reduce if too many rejects
    tvals=tvals
)

# quick acceptance report
ηs = [b.η for b in res_domain_trophic.bases]
for η in sort(unique(ηs))
    idx = findall(x -> x == η, ηs)
    acc = sum(res_domain_trophic.accepted[idx])
    rej = sum(res_domain_trophic.rejected[idx])
    @info "η=$(round(η,digits=2)) accepted=$acc rejected=$rej acc_rate=$(acc/(acc+rej+1e-9))"
end

plot_rmed_lines_by_eta(res_domain_trophic)
plot_delta_profiles_by_eta(res_domain_trophic)
plot_bump_summaries(res_domain_trophic)
