using Random
using LinearAlgebra
using Statistics
using Distributions
using CairoMakie

# -----------------------------
# Community generation helpers
# -----------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

jacobian(A,u) = Diagonal(u) * (A - I)

"""
Pairwise trophic reshuffle on a trophic interaction matrix M:

- Keep diagonal intact.
- Treat each unordered pair (i<j) as a 2-tuple: (M[i,j], M[j,i]).
- Permute these tuples across unordered locations (p<q).
- Optional: allow flipping predator/prey direction during reshuffle by swapping the tuple.

This preserves trophic requirement because each tuple remains (+/-) (or 0/0).
"""
function reshuffle_trophic_pairs(M::AbstractMatrix; rng=Random.default_rng(), allow_flip::Bool=true)
    S = size(M, 1)
    M2 = copy(Matrix(M))

    pairs = Tuple{Float64,Float64}[]
    locs  = Tuple{Int,Int}[]
    for i in 1:S-1, j in i+1:S
        push!(pairs, (M2[i,j], M2[j,i]))
        push!(locs, (i,j))
    end

    perm = randperm(rng, length(pairs))

    for k in 1:length(pairs)
        (p,q) = locs[k]
        (a,b) = pairs[perm[k]]
        if allow_flip && rand(rng) < 0.5
            a,b = b,a
        end
        M2[p,q] = a
        M2[q,p] = b
    end

    return M2
end

# -----------------------------
# Pipeline
# -----------------------------
function run_pipeline(;
    S::Int=120,
    connectance::Real=0.1,          # kept for interface parity; PPM uses (B,L,η,T=q)
    n::Int=50,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    σA::Real=0.5,
    seed::Int=1234,
    perturbation::Symbol=:biomass,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=30),
    reshuffle_allow_flip::Bool=true,

    # --- PPM trophic builder params ---
    B::Int=24,
    L::Int=2142,
    q::Real=0.2,
    η::Real=0.2,
    mag_abs::Real=0.5,
    mag_cv::Real=0.5,
    corr::Real=0.0
)
    rng = MersenneTwister(seed)
    nt = length(tvals)

    rmed_orig = fill(NaN, n, nt)
    rmed_shuf = fill(NaN, n, nt)

    for k in 1:n
        # --- ORIGINAL: build trophic topology with PPM ---
        W = build_niche_trophic(
            S; conn=connectance, mean_abs=σA, mag_cv=mag_cv,
            degree_family=:pareto, deg_param=1.0,
            rho_sym=0.0, rng=Random.default_rng()
        )
        # b = PPMBuilder()
        # set!(b; S=S, B=B, L=L, T=q, η=η)
        # net = build(b)
        # A = net.A

        # # --- turn topology into signed interaction matrix W ---
        # W = build_interaction_matrix(A;
        #     mag_abs=mag_abs,
        #     mag_cv=mag_cv,
        #     corr_aij_aji=corr,
        #     rng=rng
        # )

        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
        J = jacobian(W, u)

        # --- RESHUFFLE: pairwise trophic reshuffle (preserves +/- pair structure) ---
        # Wsh = reshuffle_trophic_pairs(W; rng=rng, allow_flip=reshuffle_allow_flip)
        # Wsh = reshuffle_offdiagonal(W; rng=rng)

        Wsh = build_niche_trophic(
            S; conn=connectance, mean_abs=σA, mag_cv=mag_cv,
            degree_family=:pareto, deg_param=1.0,
            rho_sym=0.0, rng=Random.default_rng()
        )
        Jsh = jacobian(Wsh, u)

        for (ti, t) in enumerate(tvals)
            rmed_orig[k, ti] = median_return_rate(J, u; t=t, perturbation=perturbation)
            rmed_shuf[k, ti] = median_return_rate(Jsh, u; t=t, perturbation=perturbation)
        end
    end

    mean_orig  = vec(mean(rmed_orig; dims=1))
    mean_shuf  = vec(mean(rmed_shuf; dims=1))
    mean_delta = vec(mean(rmed_orig .- rmed_shuf; dims=1))

    return (tvals=tvals,
            rmed_orig=rmed_orig, rmed_shuf=rmed_shuf,
            mean_orig=mean_orig, mean_shuf=mean_shuf, mean_delta=mean_delta)
end

# -----------------------------
# Plotting with Makie
# -----------------------------
function make_plots(results; save_prefix::Union{Nothing,String}=nothing)
    t = results.tvals
    rO = results.rmed_orig
    rS = results.rmed_shuf

    t95_orig_each = [t95_from_rmed(tvals, vec(rO[k, :])) for k in 1:size(rO, 1)]
    t95_shuf_each = [t95_from_rmed(tvals, vec(rS[k, :])) for k in 1:size(rS, 1)]
    @info "t95_orig_each: ", t95_orig_each
    @info "t95_shuf_each: ", t95_shuf_each

    if any(!isinf, t95_orig_each)
        t95_orig = median(filter(isfinite, t95_orig_each))
        t95_mean_orig = t95_from_rmed(tvals, results.mean_orig)
    else
        t95_orig = NaN
        t95_mean_orig = NaN
    end

    if any(!isinf, t95_shuf_each)
        t95_shuf = median(filter(isfinite, t95_shuf_each))
        t95_mean_shuf = t95_from_rmed(tvals, results.mean_shuf)
    else
        t95_shuf = NaN
        t95_mean_shuf = NaN
    end

    fig1 = Figure(size=(1000, 650))
    ax1 = Axis(fig1[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="rmed(t)",
        title="PPM trophic communities: original (black) vs reshuffled (red)"
    )
    for k in 1:size(rO,1)
        lines!(ax1, t, rO[k, :], color=RGBAf(0,0,0,0.35), linewidth=1)
        lines!(ax1, t, rS[k, :], color=RGBAf(1,0,0,0.35), linewidth=1)
    end
    isfinite(t95_orig) && vlines!(ax1, t95_orig; color=(:black, 0.5), linewidth=2)
    isfinite(t95_shuf) && vlines!(ax1, t95_shuf; color=(:red,   0.5), linewidth=2)

    fig2 = Figure(size=(1000, 650))
    ax2 = Axis(fig2[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="mean rmed(t)",
        title="Mean rmed(t): PPM trophic original vs trophic reshuffled"
    )
    lines!(ax2, t, results.mean_orig, color=:black, linewidth=3, label="original")
    lines!(ax2, t, results.mean_shuf, color=:red,   linewidth=3, label="reshuffled")
    axislegend(ax2; position=:rt)
    isfinite(t95_mean_orig) && vlines!(ax2, t95_mean_orig; color=(:black, 0.6), linewidth=2, linestyle=:dash)
    isfinite(t95_mean_shuf) && vlines!(ax2, t95_mean_shuf; color=(:red,   0.6), linewidth=2, linestyle=:dash)

    fig3 = Figure(size=(1000, 650))
    ax3 = Axis(fig3[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="mean Δrmed(t)",
        title="Mean Δrmed(t) = mean(original - reshuffled)"
    )
    lines!(ax3, t, abs.(results.mean_delta), linewidth=3, label="Δ")
    axislegend(ax3; position=:rt)
    isfinite(t95_mean_orig) && vlines!(ax3, t95_mean_orig; color=(:black, 0.35), linewidth=2, linestyle=:dash)
    isfinite(t95_mean_shuf) && vlines!(ax3, t95_mean_shuf; color=(:red,   0.35), linewidth=2, linestyle=:dash)

    if save_prefix !== nothing
        save("$(save_prefix)_plot1_all_lines.png", fig1)
        save("$(save_prefix)_plot2_means.png", fig2)
        save("$(save_prefix)_plot3_mean_delta.png", fig3)
    end

    display(fig1); display(fig2); display(fig3)
end

# -----------------------------
# Main
# -----------------------------
tvals = 10 .^ range(log10(0.01), log10(100.0); length=30)

results = run_pipeline(
    S=120,
    n=50,
    u_mean=1.0,
    u_cv=0.5,
    σA=0.5,
    perturbation=:biomass,
    tvals=tvals,
    reshuffle_allow_flip=true,

    # PPM params
    B=24, 
    L=2142,
    q=1.5,
    η=0.2,
    mag_abs=0.5,
    mag_cv=0.5,
    corr=0.0
)

make_plots(results; save_prefix=nothing)
