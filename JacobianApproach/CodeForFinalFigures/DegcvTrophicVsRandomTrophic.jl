using Random
using LinearAlgebra
using Statistics
using Distributions
using CairoMakie

# -----------------------------
# Community generation helpers
# -----------------------------
# u* generator (given)
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

# Jacobian (given)
jacobian(A,u) = Diagonal(u) * (A - I)

"""
Random trophic interaction matrix A with non-uniform (undirected) degree distribution.

You provide:
- connectance: baseline density target (sets the MEAN undirected degree ≈ connectance*(S-1))
- deg_cv: coefficient of variation of the intended degree propensities (heterogeneity)

Construction:
1) Draw positive "degree propensities" wᵢ from a lognormal with CV = deg_cv.
2) Rescale w so mean(w) = k̄ where k̄ = connectance*(S-1).
3) Sample an undirected simple graph via a Chung–Lu style model:
      P(edge i–j) = min(1, wᵢ*wⱼ / sum(w))
4) For each realized edge, assign a random predator/prey orientation and (+/-) magnitudes.

Notes:
- The realized degree CV will be close to deg_cv but not exact (finite-size + probability capping effects).
"""

function random_trophic_matrix_degcv(
    S::Int, connectance::Real;
    deg_cv::Real=2.0,
    σ::Real=1.0,
    rng=Random.default_rng()
)
    @assert S ≥ 2
    @assert 0 ≤ connectance ≤ 1
    @assert deg_cv ≥ 0
    @assert σ > 0

    # target mean undirected degree
    kbar = connectance * (S - 1)

    # degree propensities w with specified CV via lognormal
    # For LogNormal(μ, s): CV^2 = exp(s^2) - 1  => s = sqrt(log(1+CV^2))
    s = sqrt(log(1 + deg_cv^2))
    μ = -s^2/2                      # makes mean(w) = 1 before rescaling
    w = rand(rng, LogNormal(μ, s), S)
    w .*= (kbar / mean(w))          # rescale to hit desired mean degree

    W = sum(w)                      # ≈ 2m in Chung–Lu

    A = zeros(Float64, S, S)

    for i in 1:S-1, j in i+1:S
        pij = (W > 0) ? (w[i] * w[j] / W) : 0.0
        pij = min(1.0, pij)

        if rand(rng) < pij
            # random orientation: predator/prey
            if rand(rng) < 0.5
                pred, prey = i, j
            else
                pred, prey = j, i
            end

            # trophic magnitudes (positive), robust (no HalfNormal dependency)
            x = abs(randn(rng)) * σ   # benefit
            y = abs(randn(rng)) * σ   # harm

            A[pred, prey] = +x
            A[prey, pred] = -y
        end
    end

    return A
end

"""
Pairwise trophic reshuffle on J:

- Keep diagonal intact.
- Treat each unordered pair (i<j) as a 2-tuple: (J[i,j], J[j,i]).
- Permute these tuples across unordered locations (p<q).
- Optional: allow flipping predator/prey direction during reshuffle by swapping the tuple.

This preserves the trophic requirement because each tuple remains (+/-) (or 0/0).
"""
function reshuffle_trophic_pairs(J::AbstractMatrix; rng=Random.default_rng(), allow_flip::Bool=true)
    S = size(J, 1)
    J2 = copy(Matrix(J))

    # collect pair-values for each unordered pair
    pairs = Vector{Tuple{Float64,Float64}}(undef, 0)
    locs  = Vector{Tuple{Int,Int}}(undef, 0)

    for i in 1:S-1, j in i+1:S
        push!(pairs, (J2[i,j], J2[j,i]))
        push!(locs, (i,j))
    end

    perm = randperm(rng, length(pairs))

    for k in 1:length(pairs)
        (p,q) = locs[k]
        (a,b) = pairs[perm[k]]

        if allow_flip && rand(rng) < 0.5
            a,b = b,a
        end

        J2[p,q] = a
        J2[q,p] = b
    end

    # diagonal untouched by construction
    return J2
end

# -----------------------------
# Pipeline
# -----------------------------
function run_pipeline(;
    S::Int=120,
    connectance::Real=0.15,
    n::Int=50,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    σA::Real=0.5,
    seed::Int=1234,
    perturbation::Symbol=:biomass,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=30),
    reshuffle_allow_flip::Bool=true
)
    rng = MersenneTwister(seed)
    nt = length(tvals)

    rmed_orig = fill(NaN, n, nt)
    rmed_shuf = fill(NaN, n, nt)

    for k in 1:n
        A = random_trophic_matrix_degcv(S, connectance; σ=σA, rng=rng)
        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
        J = jacobian(A, u)
        
        # Ash = reshuffle_trophic_pairs(A; rng=rng, allow_flip=reshuffle_allow_flip)
        Ash = reshuffle_offdiagonal(A; rng=rng)
        # Ash = random_interaction_matrix(S, connectance; σ=σA, rng=rng)
        Jsh = jacobian(Ash,u)

        for (ti, t) in enumerate(tvals)
            rmed_orig[k, ti] = median_return_rate(J, u; t=t, perturbation=perturbation)
            rmed_shuf[k, ti] = median_return_rate(Jsh, u; t=t, perturbation=perturbation)
        end
    end

    mean_orig  = vec(mean(rmed_orig; dims=1))
    mean_shuf  = vec(mean(rmed_shuf; dims=1))
    mean_delta = vec(mean(rmed_orig .- rmed_shuf; dims=1))  # original - reshuffled

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
    # per-community t95s (n curves stored as rows)
    t95_orig_each = [t95_from_rmed(tvals, vec(rO[k, :])) for k in 1:size(rO, 1)]
    t95_shuf_each = [t95_from_rmed(tvals, vec(rS[k, :])) for k in 1:size(rS, 1)]
    @info "t95_orig_each: ", t95_orig_each
    @info "t95_shuf_each: ", t95_shuf_each
    if any(!isinf, t95_orig_each)
        t95_orig = median(filter(isfinite, t95_orig_each))   # or mean(...) or minimum(...)
        # also useful: t95 from the mean curves
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
    # ---- Plot 1: all trajectories ----
    fig1 = Figure(size=(1000, 650))
    ax1 = Axis(fig1[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="rmed(t)",
        title="All trophic communities: original (black) vs reshuffled (red)"
    )
    for k in 1:size(rO,1)
        lines!(ax1, t, rO[k, :], color=RGBAf(0,0,0,0.35), linewidth=1)
        lines!(ax1, t, rS[k, :], color=RGBAf(1,0,0,0.35), linewidth=1)
    end
    isfinite(t95_orig) && vlines!(ax1, t95_orig; color=(:black, 0.5), linewidth=2)
    isfinite(t95_shuf) && vlines!(ax1, t95_shuf; color=(:red,   0.5), linewidth=2)

    # ---- Plot 2: mean trajectories ----
    fig2 = Figure(size=(1000, 650))
    ax2 = Axis(fig2[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="mean rmed(t)",
        title="Mean rmed(t): trophic original vs trophic reshuffled"
    )
    lines!(ax2, t, results.mean_orig, color=:black, linewidth=3, label="original")
    lines!(ax2, t, results.mean_shuf, color=:red,   linewidth=3, label="reshuffled")
    axislegend(ax2; position=:rt)
    # Plot 2 axis: ax2
    isfinite(t95_mean_orig) && vlines!(ax2, t95_mean_orig; color=(:black, 0.6), linewidth=2, linestyle=:dash)
    isfinite(t95_mean_shuf) && vlines!(ax2, t95_mean_shuf; color=(:red,   0.6), linewidth=2, linestyle=:dash)

    # ---- Plot 3: mean delta ----
    fig3 = Figure(size=(1000, 650))
    ax3 = Axis(fig3[1,1];
        xscale=log10,
        xlabel="t",
        ylabel="mean Δrmed(t)",
        title="Mean Δrmed(t) = mean(original - reshuffled)"
    )
    lines!(ax3, t, results.mean_delta, linewidth=3, label="Δ")
    axislegend(ax3; position=:rt)
    # Plot 3 axis: ax3 (delta plot; t95 lines are just references)
    isfinite(t95_mean_orig) && vlines!(ax3, t95_mean_orig; color=(:black, 0.35), linewidth=2, linestyle=:dash)
    isfinite(t95_mean_shuf) && vlines!(ax3, t95_mean_shuf; color=(:red,   0.35), linewidth=2, linestyle=:dash)

    if save_prefix !== nothing
        save("$(save_prefix)_plot1_all_lines.png", fig1)
        save("$(save_prefix)_plot2_means.png", fig2)
        save("$(save_prefix)_plot3_mean_delta.png", fig3)
    end

    display(fig1)
    display(fig2)
    display(fig3)
end

# -----------------------------
# Main
# -----------------------------
tvals = 10 .^ range(log10(0.01), log10(100.0); length=30)

results = run_pipeline(
    S=120,
    connectance=0.1,
    n=50,
    u_mean=1.0,
    u_cv=0.5,
    σA=0.5,
    perturbation=:biomass,
    tvals=tvals,
    reshuffle_allow_flip=true
)

make_plots(results; save_prefix=nothing)