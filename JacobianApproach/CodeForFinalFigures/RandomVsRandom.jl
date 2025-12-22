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

function random_interaction_matrix(S::Int, connectance::Real;
    σ::Real=1.0, rng=Random.default_rng()
)
    A = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        if i != j && rand(rng) < connectance
            A[i,j] = rand(rng, Normal(0, σ))
        end
    end
    return A
end

A = random_interaction_matrix(30, 0.1; σ=0.5)

"""
Recover interaction matrix A from Jacobian J and biomass vector u,
assuming J = diag(u) * (A - I).
"""
function extract_interaction_matrix(J::AbstractMatrix, u::AbstractVector)
    @assert size(J, 1) == size(J, 2) == length(u)
    @assert all(u .> 0)

    Dinv = Diagonal(1.0 ./ u)
    return Dinv * J + I
end

"""
Reshuffle off-diagonal entries of J while keeping diagonal intact.
Permutes all off-diagonal values (including zeros), preserving the multiset.
"""
function reshuffle_offdiagonal(J::AbstractMatrix; rng=Random.default_rng())
    S = size(J, 1)
    J2 = copy(Matrix(J))

    vals = Float64[]
    idxs = Tuple{Int,Int}[]
    for i in 1:S, j in 1:S
        if i != j
            push!(vals, J2[i,j])
            push!(idxs, (i,j))
        end
    end

    perm = randperm(rng, length(vals))
    for k in 1:length(vals)
        (i,j) = idxs[k]
        J2[i,j] = vals[perm[k]]
    end
    return J2
end

# -----------------------------
# Pipeline
# -----------------------------
function run_pipeline(;
    S::Int=120,
    connectance::Real=0.05,
    n::Int=50,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    σA::Real=0.5,
    seed::Int=1234,
    perturbation::Symbol=:biomass,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=30)
)
    rng = MersenneTwister(seed)
    nt = length(tvals)

    rmed_orig = fill(NaN, n, nt)
    rmed_shuf = fill(NaN, n, nt)
    # u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

    for k in 1:n
        A = random_interaction_matrix(S, connectance; σ=σA, rng=rng)
        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

        J = jacobian(A, u)
        # for i in 1:S
        #     J[i,i] *= 10.0
        # end
        Ash = reshuffle_offdiagonal(A; rng=rng)   # reshuffle on A, not J
        # Ash = random_interaction_matrix(S, connectance; σ=σA, rng=rng)
        # J  = jacobian(A,  u);  for i in 1:S; J[i,i] *= 10; end
        # Ash = random_interaction_matrix(S, connectance; σ=σA, rng=rng)

        Jsh = jacobian(Ash,u)
        # Jsh = reshuffle_offdiagonal(J; rng=rng)

        for (ti, t) in enumerate(tvals)
            rmed_orig[k, ti] = median_return_rate(J, u; t=t, perturbation=perturbation)
            rmed_shuf[k, ti] = median_return_rate(Jsh, u; t=t, perturbation=perturbation)
        end
    end

    mean_orig = vec(mean(rmed_orig; dims=1))
    mean_shuf = vec(mean(rmed_shuf; dims=1))
    # mean_delta = mean_orig - mean_shuf  # original - reshuffled
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
    t95_orig_each = [t95_from_rmed(t, vec(rO[k, :])) for k in 1:size(rO, 1)]
    t95_shuf_each = [t95_from_rmed(t, vec(rS[k, :])) for k in 1:size(rS, 1)]
    @info "t95_orig_each: ", t95_orig_each
    @info "t95_shuf_each: ", t95_shuf_each
    # choose how to summarize across communities
    if any(!isinf, t95_orig_each)
        t95_orig = median(filter(isfinite, t95_orig_each))   # or mean(...) or minimum(...)
        # also useful: t95 from the mean curves
        t95_mean_orig = t95_from_rmed(t, results.mean_orig)
    else
        t95_orig = NaN
        t95_mean_orig = NaN
    end
    if any(!isinf, t95_shuf_each)
        t95_shuf = median(filter(isfinite, t95_shuf_each))
        t95_mean_shuf = t95_from_rmed(t, results.mean_shuf)
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
        title="All communities: original (black) vs reshuffled (red)"
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
        title="Mean rmed(t): original vs reshuffled"
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
    lines!(ax3, t, abs.(results.mean_delta), linewidth=3, label="Δ")
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
    σA=1.0,
    perturbation=:biomass,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=30)
)

make_plots(results; save_prefix=nothing)