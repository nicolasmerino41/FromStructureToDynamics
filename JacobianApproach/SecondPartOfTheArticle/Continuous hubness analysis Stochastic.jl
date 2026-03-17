using LinearAlgebra
using Statistics
using Random
using Distributions
using Printf
using CairoMakie
using Base.Threads

# ============================================================
# Continuous hubness analysis with T = I
# ------------------------------------------------------------
# Goal:
#   Test whether increasing degree heterogeneity ("hubness")
#   increases the contrast between perturbations centered on
#   high-centrality vs low-centrality species.
#
# Fixed throughout:
#   T = I
#   R(ω) = (im*ω*I - A)^(-1)
#
# Main class-level quantity:
#   Q(ω) = S_high(ω) / S_low(ω)
#
# where
#   S_high(ω) = || R(ω) P_high R(ω) ||_2
#   S_low(ω)  = || R(ω) P_low  R(ω) ||_2
#
# The hubness gradient is created by continuously interpolating
# between a uniform connection kernel and a heavy-tailed one.
#
# Outputs are displayed only, not saved.
# ============================================================

const DEFAULT_SEED = 1234
Random.seed!(DEFAULT_SEED)

# -------------------------
# User controls
# -------------------------
const S = 100
const NREP = 30
const CONNECTANCE = 0.12
const IS = 0.22
const STABILITY_MARGIN = 0.1

const OMEGAS = exp.(range(log(1e-2), log(1e2), length = 220))

# Continuous hubness parameter:
#   0.0 = uniform
#   1.0 = strongly heavy-tailed
const HUBNESS_LEVELS = collect(range(0.0, 1.0, length = 7))

# Frequency window for slow/intermediate scales
# Chosen here as ω <= 1, since with T = I this is the natural
# dimensionless low/intermediate regime and the high-frequency
# asymptotic regime is known to be less discriminating.
const OMEGA_WINDOW_MAX = 1.0

# -------------------------
# Utilities
# -------------------------
function trapz(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    s = 0.0
    for i in 1:length(x)-1
        s += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return s
end

function gini(x::AbstractVector{<:Real})
    n = length(x)
    xs = sort(collect(float.(x)))
    s = sum(xs)
    s == 0 && return 0.0
    num = 0.0
    for i in 1:n
        num += (2i - n - 1) * xs[i]
    end
    return num / (n * s)
end

# -------------------------
# Network generation
# -------------------------
function offdiag_mean(X::AbstractMatrix)
    S = size(X, 1)
    total = sum(X) - sum(diag(X))
    return total / (S * (S - 1))
end

function probability_matrix_uniform(S::Int, connectance::Float64)
    P = fill(connectance, S, S)
    @inbounds for i in 1:S
        P[i, i] = 0.0
    end
    return P
end

function probability_matrix_heavytail_base(
    rng::AbstractRNG,
    S::Int;
    shape::Float64 = 2.0,
    hub_fraction::Float64 = 0.15,
    hub_boost::Float64 = 3.5
)
    prop = rand(rng, Pareto(shape, 1.0), S)
    nhubs = max(1, round(Int, hub_fraction * S))
    hub_idx = partialsortperm(prop, rev = true, 1:nhubs)
    prop[hub_idx] .*= hub_boost

    B = prop * prop'
    @inbounds for i in 1:S
        B[i, i] = 0.0
    end
    return B
end

function probability_matrix_hubness(
    rng::AbstractRNG,
    S::Int,
    connectance::Float64,
    η::Float64
)
    # η = 0 => uniform
    # η = 1 => heavy-tailed
    U = ones(Float64, S, S)
    @inbounds for i in 1:S
        U[i, i] = 0.0
    end

    H = probability_matrix_heavytail_base(rng, S)

    B = (1 - η) .* U .+ η .* H
    scale = connectance / offdiag_mean(B)
    P = clamp.(scale .* B, 0.0, 1.0)

    @inbounds for i in 1:S
        P[i, i] = 0.0
    end
    return P
end

function sample_mask(rng::AbstractRNG, P::AbstractMatrix{<:Real})
    S = size(P, 1)
    M = rand(rng, S, S) .< P
    @inbounds for i in 1:S
        M[i, i] = false
    end
    return M
end

function generate_interaction_matrix(
    rng::AbstractRNG,
    S::Int,
    η::Float64;
    connectance::Float64 = 0.12,
    IS::Float64 = 0.22
)
    P = probability_matrix_hubness(rng, S, connectance, η)
    mask = sample_mask(rng, P)

    A = zeros(Float64, S, S)
    A[mask] .= rand(rng, Normal(0, IS), count(mask))

    @inbounds for i in 1:S
        A[i, i] = 0.0
    end

    return A
end

# -------------------------
# Fixed homogeneous T = I
# -------------------------
function identity_timescale_matrix(S::Int)
    return Diagonal(ones(Float64, S))
end

# -------------------------
# Dynamics
# -------------------------
function stabilize_A(A::AbstractMatrix{<:Real}; margin::Float64 = 0.1)
    J0 = Matrix(A)  # since T = I
    α = maximum(real.(eigvals(J0)))
    shift = max(0.0, α + margin)
    A_stable = Matrix(A) - shift * I
    return A_stable, shift
end

# -------------------------
# Centrality and perturbation classes
# -------------------------
function weighted_degree(A::AbstractMatrix{<:Real})
    return vec(sum(abs.(A), dims = 1)) .+ vec(sum(abs.(A), dims = 2))
end

function unweighted_degree(A::AbstractMatrix{<:Real}; tol::Float64 = 0.0)
    B = abs.(A) .> tol
    return vec(sum(B, dims = 1)) .+ vec(sum(B, dims = 2))
end

function centrality_classes(c::AbstractVector{<:Real}; q::Float64 = 0.2)
    n = length(c)
    m = max(2, round(Int, q * n))
    p = sortperm(c)
    low = sort(p[1:m])
    high = sort(p[end-m+1:end])
    return low, high
end

function perturbation_operator_class(S::Int, C::AbstractVector{<:Integer})
    M = zeros(Float64, S, S)
    Cset = Set(C)

    @inbounds for i in 1:S, j in 1:S
        if i != j && (i in Cset || j in Cset)
            M[i, j] = 1.0
        end
    end

    nf = norm(M)
    nf == 0 && error("Zero class perturbation mask.")
    return M / nf
end

# -------------------------
# Spectra
# -------------------------
function class_spectra(
    A::AbstractMatrix{<:Real},
    P_low::AbstractMatrix{<:Real},
    P_high::AbstractMatrix{<:Real},
    ωs::AbstractVector{<:Real}
)
    nω = length(ωs)

    s_low = zeros(Float64, nω)
    s_high = zeros(Float64, nω)

    Ac = ComplexF64.(A)
    Pc_low = ComplexF64.(P_low)
    Pc_high = ComplexF64.(P_high)
    Icomplex = Matrix{ComplexF64}(I, size(A,1), size(A,2))

    for (k, ω) in pairs(ωs)
        F = factorize(im * ω .* Icomplex - Ac)
        R = F \ Icomplex
        s_low[k] = opnorm(R * Pc_low * R, 2)
        s_high[k] = opnorm(R * Pc_high * R, 2)
    end

    return s_low, s_high
end

# -------------------------
# Single replicate
# -------------------------
function analyze_replicate(
    rng::AbstractRNG,
    S::Int,
    η::Float64,
    ωs::AbstractVector{<:Real}
)
    A_raw = generate_interaction_matrix(
        rng, S, η;
        connectance = CONNECTANCE,
        IS = IS
    )

    A, shift = stabilize_A(A_raw; margin = STABILITY_MARGIN)

    c = weighted_degree(A)
    lowC, highC = centrality_classes(c)

    P_low = perturbation_operator_class(S, lowC)
    P_high = perturbation_operator_class(S, highC)

    s_low, s_high = class_spectra(A, P_low, P_high, ωs)
    ratio = s_high ./ s_low

    deg = unweighted_degree(A)
    deg_gini = gini(deg)
    deg_cv = std(deg) / mean(deg)

    # summarize over low/intermediate band only
    idx_window = findall(ω -> ω <= OMEGA_WINDOW_MAX, ωs)
    ωw = ωs[idx_window]
    rw = ratio[idx_window]

    # integrated log-ratio: 0 means equality, >0 means high > low
    logratio_int = trapz(ωw, log.(rw))

    return (
        A = A,
        ωs = ωs,
        s_low = s_low,
        s_high = s_high,
        ratio = ratio,
        deg = deg,
        deg_gini = deg_gini,
        deg_cv = deg_cv,
        logratio_int = logratio_int,
        shift = shift,
        η = η
    )
end

# -------------------------
# Ensemble
# -------------------------
function build_ensemble(S::Int, ηlevels::AbstractVector{<:Real}, nrep::Int, ωs::AbstractVector{<:Real})
    nη = length(ηlevels)
    results = Matrix{NamedTuple}(undef, nη, nrep)

    Threads.@threads for idx in 1:(nη * nrep)
        iη = fld(idx - 1, nrep) + 1
        ir = mod(idx - 1, nrep) + 1

        η = ηlevels[iη]
        rng = MersenneTwister(DEFAULT_SEED + 10_000 * iη + ir)

        results[iη, ir] = analyze_replicate(rng, S, η, ωs)
    end

    return results
end

# -------------------------
# Helpers for ensemble summaries
# -------------------------
function mean_and_sd_curves(results::Matrix{<:NamedTuple}, field::Symbol, iη::Int)
    X = reduce(hcat, [getfield(results[iη, r], field) for r in 1:size(results, 2)])
    μ = vec(mean(X, dims = 2))
    σ = vec(std(X, dims = 2))
    return μ, σ
end

function collect_scalar(results::Matrix{<:NamedTuple}, field::Symbol, iη::Int)
    return [getfield(results[iη, r], field) for r in 1:size(results, 2)]
end

# -------------------------
# Plotting
# -------------------------
function plot_results(results::Matrix{<:NamedTuple}, ηlevels::AbstractVector{<:Real})
    nη, nrep = size(results)
    ωs = results[1, 1].ωs

    fig = Figure(size = (1600, 1100))

    # ------------------------------------------------
    # Panel A: mean ratio spectra across hubness levels
    # ------------------------------------------------
    ax1 = Axis(
        fig[1, 1:2],
        title = "High / low centrality sensitivity ratio across a hubness gradient",
        xlabel = "frequency ω",
        ylabel = "Q(ω) = S_high(ω) / S_low(ω)",
        xscale = log10,
        yscale = log10
    )

    cmap = cgrad(:viridis, nη, categorical = true)

    for iη in 1:nη
        μ, σ = mean_and_sd_curves(results, :ratio, iη)
        lower = max.(μ .- σ, 1e-12)
        upper = μ .+ σ

        band!(ax1, ωs, lower, upper, color = (cmap[iη], 0.12))
        lines!(ax1, ωs, μ, linewidth = 3,
               color = cmap[iη],
               label = @sprintf("η = %.2f", ηlevels[iη]))
    end

    lines!(ax1, ωs, ones(length(ωs)), linestyle = :dash, linewidth = 2, color = :black)
    vlines!(ax1, [OMEGA_WINDOW_MAX], linestyle = :dot, linewidth = 2, color = :black)
    axislegend(ax1, position = :rb)

    # ------------------------------------------------
    # Panel B: realized degree heterogeneity vs integrated log-ratio
    # ------------------------------------------------
    ax2 = Axis(
        fig[2, 1],
        title = "Realized degree heterogeneity vs class-level contrast",
        xlabel = "degree Gini coefficient",
        ylabel = "∫ log(Q(ω)) dω   for ω ≤ $(OMEGA_WINDOW_MAX)"
    )

    for iη in 1:nη
        x = collect_scalar(results, :deg_gini, iη)
        y = collect_scalar(results, :logratio_int, iη)

        scatter!(ax2, x, y, markersize = 10, color = cmap[iη], label = @sprintf("η = %.2f", ηlevels[iη]))

        μx = mean(x)
        μy = mean(y)
        scatter!(ax2, [μx], [μy], markersize = 18, color = cmap[iη])
    end

    # ------------------------------------------------
    # Panel C: mean summary vs hubness level
    # ------------------------------------------------
    ax3 = Axis(
        fig[2, 2],
        title = "Mean contrast increases with hubness?",
        xlabel = "hubness control η",
        ylabel = "mean ∫ log(Q(ω)) dω   for ω ≤ $(OMEGA_WINDOW_MAX)"
    )

    mean_y = zeros(Float64, nη)
    sd_y = zeros(Float64, nη)
    mean_g = zeros(Float64, nη)
    sd_g = zeros(Float64, nη)

    for iη in 1:nη
        yy = collect_scalar(results, :logratio_int, iη)
        gg = collect_scalar(results, :deg_gini, iη)
        mean_y[iη] = mean(yy)
        sd_y[iη] = std(yy)
        mean_g[iη] = mean(gg)
        sd_g[iη] = std(gg)
    end

    band!(ax3, ηlevels, mean_y .- sd_y, mean_y .+ sd_y, color = (:steelblue, 0.18))
    lines!(ax3, ηlevels, mean_y, linewidth = 3, color = :steelblue)
    scatter!(ax3, ηlevels, mean_y, markersize = 12, color = :steelblue)

    # ------------------------------------------------
    # Panel D: realized degree heterogeneity vs hubness level
    # ------------------------------------------------
    ax4 = Axis(
        fig[3, 1],
        title = "The control parameter actually changes degree heterogeneity",
        xlabel = "hubness control η",
        ylabel = "degree Gini coefficient"
    )

    band!(ax4, ηlevels, mean_g .- sd_g, mean_g .+ sd_g, color = (:darkorange, 0.18))
    lines!(ax4, ηlevels, mean_g, linewidth = 3, color = :darkorange)
    scatter!(ax4, ηlevels, mean_g, markersize = 12, color = :darkorange)

    # ------------------------------------------------
    # Panel E: degree histograms for low / medium / high η
    # ------------------------------------------------
    histgrid = GridLayout()
    fig[3, 2] = histgrid

    idx_examples = [1, cld(nη, 2), nη]
    titles = ["low hubness", "medium hubness", "high hubness"]

    for (k, iη) in enumerate(idx_examples)
        ax = Axis(
            histgrid[1, k],
            title = titles[k],
            xlabel = "unweighted degree",
            ylabel = k == 1 ? "count" : ""
        )
        deg_example = results[iη, 1].deg
        hist!(ax, deg_example, bins = 16, color = cmap[iη])
    end

    Label(
        fig[0, :],
        "Continuous hubness analysis with homogeneous timescales (T = I)",
        fontsize = 22
    )

    display(fig)
end

# -------------------------
# Main
# -------------------------
function main()
    println("Running continuous hubness analysis")
    println(@sprintf("S = %d, NREP = %d, connectance = %.3f, IS = %.3f", S, NREP, CONNECTANCE, IS))
    println("T is fixed homogeneous: T = I")
    println("Hubness levels: ", HUBNESS_LEVELS)
    println("Using $(Threads.nthreads()) thread(s)")

    results = build_ensemble(S, HUBNESS_LEVELS, NREP, OMEGAS)
    plot_results(results, HUBNESS_LEVELS)
end

main()