using LinearAlgebra
using Random
using Distributions
using CairoMakie
using Base.Threads

# ============================================================
# Raw intrinsic profiles only
# ------------------------------------------------------------
# Builds synthetic communities with different structures
# and displays intrinsic sensitivity profiles:
#
#     S(ω) = || (im*ω*T - A)^(-1) ||_2
#
# Notation:
#   A = interaction matrix
#   T = diagonal matrix of intrinsic timescales
#   J = T \ A
#
# This script does only one thing:
#   display raw profiles for visual inspection
# ============================================================

const DEFAULT_SEED = 1234
Random.seed!(DEFAULT_SEED)

# -------------------------
# User controls
# -------------------------
const S = 100
const N_UNIFORM = 10
const N_HEAVY = 10
const CONNECTANCE = 0.12
const IS = 0.22
const TS_SIGMA_LOG = 0.15
const STABILITY_MARGIN = 0.1
const OMEGAS = exp.(range(log(1e-2), log(1e2), length = 220))

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

function probability_matrix_heavytail(
    S::Int,
    connectance::Float64;
    shape::Float64 = 2.0,
    hub_fraction::Float64 = 0.15,
    hub_boost::Float64 = 3.5
)
    prop = rand(Pareto(shape, 1.0), S)
    nhubs = max(1, round(Int, hub_fraction * S))
    hub_idx = partialsortperm(prop, rev = true, 1:nhubs)
    prop[hub_idx] .*= hub_boost

    base = prop * prop'
    scale = connectance / offdiag_mean(base)
    P = clamp.(scale .* base, 0.0, 1.0)

    @inbounds for i in 1:S
        P[i, i] = 0.0
    end
    return P
end

function sample_mask(P::AbstractMatrix{<:Real})
    S = size(P, 1)
    M = rand(S, S) .< P
    @inbounds for i in 1:S
        M[i, i] = false
    end
    return M
end

function generate_interaction_matrix(
    S::Int,
    network_type::Symbol;
    connectance::Float64 = 0.12,
    IS::Float64 = 0.22
)
    P = if network_type == :uniform
        probability_matrix_uniform(S, connectance)
    elseif network_type == :heavytail
        probability_matrix_heavytail(S, connectance)
    else
        error("Unknown network_type = $network_type")
    end

    mask = sample_mask(P)
    A = zeros(Float64, S, S)
    A[mask] .= rand(Normal(0, IS), count(mask))

    @inbounds for i in 1:S
        A[i, i] = 0.0
    end

    return A
end

function generate_timescales(S::Int; sigma_log::Float64 = 0.15)
    rand(LogNormal(-0.5 * sigma_log^2, sigma_log), S)
end

# -------------------------
# Dynamics
# -------------------------
function stabilize_A(A::AbstractMatrix{<:Real}, T::AbstractMatrix{<:Real}; margin::Float64 = 0.1)
    J0 = T \ Matrix(A)
    α = maximum(real.(eigvals(J0)))
    shift = max(0.0, α + margin)
    A_stable = Matrix(A) - shift * T
    return A_stable
end

function intrinsic_profile(A::AbstractMatrix{<:Real},
                           T::AbstractMatrix{<:Real},
                           ωs::AbstractVector{<:Real})
    nω = length(ωs)
    vals = zeros(Float64, nω)

    # convert once per community
    Ac = ComplexF64.(A)
    Tc = ComplexF64.(T)

    for (k, ω) in pairs(ωs)
        F = factorize(im * ω .* Tc - Ac)
        R = F \ I
        vals[k] = opnorm(R, 2)
    end

    return vals
end

# -------------------------
# Build one profile
# -------------------------
function build_profile(S::Int, network_type::Symbol)
    A_raw = generate_interaction_matrix(
        S, network_type;
        connectance = CONNECTANCE,
        IS = IS
    )

    t = generate_timescales(S; sigma_log = TS_SIGMA_LOG)
    T = Diagonal(t)

    A = stabilize_A(A_raw, T; margin = STABILITY_MARGIN)
    prof = intrinsic_profile(A, T, OMEGAS)

    return (A = A, T = T, profile = prof)
end

# -------------------------
# Build ensembles in parallel
# -------------------------
function build_profiles(S::Int, network_type::Symbol, nrep::Int)
    results = Vector{NamedTuple}(undef, nrep)

    Threads.@threads for r in 1:nrep
        # lightweight per-thread reseeding to avoid identical streams
        Random.seed!(DEFAULT_SEED + 1000 * threadid() + r)
        results[r] = build_profile(S, network_type)
    end

    return results
end

# -------------------------
# Display only
# -------------------------
function main()
    println("Building raw intrinsic profiles...")
    println("Uniform communities: $N_UNIFORM")
    println("Heavy-tailed communities: $N_HEAVY")
    println("Species: $S")
    println("Threads: $(Threads.nthreads())")

    uniform_profiles = build_profiles(S, :uniform, N_UNIFORM)
    heavy_profiles = build_profiles(S, :heavytail, N_HEAVY)

    # -------------------------------------------------
    # Compute global y-axis range across all profiles
    # -------------------------------------------------
    all_profiles = vcat(
        [p.profile for p in uniform_profiles],
        [p.profile for p in heavy_profiles]
    )

    ymin = minimum(minimum.(all_profiles))
    ymax = maximum(maximum.(all_profiles))

    total = N_UNIFORM + N_HEAVY
    ncols = 4
    nrows = ceil(Int, total / ncols)

    fig = Figure(size = (420 * ncols, 260 * nrows))

    # -----------------
    # Uniform networks
    # -----------------
    for idx in 1:N_UNIFORM
        row = ceil(Int, idx / ncols)
        col = idx - (row - 1) * ncols

        ax = Axis(
            fig[row, col],
            title = "uniform $(idx)",
            xlabel = "ω",
            ylabel = "S(ω)",
            xscale = log10,
            yscale = log10,
            limits = (nothing, (ymin, ymax))
        )

        lines!(ax, OMEGAS, uniform_profiles[idx].profile, linewidth = 3)
    end

    # -----------------
    # Heavy-tailed networks
    # -----------------
    for jdx in 1:N_HEAVY
        idx = N_UNIFORM + jdx
        row = ceil(Int, idx / ncols)
        col = idx - (row - 1) * ncols

        ax = Axis(
            fig[row, col],
            title = "heavy-tail $(jdx)",
            xlabel = "ω",
            ylabel = "S(ω)",
            xscale = log10,
            yscale = log10,
            limits = (nothing, (ymin, ymax))
        )

        lines!(ax, OMEGAS, heavy_profiles[jdx].profile, linewidth = 3)
    end

    display(fig)
end

main()