using LinearAlgebra
using Random
using Distributions
using CairoMakie

# ============================================================
# Intrinsic profiles under varying timescale heterogeneity
# ------------------------------------------------------------
# 5 network realizations  ×  5 levels of timescale heterogeneity
#
# For each row:
#   - same interaction matrix A
#   - same underlying timescale random template
#   - only the heterogeneity level of T changes across columns
#
# Resolvent:
#   R(ω) = (im * ω * T - A)^(-1)
#
# Profile:
#   S(ω) = ||R(ω)||_2
#
# All panels share the same y-axis range.
# ============================================================
const DEFAULT_SEED = 1234
Random.seed!(DEFAULT_SEED)

# -------------------------
# User controls
# -------------------------
const S = 100
const N_NETWORKS = 5
const NETWORK_TYPE = :uniform      # choose :uniform or :heavytail
const CONNECTANCE = 0.12
const IS = 0.22
const STABILITY_MARGIN = 0.1

const OMEGAS = exp.(range(log(1e-2), log(1e2), length = 220))

# Five levels of timescale heterogeneity
# t_i = exp(σ z_i - σ^2/2), with z_i fixed within a row
# so mean scale stays near 1 while heterogeneity increases with σ
const TS_SIGMAS = [0.0, 0.05, 0.4, 1.0, 2.0]
const TS_LABELS = ["zero", "a bit", "medium", "quite a lot", "a lot"]

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

# -------------------------
# Timescales
# -------------------------
function timescales_from_template(z::AbstractVector{<:Real}, sigma_log::Float64)
    # lognormal with mean approximately 1 for any sigma_log
    return exp.(sigma_log .* z .- 0.5 * sigma_log^2)
end

# -------------------------
# Dynamics
# -------------------------
function stabilize_A(A::AbstractMatrix{<:Real}, T::AbstractMatrix{<:Real}; margin::Float64 = 0.1)
    J0 = T \ Matrix(A)
    α = maximum(real.(eigvals(J0)))
    shift = max(0.0, α + margin)
    return Matrix(A) - shift * T
end

function intrinsic_profile(A::AbstractMatrix{<:Real},
                           T::AbstractMatrix{<:Real},
                           ωs::AbstractVector{<:Real})
    vals = zeros(Float64, length(ωs))

    # convert once per panel
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
# Build all profiles
# -------------------------
function build_profile_grid()
    profiles = Matrix{Vector{Float64}}(undef, N_NETWORKS, length(TS_SIGMAS))

    # same row = same A and same z-template
    for r in 1:N_NETWORKS
        println("Building network $r / $N_NETWORKS")
        A_raw = generate_interaction_matrix(
            S, NETWORK_TYPE;
            connectance = CONNECTANCE,
            IS = IS
        )

        z = randn(S)

        for c in eachindex(TS_SIGMAS)
            σ = TS_SIGMAS[c]
            t = timescales_from_template(z, σ)
            T = Diagonal(t)
            A = stabilize_A(A_raw, T; margin = STABILITY_MARGIN)
            profiles[r, c] = intrinsic_profile(A, T, OMEGAS)
        end
    end

    return profiles
end

# -------------------------
# Plot
# -------------------------
function plot_profile_grid(profiles::Matrix{Vector{Float64}})
    nrows, ncols = size(profiles)

    allvals = vcat([profiles[r, c] for r in 1:nrows, c in 1:ncols]...)
    ymax = maximum(allvals)*1.05
    ymin = minimum(allvals[allvals .> 0])

    fig = Figure(size = (330 * ncols, 220 * nrows))

    for r in 1:nrows
        for c in 1:ncols
            title_str = r == 1 ? TS_LABELS[c] : ""
            ax = Axis(
                fig[r, c],
                title = title_str,
                xlabel = r == nrows ? "ω" : "",
                ylabel = c == 1 ? "network $r\nS(ω)" : "",
                xscale = log10,
                yscale = log10
            )

            lines!(ax, OMEGAS, profiles[r, c], linewidth = 3)
            ylims!(ax, ymin, ymax)
        end
    end

    Label(fig[0, :],
        "Intrinsic profiles for the same networks under increasing timescale heterogeneity ($(NETWORK_TYPE))",
        fontsize = 22)

    display(fig)
end

function main()
    println("Network type: $NETWORK_TYPE")
    println("Species: $S")
    println("Timescale heterogeneity levels: $(TS_SIGMAS)")
    profiles = build_profile_grid()
    plot_profile_grid(profiles)
end

main()