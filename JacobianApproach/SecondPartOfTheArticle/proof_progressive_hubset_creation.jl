using LinearAlgebra
using Statistics
using Random
using CairoMakie

# ============================================================
# 3) Progressive hub-set rewiring proof of concept
# ------------------------------------------------------------
# Deterministic family of matrices A^(λ):
#
#   λ = 0   -> regular directed graph (homogeneous degree structure)
#   λ = 1   -> same edge count, but non-hub nodes preferentially
#              point to a small fixed hub set
#
# Fixed throughout:
#   T = I
#   R(ω) = (im*ω*I - A)^(-1)
#
# Goal:
#   Create a much stronger topological contrast than the previous
#   "single-hub mild rewiring" construction.
# ============================================================

Random.seed!(1234)

const OMEGAS = exp.(range(log(1e-2), log(1e2), length = 250))
const STABILITY_MARGIN = 0.2

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
function stabilize_A(A::AbstractMatrix{<:Real}; margin::Float64 = STABILITY_MARGIN)
    α = maximum(real.(eigvals(Matrix(A))))
    shift = max(0.0, α + margin)
    return Matrix(A) - shift * I, shift
end

function structured_spectrum(A::AbstractMatrix{<:Real}, P::AbstractMatrix{<:Real}, ωs::AbstractVector{<:Real})
    n = size(A, 1)
    Icomplex = Matrix{ComplexF64}(I, n, n)
    Ac = ComplexF64.(A)
    Pc = ComplexF64.(P)
    out = zeros(Float64, length(ωs))

    for (k, ω) in pairs(ωs)
        F = factorize(im * ω .* Icomplex - Ac)
        R = F \ Icomplex
        out[k] = opnorm(R * Pc * R, 2)
    end

    out
end

function weighted_degree(A::AbstractMatrix{<:Real})
    vec(sum(abs.(A), dims = 1)) .+ vec(sum(abs.(A), dims = 2))
end

function unweighted_indegree(A::AbstractMatrix{<:Real}; tol::Float64 = 0.0)
    B = abs.(A) .> tol
    vec(sum(B, dims = 1))
end

function unweighted_outdegree(A::AbstractMatrix{<:Real}; tol::Float64 = 0.0)
    B = abs.(A) .> tol
    vec(sum(B, dims = 2))
end

function class_from_rank(c::AbstractVector{<:Real}; q::Float64 = 0.2)
    n = length(c)
    m = max(2, round(Int, q * n))
    p = sortperm(c)
    low = sort(p[1:m])
    high = sort(p[end-m+1:end])
    return low, high
end

function perturbation_operator_class(n::Int, C::AbstractVector{<:Integer})
    M = zeros(Float64, n, n)
    Cset = Set(C)

    @inbounds for i in 1:n, j in 1:n
        if i != j && (i in Cset || j in Cset)
            M[i, j] = 1.0
        end
    end

    M / norm(M)
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
    num / (n * s)
end

# ------------------------------------------------------------
# Base regular directed graph
# ------------------------------------------------------------
# Each node points to the next kout nodes on the ring.
# So outdegree is exactly fixed and homogeneous.
# ------------------------------------------------------------
function regular_directed_mask(n::Int, kout::Int)
    B = falses(n, n)
    for i in 1:n
        for s in 1:kout
            j = ((i + s - 1) % n) + 1
            if j != i
                B[i, j] = true
            end
        end
    end
    B
end

# ------------------------------------------------------------
# New rewiring rule: regular graph -> hub-set graph
# ------------------------------------------------------------
# H is a fixed small hub set.
#
# For each non-hub source i:
#   - identify its outgoing non-hub edges
#   - rewire a fraction λ of those edges so they point to hubs
#   - preserve outdegree exactly
#
# Hub sources themselves are left unchanged for clarity.
#
# This creates a strong in-degree concentration onto the hub set.
# ------------------------------------------------------------
function rewire_toward_hubset(B0::BitMatrix, hubs::Vector{Int}, λ::Float64)
    n = size(B0, 1)
    B = copy(B0)
    H = Set(hubs)

    nonhubs = [i for i in 1:n if !(i in H)]

    for i in nonhubs
        # current outgoing edges
        targets = [j for j in 1:n if B[i, j]]

        # only edges currently pointing to non-hubs are candidates
        current_nonhub_targets = [j for j in targets if !(j in H) && j != i]

        # hub targets not already occupied
        available_hubs = [h for h in hubs if h != i && !B[i, h]]

        nrewire = min(round(Int, λ * length(current_nonhub_targets)), length(available_hubs))

        if nrewire > 0
            # deterministically replace the first nrewire non-hub targets
            # by the first nrewire available hub targets
            for k in 1:nrewire
                oldt = current_nonhub_targets[k]
                newt = available_hubs[k]
                B[i, oldt] = false
                B[i, newt] = true
            end
        end
    end

    return B
end

function weights_on_mask(B::BitMatrix; w::Float64 = 0.8)
    A = zeros(Float64, size(B)...)
    A[B] .= w
    A
end

# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
function heatmap_axis(figpos, A, title_text)
    ax = Axis(figpos, title = title_text, xlabel = "j", ylabel = "i", aspect = DataAspect())
    heatmap!(ax, A)
    ax
end

# ------------------------------------------------------------
# Main proof-of-concept
# ------------------------------------------------------------
function proof_progressive_hubset_creation()
    n = 50
    kout = 6

    # choose a small deterministic hub set
    hubs = [1, 2, 3, 4]
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    B0 = regular_directed_mask(n, kout)

    As = Matrix{Float64}[]
    ratios = Vector{Float64}[]
    highs = Vector{Float64}[]
    lows = Vector{Float64}[]
    indeg_ginis = Float64[]
    indegs = Vector{Float64}[]

    for lev in levels
        B = rewire_toward_hubset(B0, hubs, lev)
        A = weights_on_mask(B; w = 0.8)
        A, _ = stabilize_A(A)
        push!(As, A)

        indeg = Float64.(unweighted_indegree(A))
        push!(indegs, indeg)
        push!(indeg_ginis, gini(indeg))

        c = weighted_degree(A)
        lowC, highC = class_from_rank(c; q = 0.2)

        P_low = perturbation_operator_class(n, lowC)
        P_high = perturbation_operator_class(n, highC)

        H = structured_spectrum(A, P_high, OMEGAS)
        L = structured_spectrum(A, P_low, OMEGAS)

        push!(highs, H)
        push!(lows, L)
        push!(ratios, H ./ L)
    end

    fig = Figure(size = (1700, 1200))
    Label(fig[0, :], "3) Progressive hub-set rewiring proof of concept", fontsize = 24)

    # --------------------------------------------------------
    # Row 1: matrices
    # --------------------------------------------------------
    for (idx, lev) in enumerate(levels)
        A = As[idx]
        ttl = @sprintf("rewiring = %.2f", lev)
        ax = heatmap_axis(fig[1, idx], A, ttl)
        if idx != 1
            ax.ylabel = ""
        end
    end

    # --------------------------------------------------------
    # Row 2: indegree histograms
    # --------------------------------------------------------
    for (idx, lev) in enumerate(levels)
        ax = Axis(fig[2, idx],
            title = @sprintf("in-degree distribution, λ = %.2f", lev),
            xlabel = "unweighted in-degree",
            ylabel = idx == 1 ? "count" : "")
        hist!(ax, indegs[idx], bins = 18)
    end

    # --------------------------------------------------------
    # Row 3: high/low ratio
    # --------------------------------------------------------
    axr = Axis(fig[3, 1:5],
        title = "High / low centrality sensitivity ratio",
        xlabel = "frequency ω",
        ylabel = "S_high(ω) / S_low(ω)",
        xscale = log10,
        yscale = identity)

    for (idx, lev) in enumerate(levels)
        lbl = @sprintf("rewiring = %.2f  (Gini = %.2f)", lev, indeg_ginis[idx])
        lines!(axr, OMEGAS, ratios[idx], linewidth = 3, label = lbl)
    end
    lines!(axr, OMEGAS, ones(length(OMEGAS)), linestyle = :dash, linewidth = 2, color = :black)
    axislegend(axr, position = :rb)

    # --------------------------------------------------------
    # Row 4: structured spectra for high-centrality class
    # --------------------------------------------------------
    axs = Axis(fig[4, 1:5],
        title = "Structured spectra for the hub-set rewiring sequence (high-centrality class)",
        xlabel = "frequency ω",
        ylabel = "S_P(ω)",
        xscale = log10,
        yscale = identity)

    for (idx, lev) in enumerate(levels)
        lbl = @sprintf("high class, %.2f", lev)
        lines!(axs, OMEGAS, highs[idx], linewidth = 3, label = lbl)
    end
    axislegend(axs, position = :rb)

    display(fig)
end

proof_progressive_hubset_creation()