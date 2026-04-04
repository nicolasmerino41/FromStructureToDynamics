using LinearAlgebra
using Statistics
using Random, Printf
using CairoMakie
using StatsBase

# ============================================================
# 3) Progressive hub-set rewiring proof of concept
# ------------------------------------------------------------
# Smoother deterministic transition from regular graph to
# hub-set graph.
#
# Added:
#   - one row for central-class RPR profiles
#   - one row for satellite-class RPR profiles
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

function structured_spectrum(A::AbstractMatrix{<:Real},
                             P::AbstractMatrix{<:Real},
                             ωs::AbstractVector{<:Real},
                             T::AbstractMatrix{<:Real})
    n = size(A, 1)
    Icomplex = Matrix{ComplexF64}(I, n, n)
    Ac = ComplexF64.(A)
    Pc = ComplexF64.(P)
    Tc = ComplexF64.(T)
    out = zeros(Float64, length(ωs))

    for (k, ω) in pairs(ωs)
        F = factorize(im * ω .* Tc - Ac)
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

function class_from_rank(c::AbstractVector{<:Real}; q::Float64 = 0.2)
    n = length(c)
    m = max(2, round(Int, q * n))
    p = sortperm(c)
    low = sort(p[1:m])
    high = sort(p[end-m+1:end])
    return low, high
end

function perturbation_operator_class_allP(n::Int, C::AbstractVector{<:Integer})
    M = zeros(Float64, n, n)
    Cset = Set(C)

    @inbounds for i in 1:n, j in 1:n
        if i != j && (i in Cset || j in Cset)
            M[i, j] = 1.0
        end
    end

    return M / norm(M)
end

function perturbation_operator_class(A::AbstractMatrix{<:Real}, C::AbstractVector{<:Integer}; tol::Float64 = 0.0)
    n = size(A, 1)
    M = zeros(Float64, n, n)
    Cset = Set(C)

    @inbounds for i in 1:n, j in 1:n
        if i != j && abs(A[i, j]) > tol && (i in Cset || j in Cset)
            M[i, j] = 1.0
        end
    end

    nf = norm(M)
    nf == 0 && error("Class perturbation mask has zero Frobenius norm.")
    return M / nf
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
# Smoother rewiring rule
# ------------------------------------------------------------
function rewire_toward_hubset_smooth(B0::BitMatrix, hubs::Vector{Int}, λ::Float64)
    n = size(B0, 1)
    B = copy(B0)
    H = Set(hubs)

    nonhubs = [i for i in 1:n if !(i in H)]
    ns = length(nonhubs)

    max_rewires = Dict{Int, Int}()
    for i in nonhubs
        targets = [j for j in 1:n if B[i, j]]
        current_nonhub_targets = [j for j in targets if !(j in H) && j != i]
        available_hubs = [h for h in hubs if h != i && !B[i, h]]
        max_rewires[i] = min(length(current_nonhub_targets), length(available_hubs))
    end

    nrewire_per_source = Dict{Int, Int}()
    remainders = Float64[]
    for i in nonhubs
        x = λ * max_rewires[i]
        base = floor(Int, x)
        nrewire_per_source[i] = base
        push!(remainders, x - base)
    end

    order = sortperm(remainders, rev = true)
    for idx in order
        i = nonhubs[idx]
        x = λ * max_rewires[i]
        frac = x - floor(Int, x)
        if frac > 0
            if frac >= 0.5
                nrewire_per_source[i] += 1
            end
        end
    end

    for (src_idx, i) in enumerate(nonhubs)
        r = nrewire_per_source[i]
        r == 0 && continue

        targets = [j for j in 1:n if B[i, j]]
        current_nonhub_targets = [j for j in targets if !(j in H) && j != i]

        rotated_hubs = [hubs[((k + src_idx - 2) % length(hubs)) + 1] for k in 1:length(hubs)]
        available_hubs = [h for h in rotated_hubs if h != i && !B[i, h]]

        r = min(r, length(current_nonhub_targets), length(available_hubs))

        for k in 1:r
            oldt = current_nonhub_targets[k]
            newt = available_hubs[k]
            B[i, oldt] = false
            B[i, newt] = true
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

function intrinsic_spectrum(A::AbstractMatrix{<:Real},
                            ωs::AbstractVector{<:Real},
                            T::AbstractMatrix{<:Real})
    n = size(A, 1)
    Ac = ComplexF64.(A)
    Tc = ComplexF64.(T)
    Icomplex = Matrix{ComplexF64}(I, n, n)
    out = zeros(Float64, length(ωs))

    for (k, ω) in pairs(ωs)
        F = factorize(im * ω .* Tc - Ac)
        R = F \ Icomplex
        out[k] = opnorm(R, 2)
    end

    out
end
function make_timescale_matrix(τ::AbstractVector{<:Real})
    all(τ .> 0) || error("All timescales must be positive.")
    return Diagonal(Float64.(τ))
end
τ_uniform = ones(50)                      # baseline: T = I
τ_slow_hubs = ones(50)
τ_slow_hubs[1:4] .= 3.0                   # hubs 3x slower

τ_fast_hubs = ones(50)
τ_fast_hubs[1:4] .= 0.4                   # hubs faster

τ_gradient = Random.rand(50)              # random gradient
# ------------------------------------------------------------
# Main proof-of-concept
# ------------------------------------------------------------
function proof_progressive_hubset_creation(; all_P = true)
    n = 50
    kout = 6
    hubs = [1, 2, 3, 4]
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    τ = ones(n)
    τ[hubs] .= 3.0    # example: hubs are slower
    T = make_timescale_matrix(τ_gradient)

    B0 = regular_directed_mask(n, kout)

    As = Matrix{Float64}[]
    ratios = Vector{Float64}[]
    highs = Vector{Float64}[]
    lows = Vector{Float64}[]
    indeg_ginis = Float64[]
    indegs = Vector{Float64}[]
    intrinsics = Vector{Float64}[]

    for lev in levels
        B = rewire_toward_hubset_smooth(B0, hubs, lev)
        A = weights_on_mask(B; w = 0.8)
        A, diagonal_value = stabilize_A(A)
        println("diagonal value: ", diagonal_value)
        push!(As, A)

        indeg = Float64.(unweighted_indegree(A))
        push!(indegs, indeg)
        push!(indeg_ginis, gini(indeg))

        c = weighted_degree(A)
        lowC, highC = class_from_rank(c; q = 0.2)

        if all_P
            P_low = perturbation_operator_class_allP(n, lowC)
            P_high = perturbation_operator_class_allP(n, highC)
        else
            P_low = perturbation_operator_class(A, lowC)
            P_high = perturbation_operator_class(A, highC)
        end

        H = structured_spectrum(A, P_high, OMEGAS, T)
        L = structured_spectrum(A, P_low, OMEGAS, T)

        push!(highs, H)
        push!(lows, L)
        push!(ratios, H ./ L)

        S = intrinsic_spectrum(A, OMEGAS, T)
        push!(intrinsics, S)
    end

    fig = Figure(size = (1700, 1800))

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

        vals = indegs[idx]
        counts = countmap(vals)

        kmin = minimum(vals) - 1
        kmax = maximum(vals) + 1

        xs = collect(kmin:kmax)
        ys = [get(counts, k, 0) for k in xs]

        barplot!(ax, xs, ys, width = 0.6)

        if length(unique(vals)) == 1
            ax.xticks = (xs, string.(xs))
        end
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
    axislegend(axr, position = :rt)

    # --------------------------------------------------------
    # Row 4: central-class RPR profile
    # --------------------------------------------------------
    axc = Axis(fig[4, 1:5],
        title = "Central-class RPR profile",
        xlabel = "frequency ω",
        ylabel = "||R(ω) P_central R(ω)||",
        xscale = log10,
        yscale = identity)

    for (idx, lev) in enumerate(levels)
        lbl = @sprintf("rewiring = %.2f", lev)
        lines!(axc, OMEGAS, highs[idx], linewidth = 3, label = lbl)
    end
    axislegend(axc, position = :rt)

    # --------------------------------------------------------
    # Row 5: satellite-class RPR profile
    # --------------------------------------------------------
    axl = Axis(fig[5, 1:5],
        title = "Satellite-class RPR profile",
        xlabel = "frequency ω",
        ylabel = "||R(ω) P_satellite R(ω)||",
        xscale = log10,
        yscale = identity)

    for (idx, lev) in enumerate(levels)
        lbl = @sprintf("rewiring = %.2f", lev)
        lines!(axl, OMEGAS, lows[idx], linewidth = 3, label = lbl)
    end
    axislegend(axl, position = :rt)

    # --------------------------------------------------------
    # Row 6: intrinsic spectra
    # --------------------------------------------------------
    axs = Axis(
        fig[6, 1:5],
        title = "Intrinsic sensitivity across rewiring levels",
        xlabel = "frequency ω",
        ylabel = "S(ω)",
        xscale = log10,
        yscale = identity
    )
    for (idx, lev) in enumerate(levels)
        lbl = @sprintf("rewiring = %.2f", lev)
        lines!(axs, OMEGAS, intrinsics[idx], linewidth = 3, label = lbl)
    end
    axislegend(axs, position = :rt)

    display(fig)
end

proof_progressive_hubset_creation(; all_P = false)