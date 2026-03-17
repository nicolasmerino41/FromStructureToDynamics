using LinearAlgebra
using Statistics
using Random
using CairoMakie

# ============================================================
# Deterministic proofs of concept for frequency-resolved
# structural sensitivity, with homogeneous timescales T = I.
#
# Resolvent:
#   R(ω) = (im*ω*I - A)^(-1)
#
# Intrinsic sensitivity:
#   S(ω) = ||R(ω)||_2
#
# Structured sensitivity:
#   S_P(ω) = ||R(ω) P R(ω)||_2
#
# Three proofs of concept:
#   1) non-normality
#   2) strength localization
#   3) progressive hub creation
# ============================================================

Random.seed!(1234)

const OMEGAS = exp.(range(log(1e-2), log(1e2), length=250))
const STABILITY_MARGIN = 0.2

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
function stabilize_A(A::AbstractMatrix{<:Real}; margin::Float64 = STABILITY_MARGIN)
    α = maximum(real.(eigvals(Matrix(A))))
    shift = max(0.0, α + margin)
    return Matrix(A) - shift * I, shift
end

function intrinsic_spectrum(A::AbstractMatrix{<:Real}, ωs::AbstractVector{<:Real})
    n = size(A, 1)
    Icomplex = Matrix{ComplexF64}(I, n, n)
    Ac = ComplexF64.(A)
    out = zeros(Float64, length(ωs))
    for (k, ω) in pairs(ωs)
        F = factorize(im * ω .* Icomplex - Ac)
        R = F \ Icomplex
        out[k] = opnorm(R, 2)
    end
    out
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
    vec(sum(abs.(A), dims=1)) .+ vec(sum(abs.(A), dims=2))
end

function class_from_rank(c::AbstractVector{<:Real}; q::Float64=0.2)
    n = length(c)
    m = max(2, round(Int, q*n))
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

function edgecount_offdiag(B::BitMatrix)
    count(B)
end

function make_heatmap_axis(figpos, A, title_text)
    ax = Axis(figpos, title=title_text, xlabel="j", ylabel="i", aspect=DataAspect())
    heatmap!(ax, A)
    ax
end

# ------------------------------------------------------------
# 1) NON-NORMALITY PROOF OF CONCEPT
# ------------------------------------------------------------
#
# Same size, similar interaction scale, but different geometry:
#   - symmetric chain: near-normal
#   - feedforward chain: strongly non-normal
#
# We compare intrinsic spectra and a focal perturbation class.
# ------------------------------------------------------------
function build_symmetric_chain(n::Int; g::Float64=1.0)
    A = zeros(Float64, n, n)
    for i in 1:n-1
        A[i, i+1] = g
        A[i+1, i] = g
    end
    A
end

function build_feedforward_chain(n::Int; g::Float64=1.0)
    A = zeros(Float64, n, n)
    for i in 1:n-1
        A[i, i+1] = g
    end
    A
end

function proof_non_normality()
    n = 8
    A_sym = build_symmetric_chain(n; g=1.0)
    A_ff  = build_feedforward_chain(n; g=1.0)

    A_sym, shift_sym = stabilize_A(A_sym)
    A_ff,  shift_ff  = stabilize_A(A_ff)

    c_sym = weighted_degree(A_sym)
    c_ff  = weighted_degree(A_ff)

    low_sym, high_sym = class_from_rank(c_sym; q=0.25)
    low_ff,  high_ff  = class_from_rank(c_ff; q=0.25)

    P_sym = perturbation_operator_class(n, high_sym)
    P_ff  = perturbation_operator_class(n, high_ff)

    S_sym = intrinsic_spectrum(A_sym, OMEGAS)
    S_ff  = intrinsic_spectrum(A_ff, OMEGAS)

    SP_sym = structured_spectrum(A_sym, P_sym, OMEGAS)
    SP_ff  = structured_spectrum(A_ff, P_ff, OMEGAS)

    fig = Figure(size=(1400, 900))
    Label(fig[0, :], "1) Non-normality proof of concept", fontsize=24)

    make_heatmap_axis(fig[1, 1], A_sym, "Symmetric chain (near-normal)")
    make_heatmap_axis(fig[1, 2], A_ff,  "Feedforward chain (non-normal)")

    ax = Axis(fig[2, 1:2],
        title = "Intrinsic spectra",
        xlabel = "frequency ω",
        ylabel = "S(ω)",
        xscale = log10,
        yscale = log10)
    lines!(ax, OMEGAS, S_sym, linewidth=3, label="symmetric")
    lines!(ax, OMEGAS, S_ff, linewidth=3, label="feedforward")
    axislegend(ax, position=:rb)

    ax2 = Axis(fig[3, 1:2],
        title = "Structured spectra for high-centrality perturbation class",
        xlabel = "frequency ω",
        ylabel = "S_P(ω)",
        xscale = log10,
        yscale = log10)
    lines!(ax2, OMEGAS, SP_sym, linewidth=3, label="symmetric")
    lines!(ax2, OMEGAS, SP_ff, linewidth=3, label="feedforward")
    axislegend(ax2, position=:rb)

    display(fig)
end

# ------------------------------------------------------------
# 2) STRENGTH LOCALIZATION PROOF OF CONCEPT
# ------------------------------------------------------------
#
# Same adjacency pattern and same total off-diagonal weight,
# but strong weights are either:
#   - concentrated around one focal species
#   - distributed diffusely over all existing links
#
# This isolates "where the IS are placed".
# ------------------------------------------------------------
function ring_adjacency(n::Int)
    B = falses(n, n)
    for i in 1:n
        j1 = i == n ? 1 : i + 1
        j2 = i == 1 ? n : i - 1
        B[i, j1] = true
        B[j1, i] = true
        B[i, j2] = true
        B[j2, i] = true
    end
    B
end

function matrix_from_weights(B::BitMatrix, W::AbstractMatrix{<:Real})
    A = zeros(Float64, size(B)...)
    A[B] .= W[B]
    A
end

function make_localized_weights(B::BitMatrix, focal::Int; strong::Float64=1.8, weak::Float64=0.35)
    n = size(B, 1)
    W = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        if B[i, j]
            W[i, j] = (i == focal || j == focal) ? strong : weak
        end
    end
    W
end

function make_diffuse_weights(B::BitMatrix; total_weight::Float64)
    m = edgecount_offdiag(B)
    w = total_weight / m
    W = zeros(Float64, size(B)...)
    W[B] .= w
    W
end

function proof_strength_localization()
    n = 10
    focal = 1
    B = ring_adjacency(n)

    W_loc = make_localized_weights(B, focal; strong=1.8, weak=0.35)
    total_weight = sum(W_loc)
    W_dif = make_diffuse_weights(B; total_weight=total_weight)

    A_loc = matrix_from_weights(B, W_loc)
    A_dif = matrix_from_weights(B, W_dif)

    A_loc, _ = stabilize_A(A_loc)
    A_dif, _ = stabilize_A(A_dif)

    c_loc = weighted_degree(A_loc)
    c_dif = weighted_degree(A_dif)

    low_loc, high_loc = class_from_rank(c_loc; q=0.2)
    low_dif, high_dif = class_from_rank(c_dif; q=0.2)

    P_high_loc = perturbation_operator_class(n, high_loc)
    P_low_loc  = perturbation_operator_class(n, low_loc)
    P_high_dif = perturbation_operator_class(n, high_dif)
    P_low_dif  = perturbation_operator_class(n, low_dif)

    S_loc = intrinsic_spectrum(A_loc, OMEGAS)
    S_dif = intrinsic_spectrum(A_dif, OMEGAS)

    H_loc = structured_spectrum(A_loc, P_high_loc, OMEGAS)
    L_loc = structured_spectrum(A_loc, P_low_loc, OMEGAS)
    H_dif = structured_spectrum(A_dif, P_high_dif, OMEGAS)
    L_dif = structured_spectrum(A_dif, P_low_dif, OMEGAS)

    fig = Figure(size=(1400, 1000))
    Label(fig[0, :], "2) Strength localization proof of concept", fontsize=24)

    make_heatmap_axis(fig[1, 1], A_dif, "Diffuse strengths")
    make_heatmap_axis(fig[1, 2], A_loc, "Strengths concentrated around focal species")

    ax = Axis(fig[2, 1:2],
        title = "Intrinsic spectra",
        xlabel = "frequency ω",
        ylabel = "S(ω)",
        xscale = log10,
        yscale = log10)
    lines!(ax, OMEGAS, S_dif, linewidth=3, label="diffuse")
    lines!(ax, OMEGAS, S_loc, linewidth=3, label="localized")
    axislegend(ax, position=:rb)

    ax2 = Axis(fig[3, 1],
        title = "Diffuse case: high vs low centrality perturbations",
        xlabel = "frequency ω",
        ylabel = "S_P(ω)",
        xscale = log10,
        yscale = log10)
    lines!(ax2, OMEGAS, H_dif, linewidth=3, label="high-centrality")
    lines!(ax2, OMEGAS, L_dif, linewidth=3, label="low-centrality")
    axislegend(ax2, position=:rb)

    ax3 = Axis(fig[3, 2],
        title = "Localized case: high vs low centrality perturbations",
        xlabel = "frequency ω",
        ylabel = "S_P(ω)",
        xscale = log10,
        yscale = log10)
    lines!(ax3, OMEGAS, H_loc, linewidth=3, label="high-centrality")
    lines!(ax3, OMEGAS, L_loc, linewidth=3, label="low-centrality")
    axislegend(ax3, position=:rb)

    display(fig)
end

# ------------------------------------------------------------
# 3) PROGRESSIVE HUB CREATION PROOF OF CONCEPT
# ------------------------------------------------------------
#
# Start from a regular directed random pattern with fixed outdegree.
# Progressively rewire some edges so they point to one focal hub.
# Connectance remains fixed; only concentration around one hub grows.
# ------------------------------------------------------------
function regular_directed_mask(n::Int, kout::Int)
    B = falses(n, n)
    for i in 1:n
        targets = Int[]
        j = i + 1
        while length(targets) < kout
            jj = ((j - 1) % n) + 1
            if jj != i
                push!(targets, jj)
            end
            j += 1
        end
        for t in targets
            B[i, t] = true
        end
    end
    B
end

function rewire_toward_hub(B0::BitMatrix, hub::Int, level::Float64)
    n = size(B0, 1)
    B = copy(B0)
    edges = [(i, j) for i in 1:n for j in 1:n if B[i, j] && j != hub]
    nrewire = round(Int, level * length(edges))

    for k in 1:nrewire
        i, j = edges[k]
        if i != hub && !B[i, hub]
            B[i, j] = false
            B[i, hub] = true
        end
    end
    B
end

function weights_on_mask(B::BitMatrix; w::Float64=0.8)
    A = zeros(Float64, size(B)...)
    A[B] .= w
    A
end

function proof_progressive_hub_creation()
    n = 10
    kout = 2
    hub = 1
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    B0 = regular_directed_mask(n, kout)
    As = Matrix{Float64}[]
    ratios = Vector{Float64}[]
    highs = Vector{Float64}[]
    lows = Vector{Float64}[]

    fig = Figure(size=(1600, 1100))
    Label(fig[0, :], "3) Progressive hub creation proof of concept", fontsize=24)

    for (idx, lev) in enumerate(levels)
        B = rewire_toward_hub(B0, hub, lev)
        A = weights_on_mask(B; w=0.8)
        A, _ = stabilize_A(A)
        push!(As, A)

        c = weighted_degree(A)
        lowC, highC = class_from_rank(c; q=0.1)
        P_low = perturbation_operator_class(n, lowC)
        P_high = perturbation_operator_class(n, highC)

        H = structured_spectrum(A, P_high, OMEGAS)
        L = structured_spectrum(A, P_low, OMEGAS)
        push!(highs, H)
        push!(lows, L)
        push!(ratios, H ./ L)

        axh = Axis(fig[1, idx], title = "rewiring = $(lev)", xlabel="j", ylabel=idx==1 ? "i" : "", aspect=DataAspect())
        heatmap!(axh, A)
    end

    ax = Axis(fig[2, 1:5],
        title = "High / low centrality sensitivity ratio",
        xlabel = "frequency ω",
        ylabel = "S_high(ω) / S_low(ω)",
        xscale = log10,
        yscale = log10)
    for (idx, lev) in enumerate(levels)
        lines!(ax, OMEGAS, ratios[idx], linewidth=3, label="rewiring = $(lev)")
    end
    lines!(ax, OMEGAS, ones(length(OMEGAS)), linestyle=:dash, linewidth=2, color=:black)
    axislegend(ax, position=:rb)

    ax2 = Axis(fig[3, 1:5],
        title = "Structured spectra for the progressive hub-creation sequence",
        xlabel = "frequency ω",
        ylabel = "S_P(ω)",
        xscale = log10,
        yscale = log10)
    for (idx, lev) in enumerate(levels)
        lines!(ax2, OMEGAS, highs[idx], linewidth=3, label="high class, $(lev)")
    end
    axislegend(ax2, position=:rb)

    display(fig)
end

# ------------------------------------------------------------
# Run one by one
# ------------------------------------------------------------
proof_non_normality()
proof_strength_localization()
proof_progressive_hub_creation()