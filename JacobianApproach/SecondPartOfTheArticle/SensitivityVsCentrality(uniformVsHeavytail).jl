using LinearAlgebra
using Statistics
using Random
using Distributions
using StatsBase
using Printf
using CairoMakie

# ============================================================
# Structural sensitivity analysis for synthetic ecological networks
# ------------------------------------------------------------
# Choices fixed for clarity and minimal assumptions:
# 1) Interaction matrix: A
# 2) Intrinsic timescale matrix: T = Diagonal(t)
# 3) Full Jacobian: J = T \ A
# 4) Resolvent: R(ω) = (im * ω * T - A)^(-1)
# 5) Sensitivity norm: operator / spectral 2-norm
# 6) Stability: shift diag(A) by the minimum amount needed so that
#    max real part of eig(T \ A) equals -margin
# 7) Centrality: weighted degree = row sum abs(A) + column sum abs(A)
# 8) Perturbation operator for focal class C:
#       P^(C) = M^(C) / ||M^(C)||_F
#    where M^(C)_{ij}=1 iff i in C or j in C, excluding diagonal
# 9) Network types:
#       - uniform degree structure
#       - heavy-tailed degree structure
# 10) Interaction values: Normal(0, IS)
# 11) Timescales: mildly heterogeneous lognormal around mean ~1
# ============================================================
const DEFAULT_SEED = 1234
Random.seed!(DEFAULT_SEED)

# -------------------------
# Utility functions
# -------------------------
function rank_average(x::AbstractVector{<:Real})
    p = sortperm(x)
    r = zeros(Float64, length(x))
    i = 1
    while i <= length(x)
        j = i
        while j < length(x) && x[p[j+1]] == x[p[i]]
            j += 1
        end
        avg_rank = (i + j) / 2
        for k in i:j
            r[p[k]] = avg_rank
        end
        i = j + 1
    end
    return r
end

function spearman_corr(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    rx = rank_average(collect(x))
    ry = rank_average(collect(y))
    sx = std(rx)
    sy = std(ry)
    if sx == 0 || sy == 0
        return NaN
    end
    return cor(rx, ry)
end

function trapz(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y)
    s = 0.0
    for i in 1:length(x)-1
        s += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return s
end

geom_midpoint(ωs::AbstractVector{<:Real}) = exp((log(first(ωs)) + log(last(ωs))) / 2)

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
    connectance::Float64 = 0.15,
    IS::Float64 = 0.25
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
    t = ones(Float64, S)
    return t
end

# -------------------------
# Dynamics and resolvent
# -------------------------
function stabilize_A(A::AbstractMatrix{<:Real}, T::AbstractMatrix{<:Real}; margin::Float64 = 0.1)
    S = size(A, 1)
    J0 = T \ Matrix(A)
    α = maximum(real.(eigvals(J0)))
    shift = max(0.0, α + margin)
    A_stable = Matrix(A) - shift * T
    J = T \ A_stable
    return A_stable, J, shift
end

function intrinsic_spectrum(A::AbstractMatrix{<:Real}, T::AbstractMatrix{<:Real},
                            ωs::AbstractVector{<:Real})
    vals = zeros(Float64, length(ωs))
    Ac = ComplexF64.(A)
    Tc = ComplexF64.(T)
    for (k, ω) in pairs(ωs)
        R = (im * ω .* Tc - Ac) \ I
        vals[k] = opnorm(R, 2)
    end
    return vals
end

function structured_spectrum(A::AbstractMatrix{<:Real}, T::AbstractMatrix{<:Real},
                             P::AbstractMatrix{<:Real},
                             ωs::AbstractVector{<:Real})
    vals = zeros(Float64, length(ωs))
    Ac = ComplexF64.(A)
    Tc = ComplexF64.(T)
    Pc = ComplexF64.(P)
    for (k, ω) in pairs(ωs)
        R = (im * ω .* Tc - Ac) \ I
        vals[k] = opnorm(R * Pc * R, 2)
    end
    return vals
end

# -------------------------
# Centrality and perturbation operators
# -------------------------
function weighted_degree(A::AbstractMatrix{<:Real})
    return vec(sum(abs.(A), dims = 1)) .+ vec(sum(abs.(A), dims = 2))
end

function unweighted_degree(A::AbstractMatrix{<:Real}; tol::Float64 = 0.0)
    B = abs.(A) .> tol
    return vec(sum(B, dims = 1)) .+ vec(sum(B, dims = 2))
end

function centrality_classes(c::AbstractVector{<:Real}; q::Float64 = 0.2)
    S = length(c)
    m = max(2, round(Int, q * S))
    p = sortperm(c)
    low = p[1:m]
    high = p[end-m+1:end]
    return sort(low), sort(high)
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
    nf == 0 && error("Class perturbation mask has zero Frobenius norm.")
    return M / nf
end

function perturbation_operator_species(S::Int, i::Int)
    M = zeros(Float64, S, S)
    @inbounds for j in 1:S
        if j != i
            M[i, j] = 1.0
            M[j, i] = 1.0
        end
    end
    nf = norm(M)
    return M / nf
end

# -------------------------
# Spectrum summaries
# -------------------------
function spectrum_summary(ωs::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    kmax = argmax(s)
    ωmid = geom_midpoint(ωs)
    low_idx = findall(≤(ωmid), ωs)
    high_idx = findall(>(ωmid), ωs)
    return (
        peak = s[kmax],
        peak_freq = ωs[kmax],
        low_int = trapz(ωs[low_idx], s[low_idx]),
        high_int = trapz(ωs[high_idx], s[high_idx]),
        ωmid = ωmid
    )
end

# -------------------------
# Single-network analysis
# -------------------------
function analyze_network(A_raw::AbstractMatrix{<:Real},
                         t::AbstractVector{<:Real},
                         ωs::AbstractVector{<:Real})

    S = size(A_raw, 1)
    T = Diagonal(t)

    A, J, shift = stabilize_A(A_raw, T)

    c = weighted_degree(A)
    lowC, highC = centrality_classes(c)

    P_low = perturbation_operator_class(S, lowC)
    P_high = perturbation_operator_class(S, highC)

    nω = length(ωs)

    s_intr = zeros(Float64, nω)
    s_low  = zeros(Float64, nω)
    s_high = zeros(Float64, nω)

    species_spec = zeros(Float64, S, nω)

    # convert once
    Ac = ComplexF64.(A)
    Tc = ComplexF64.(T)
    Pc_low  = ComplexF64.(P_low)
    Pc_high = ComplexF64.(P_high)

    onesv = ones(ComplexF64, S)

    for (k, ω) in pairs(ωs)

        F = factorize(im * ω .* Tc - Ac)
        R = F \ I

        s_intr[k] = opnorm(R, 2)

        s_low[k]  = opnorm(R * Pc_low * R, 2)
        s_high[k] = opnorm(R * Pc_high * R, 2)

        for i in 1:S
            species_spec[i, k] = species_sensitivity_fast(R, i)
        end
    end

    sum_intr = spectrum_summary(ωs, s_intr)
    sum_low  = spectrum_summary(ωs, s_low)
    sum_high = spectrum_summary(ωs, s_high)

    H = zeros(Float64, S)
    L = zeros(Float64, S)
    U = zeros(Float64, S)
    peakfreq = zeros(Float64, S)

    for i in 1:S
        si = view(species_spec, i, :)
        sm = spectrum_summary(ωs, si)

        H[i] = sm.peak
        L[i] = sm.low_int
        U[i] = sm.high_int
        peakfreq[i] = sm.peak_freq
    end

    rank_stats = (
        rho_peak = spearman_corr(c, H),
        rho_low = spearman_corr(c, L),
        rho_high = spearman_corr(c, U),
        rho_peakfreq = spearman_corr(c, peakfreq)
    )

    return (
        A_raw = A_raw,
        A = A,
        T = T,
        t = t,
        J = J,
        stabilization_shift = shift,
        centrality = c,
        low_class = lowC,
        high_class = highC,
        P_low = P_low,
        P_high = P_high,
        ωs = ωs,
        intrinsic = s_intr,
        low_spectrum = s_low,
        high_spectrum = s_high,
        summary_intrinsic = sum_intr,
        summary_low = sum_low,
        summary_high = sum_high,
        species_peak = H,
        species_low = L,
        species_high = U,
        species_peakfreq = peakfreq,
        rank_stats = rank_stats
    )
end

# -------------------------
# Ensemble analysis
# -------------------------
function species_sensitivity_fast(R::AbstractMatrix{<:Complex}, i::Int)
    S = size(R, 1)

    ei = zeros(eltype(R), S)
    ei[i] = 1

    onesv = ones(eltype(R), S)

    u = R * ei
    v = R' * onesv

    # Gram matrix of rank-2 operator
    g11 = real(dot(u, u))
    g22 = real(dot(v, v))
    g12 = real(dot(u, v))

    G = [g11 g12; g12 g22]

    return sqrt(maximum(eigvals(G)))
end

using Base.Threads
using Base.Threads

function build_network_ensemble(S::Int, network_type::Symbol, nrep::Int;
                                connectance::Float64 = 0.15,
                                IS::Float64 = 0.25,
                                ts_sigma_log::Float64 = 0.15,
                                ωmin::Float64 = 1e-2,
                                ωmax::Float64 = 1e2,
                                nω::Int = 220)

    ωs = exp.(range(log(ωmin), log(ωmax), length = nω))
    results = Vector{NamedTuple}(undef, nrep)

    Threads.@threads for r in 1:nrep

        A_raw = generate_interaction_matrix(
            S, network_type;
            connectance = connectance,
            IS = IS
        )

        t = generate_timescales(S; sigma_log = ts_sigma_log)

        results[r] = analyze_network(A_raw, t, ωs)
    end

    return results
end

function ensemble_spectrum_mean(results::Vector{<:NamedTuple}, field::Symbol)
    X = reduce(hcat, [getfield(r, field) for r in results])
    μ = vec(mean(X, dims = 2))
    sd = vec(std(X, dims = 2))
    return μ, sd
end

function extract_summaries(results::Vector{<:NamedTuple}, summary_field::Symbol, key::Symbol)
    return [getfield(getfield(r, summary_field), key) for r in results]
end

function extract_rankstat(results::Vector{<:NamedTuple}, key::Symbol)
    return [getfield(r.rank_stats, key) for r in results]
end

# -------------------------
# Plotting helpers (Makie)
# -------------------------
function save_mean_spectra(results_uniform, results_heavy; outdir = "results")
    mkpath(outdir)
    ωs = results_uniform[1].ωs

    μI_u, sdI_u = ensemble_spectrum_mean(results_uniform, :intrinsic)
    μL_u, sdL_u = ensemble_spectrum_mean(results_uniform, :low_spectrum)
    μH_u, sdH_u = ensemble_spectrum_mean(results_uniform, :high_spectrum)

    μI_h, sdI_h = ensemble_spectrum_mean(results_heavy, :intrinsic)
    μL_h, sdL_h = ensemble_spectrum_mean(results_heavy, :low_spectrum)
    μH_h, sdH_h = ensemble_spectrum_mean(results_heavy, :high_spectrum)

    fig = Figure(size = (1300, 520))

    ax1 = Axis(fig[1, 1],
        title = "Uniform networks",
        xlabel = "frequency ω",
        ylabel = "sensitivity",
        xscale = log10,
        yscale = log10)

    band!(ax1, ωs, max.(μI_u .- sdI_u, 1e-12), μI_u .+ sdI_u, alpha = 0.18)
    lines!(ax1, ωs, μI_u, linewidth = 3, label = "intrinsic S(ω)")

    band!(ax1, ωs, max.(μH_u .- sdH_u, 1e-12), μH_u .+ sdH_u, alpha = 0.18)
    lines!(ax1, ωs, μH_u, linewidth = 3, label = "high-centrality Sₚ(ω)")

    band!(ax1, ωs, max.(μL_u .- sdL_u, 1e-12), μL_u .+ sdL_u, alpha = 0.18)
    lines!(ax1, ωs, μL_u, linewidth = 3, label = "low-centrality Sₚ(ω)")

    axislegend(ax1, position = :rb)

    ax2 = Axis(fig[1, 2],
        title = "Heavy-tailed networks",
        xlabel = "frequency ω",
        ylabel = "sensitivity",
        xscale = log10,
        yscale = log10)

    band!(ax2, ωs, max.(μI_h .- sdI_h, 1e-12), μI_h .+ sdI_h, alpha = 0.18)
    lines!(ax2, ωs, μI_h, linewidth = 3, label = "intrinsic S(ω)")

    band!(ax2, ωs, max.(μH_h .- sdH_h, 1e-12), μH_h .+ sdH_h, alpha = 0.18)
    lines!(ax2, ωs, μH_h, linewidth = 3, label = "high-centrality Sₚ(ω)")

    band!(ax2, ωs, max.(μL_h .- sdL_h, 1e-12), μL_h .+ sdL_h, alpha = 0.18)
    lines!(ax2, ωs, μL_h, linewidth = 3, label = "low-centrality Sₚ(ω)")

    axislegend(ax2, position = :rb)

    save(joinpath(outdir, "mean_spectra.png"), fig)
    return fig
end

function save_ratio_spectra(results_uniform, results_heavy; outdir = "results")
    mkpath(outdir)
    ωs = results_uniform[1].ωs

    ratio_u = reduce(hcat, [r.high_spectrum ./ r.low_spectrum for r in results_uniform])
    ratio_h = reduce(hcat, [r.high_spectrum ./ r.low_spectrum for r in results_heavy])

    μu = vec(mean(ratio_u, dims = 2))
    sdu = vec(std(ratio_u, dims = 2))
    μh = vec(mean(ratio_h, dims = 2))
    sdh = vec(std(ratio_h, dims = 2))

    fig = Figure(size = (900, 520))
    ax = Axis(fig[1, 1],
        title = "Relative importance of high-centrality perturbations",
        xlabel = "frequency ω",
        ylabel = "high / low sensitivity",
        xscale = log10,
        yscale = log10)

    band!(ax, ωs, max.(μu .- sdu, 1e-12), μu .+ sdu, alpha = 0.18)
    lines!(ax, ωs, μu, linewidth = 3, label = "uniform")

    band!(ax, ωs, max.(μh .- sdh, 1e-12), μh .+ sdh, alpha = 0.18)
    lines!(ax, ωs, μh, linewidth = 3, label = "heavy-tail")

    lines!(ax, ωs, ones(length(ωs)), linestyle = :dash, linewidth = 2, label = "equal")
    axislegend(ax, position = :rb)

    save(joinpath(outdir, "ratio_high_to_low.png"), fig)
    return fig
end

function save_rank_scatter(example_result::NamedTuple; outdir = "results", prefix = "example")
    mkpath(outdir)

    c = example_result.centrality
    H = example_result.species_peak
    L = example_result.species_low
    U = example_result.species_high

    fig = Figure(size = (1350, 430))

    ax1 = Axis(fig[1, 1],
        xlabel = "weighted degree centrality",
        ylabel = "species peak sensitivity",
        title = @sprintf("peak: ρ = %.2f", example_result.rank_stats.rho_peak))
    scatter!(ax1, c, H, markersize = 9)

    ax2 = Axis(fig[1, 2],
        xlabel = "weighted degree centrality",
        ylabel = "low-frequency integrated sensitivity",
        title = @sprintf("low band: ρ = %.2f", example_result.rank_stats.rho_low))
    scatter!(ax2, c, L, markersize = 9)

    ax3 = Axis(fig[1, 3],
        xlabel = "weighted degree centrality",
        ylabel = "high-frequency integrated sensitivity",
        title = @sprintf("high band: ρ = %.2f", example_result.rank_stats.rho_high))
    scatter!(ax3, c, U, markersize = 9)

    save(joinpath(outdir, "$(prefix)_rank_scatter.png"), fig)
    return fig
end

function save_example_spectrum(example_result::NamedTuple; outdir = "results", prefix = "example")
    mkpath(outdir)
    ωs = example_result.ωs

    fig = Figure(size = (900, 520))
    ax = Axis(fig[1, 1],
        title = "Single-network spectra",
        xlabel = "frequency ω",
        ylabel = "sensitivity",
        xscale = log10,
        yscale = log10)

    lines!(ax, ωs, example_result.intrinsic, linewidth = 3, label = "intrinsic S(ω)")
    lines!(ax, ωs, example_result.high_spectrum, linewidth = 3, label = "high-centrality Sₚ(ω)")
    lines!(ax, ωs, example_result.low_spectrum, linewidth = 3, label = "low-centrality Sₚ(ω)")
    axislegend(ax, position = :rb)

    save(joinpath(outdir, "$(prefix)_spectra.png"), fig)
    return fig
end

function save_degree_histograms(example_uniform::NamedTuple, example_heavy::NamedTuple; outdir = "results")
    mkpath(outdir)
    cu = unweighted_degree(example_uniform.A)
    ch = unweighted_degree(example_heavy.A)

    fig = Figure(size = (1100, 420))

    ax1 = Axis(fig[1, 1],
        xlabel = "unweighted degree",
        ylabel = "count",
        title = "Uniform network degree distribution")
    hist!(ax1, cu, bins = 15)

    ax2 = Axis(fig[1, 2],
        xlabel = "unweighted degree",
        ylabel = "count",
        title = "Heavy-tailed network degree distribution")
    hist!(ax2, ch, bins = 15)

    save(joinpath(outdir, "degree_histograms.png"), fig)
    return fig
end

# -------------------------
# Printed summaries
# -------------------------
function summarize_ensemble(results::Vector{<:NamedTuple}, label::String)
    peak_high = extract_summaries(results, :summary_high, :peak)
    peak_low = extract_summaries(results, :summary_low, :peak)
    low_high = extract_summaries(results, :summary_high, :low_int)
    low_low = extract_summaries(results, :summary_low, :low_int)
    high_high = extract_summaries(results, :summary_high, :high_int)
    high_low = extract_summaries(results, :summary_low, :high_int)

    ρH = extract_rankstat(results, :rho_peak)
    ρL = extract_rankstat(results, :rho_low)
    ρU = extract_rankstat(results, :rho_high)

    println("\n==================== $(label) ====================")
    println(@sprintf("Mean peak sensitivity, high-centrality class: %.4f ± %.4f", mean(peak_high), std(peak_high)))
    println(@sprintf("Mean peak sensitivity, low-centrality class : %.4f ± %.4f", mean(peak_low), std(peak_low)))
    println(@sprintf("Mean peak ratio high/low                   : %.4f", mean(peak_high ./ peak_low)))
    println(@sprintf("Mean low-band ratio high/low               : %.4f", mean(low_high ./ low_low)))
    println(@sprintf("Mean high-band ratio high/low              : %.4f", mean(high_high ./ high_low)))
    println(@sprintf("Spearman centrality vs species peak        : %.4f ± %.4f", mean(ρH), std(ρH)))
    println(@sprintf("Spearman centrality vs species low-band    : %.4f ± %.4f", mean(ρL), std(ρL)))
    println(@sprintf("Spearman centrality vs species high-band   : %.4f ± %.4f", mean(ρU), std(ρU)))
end

# -------------------------
# Main script
# -------------------------
function main()
    # -------------------------
    # User-facing parameters
    # -------------------------
    S = 100
    nrep = 25
    connectance = 0.12
    IS = 0.22
    ts_sigma_log = 0.15
    ωmin = 1e-2
    ωmax = 1e2
    nω = 220
    outdir = "JacobianApproach/SecondPartOfTheArticle/results"

    println("Running structural sensitivity analysis...")
    println(@sprintf("S = %d, nrep = %d, connectance = %.3f, IS = %.3f, timescale log-sd = %.3f",
        S, nrep, connectance, IS, ts_sigma_log))

    results_uniform = build_network_ensemble(
        S, :uniform, nrep;
        connectance = connectance,
        IS = IS,
        ts_sigma_log = ts_sigma_log,
        ωmin = ωmin,
        ωmax = ωmax,
        nω = nω
    )

    results_heavy = build_network_ensemble(
        S, :heavytail, nrep;
        connectance = connectance,
        IS = IS,
        ts_sigma_log = ts_sigma_log,
        ωmin = ωmin,
        ωmax = ωmax,
        nω = nω
    )

    summarize_ensemble(results_uniform, "UNIFORM")
    summarize_ensemble(results_heavy, "HEAVY-TAILED")

    ex_uniform = results_uniform[1]
    ex_heavy = results_heavy[1]

    save_degree_histograms(ex_uniform, ex_heavy; outdir = outdir)
    save_mean_spectra(results_uniform, results_heavy; outdir = outdir)
    save_ratio_spectra(results_uniform, results_heavy; outdir = outdir)
    save_example_spectrum(ex_uniform; outdir = outdir, prefix = "uniform_example")
    save_example_spectrum(ex_heavy; outdir = outdir, prefix = "heavytail_example")
    save_rank_scatter(ex_uniform; outdir = outdir, prefix = "uniform_example")
    save_rank_scatter(ex_heavy; outdir = outdir, prefix = "heavytail_example")

    println("\nSaved plots to: $(abspath(outdir))")
    println("Files:")
    println("  - degree_histograms.png")
    println("  - mean_spectra.png")
    println("  - ratio_high_to_low.png")
    println("  - uniform_example_spectra.png")
    println("  - heavytail_example_spectra.png")
    println("  - uniform_example_rank_scatter.png")
    println("  - heavytail_example_rank_scatter.png")

    return (
        results_uniform = results_uniform,
        results_heavy = results_heavy,
        example_uniform = ex_uniform,
        example_heavy = ex_heavy
    )
end

main()
