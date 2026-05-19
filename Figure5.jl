using LinearAlgebra
using Statistics
using CairoMakie
using Printf

"""
Figure 5: timescale alignment and frequency-dependent sensitivity.

This script builds a six-species illustrative interaction matrix made of three
coupled oscillatory modules. It compares three diagonal timescale choices:
muted, homogeneous, and amplified. The amplified case aligns the modules'
effective frequencies, producing a larger resolvent peak; the muted case dampens sensitivity. 
The figure shows the interaction matrix, the imposed
timescales, and the resulting sensitivity profiles.
"""

const COL_ORANGE = "#D95F02"
const COL_GREEN  = "#1B9E77"
const COL_BLUE   = "#2C7FB8"
const COL_GREY   = "#969696"

function illustrative_matrix()
    Ω = [0.35, 1.20, 4.00]
    g = 2.5
    A = zeros(Float64, 6, 6)

    for b in 1:3
        i = 2b - 1
        ω = Ω[b]

        A[i, i] = -1.0
        A[i + 1, i + 1] = -1.0
        A[i, i + 1] = -ω
        A[i + 1, i] = ω
    end

    A[3, 1] += g
    A[4, 2] += g
    A[3, 2] += 0.4g
    A[4, 1] -= 0.4g

    A[5, 3] += g
    A[6, 4] += g
    A[5, 4] += 0.4g
    A[6, 3] -= 0.4g

    return A, Ω
end

resolvent(A, T, ω) = inv(im * ω * T - A)
response_sensitivity(A, T, ω) = opnorm(resolvent(A, T, ω), 2)
sensitivity_profile(A, T, ωs) = [response_sensitivity(A, T, ω) for ω in ωs]

function log_frequency_area(ωs, profile)
    x = log.(ωs)
    sum(0.5 * (profile[k] + profile[k + 1]) * (x[k + 1] - x[k]) for k in 1:length(x)-1)
end

function smooth_log_profile(y; window = 21)
    z = log.(y .+ eps(Float64))
    out = similar(z)
    half = div(window, 2)

    for i in eachindex(z)
        lo = max(firstindex(z), i - half)
        hi = min(lastindex(z), i + half)
        out[i] = mean(z[lo:hi])
    end

    return exp.(out)
end

function timescale_configurations(Ω)
    target = 1.20
    τ_hom = ones(6)
    τ_amp = repeat(Ω ./ target, inner = 2)
    τ_mut = repeat([4.0, 1.0, 0.25], inner = 2)

    return τ_mut, τ_hom, τ_amp
end

function reorder_system(A, τ_mut, τ_hom, τ_amp, perm)
    return A[perm, perm], τ_mut[perm], τ_hom[perm], τ_amp[perm]
end

function print_summary(τ_mut, τ_hom, τ_amp, area_ratios, peak_ratios, low_frequency_values)
    println("Timescale configurations:")
    println("muted       τ = ", round.(τ_mut, digits = 3))
    println("homogeneous τ = ", round.(τ_hom, digits = 3))
    println("amplified   τ = ", round.(τ_amp, digits = 3))

    println("\nArea ratios over ω ≥ 10^-2:")
    @printf("muted / homogeneous     = %.3f\n", area_ratios.muted)
    @printf("amplified / homogeneous = %.3f\n", area_ratios.amplified)

    println("\nPeak ratios over ω ≥ 10^-2:")
    @printf("muted / homogeneous     = %.3f\n", peak_ratios.muted)
    @printf("amplified / homogeneous = %.3f\n", peak_ratios.amplified)

    println("\nLow-frequency convergence:")
    @printf("R(0)                  = %.4f\n", low_frequency_values.static)
    @printf("muted at min ω        = %.4f\n", low_frequency_values.muted)
    @printf("homogeneous at min ω  = %.4f\n", low_frequency_values.homogeneous)
    @printf("amplified at min ω    = %.4f\n", low_frequency_values.amplified)
end

function build_figure(A, τ_mut, τ_hom, τ_amp, ωs, profiles, static_sensitivity)
    fig = Figure(
        size = (1200, 500),
        backgroundcolor = :white,
        figure_padding = (16, 18, 34, 12)
    )

    qA = GridLayout()
    qB = GridLayout()
    qC = GridLayout()

    fig[1, 1] = qA
    fig[1, 2] = qB
    fig[1, 3] = qC

    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 20)

    species_labels = string.(1:6)
    axA = Axis(
        qA[1, 1],
        aspect = DataAspect(),
        backgroundcolor = :transparent,
        xgridvisible = false,
        ygridvisible = false,
        xticks = (1:6, species_labels),
        yticks = (1:6, species_labels),
        xticklabelrotation = π / 4,
        xticklabelsize = 10,
        yticklabelsize = 10
    )

    maxabsA = maximum(abs.(A))
    hmA = heatmap!(
        axA,
        1:6,
        1:6,
        A',
        colormap = :balance,
        colorrange = (-maxabsA, maxabsA)
    )

    Colorbar(
        qA[1, 2],
        hmA,
        width = 15,
        height = Relative(0.6),
        valign = :center,
        label = "Interaction strength",
        labelsize = 13,
        ticklabelsize = 11
    )

    xlims!(axA, 0.5, 6.5)
    ylims!(axA, 6.5, 0.5)
    hidespines!(axA, :t, :r)
    colsize!(qA, 1, Relative(0.88))
    colsize!(qA, 2, Fixed(38))
    colgap!(qA, 8)

    axBars = Axis(
        qB[1, 1],
        xlabel = "log τᵢ",
        backgroundcolor = :transparent,
        xgridvisible = false,
        ygridvisible = false,
        yticks = ([1, 2, 3], ["muted", "homogeneous", "amplified"]),
        xlabelsize = 15,
        xticklabelsize = 12,
        yticklabelsize = 13
    )

    offsets = range(-0.35, 0.35, length = 6)
    timescale_sets = (
        (τ = τ_mut, y = 1, color = COL_GREEN),
        (τ = τ_hom, y = 2, color = COL_BLUE),
        (τ = τ_amp, y = 3, color = COL_ORANGE)
    )

    for config in timescale_sets, k in 1:6
        y = config.y + offsets[k]
        x = log(config.τ[k])

        lines!(axBars, [0, x], [y, y], color = config.color, linewidth = 9)
        scatter!(axBars, [x], [y], color = config.color, markersize = 10)
    end

    vlines!(axBars, [0], color = COL_GREY, linewidth = 1.3, linestyle = :dash)
    xlims!(axBars, -1.8, 1.8)
    ylims!(axBars, 0.45, 3.55)
    hidespines!(axBars, :t, :r)

    axProf = Axis(
        qC[1, 1],
        xlabel = "Frequency ω",
        ylabel = "Sensitivity ‖R(ω)‖₂",
        xscale = log10,
        backgroundcolor = :transparent,
        xgridvisible = false,
        ygridvisible = false,
        xminorticksvisible = false,
        yminorticksvisible = false,
        xlabelsize = 15,
        ylabelsize = 15,
        xticklabelsize = 12,
        yticklabelsize = 12
    )

    lines!(axProf, ωs, profiles.amplified, linewidth = 4, color = COL_ORANGE, label = "amplified")
    lines!(axProf, ωs, profiles.homogeneous, linewidth = 4, color = COL_BLUE, label = "homogeneous")
    lines!(axProf, ωs, profiles.muted, linewidth = 4, color = COL_GREEN, label = "muted")

    hlines!(axProf, [static_sensitivity], color = COL_GREY, linewidth = 2, linestyle = :dash)
    text!(
        axProf,
        minimum(ωs) * 1.15,
        static_sensitivity * 0.985,
        text = "shared ω → 0 limit",
        color = COL_GREY,
        fontsize = 12,
        align = (:left, :top)
    )

    axislegend(axProf, position = :lb, framevisible = false, labelsize = 12)
    hidespines!(axProf, :t, :r)

    return fig
end

A, Ω = illustrative_matrix()
τ_mut, τ_hom, τ_amp = timescale_configurations(Ω)
A, τ_mut, τ_hom, τ_amp = reorder_system(A, τ_mut, τ_hom, τ_amp, [1, 3, 2, 4, 5, 6])

T_mut = Diagonal(τ_mut)
T_hom = Diagonal(τ_hom)
T_amp = Diagonal(τ_amp)

ωs = 10 .^ range(-2.5, 1.5, length = 150)
visible_mask = ωs .>= 1e-2

profile_muted = sensitivity_profile(A, T_mut, ωs)
profile_hom = sensitivity_profile(A, T_hom, ωs)
profile_amp = sensitivity_profile(A, T_amp, ωs)

profile_muted_s = smooth_log_profile(profile_muted; window = 17)
profile_hom_s = smooth_log_profile(profile_hom; window = 17)
profile_amp_s = smooth_log_profile(profile_amp; window = 17)

static_sensitivity = response_sensitivity(A, T_hom, 0.0)
area_muted = log_frequency_area(ωs[visible_mask], profile_muted[visible_mask])
area_hom = log_frequency_area(ωs[visible_mask], profile_hom[visible_mask])
area_amp = log_frequency_area(ωs[visible_mask], profile_amp[visible_mask])
peak_muted = maximum(profile_muted[visible_mask])
peak_hom = maximum(profile_hom[visible_mask])
peak_amp = maximum(profile_amp[visible_mask])

print_summary(
    τ_mut,
    τ_hom,
    τ_amp,
    (muted = area_muted / area_hom, amplified = area_amp / area_hom),
    (muted = peak_muted / peak_hom, amplified = peak_amp / peak_hom),
    (
        static = static_sensitivity,
        muted = profile_muted[1],
        homogeneous = profile_hom[1],
        amplified = profile_amp[1]
    )
)

begin
    fig = build_figure(
        A,
        τ_mut,
        τ_hom,
        τ_amp,
        ωs,
        (muted = profile_muted_s, homogeneous = profile_hom_s, amplified = profile_amp_s),
        static_sensitivity
    )

    save("Figure5.png", fig, px_per_unit = 3)
    display(fig)
end
