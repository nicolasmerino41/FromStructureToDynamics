begin
using LinearAlgebra
using CairoMakie
using Statistics
using Printf

# ============================================================
# FINAL FIGURE 2 (revised)
# ------------------------------------------------------------
# - Panel B removed as separate profile panel
# - Valley/peak markers kept in Panel A
# - Frequency content shifted left uniformly
# - Top-right empty space used for a schematic network with
#   two highlighted modified regions matching class colors
# ============================================================

# -------------------------
# Global frequency shift
# -------------------------
# Uniform rescaling: preserves qualitative shape, moves everything left
const FREQ_SHIFT = 0.62

# -------------------------
# Frequency grid
# -------------------------
const OMEGAS = FREQ_SHIFT .* exp.(range(log(0.08), log(2.5), length = 1800))

# -------------------------
# Time-domain simulation parameters
# -------------------------
# Slightly longer horizon so the slower forcing still shows clearly
const DT = 0.03
const TMAX = 78.0 / FREQ_SHIFT
const FORCE_START = 0.12 * TMAX
const FORCE_AMPLITUDE = 0.18
const EPS_P = 0.1

# ============================================================
# Helpers
# ============================================================
function resolvent(A::AbstractMatrix{<:Real}, ω::Real)
    n = size(A, 1)
    Icomplex = Matrix{ComplexF64}(I, n, n)
    Ac = ComplexF64.(A)
    F = factorize(im * ω .* Icomplex - Ac)
    return F \ Icomplex
end

function intrinsic_profile(A::AbstractMatrix{<:Real}, ωs::AbstractVector{<:Real})
    vals = zeros(Float64, length(ωs))
    for (k, ω) in pairs(ωs)
        vals[k] = opnorm(resolvent(A, ω), 2)
    end
    return vals
end

function rpr_profile(A::AbstractMatrix{<:Real}, P::AbstractMatrix{<:Real}, ωs::AbstractVector{<:Real})
    vals = zeros(Float64, length(ωs))
    for (k, ω) in pairs(ωs)
        R = resolvent(A, ω)
        vals[k] = opnorm(R * P * R, 2)
    end
    return vals
end

function choose_peak_in_band(S::AbstractVector, ωs::AbstractVector, band::Tuple{Float64,Float64})
    idxs = findall((ωs .>= band[1]) .& (ωs .<= band[2]))
    isempty(idxs) && error("No frequencies in band $(band)")
    return idxs[argmax(S[idxs])]
end

function choose_valley_in_band(S::AbstractVector, ωs::AbstractVector, band::Tuple{Float64,Float64})
    idxs = findall((ωs .>= band[1]) .& (ωs .<= band[2]))
    isempty(idxs) && error("No frequencies in band $(band)")
    return idxs[argmin(S[idxs])]
end

# dx/dt = A*x + u(t)*b
function simulate_forced_system(
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    ω::Float64;
    dt::Float64 = DT,
    tmax::Float64 = TMAX,
    forcing_amplitude::Float64 = FORCE_AMPLITUDE,
    forcing_start::Float64 = FORCE_START
)
    ts = collect(0:dt:tmax)

    n = size(A, 1)
    X = zeros(Float64, n, length(ts))

    forcing_scalar(t) = t < forcing_start ? 0.0 :
                        forcing_amplitude * sin(ω * (t - forcing_start))

    forcing_at_time(t) = forcing_scalar(t) .* b
    f(x, t) = A * x + forcing_at_time(t)

    x = zeros(Float64, n)
    for k in 1:length(ts)-1
        t = ts[k]
        k1 = f(x, t)
        k2 = f(x .+ 0.5dt .* k1, t + 0.5dt)
        k3 = f(x .+ 0.5dt .* k2, t + 0.5dt)
        k4 = f(x .+ dt .* k3, t + dt)
        x = x .+ (dt / 6.0) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        X[:, k+1] .= x
    end

    forcing_signal = [forcing_scalar(t) for t in ts]
    return ts, X, forcing_signal
end

function profile_ylim(Ss::AbstractVector...)
    ymin = minimum(minimum(S) for S in Ss)
    ymax = maximum(maximum(S) for S in Ss)
    ymin = max(ymin, 1e-4)
    ymax = max(ymax, 10ymin)
    return (0.95ymin, 1.08ymax)
end

function community_ylim(ys::AbstractVector...; q = 0.985)
    vals = abs.(vcat(ys...))
    m = quantile(vals, q)
    m = max(m, 0.05)
    return (-1.12m, 1.12m)
end

# ============================================================
# Model for Fig. 2
# ------------------------------------------------------------
# Uniform scaling by FREQ_SHIFT moves all profiles left without
# changing their qualitative geometry.
# ============================================================
A_raw = [
    -0.22  -0.55   0.10   0.00
     0.55  -0.22   0.00   0.10
     0.10   0.00  -0.16  -1.50
     0.00   0.10   1.50  -0.16
]

A = FREQ_SHIFT .* A_raw
const TIME_SCALE = 0.65
A = TIME_SCALE .* A
A_mod = A + EPS_P .* P_focus
# Structural class A: change within slow block
P_A = [
    0.0  1.0  0.0  0.0
    1.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0
]

# Structural class B: change within fast block
P_B = [
    0.0   0.0   0.0   0.0
    0.0   0.0   0.0   0.0
    0.0   0.0   0.35  1.10
    0.0   0.0   0.55 -0.25
]

const EPS_A = TIME_SCALE * 0.25
const EPS_B = TIME_SCALE * 0.25
A_A = A + EPS_A .* P_A
A_B = A + EPS_B .* P_B

println("Eigenvalues of A:")
println(eigvals(A))
println("\nEigenvalues of A_mod:")
println(eigvals(A_mod))

# ============================================================
# Frequency-domain analysis
# ============================================================
S_intr = intrinsic_profile(A, OMEGAS)
S_A = rpr_profile(A, P_A, OMEGAS)
S_B = rpr_profile(A, P_B, OMEGAS)

# Shift the selection bands consistently
idx_valley = choose_valley_in_band(S_B, OMEGAS, FREQ_SHIFT .* (0.28, 0.48))
idx_peak   = choose_peak_in_band(S_B, OMEGAS, FREQ_SHIFT .* (0.95, 1.35))
# idx_peak   = 1700

ω_valley = OMEGAS[idx_valley]
ω_peak   = OMEGAS[idx_peak]

println()
println(@sprintf("Chosen valley frequency: ω = %.4f | S_B = %.4f", ω_valley, S_B[idx_valley]))
println(@sprintf("Chosen peak frequency:   ω = %.4f | S_B = %.4f", ω_peak,   S_B[idx_peak]))
println(@sprintf("Peak/valley ratio:       %.2f", S_B[idx_peak] / S_B[idx_valley]))

# ============================================================
# Time-domain demonstration
# ============================================================
b = [1.0, -0.65, 0.55, -0.25]
b ./= norm(b)

c = [1.0, 0.8, 1.1, 0.9]

# Valley frequency
ts, X_valley_base, _ = simulate_forced_system(A,   b, ω_valley)
_,  X_valley_A,    _ = simulate_forced_system(A_A, b, ω_valley)
_,  X_valley_B,    _ = simulate_forced_system(A_B, b, ω_valley)

# Peak frequency
_,  X_peak_base, _ = simulate_forced_system(A,   b, ω_peak)
_,  X_peak_A,    _ = simulate_forced_system(A_A, b, ω_peak)
_,  X_peak_B,    _ = simulate_forced_system(A_B, b, ω_peak)

# Aggregate community signals
Y_valley_base = vec(c' * X_valley_base)
Y_valley_A    = vec(c' * X_valley_A)
Y_valley_B    = vec(c' * X_valley_B)

Y_peak_base   = vec(c' * X_peak_base)
Y_peak_A      = vec(c' * X_peak_A)
Y_peak_B      = vec(c' * X_peak_B)

# Full-state differences relative to baseline
Δ_valley_A = [norm(X_valley_A[:, i] - X_valley_base[:, i]) for i in eachindex(ts)]
Δ_valley_B = [norm(X_valley_B[:, i] - X_valley_base[:, i]) for i in eachindex(ts)]

Δ_peak_A = [norm(X_peak_A[:, i] - X_peak_base[:, i]) for i in eachindex(ts)]
Δ_peak_B = [norm(X_peak_B[:, i] - X_peak_base[:, i]) for i in eachindex(ts)]

println(@sprintf("Max ||Δx|| at valley: %.4f", maximum(Δ_valley)))
println(@sprintf("Max ||Δx|| at peak:   %.4f", maximum(Δ_peak)))
println(@sprintf("Peak/valley ||Δx|| ratio: %.2f", maximum(Δ_peak) / maximum(Δ_valley)))

# Shared visual limits
y_prof = profile_ylim(S_intr, S_A, S_B)
y_comm = community_ylim(
    Y_valley_base, Y_valley_A, Y_valley_B, Y_peak_base, Y_peak_A, Y_peak_B
)

y_delta = (
    0.0,
    1.05 * maximum(vcat(Δ_valley_A, Δ_valley_B, Δ_peak_A, Δ_peak_B))
)

# ============================================================
# Network schematic data
# ============================================================
const COL_BG   = RGBf(0.985, 0.985, 0.985)
const COL_EDGE = RGBf(0.67, 0.67, 0.67)
const COL_NODE = RGBf(0.18, 0.18, 0.18)
const COL_A    = Makie.to_color(:darkorange)
const COL_B    = Makie.to_color(:teal)

pts = [
    (0.08, 0.63),  # 1
    (0.18, 0.82),  # 2
    (0.29, 0.60),  # 3
    (0.22, 0.34),  # 4
    (0.42, 0.80),  # 5
    (0.50, 0.54),  # 6
    (0.46, 0.22),  # 7
    (0.70, 0.74),  # 8
    (0.83, 0.54),  # 9
    (0.73, 0.28),  # 10
    (0.98, 0.73),  # 11
    (1.03, 0.38)   # 12
]

base_edges = [
    (1,2), (1,3), (1,4),
    (2,3), (2,5),
    (3,4), (3,6), (3,5),
    (4,7),
    (5,6), (5,8),
    (6,7), (6,8), (6,9), (6,10),
    (7,10),
    (8,9), (8,11),
    (9,10), (9,11), (9,12),
    (10,12),
    (11,12)
]

# highlighted region A (slow-block analogue)
mod_edges_A = [(1,2), (2,3), (1,3), (3,4)]

# highlighted region B (fast-block analogue)
mod_edges_B = [(8,9), (9,11), (9,12), (11,12)]

regionA_nodes = [1,2,3,4]
regionB_nodes = [8,9,11,12]

# ============================================================
# Plot
# ============================================================
begin
    fig = Figure(size = (1550, 980))

    Label(
        fig[0, 1:2],
        "How to read structural sensitivity ecologically",
        fontsize = 28
    )

    # --------------------------------------------------------
    # Panel A
    # --------------------------------------------------------
    axA = Axis(
        fig[1, 2],
        title = "A. Different structural changes matter in different frequency bands",
        xlabel = "frequency ω",
        ylabel = "sensitivity",
        xscale = log10,
        yscale = log10
    )
    ylims!(axA, y_prof...)

    # Wider left-side breathing room
    xlims!(axA, minimum(OMEGAS), maximum(OMEGAS))

    # subtle band shading
    vspan!(axA, minimum(OMEGAS), FREQ_SHIFT * 0.55, color = (:gray, 0.06))
    vspan!(axA, FREQ_SHIFT * 0.55, FREQ_SHIFT * 1.35, color = (:gray, 0.10))
    vspan!(axA, FREQ_SHIFT * 1.35, maximum(OMEGAS), color = (:gray, 0.06))

    lines!(axA, OMEGAS, S_intr, color = :black, linewidth = 4, label = "intrinsic sensitivity")
    lines!(axA, OMEGAS, S_A, color = :darkorange, linewidth = 4, label = "structural class A")
    lines!(axA, OMEGAS, S_B, color = :teal, linewidth = 4, label = "structural class B")
    
    function boxed_text_log!(ax, x, y, s;
        textcolor = :black,
        boxcolor = :white,
        boxalpha = 0.95,
        fontsize = 18,
        xfac = 1.22,
        yfac = 1.18
    )
        poly!(ax,
            [Point2f(x / xfac, y / yfac),
            Point2f(x * xfac, y / yfac),
            Point2f(x * xfac, y * yfac),
            Point2f(x / xfac, y * yfac)],
            color = (boxcolor, boxalpha),
            strokecolor = :gray70,
            strokewidth = 1.5
        )

        text!(ax, x, y,
            text = s,
            color = textcolor,
            fontsize = fontsize,
            align = (:center, :center),
            justification = :center
        )
    end
    # --------------------------------------------------------
    # Automatic arrows from fixed label positions
    # Arrow targets = intersection of each curve with its own
    # vertical marker line (ω_valley for A, ω_peak for B)
    # --------------------------------------------------------
    
    # Fixed label positions (exactly yours)
    xA_lab = FREQ_SHIFT * 0.14
    yA_lab = 0.51 * maximum(S_A)

    xB_lab = FREQ_SHIFT * 0.58
    yB_lab = 0.52 * maximum(S_B)

    # Target x-positions = existing vertical lines
    xA_target = ω_valley
    xB_target = ω_peak

    # Nearest sampled indices on frequency grid
    idxA = argmin(abs.(OMEGAS .- xA_target))
    idxB = argmin(abs.(OMEGAS .- xB_target))

    # y-values where vertical lines intersect the corresponding curves
    yA_target = S_A[idxA]
    yB_target = S_B[idxB]

    # Arrows
    arrows!(axA,
        [Point2f(xA_lab, yA_lab)],
        [Vec2f(xA_target - xA_lab, yA_target - yA_lab)],
        color = :darkorange,
        linewidth = 2.4,
        arrowsize = 17
    )

    arrows!(axA,
        [Point2f(xB_lab, yB_lab)],
        [Vec2f(xB_target - xB_lab, yB_target - yB_lab)],
        color = :teal,
        linewidth = 2.4,
        arrowsize = 17
    )

    # Labels at your exact positions
    boxed_text_log!(axA, xA_lab, yA_lab,
        "A matters more here",
        textcolor = :darkorange,
        boxcolor = :white,
        fontsize = 18,
        xfac = 1.65
    )

    boxed_text_log!(axA, xB_lab, yB_lab,
        "B matters more here",
        textcolor = :teal,
        boxcolor = :white,
        fontsize = 18,
        xfac = 1.65
    )
    # valley/peak markers kept in Panel A
    vlines!(axA, [ω_valley], color = :firebrick, linestyle = :dash, linewidth = 3)
    vlines!(axA, [ω_peak],   color = :navy, linestyle = :dash, linewidth = 3)

    # scatter!(axA, [ω_valley], [S_B[idx_valley]], color = :firebrick, markersize = 15)
    # scatter!(axA, [ω_peak],   [S_B[idx_peak]],   color = :navy, markersize = 15)

    axislegend(axA, position = :lb)

    # --------------------------------------------------------
    # Top-right network schematic (uses former Panel B space)
    # --------------------------------------------------------
    ax_net = Axis(
        fig[1, 1],
        title = "Structural changes represented as localized modified regions",
        xticksvisible = false,
        yticksvisible = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        leftspinevisible = false,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = false,
        xgridvisible = false,
        ygridvisible = false,
        backgroundcolor = COL_BG
    )

    xlims!(ax_net, 0.00, 1.1)
    ylims!(ax_net, 0.0, 0.96)
    Δy_net = -0.08
    pts_net = [(x, y + Δy_net) for (x, y) in pts]
    scatter!(ax_net, [0.21], [0.60 + Δy_net],
        marker = :circle,
        markersize = 200,
        color = (COL_A, 0.13),
        strokecolor = (COL_A, 0.65),
        strokewidth = 2.0
    )

    scatter!(ax_net, [0.90], [0.58 + Δy_net],
        marker = :circle,
        markersize = 200,
        color = (COL_B, 0.13),
        strokecolor = (COL_B, 0.65),
        strokewidth = 2.0
    )

    # base network
    for (i, j) in base_edges
        p1, p2 = pts_net[i], pts_net[j]
        lines!(ax_net, [p1[1], p2[1]], [p1[2], p2[2]], color = COL_EDGE, linewidth = 2.4)
    end

    # highlighted modified edges for region A
    for (i, j) in mod_edges_A
        p1, p2 = pts_net[i], pts_net[j]
        lines!(ax_net, [p1[1], p2[1]], [p1[2], p2[2]], color = COL_A, linewidth = 5.0)
    end

    # highlighted modified edges for region B
    for (i, j) in mod_edges_B
        p1, p2 = pts_net[i], pts_net[j]
        lines!(ax_net, [p1[1], p2[1]], [p1[2], p2[2]], color = COL_B, linewidth = 5.0)
    end

    xs = first.(pts_net)
    ys_pts = last.(pts_net)

    scatter!(ax_net, xs, ys_pts, color = COL_NODE, markersize = 24)

    scatter!(ax_net, xs[regionA_nodes], ys_pts[regionA_nodes],
        color = COL_A, markersize = 28)

    scatter!(ax_net, xs[regionB_nodes], ys_pts[regionB_nodes],
        color = COL_B, markersize = 28)

    text!(ax_net, 0.18, 0.91 + Δy_net, text = "modified region A", color = COL_A, fontsize = 18)
    text!(ax_net, 0.76, 0.91 + Δy_net, text = "modified region B", color = COL_B, fontsize = 18)

    # --------------------------------------------------------
    # Panel C
    # --------------------------------------------------------
    axC = Axis(
        fig[2, 1],
        title = @sprintf("C. Low-impact regime (ω = %.3f): responses under two structural modifications", ω_valley),
        xlabel = "time",
        ylabel = "community signal"
    )
    xlims!(axC, first(ts), last(ts))
    ylims!(axC, y_comm...)
    vlines!(axC, [FORCE_START], color = :gray35, linestyle = :dash, linewidth = 2)

    lines!(axC, ts, Y_valley_base, color = :black, linewidth = 3.0, label = "baseline")
    lines!(axC, ts, Y_valley_A, color = :darkorange, linewidth = 2.8, linestyle = :dot, label = "structural class A")
    lines!(axC, ts, Y_valley_B, color = :teal, linewidth = 2.8, linestyle = :dot, label = "structural class B")

    axC_r = Axis(
        fig[2, 1],
        yaxisposition = :right,
        ylabel = "‖Δx(t)‖"
    )
    hidespines!(axC_r)
    hidexdecorations!(axC_r)
    ylims!(axC_r, y_delta...)
    # lines!(axC_r, ts, Δ_valley_A, color = (:darkorange, 0.65), linewidth = 3)
    # lines!(axC_r, ts, Δ_valley_B, color = (:teal, 0.65), linewidth = 3)

    axislegend(axC, position = :rt)

    # --------------------------------------------------------
    # Panel D
    # --------------------------------------------------------
    axD = Axis(
        fig[2, 2],
        title = @sprintf("D. High-impact regime (ω = %.3f): responses under two structural modifications", ω_peak),
        xlabel = "time",
        ylabel = "community signal"
    )
    xlims!(axD, first(ts), last(ts))
    ylims!(axD, y_comm...)
    vlines!(axD, [FORCE_START], color = :gray35, linestyle = :dash, linewidth = 2)

    lines!(axD, ts, Y_peak_base, color = :black, linewidth = 3.0, label = "baseline")
    lines!(axD, ts, Y_peak_A, color = :darkorange, linewidth = 2.8, linestyle = :dot, label = "structural class A")
    lines!(axD, ts, Y_peak_B, color = :teal, linewidth = 2.8, linestyle = :dot, label = "structural class B")

    axD_r = Axis(
        fig[2, 2],
        yaxisposition = :right,
        ylabel = "‖Δx(t)‖"
    )
    hidespines!(axD_r)
    hidexdecorations!(axD_r)
    ylims!(axD_r, y_delta...)
    # lines!(axD_r, ts, Δ_peak_A, color = (:darkorange, 0.65), linewidth = 3)
    # lines!(axD_r, ts, Δ_peak_B, color = (:teal, 0.65), linewidth = 3)

    axislegend(axD, position = :rt)

    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 26)

    display(fig)
end
end