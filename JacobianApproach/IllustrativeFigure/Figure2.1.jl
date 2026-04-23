begin
using LinearAlgebra
using CairoMakie
using Statistics
using Printf

# ============================================================
# FINAL FIGURE 2
# ------------------------------------------------------------
# How to read structural sensitivity ecologically
#
# Panel A: different structural changes matter in different bands
# Panel B: one structural change, with a valley and a peak selected
# Panel C: valley forcing -> small time-domain impact
# Panel D: peak forcing   -> large time-domain impact
#
# Clean, explanatory, low-clutter figure.
# ============================================================
# -------------------------
# Frequency grid
# -------------------------   
const OMEGAS = exp.(range(log(0.08), log(2.5), length = 1800))

# -------------------------
# Time-domain simulation parameters
# -------------------------
const DT = 0.03
const TMAX = 78.0
const FORCE_START = 0.12 * TMAX
const FORCE_AMPLITUDE = 0.18
const EPS_P = 0.45

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

    x = zeros(Float64, n)  # starts at equilibrium
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

function community_ylim(y1::AbstractVector, y2::AbstractVector, y3::AbstractVector, y4::AbstractVector; q = 0.985)
    vals = abs.(vcat(y1, y2, y3, y4))
    m = quantile(vals, q)
    m = max(m, 0.05)
    return (-1.12m, 1.12m)
end

# ============================================================
# Model for Fig. 2
# ------------------------------------------------------------
# 4-state system: two coupled oscillatory blocks
# - slow block around ~0.55
# - faster block around ~1.20
#
# This gives a clean intrinsic profile and two clearly distinct
# structural sensitivity classes.
# ============================================================
A = [
    -0.22  -0.55   0.10   0.00
     0.55  -0.22   0.00   0.10
     0.10   0.00  -0.18  -1.20
     0.00   0.10   1.20  -0.18
]

# Structural class A: change within slow block
P_A = [
    0.0  1.0  0.0  0.0
    1.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0
]

# Structural class B: change within fast block
P_B = [
    0.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.0
    0.0  0.0  0.0  1.0
    0.0  0.0  1.0  0.0
]

# We'll use class B for the time-domain demonstration
P_focus = P_B
A_mod = A + EPS_P .* P_focus

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

# Choose one valley and one peak on the focused structural class
idx_valley = choose_valley_in_band(S_B, OMEGAS, (0.28, 0.48))
idx_peak   = choose_peak_in_band(S_B, OMEGAS, (0.95, 1.35))

ω_valley = OMEGAS[idx_valley]
ω_peak   = OMEGAS[idx_peak]

println()
println(@sprintf("Chosen valley frequency: ω = %.4f | S_B = %.4f", ω_valley, S_B[idx_valley]))
println(@sprintf("Chosen peak frequency:   ω = %.4f | S_B = %.4f", ω_peak,   S_B[idx_peak]))
println(@sprintf("Peak/valley ratio:       %.2f", S_B[idx_peak] / S_B[idx_valley]))

# ============================================================
# Time-domain demonstration
# ============================================================
# Same forcing direction in both regimes
b = [1.0, -0.65, 0.55, -0.25]
b ./= norm(b)

# Community-level observable
c = [1.0, 0.8, 1.1, 0.9]

ts, X_valley_base, _ = simulate_forced_system(A,     b, ω_valley)
_,  X_valley_mod,  _ = simulate_forced_system(A_mod, b, ω_valley)

_,  X_peak_base,   _ = simulate_forced_system(A,     b, ω_peak)
_,  X_peak_mod,    _ = simulate_forced_system(A_mod, b, ω_peak)

# Aggregate community signals
Y_valley_base = vec(c' * X_valley_base)
Y_valley_mod  = vec(c' * X_valley_mod)

Y_peak_base   = vec(c' * X_peak_base)
Y_peak_mod    = vec(c' * X_peak_mod)

# Full-state difference
Δ_valley = [norm(X_valley_mod[:, i] - X_valley_base[:, i]) for i in eachindex(ts)]
Δ_peak   = [norm(X_peak_mod[:, i]   - X_peak_base[:, i])   for i in eachindex(ts)]

println(@sprintf("Max ||Δx|| at valley: %.4f", maximum(Δ_valley)))
println(@sprintf("Max ||Δx|| at peak:   %.4f", maximum(Δ_peak)))
println(@sprintf("Peak/valley ||Δx|| ratio: %.2f", maximum(Δ_peak) / maximum(Δ_valley)))

# Shared visual limits
y_prof = profile_ylim(S_intr, S_A, S_B)
y_comm = community_ylim(Y_valley_base, Y_valley_mod, Y_peak_base, Y_peak_mod)
y_delta = (0.0, 1.05 * maximum(Δ_peak))

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
        fig[1, 1],
        title = "A. Different structural changes matter in different frequency bands",
        xlabel = "frequency ω",
        ylabel = "sensitivity",
        xscale = log10,
        yscale = log10
    )
    ylims!(axA, y_prof...)

    # subtle band shading
    vspan!(axA, minimum(OMEGAS), 0.55, color = (:gray, 0.06))
    vspan!(axA, 0.55, 1.35, color = (:gray, 0.10))
    vspan!(axA, 1.35, maximum(OMEGAS), color = (:gray, 0.06))

    lines!(axA, OMEGAS, S_intr, color = :black, linewidth = 4, label = "intrinsic sensitivity")
    lines!(axA, OMEGAS, S_A, color = :darkorange, linewidth = 4, label = "structural class A")
    lines!(axA, OMEGAS, S_B, color = :teal, linewidth = 4, label = "structural class B")

    text!(axA, 0.23, 0.95 * maximum(S_A), text = "A matters more here", color = :darkorange, fontsize = 18)
    text!(axA, 1.05, 0.75 * maximum(S_B), text = "B matters more here", color = :teal, fontsize = 18)

    axislegend(axA, position = :rb)

    # --------------------------------------------------------
    # Panel B
    # --------------------------------------------------------
    # axB = Axis(
    #     fig[1, 2],
    #     title = "B. The same structural change can have low or high impact depending on frequency",
    #     xlabel = "frequency ω",
    #     ylabel = "sensitivity",
    #     xscale = log10,
    #     yscale = log10
    # )
    # ylims!(axB, y_prof...)

    # lines!(axB, OMEGAS, S_intr, color = (:black, 0.30), linewidth = 2.5, label = "intrinsic sensitivity")
    # lines!(axB, OMEGAS, S_B, color = :purple4, linewidth = 4, label = "RPR for structural class B")

    vlines!(axA, [ω_valley], color = :firebrick, linestyle = :dash, linewidth = 3)
    vlines!(axA, [ω_peak],   color = :navy, linestyle = :dash, linewidth = 3)

    scatter!(axA, [ω_valley], [S_B[idx_valley]], color = :firebrick, markersize = 15)
    scatter!(axA, [ω_peak],   [S_B[idx_peak]],   color = :navy, markersize = 15)

    # text!(ax, ω_valley, S_B[idx_valley] * 1.35, text = "valley", color = :firebrick,
    #       align = (:center, :bottom), fontsize = 17)
    # text!(axB, ω_peak, S_B[idx_peak] * 1.18, text = "peak", color = :navy,
    #       align = (:center, :bottom), fontsize = 17)

    # axislegend(axB, position = :rb)

    # --------------------------------------------------------
    # Panel C
    # --------------------------------------------------------
    axC = Axis(
        fig[2, 1],
        title = @sprintf("C. Low-impact regime (ω = %.3f): little change after structural modification", ω_valley),
        xlabel = "time",
        ylabel = "community signal"
    )
    xlims!(axC, first(ts), last(ts))
    ylims!(axC, y_comm...)
    vlines!(axC, [FORCE_START], color = :gray35, linestyle = :dash, linewidth = 2)

    lines!(axC, ts, Y_valley_base, color = :black, linewidth = 2.8, label = "baseline community response")
    lines!(axC, ts, Y_valley_mod,  color = :purple4, linewidth = 2.8, linestyle = :dash,
           label = "modified community response")

    axC_r = Axis(
        fig[2, 1],
        yaxisposition = :right,
        ylabel = "‖Δx(t)‖"
    )
    hidespines!(axC_r)
    hidexdecorations!(axC_r)
    ylims!(axC_r, y_delta...)
    lines!(axC_r, ts, Δ_valley, color = :firebrick, linewidth = 4, label = "full-state change")

    axislegend(axC, position = :rt)

    # --------------------------------------------------------
    # Panel D
    # --------------------------------------------------------
    axD = Axis(
        fig[2, 2],
        title = @sprintf("D. High-impact regime (ω = %.3f): large change after structural modification", ω_peak),
        xlabel = "time",
        ylabel = "community signal"
    )
    xlims!(axD, first(ts), last(ts))
    ylims!(axD, y_comm...)
    vlines!(axD, [FORCE_START], color = :gray35, linestyle = :dash, linewidth = 2)

    lines!(axD, ts, Y_peak_base, color = :black, linewidth = 2.8, label = "baseline community response")
    lines!(axD, ts, Y_peak_mod,  color = :purple4, linewidth = 2.8, linestyle = :dash,
           label = "modified community response")

    axD_r = Axis(
        fig[2, 2],
        yaxisposition = :right,
        ylabel = "‖Δx(t)‖"
    )
    hidespines!(axD_r)
    hidexdecorations!(axD_r)
    ylims!(axD_r, y_delta...)
    lines!(axD_r, ts, Δ_peak, color = :navy, linewidth = 4, label = "full-state change")

    axislegend(axD, position = :rt)

    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 26)

    display(fig)
end
end