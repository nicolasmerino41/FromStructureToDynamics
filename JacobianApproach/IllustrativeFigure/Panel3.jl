using CairoMakie

# ============================================================
# PANEL 3 — Re-expression by temporal scale
# ============================================================

# -------------------------
# Helpers
# -------------------------
gauss(t, μ, σ, A=1.0) = A .* exp.(-0.5 .* ((t .- μ) ./ σ).^2)

function gamma_bump(t; t0=0.0, k=3.0, θ=1.0, A=1.0)
    x = max.(t .- t0, 0.0)
    y = (x .^ (k - 1)) .* exp.(-x ./ θ)
    m = maximum(y)
    m > 0 ? A .* (y ./ m) : zero.(t)
end

# smooth oscillatory packet
function wavepacket(t; μ=0.0, σ=1.0, ω=1.0, A=1.0, phase=0.0)
    A .* exp.(-0.5 .* ((t .- μ) ./ σ).^2) .* cos.(ω .* (t .- μ) .+ phase)
end

# -------------------------
# Colors
# -------------------------
const COL_BG      = :white
const COL_TEXT    = RGBf(0.10, 0.10, 0.10)
const COL_AXIS    = RGBf(0.20, 0.20, 0.20)

const COL_ORIG    = RGBf(0.40, 0.49, 0.61)     # muted blue-gray
const COL_MOD     = RGBf(0.72, 0.50, 0.30)     # muted orange-brown

const BAND_SLOW   = RGBAf(0.56, 0.63, 0.78, 0.23)
const BAND_MID    = RGBAf(0.76, 0.82, 0.72, 0.23)
const BAND_FAST   = RGBAf(0.72, 0.82, 0.66, 0.23)

const COL_SLOW    = RGBf(0.40, 0.49, 0.61)
const COL_MID     = RGBf(0.46, 0.60, 0.52)
const COL_FAST    = RGBf(0.42, 0.55, 0.38)

# -------------------------
# Left: same time-domain curves as panel 2
# -------------------------
t = range(0, 10, length=700)
y_orig = gamma_bump(t; t0=1.2,  k=3.2, θ=0.78, A=1.0)
y_mod  = gamma_bump(t; t0=1.35, k=3.6, θ=0.84, A=0.88)

# -------------------------
# Right: timescale-organized representative content
# Use separate local x coordinates in [0,1] for each strip
# -------------------------
u = range(0, 1, length=700)

slow_orig = 0.10 .+ 0.23 .* gauss(u, 0.38, 0.16) .+ 0.07 .* gauss(u, 0.72, 0.22)
slow_mod  = 0.10 .+ 0.18 .* gauss(u, 0.46, 0.18) .+ 0.05 .* gauss(u, 0.77, 0.22)

mid_orig = 0.02 .+ 0.20 .* gauss(u, 0.22, 0.09) .- 0.12 .* gauss(u, 0.48, 0.10) .+ 0.10 .* gauss(u, 0.72, 0.11)
mid_mod  = 0.03 .+ 0.13 .* gauss(u, 0.26, 0.09) .- 0.08 .* gauss(u, 0.50, 0.11) .+ 0.08 .* gauss(u, 0.70, 0.12)

fast_orig = 0.12 .* wavepacket(u; μ=0.45, σ=0.15, ω=26, phase=0.0) .+ 0.06 .* wavepacket(u; μ=0.75, σ=0.10, ω=34, phase=0.5)
fast_mod  = 0.09 .* wavepacket(u; μ=0.47, σ=0.15, ω=24, phase=0.15) .+ 0.05 .* wavepacket(u; μ=0.76, σ=0.10, ω=31, phase=0.6)

# -------------------------
# Plot
# -------------------------
begin
    fig = Figure(size = (1100, 480), backgroundcolor = COL_BG)

    ax = Axis(
        fig[1, 1],
        backgroundcolor = COL_BG,
        leftspinevisible = false,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = false,
        # xticks = [],
        # yticks = []
    )

    xlims!(ax, 0, 1)
    ylims!(ax, 0, 1)

    hidedecorations!(ax)
    hidespines!(ax)

    # --------------------------------------------------------
    # Left time-domain inset
    # --------------------------------------------------------
    x0, x1 = 0.06, 0.33
    y0, y1 = 0.22, 0.70

    # frame
    lines!(ax, [x0, x1], [y0, y0], color = COL_AXIS, linewidth = 2)
    lines!(ax, [x0, x0], [y0, y1], color = COL_AXIS, linewidth = 2)
    lines!(ax, [x0, x1], [y1, y1], color = RGBAf(0,0,0,0.25), linewidth = 1)
    lines!(ax, [x1, x1], [y0, y1], color = RGBAf(0,0,0,0.25), linewidth = 1)

    # pulse arrow inside inset
    arrows!(
        ax,
        [x0 + 0.045], [y1 - 0.02],
        [0.0], [-0.18],
        color = :black,
        linewidth = 2.0,
        arrowsize = 12,
        lengthscale = 1.0
    )

    # map curves into inset coordinates
    xt = x0 .+ (x1 - x0) .* ((t .- minimum(t)) ./ (maximum(t) - minimum(t)))
    yt_orig = y0 .+ 0.84 .* (y1 - y0) .* y_orig
    yt_mod  = y0 .+ 0.84 .* (y1 - y0) .* y_mod

    lines!(ax, xt, yt_orig, color = COL_ORIG, linewidth = 3)
    lines!(ax, xt, yt_mod,  color = COL_MOD,  linewidth = 3)

    text!(ax, (x0 + x1)/2, y0 - 0.055,
        text = "Time domain",
        fontsize = 20,
        color = COL_TEXT,
        align = (:center, :top)
    )

    # --------------------------------------------------------
    # Transform arrow
    # --------------------------------------------------------
    # arrows!(
    #     ax,
    #     [0.40], [0.46],
    #     [0.10], [0.0],
    #     color = COL_TEXT,
    #     linewidth = 2.4,
    #     arrowsize = 16,
    #     lengthscale = 1.0
    # )

    # optional transform cue
    # text!(ax, 0.455, 0.54,
    #     text = "re-express\nby scale",
    #     fontsize = 18,
    #     color = COL_TEXT,
    #     align = (:center, :bottom)
    # )

    # --------------------------------------------------------
    # Right timescale strips
    # --------------------------------------------------------
    xr0, xr1 = 0.56, 0.96
    strip_h = 0.18

    ys = [0.73, 0.50, 0.27]
    band_cols = [BAND_SLOW, BAND_MID, BAND_FAST]
    band_labels = ["slow", "intermediate", "fast"]

    # background strips + labels
    for k in 1:3
        yb = ys[k] - strip_h/2
        poly!(
            ax,
            Rect2f(xr0, yb, xr1 - xr0, strip_h),
            color = band_cols[k],
            strokecolor = :transparent
        )
        text!(ax, xr0 - 0.03, ys[k],
            text = band_labels[k],
            fontsize = 22,
            color = COL_TEXT,
            align = (:right, :center)
        )
    end

    # local horizontal coordinate mapped into right area
    xu = xr0 .+ (xr1 - xr0) .* u

    # slow band curves
    lines!(ax, xu, ys[1] .+ 0.22 .* slow_orig, color = COL_ORIG, linewidth = 3)
    lines!(ax, xu, ys[1] .+ 0.22 .* slow_mod,  color = COL_MOD,  linewidth = 3)

    # intermediate band curves
    lines!(ax, xu, ys[2] .+ 0.35 .* mid_orig, color = COL_ORIG, linewidth = 3)
    lines!(ax, xu, ys[2] .+ 0.35 .* mid_mod,  color = COL_MOD,  linewidth = 3)

    # fast band curves
    lines!(ax, xu, ys[3] .+ 0.55 .* fast_orig, color = COL_ORIG, linewidth = 2.5)
    lines!(ax, xu, ys[3] .+ 0.55 .* fast_mod,  color = COL_MOD,  linewidth = 2.5)

    # small label underneath right block
    text!(ax, (xr0 + xr1)/2, 0.08,
        text = "Timescale domain",
        fontsize = 20,
        color = COL_TEXT,
        align = (:center, :center)
    )

    display(fig)

    # save("panel3_reexpress_by_scale.png", fig, px_per_unit = 2)
end