using CairoMakie

# ============================================================
# PANEL 2 — Pulse propagation in time
# ============================================================
begin
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

    # -------------------------
    # Colors
    # -------------------------
    const COL_BG      = :white
    const COL_TEXT    = RGBf(0.10, 0.10, 0.10)
    const COL_AXIS    = RGBf(0.15, 0.15, 0.15)

    const COL_ORIG    = RGBf(0.40, 0.49, 0.61)   # muted blue-gray
    const COL_MOD     = RGBf(0.72, 0.50, 0.30)   # muted orange-brown
    const COL_FAINT   = RGBAf(0.7, 0.7, 0.7, 1.0)
    const COL_PULSE   = RGBf(0.45, 0.53, 0.65)

    # -------------------------
    # Time axis and curves
    # -------------------------
    t = range(0, 10, length=900)

    # Original community: slightly earlier, sharper
    y_orig = gamma_bump(t; t0=1.2, k=3.2, θ=0.78, A=1.0)

    # Modified by P: slightly delayed, broader
    y_mod  = gamma_bump(t; t0=1.35, k=3.6, θ=0.84, A=0.88)

    # Faint hidden components to suggest entanglement
    c1 = 0.42 .* gauss(t, 2.4, 0.65)
    c2 = 0.34 .* gauss(t, 3.6, 0.90)
    c3 = 0.24 .* gauss(t, 5.0, 1.10)
    c4 = 0.18 .* gauss(t, 6.8, 1.20)

    # Pulse perturbation location
    tpulse = 1.0

    # -------------------------
    # Plot
    # -------------------------

    fig = Figure(size = (900, 520), backgroundcolor = COL_BG)

    ax = Axis(
        fig[1, 1],
        xlabel = "Time",
        ylabel = "Return rate",
        backgroundcolor = COL_BG,
        leftspinevisible = true,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = true,
        xlabelsize = 24,
        ylabelsize = 24,
    )
    hidedecorations!(ax, ticks = false, ticklabels = false, label = false)

    ax.yticksvisible = false
    ax.yticklabelsvisible = false
    ax.xticksvisible = false
    ax.xticklabelsvisible = false
    
    # Limits
    xlims!(ax, 0, 10)
    ylims!(ax, 0, 1.18)

    # Faint component curves
    lines!(ax, t, c1, color = COL_FAINT, linewidth = 3)
    lines!(ax, t, c2, color = COL_FAINT, linewidth = 3)
    lines!(ax, t, c3, color = COL_FAINT, linewidth = 3)
    lines!(ax, t, c4, color = COL_FAINT, linewidth = 3)

    # Main response curves with legend labels
    lines!(ax, t, y_orig, color = COL_ORIG, linewidth = 4, label = "Original community")
    lines!(ax, t, y_mod,  color = COL_MOD,  linewidth = 4, label = "Modified community")

    # Pulse perturbation arrow
    arrows!(
        ax,
        [tpulse*1.1], [1.02],   # tail
        [0.0], [-0.78],     # direction
        color = :black,
        linewidth = 2.5,
        arrowsize = 18,
        lengthscale = 1.0
    )

    text!(
        ax, tpulse*1.1, 1.08,
        text = "Pulse perturbation",
        fontsize = 22,
        color = COL_TEXT,
        align = (:center, :bottom)
    )

    # Legend inside the axis
    axislegend(
        ax;
        position = :rt,
        framevisible = false,
        labelsize = 20,
        patchsize = (35, 18)
    )

    # Slight styling cleanup
    hidespines!(ax, :t, :r)
    hidedecorations!(ax, ticks = false, ticklabels = false, label = false)
    ax.xlabelvisible = true

    display(fig)

    # save("panel2_time_response.png", fig, px_per_unit = 2)
end