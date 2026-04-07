begin
    using CairoMakie
    # ============================================================
    # PANEL 4 — Frequency-domain sensitivity profile
    # ============================================================

    # -------------------------
    # Helpers
    # -------------------------
    gauss(x, μ, σ, A=1.0) = A .* exp.(-0.5 .* ((x .- μ) ./ σ).^2)

    # -------------------------
    # Colors
    # -------------------------
    const COL_BG      = :white
    const COL_TEXT    = RGBf(0.10, 0.10, 0.10)
    const COL_AXIS    = RGBf(0.15, 0.15, 0.15)

    const COL_ORIG    = RGBf(0.40, 0.49, 0.61)   # muted blue-gray
    const COL_MOD     = RGBf(0.72, 0.50, 0.30)   # muted orange-brown

    # Stronger contrast between sections
    const BAND_SLOW   = RGBAf(0.50, 0.60, 0.82, 0.30)
    const BAND_MID    = RGBAf(0.90, 0.78, 0.50, 0.28)
    const BAND_FAST   = RGBAf(0.60, 0.78, 0.58, 0.30)

    # -------------------------
    # Frequency axis and curves
    # Use log-frequency coordinate internally for smooth shapes
    # -------------------------
    ω = exp.(range(log(1e-2), log(1e2), length=900))
    x = log10.(ω)

    # Original community:
    # broad low/intermediate response with a smaller fast shoulder
    S_orig =
        0.2 .+
        gauss(x, -0.95, 0.32, 0.34) .+
        gauss(x, -0.10, 0.22, 0.58) .+
        gauss(x,  0.70, 0.26, 0.22)

    # Impact of structural change P:
    # low-frequency dominated, heavy-tail-like decay, much smaller overall
    S_mod =
        0.02 .+
        0.16 ./ (1 .+ (ω ./ 0.10).^0.85) .+   # dominant slow-timescale impact
        0.035 .* exp.(-(ω ./ 1.8).^0.7)       # gentle extra shoulder, still decaying

    fig = Figure(size = (900, 560), backgroundcolor = COL_BG)

    ax = Axis(
        fig[1, 1],
        xlabel = "Frequency ω",
        ylabel = "Sensitivity R(ω) / stability",
        xscale = log10,
        backgroundcolor = COL_BG,
        leftspinevisible = true,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = true,
        xlabelsize = 24,
        ylabelsize = 24,
    )

    xlims!(ax, minimum(ω), maximum(ω))
    ylims!(ax, 0, 1.05)
    hidedecorations!(ax, ticks = false, ticklabels = false, label = false)

    ax.yticksvisible = false
    ax.yticklabelsvisible = false
    ax.xticksvisible = false
    ax.xticklabelsvisible = false
    

    # Background bands
    vspan!(ax, 1e-2, 1e-1, color = BAND_SLOW)
    vspan!(ax, 1e-1, 1e0, color = BAND_MID)
    vspan!(ax, 1e0, 1e2, color = BAND_FAST)

    # Curves
    l1 = lines!(ax, ω, S_orig, color = COL_ORIG, linewidth = 4)
    l2 = lines!(ax, ω, S_mod,  color = COL_MOD,  linewidth = 4)

    # Band centers on a log scale
    ωslow = sqrt(1e-2 * 1e-1)
    ωmid  = sqrt(1e-1 * 1e0)
    ωfast = sqrt(1e0 * 1e2)

    # Band labels
    text!(ax, ωslow, 1.00,
        text = "slow",
        fontsize = 22,
        color = COL_TEXT,
        align = (:center, :top)
    )

    text!(ax, ωmid, 1.00,
        text = "intermediate",
        fontsize = 22,
        color = COL_TEXT,
        align = (:center, :top)
    )

    text!(ax, ωfast, 1.00,
        text = "fast",
        fontsize = 22,
        color = COL_TEXT,
        align = (:center, :top)
    )

    # Legend
    axislegend(
        ax,
        [l1, l2],
        ["Original community", "Impact of structural change P"],
        position = :rc,
        framevisible = true,
        bgcolor = RGBAf(1, 1, 1, 0.90),
        labelsize = 18,
        padding = (10, 10, 10, 10)
    )

    # Clean up decorations
    hidespines!(ax, :t, :r)
    hidedecorations!(ax, ticks = false, ticklabels = false, minorgrid = false, minorticks = false)

    ax.xlabelvisible = true
    ax.ylabelvisible = true

    display(fig)

    # save("panel4_frequency_sensitivity.png", fig, px_per_unit = 2)
end