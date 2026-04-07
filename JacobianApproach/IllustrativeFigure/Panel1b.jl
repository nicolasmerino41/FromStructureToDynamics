begin
    using CairoMakie

    # ============================================================
    # CLEAN MATRIX SCHEMATIC
    # ============================================================

    # -------------------------
    # Colors
    # -------------------------
    const COL_BG        = :white
    const COL_TEXT      = RGBf(0.10, 0.10, 0.10)
    const COL_LIGHT     = RGBf(0.92, 0.92, 0.92)
    const COL_MID       = RGBf(0.82, 0.82, 0.82)
    const COL_DARK      = RGBf(0.70, 0.70, 0.70)
    const COL_ORANGE    = RGBf(0.83, 0.57, 0.32)
    const COL_ORANGE_L  = RGBAf(0.83, 0.57, 0.32, 0.10)
    const COL_ORANGE_S  = RGBAf(0.83, 0.57, 0.32, 0.75)

    # -------------------------
    # Matrix size
    # -------------------------
    n = 9

    # Build a pale matrix with a few structured entries
    M = fill(0.0, n, n)

    # dark corner-ish blocks
    M[1,1] = 0.85
    M[1,2] = 0.45
    M[1,3] = 0.20
    M[1,8] = 0.35
    M[1,9] = 0.78

    M[2,2] = 0.95   # orange
    M[3,3] = 0.95   # orange
    M[3,7] = 0.35
    M[3,8] = 0.62

    M[5,3] = 0.30
    M[5,6] = 0.95   # orange
    M[5,9] = 0.42

    M[6,7] = 0.95   # orange
    M[6,9] = 0.48

    M[8,1] = 0.72
    M[8,2] = 0.40
    M[8,3] = 0.18
    M[8,8] = 0.52
    M[8,9] = 0.78

    highlight_row = 5

    # helper
    cellrect(x, y, w=1, h=1) = Rect(x, y, w, h)

    # -------------------------
    # Figure
    # -------------------------

    fig = Figure(size = (700, 760), backgroundcolor = COL_BG)
    ax = Axis(
        fig[1, 1],
        aspect = DataAspect(),
        xreversed = false,
        yreversed = true,
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
        xminorgridvisible = false,
        yminorgridvisible = false,
        backgroundcolor = COL_BG
    )

    xlims!(ax, -1.2, n + 1.2)
    ylims!(ax, -1.2, n + 2.2)

    # -------------------------
    # Pale background cells
    # -------------------------
    for i in 1:n, j in 1:n
        poly!(
            ax,
            cellrect(j - 1, i - 1, 1, 1),
            color = COL_LIGHT,
            strokecolor = COL_BG,
            strokewidth = 2
        )
    end

    # -------------------------
    # Highlighted row band
    # -------------------------
    poly!(
        ax,
        cellrect(0, highlight_row - 1, n, 1),
        color = COL_ORANGE_L,
        strokecolor = COL_ORANGE_S,
        strokewidth = 2
    )

    # -------------------------
    # Draw selected matrix entries
    # -------------------------
   for i in 1:n, j in 1:n
        v = M[i, j]
        if v > 0
            c =
                if i == highlight_row
                    COL_ORANGE
                else
                    v > 0.65 ? COL_DARK :
                    v > 0.35 ? COL_MID  :
                            COL_LIGHT
                end

            pad = 0.12
            poly!(
                ax,
                cellrect((j - 1) + pad, (i - 1) + pad, 1 - 2pad, 1 - 2pad),
                color = c,
                strokecolor = :transparent
            )
        end
    end

    # -------------------------
    # Outer square bracket
    # -------------------------
    lw = 3.0
    lines!(ax, [-0.35, -0.35], [0, n], color = COL_TEXT, linewidth = lw)
    lines!(ax, [-0.35, 0.15], [0, 0], color = COL_TEXT, linewidth = lw)
    lines!(ax, [-0.35, 0.15], [n, n], color = COL_TEXT, linewidth = lw)

    lines!(ax, [n+0.35, n+0.35], [0, n], color = COL_TEXT, linewidth = lw)
    lines!(ax, [n-0.15, n+0.35], [0, 0], color = COL_TEXT, linewidth = lw)
    lines!(ax, [n-0.15, n+0.35], [n, n], color = COL_TEXT, linewidth = lw)

    # -------------------------
    # Ellipses
    # -------------------------
    # text!(ax, n/2, 0.9, text = "⋯", fontsize = 28, color = COL_TEXT, align = (:center, :center))
    # text!(ax, n/2, 7.9, text = "⋯", fontsize = 28, color = COL_TEXT, align = (:center, :center))

    # -------------------------
    # Labels
    # -------------------------
    text!(
        ax, n/2, -0.75,
        text = "Community matrix A",
        fontsize = 28,
        color = COL_TEXT,
        align = (:center, :center)
    )

    # text!(
    #     ax, n/2, n + 0.95,
    #     text = "structure enters here.",
    #     fontsize = 24,
    #     color = COL_TEXT,
    #     align = (:center, :center)
    # )

    # arrows!(
    #     ax,
    #     [n/2], [n + 0.55],
    #     [0.0], [-0.55],
    #     color = COL_TEXT,
    #     linewidth = 2.0,
    #     arrowsize = 14,
    #     lengthscale = 1.0
    # )

    display(fig)
    # save("clean_matrix_panel.png", fig, px_per_unit = 2)
end