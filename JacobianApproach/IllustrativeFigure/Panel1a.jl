begin
    using CairoMakie
    using Random

    # ============================================================
    # PANEL 1 — Community structure with structural modification P
    # ============================================================

    Random.seed!(12)

    # -------------------------
    # Styling
    # -------------------------
    const COL_BG      = :white
    const COL_NODE    = RGBf(0.70, 0.70, 0.70)
    const COL_EDGE    = RGBf(0.65, 0.65, 0.65)
    const COL_EDGE_A  = RGBf(0.15, 0.15, 0.15)
    const COL_MOD     = RGBf(0.77, 0.55, 0.33)   # muted orange
    const COL_TEXT    = RGBf(0.10, 0.10, 0.10)

    # -------------------------
    # Node positions
    # Hand-placed for a clean balanced layout
    # -------------------------
    pts = [
        Point2f(0.10, 0.52),
        Point2f(0.22, 0.80),
        Point2f(0.48, 0.88),
        Point2f(0.76, 0.78),
        Point2f(0.90, 0.55),
        Point2f(0.78, 0.24),
        Point2f(0.46, 0.12),
        Point2f(0.18, 0.22),
        Point2f(0.36, 0.52),
        Point2f(0.60, 0.54),
    ]

    # -------------------------
    # Base network edges
    # -------------------------
    edges = [
        (1,2), (1,8), (1,9),
        (2,3), (2,9), (2,8),
        (3,4), (3,9), (3,10),
        (4,5), (4,10), (4,9),
        (5,6), (5,10), (5,9),
        (6,7), (6,10), (6,9),
        (7,8), (7,9),
        (8,9),
        (9,10),
        (3,5), (2,4), (4,6), (1,7)
    ]

    # Subset highlighted as structural modification P
    mod_edges = [
        (9,10),
        (10,5),
        (10,6),
        (9,5)
    ]

    # A few darker original edges for visual depth
    dark_edges = [
        (1,9),
        (8,9),
        (1,7),
        (7,9)
    ]

    # -------------------------
    # Figure
    # -------------------------
    fig = Figure(size = (800, 500), backgroundcolor = COL_BG)
    ax = Axis(
        fig[1, 1],
        aspect = DataAspect(),
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
        # backgroundcolor = COL_BG
    )

    xlims!(ax, -0.08, 1.18)
    ylims!(ax, -0.05, 1.02)
    

    # -------------------------
    # Draw base edges
    # -------------------------
    for (i, j) in edges
        p1, p2 = pts[i], pts[j]
        lines!(
            ax,
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color = COL_EDGE,
            linewidth = 2
        )
    end

    # Darker structural edges
    for (i, j) in dark_edges
        p1, p2 = pts[i], pts[j]
        lines!(
            ax,
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color = COL_EDGE,
            linewidth = 2
        )
    end

    # Highlighted modification P
    for (i, j) in mod_edges
        p1, p2 = pts[i], pts[j]
        lines!(
            ax,
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color = COL_MOD,
            linewidth = 4
        )
    end

    # -------------------------
    # Nodes
    # -------------------------
    xs = first.(pts)
    ys = last.(pts)

    scatter!(
        ax, xs, ys,
        color = COL_NODE,
        markersize = 16,
        strokecolor = :black,
        strokewidth = 1.5
    )

    # Highlight nodes touched by P
    mod_nodes = unique(vcat(first.(mod_edges), last.(mod_edges)))
    scatter!(
        ax, xs[mod_nodes], ys[mod_nodes],
        color = COL_MOD,
        markersize = 16,
        strokecolor = :black,
        strokewidth = 1.5
    )

    # -------------------------
    # Labels
    # -------------------------
    # text!(ax, -0.02, 0.50,
    #     text = "A",
    #     fontsize = 34,
    #     font = :bold,
    #     color = COL_TEXT,
    #     align = (:right, :center)
    # )

    # text!(ax, 0.98, 0.56,
    #     text = "P",
    #     fontsize = 34,
    #     font = :bold,
    #     color = COL_MOD,
    #     align = (:left, :center)
    # )

    # # Subtle annotation
    # text!(ax, 0.86, 0.18,
    #     text = "structure\nenters here",
    #     fontsize = 18,
    #     color = COL_TEXT,
    #     align = (:left, :center)
    # )

    # arrows!(
    #     ax,
    #     [0.92], [0.23],          # arrow tail
    #     [-0.10], [0.10],         # direction
    #     color = COL_TEXT,
    #     linewidth = 1.5,
    #     arrowsize = 12,
    #     lengthscale = 1
    # )

    display(fig)

end
# save("panel1_structure.png", fig, px_per_unit = 2)