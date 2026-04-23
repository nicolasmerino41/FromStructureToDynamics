using CairoMakie
using Random

begin
    # ============================================================
    # MAIN ILLUSTRATION — Structure → time response → frequency decomposition
    # ============================================================

    Random.seed!(12)

    # ------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------
    gauss(x, μ, σ, A=1.0) = A .* exp.(-0.5 .* ((x .- μ) ./ σ).^2)

    function gamma_bump(t; t0=0.0, k=3.0, θ=1.0, A=1.0)
        x = max.(t .- t0, 0.0)
        y = (x .^ (k - 1)) .* exp.(-x ./ θ)
        m = maximum(y)
        m > 0 ? A .* (y ./ m) : zero.(t)
    end

    bandwave(u; f=1.0, A=1.0, phase=0.0, offset=0.0) =
        offset .+ A .* cos.(2π * f .* u .+ phase)

    cellrect(x, y, w=1, h=1) = Rect2f(x, y, w, h)

    # ------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------
    COL_BG      = :white
    COL_TEXT    = RGBf(0.10, 0.10, 0.10)

    COL_NODE    = RGBf(0.70, 0.70, 0.70)
    COL_EDGE    = RGBf(0.68, 0.68, 0.68)

    COL_ORIG    = RGBf(0.40, 0.49, 0.61)
    COL_MOD     = RGBf(0.72, 0.50, 0.30)

    COL_LIGHT   = RGBf(0.92, 0.92, 0.92)
    COL_MID     = RGBf(0.82, 0.82, 0.82)
    COL_DARK    = RGBf(0.70, 0.70, 0.70)

    COL_ORANGE_L = RGBAf(0.83, 0.57, 0.32, 0.10)
    COL_ORANGE_S = RGBAf(0.83, 0.57, 0.32, 0.78)

    COL_SPECIES_GREY = RGBAf(0.72, 0.72, 0.72, 0.95)
    COL_SPECIES_ORNG = RGBAf(0.72, 0.50, 0.30, 0.95)

    BAND_SLOW   = RGBAf(0.50, 0.60, 0.82, 0.30)
    BAND_MID    = RGBAf(0.90, 0.78, 0.50, 0.28)
    BAND_FAST   = RGBAf(0.60, 0.78, 0.58, 0.30)

    # ------------------------------------------------------------
    # Data for panel 1a: 4-species network
    # ------------------------------------------------------------
    pts = [
        # -------------------------
        # Outer ring (8 species)
        # same overall shape, more homogeneous spacing
        # -------------------------
        Point2f(0.50, 0.91),  # 1
        Point2f(0.72, 0.83),  # 2
        Point2f(0.86, 0.66),  # 3
        Point2f(0.87, 0.43),  # 4
        Point2f(0.73, 0.22),  # 5
        Point2f(0.28, 0.78),  # 6
        Point2f(0.23, 0.23),  # 7
        Point2f(0.10, 0.47),  # 8

        # -------------------------
        # Inner ring (5 species)
        # same relative placement, slightly more even
        # -------------------------
        Point2f(0.50, 0.68),  # 9
        Point2f(0.66, 0.58),  # 10
        Point2f(0.60, 0.39),  # 11  <- focal species
        Point2f(0.41, 0.34),  # 12
        Point2f(0.33, 0.56),  # 13

        # -------------------------
        # Core (2 species)
        # -------------------------
        Point2f(0.47, 0.49),  # 14
        Point2f(0.55, 0.51),  # 15
    ]
    # focal species
    mod_node = 11

    # Sparse, local, imperfect connectivity
    edges = [
        # -------------------------
        # Outer ring: only some contiguous links
        # -------------------------
        (1,2),
        (2,3),
        # (4,5),
        (6,7),
        (7,8),
        # (8,1),

        # a few short outer skips for realism
        (2,4),
        (5,7),

        # -------------------------
        # Inner ring: only some contiguous links
        # -------------------------
        (9,10),
        (10,11),
        (12,13),
        (13,9),

        # one inner skip
        (10,13),

        # -------------------------
        # Core connections
        # -------------------------
        (14,15),
        (14,9),
        (15,10),
        (14,12),

        # -------------------------
        # Outer ↔ inner local bridges
        # -------------------------
        (1,9),
        (2,10),
        (4,10),
        (5,11),
        (7,12),
        (8,13),

        # -------------------------
        # Focal species connections: 4 nearby nodes
        # inner-ring species, not core, not outer-only
        # -------------------------
        (11,10),
        (11,12),
        (11,5),
        (11,15),
    ]

    mod_edges = [(i, j) for (i, j) in edges if i == mod_node || j == mod_node]
    base_edges = [(i, j) for (i, j) in edges if !(i == mod_node || j == mod_node)]
    # Arrow and annotation for the focal structural change

    # ------------------------------------------------------------
    # Data for panel 1b: matrix
    # ------------------------------------------------------------
    n = length(pts)
    M = fill(0.0, n, n)

    for i in 1:n
        M[i, i] = 0.55
    end

    for (i, j) in base_edges
        M[i, j] = 0.72
        M[j, i] = 0.72
    end

    for (i, j) in mod_edges
        M[i, j] = 0.98
        M[j, i] = 0.98
    end

    highlight_row = mod_node

    # ------------------------------------------------------------
    # Data for panel 2: time response
    # ------------------------------------------------------------
    t = range(0, 10, length=900)

    # Displacement trajectories:
    # original = mostly monotone relaxation
    # modified = transient amplification before relaxation

    # Both start from the same initial displacement:
    y0 = 0.75

    # Original: nearly monotone decay
    y_orig = y0 .* exp.(-t ./ 2.15)

    # Modified: same intercept, transient amplification, then decay
    y_mod =
        y0 .* exp.(-t ./ 3.2) .+
    1.05 .* (1 .- exp.(-t ./ 0.9)) .* exp.(-t ./ 1.5)

    # ------------------------------------------------------------
    # Modal-participation inset
    # Each bar = one mode; stacked by species contribution magnitude
    # Species 3 is the touched species and contributes across several modes
    # ------------------------------------------------------------
    mode_names = ["slow", "intermediate", "fast"]

    # rows = species, cols = modes
    Wmag = [
        0.30  0.10  0.06  0.14;
        0.24  0.28  0.12  0.10;
        0.22  0.34  0.31  0.18;   # touched species (orange)
        0.24  0.28  0.51  0.58
    ]

    # normalize each mode to height 1
    for j in 1:size(Wmag, 2)
        Wmag[:, j] ./= sum(Wmag[:, j])
    end
        # ------------------------------------------------------------
    # Data for bridge: timescale-separated wave strips
    # ------------------------------------------------------------
    u = range(0, 1, length=900)

    f_slow = 1.0
    f_mid  = 4.2
    f_fast = 11.5

    slow_orig = bandwave(u; f=f_slow, A=0.050, phase=0.10)
    mid_orig  = bandwave(u; f=f_mid,  A=0.034, phase=0.30)
    fast_orig = bandwave(u; f=f_fast, A=0.020, phase=0.12)

    slow_mod = bandwave(u; f=f_slow, A=0.032, phase=0.60)
    mid_mod  = bandwave(u; f=f_mid,  A=0.021, phase=0.75)
    fast_mod = bandwave(u; f=f_fast, A=0.011, phase=0.95)

    # ------------------------------------------------------------
    # Data for panel 4: frequency profile
    # ------------------------------------------------------------
    ω = exp.(range(log(1e-2), log(1e2), length=900))
    x = log10.(ω)

    S_orig =
        0.40 .+
        gauss(x, -0.95, 0.25, 0.18) .+
        gauss(x, -0.15, 0.28, 0.46) .-
        0.10 ./ (1 .+ exp.(-4.0 .* (x .- 0.45)))

    S_mod =
        0.19 .+
        0.16 ./ (1 .+ (ω ./ 0.10).^0.85) .+
        0.035 .* exp.(-(ω ./ 1.8).^0.7)

    ωslow = sqrt(1e-2 * 1e-1)
    ωmid  = sqrt(1e-1 * 1e0)
    ωfast = sqrt(1e0 * 1e2)

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    fig = Figure(size = (1820, 1020), backgroundcolor = COL_BG)

    top = fig[1, 1] = GridLayout()
    bottom = fig[2, 1] = GridLayout()

    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 26)

    # ============================================================
    # TOP LEFT — NETWORK
    # ============================================================
    ax_net = Axis(
        top[1, 1],
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
        backgroundcolor = COL_BG
    )

    xlims!(ax_net, -0.05, 1.15)
    ylims!(ax_net, -0.02, 1.02)

    for (i, j) in base_edges
        p1, p2 = pts[i], pts[j]
        lines!(ax_net, [p1[1], p2[1]], [p1[2], p2[2]], color = COL_EDGE, linewidth = 3.0)
    end

    for (i, j) in mod_edges
        p1, p2 = pts[i], pts[j]
        lines!(ax_net, [p1[1], p2[1]], [p1[2], p2[2]], color = COL_MOD, linewidth = 5.0)
    end

    xs = first.(pts)
    ys_pts = last.(pts)

    scatter!(ax_net, xs, ys_pts, color = COL_NODE, markersize = 34)
    scatter!(ax_net, [xs[mod_node]], [ys_pts[mod_node]], color = COL_MOD, markersize = 40)

    text!(
        ax_net,
        xs[mod_node] + 0.25, ys_pts[mod_node] - 0.1,
        text = "Modified\nstructure",
        fontsize = 22,
        color = COL_TEXT,
        align = (:left, :center)
    )

    arrows!(
        ax_net,
        [xs[mod_node] + 0.03], [ys_pts[mod_node] + 0.00],
        [0.2], [-0.1 0],
        color = COL_TEXT,
        linewidth = 1.8,
        arrowsize = 14,
        lengthscale = 1.0
    )

    # ============================================================
    # TOP RIGHT — MATRIX
    # ============================================================
    ax_mat = Axis(
        top[1, 2],
        aspect = DataAspect(),
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
        backgroundcolor = COL_BG
    )

    xlims!(ax_mat, -1.0, n + 1.0)
    ylims!(ax_mat, -1.0, n + 1.1)

    for i in 1:n, j in 1:n
        poly!(ax_mat, cellrect(j - 1, i - 1, 1, 1),
            color = COL_LIGHT, strokecolor = COL_BG, strokewidth = 2)
    end

    poly!(ax_mat, cellrect(0, highlight_row - 1, n, 1),
        color = COL_ORANGE_L, strokecolor = COL_ORANGE_S, strokewidth = 2)
    poly!(ax_mat, cellrect(mod_node - 1, 0, 1, n),
        color = COL_ORANGE_L, strokecolor = COL_ORANGE_S, strokewidth = 2)

    for i in 1:n, j in 1:n
        v = M[i, j]
        if v > 0
            c = if i == highlight_row || j == mod_node
                COL_MOD
            else
                v > 0.78 ? COL_DARK :
                v > 0.60 ? COL_MID  :
                           COL_LIGHT
            end
            pad = 0.12
            poly!(
                ax_mat,
                cellrect((j - 1) + pad, (i - 1) + pad, 1 - 2pad, 1 - 2pad),
                color = c, strokecolor = :transparent
            )
        end
    end

    lw = 3.0
    lines!(ax_mat, [-0.30, -0.30], [0, n], color = COL_TEXT, linewidth = lw)
    lines!(ax_mat, [-0.30, 0.12], [0, 0], color = COL_TEXT, linewidth = lw)
    lines!(ax_mat, [-0.30, 0.12], [n, n], color = COL_TEXT, linewidth = lw)
    lines!(ax_mat, [n + 0.30, n + 0.30], [0, n], color = COL_TEXT, linewidth = lw)
    lines!(ax_mat, [n - 0.12, n + 0.30], [0, 0], color = COL_TEXT, linewidth = lw)
    lines!(ax_mat, [n - 0.12, n + 0.30], [n, n], color = COL_TEXT, linewidth = lw)

    text!(ax_mat, n/2, -0.60,
        text = "Community matrix",
        fontsize = 24, color = COL_TEXT,
        align = (:center, :center))

    # ============================================================
    # BOTTOM LEFT — TIME RESPONSE WITH INSET HISTOGRAM
    # ============================================================
    ax_time = Axis(
        bottom[1, 1],
        xlabel = "Time following a perturbation",
        ylabel = "Distance to equilibrium",
        backgroundcolor = COL_BG,
        leftspinevisible = true,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = true,
        xlabelsize = 24,
        ylabelsize = 24,
    )

    xlims!(ax_time, 0, 10)
    ylims!(ax_time, 0, 1.18)

    ax_time.yticksvisible = false
    ax_time.yticklabelsvisible = false
    ax_time.xticksvisible = false
    ax_time.xticklabelsvisible = false
    hidedecorations!(ax_time, ticks = false, ticklabels = false, label = false)
    hidespines!(ax_time, :t, :r)

    lines!(ax_time, t, y_orig, color = COL_ORIG, linewidth = 4, label = "Original community")
    lines!(ax_time, t, y_mod,  color = COL_MOD,  linewidth = 4, label = "Modified community")

    arrows!(ax_time, [1.0], [1.1], [0.0], [-0.78],
        color = :black, linewidth = 2.5, arrowsize = 18, lengthscale = 1.0)

    text!(ax_time, 1.15, 1.04,
        text = "Pulse perturbation",
        fontsize = 15, color = COL_TEXT,
        align = (:center, :bottom))

    # text!(ax_time, 4.95, 0.96,
    #     text = "hard to read off\nwhich modes matter",
    #     fontsize = 18, color = COL_TEXT,
    #     align = (:center, :center))

    axislegend(
        ax_time;
        position = (1.0, 0.20),
        framevisible = false,
        labelsize = 18,
        patchsize = (34, 18)
    )

    # --- inset histogram inside time plot, top-right ---
    ax_inset = Axis(
        bottom[1, 1],
        width = Relative(0.30),
        height = Relative(0.40),
        halign = 0.97,
        valign = 0.88,
        backgroundcolor = RGBAf(1, 1, 1, 0.92),
        xgridvisible = false,
        ygridvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        xlabelvisible = false,
        ylabelvisible = false
    )

    hidespines!(ax_inset, :t, :r)

    xlims!(ax_inset, 0.35, 4.65)
    ylims!(ax_inset, 0, 1.34)

    ax_inset.xticks = ([1, 2, 3, 4], ["Mode 1", "Mode 2", "   ...   ", "Mode N"])
    ax_inset.xticklabelrotation = π/4
    ax_inset.xticksvisible = true
    ax_inset.xticklabelsvisible = true
    ax_inset.xticklabelsize = 10

    # orange fraction = contribution of the touched species to each selected mode
    orange_frac = Dict(
        1 => 0.1,   # Mode 1
        2 => 0.4,   # Mode 2
        4 => 0.18    # Mode N
    )

    bar_x = [1, 2, 4]
    bar_w = 0.58

    # draw bars with orange slice in the middle
    for j in bar_x
        h_orange = orange_frac[j]
        h_gray_total = 1.0 - h_orange

        # split remaining gray into bottom and top parts
        h_gray_bottom = 0.55 * h_gray_total
        h_gray_top    = 0.45 * h_gray_total

        # bottom gray
        poly!(
            ax_inset,
            Rect2f(j - bar_w/2, 0.0, bar_w, h_gray_bottom),
            color = COL_SPECIES_GREY,
            strokecolor = :transparent
        )

        # orange middle
        poly!(
            ax_inset,
            Rect2f(j - bar_w/2, h_gray_bottom, bar_w, h_orange),
            color = COL_SPECIES_ORNG,
            strokecolor = :transparent
        )

        # top gray
        poly!(
            ax_inset,
            Rect2f(j - bar_w/2, h_gray_bottom + h_orange, bar_w, h_gray_top),
            color = COL_SPECIES_GREY,
            strokecolor = :transparent
        )
    end

    # outlines for the real bars
    for j in bar_x
        lines!(
            ax_inset,
            [j - bar_w/2, j - bar_w/2, j + bar_w/2, j + bar_w/2, j - bar_w/2],
            [0, 1, 1, 0, 0],
            color = RGBAf(0.20, 0.20, 0.20, 0.9),
            linewidth = 1.0
        )
    end

    # diagonal ellipsis at x = 3 instead of a bar
    scatter!(
        ax_inset,
        [2.88, 3.00, 3.12],
        [0.36, 0.50, 0.64],
        color = RGBAf(0.35, 0.35, 0.35, 0.9),
        markersize = 8
    )

    text!(
        ax_inset,
        2.5, 1.14,
        text = "Species contributions to each mode",
        fontsize = 12,
        color = COL_TEXT,
        align = (:center, :bottom)
    )

    # ============================================================
    # BOTTOM MIDDLE — TIMESCALE DECOMPOSITION
    # ============================================================
    ax_bridge = Axis(
        bottom[1, 2],
        backgroundcolor = COL_BG,
        leftspinevisible = false,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = false,
    )

    xlims!(ax_bridge, -0.15, 1.15)
    ylims!(ax_bridge, 0, 1)
    hidedecorations!(ax_bridge)
    hidespines!(ax_bridge)

    ys_bridge = [0.77, 0.45, 0.13]
    strip_h = 0.25

    x_strip0 = 0.20
    x_strip1 = 0.80
    x_arrow_l = 0.00
    x_arrow_r = 0.88

    for (y, col, lbl) in zip(ys_bridge, [BAND_SLOW, BAND_MID, BAND_FAST], ["Slow", "Intermediate", "Fast"])
        poly!(
            ax_bridge,
            Rect2f(x_strip0, y - strip_h/2, x_strip1 - x_strip0, strip_h),
            color = col,
            strokecolor = :transparent
        )

        text!(
            ax_bridge,
            (x_strip0 + x_strip1) / 2, y + strip_h/2 + 0.008,
            text = lbl,
            fontsize = 18,
            color = COL_TEXT,
            align = (:center, :bottom)
        )
    end

    xu = x_strip0 .+ (x_strip1 - x_strip0) .* u
    lines!(ax_bridge, xu, ys_bridge[1] .+ slow_orig, color = COL_ORIG, linewidth = 2.5)
    lines!(ax_bridge, xu, ys_bridge[1] .+ slow_mod,  color = COL_MOD,  linewidth = 2.5)
    lines!(ax_bridge, xu, ys_bridge[2] .+ mid_orig,  color = COL_ORIG, linewidth = 2.5)
    lines!(ax_bridge, xu, ys_bridge[2] .+ mid_mod,   color = COL_MOD,  linewidth = 2.5)
    lines!(ax_bridge, xu, ys_bridge[3] .+ fast_orig, color = COL_ORIG, linewidth = 2.2)
    lines!(ax_bridge, xu, ys_bridge[3] .+ fast_mod,  color = COL_MOD,  linewidth = 2.2)

    arrows!(ax_bridge, [x_arrow_l], [0.5], [0.12], [0.0],
        color = COL_TEXT, linewidth = 2.2, arrowsize = 14, lengthscale = 1.0)
    arrows!(ax_bridge, [x_arrow_r], [0.50], [0.12], [0.0],
        color = COL_TEXT, linewidth = 2.2, arrowsize = 14, lengthscale = 1.0)

    text!(
        ax_bridge,
        0.50, 0.96,
        text = "frequency domain separates\n the modal content by timescale",
        fontsize = 18,
        color = COL_TEXT,
        align = (:center, :top)
    )

    # ============================================================
    # BOTTOM RIGHT — FREQUENCY PROFILE
    # ============================================================
    ax_freq = Axis(
        bottom[1, 3],
        xlabel = "Frequency ω",
        ylabel = "Sensitivity R",
        xscale = log10,
        backgroundcolor = COL_BG,
        leftspinevisible = true,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = true,
        xgridvisible = false,
        ygridvisible = false,
        xlabelsize = 24,
        ylabelsize = 24,
    )

    xlims!(ax_freq, minimum(ω), maximum(ω))
    ylims!(ax_freq, 0, 1.05)

    ax_freq.yticksvisible = false
    ax_freq.yticklabelsvisible = false
    ax_freq.xticksvisible = false
    ax_freq.xticklabelsvisible = false

    vspan!(ax_freq, 1e-2, 1e-1, color = BAND_SLOW)
    vspan!(ax_freq, 1e-1, 1e0,  color = BAND_MID)
    vspan!(ax_freq, 1e0,  1e2,  color = BAND_FAST)

    q1 = lines!(ax_freq, ω, S_orig, color = COL_ORIG, linewidth = 4)
    q2 = lines!(ax_freq, ω, S_mod,  color = COL_MOD,  linewidth = 4)

    text!(ax_freq, ωslow, 1.00, text = "slow", fontsize = 20, color = COL_TEXT, align = (:center, :top))
    text!(ax_freq, ωmid,  1.00, text = "intermediate", fontsize = 20, color = COL_TEXT, align = (:center, :top))
    text!(ax_freq, ωfast, 1.00, text = "fast", fontsize = 20, color = COL_TEXT, align = (:center, :top))

    axislegend(
        ax_freq,
        [q1, q2],
        [
            L"\mathrm{Original\ community}\,\left(R_{\omega}\right)",
            L"\mathrm{Impact\ of\ structural\ change}\ P\,\left(R_{\omega} P R_{\omega}\right)"
        ],
        position = (0.0, 0.0),
        framevisible = true,
        bgcolor = RGBAf(1, 1, 1, 0.90),
        labelsize = 18,
        padding = (10, 10, 10, 10)
    )

    textlabel!(
        ax_freq,
        [12.0], [0.75],
        text = [L"R_{\omega} = (i\omega - A)^{-1}"],
        fontsize = 20,
        text_color = COL_TEXT,
        text_align = (:center, :center),
        shape = Rect2f(0, 0, 1, 1),
        background_color = RGBAf(1, 1, 1, 0.95),
        strokecolor = COL_TEXT,
        strokewidth = 1.2,
        padding = 8
    )

    text!(
        ax_freq,
        -0.115, 0.395,
        text = "ω",
        space = :relative,
        rotation = π/2,
        fontsize = 15,
        color = COL_TEXT,
        align = (:center, :center)
    )

    # ------------------------------------------------------------
    # Column sizing
    # ------------------------------------------------------------
    colsize!(top, 1, Relative(0.40))
    colsize!(top, 2, Relative(0.60))

    colsize!(bottom, 1, Relative(0.39))
    colsize!(bottom, 2, Relative(0.23))
    colsize!(bottom, 3, Relative(0.38))

    # ------------------------------------------------------------
    # Row headers
    # ------------------------------------------------------------
    Label(fig[0, 1], "Structure", fontsize = 28, color = COL_TEXT, tellwidth = false)
    Label(fig[2, 1, Top()], "Dynamics", fontsize = 28, color = COL_TEXT, tellwidth = false)

    display(fig)

    # save("main_illustration_inset_histogram.png", fig, px_per_unit = 2)
end