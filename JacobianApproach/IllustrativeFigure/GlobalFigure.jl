using CairoMakie
using Random

begin
    # ============================================================
    # MAIN ILLUSTRATION — Structure → dynamics → scale
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
    COL_AXIS    = RGBf(0.15, 0.15, 0.15)

    COL_NODE    = RGBf(0.70, 0.70, 0.70)
    COL_EDGE    = RGBf(0.68, 0.68, 0.68)

    COL_ORIG    = RGBf(0.40, 0.49, 0.61)
    COL_MOD     = RGBf(0.72, 0.50, 0.30)

    COL_LIGHT   = RGBf(0.92, 0.92, 0.92)
    COL_MID     = RGBf(0.82, 0.82, 0.82)
    COL_DARK    = RGBf(0.70, 0.70, 0.70)

    COL_ORANGE_L = RGBAf(0.83, 0.57, 0.32, 0.10)
    COL_ORANGE_S = RGBAf(0.83, 0.57, 0.32, 0.75)

    COL_FAINT   = RGBAf(0.7, 0.7, 0.7, 1.0)

    BAND_SLOW   = RGBAf(0.50, 0.60, 0.82, 0.30)
    BAND_MID    = RGBAf(0.90, 0.78, 0.50, 0.28)
    BAND_FAST   = RGBAf(0.60, 0.78, 0.58, 0.30)

    # ------------------------------------------------------------
    # Data for panel 1a: network
    # ------------------------------------------------------------
    # More heterogeneous, sparse, realistic-looking layout

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
#---------------------------------------------------------
    # Data for panel 1b: matrix
    # Match matrix pattern to the network above
    # ------------------------------------------------------------
    n = length(pts)
    M = fill(0.0, n, n)

    # diagonal
    for i in 1:n
        M[i, i] = 0.55
    end

    # network-consistent interactions
    for (i, j) in base_edges
        M[i, j] = 0.72
        M[j, i] = 0.72
    end

    for (i, j) in mod_edges
        M[i, j] = 0.98
        M[j, i] = 0.98
    end

    # slight intensity variation for a cleaner non-uniform look
    for (i, j) in [(1,2), (2,3), (3,4), (6,7), (7,8), (4,10), (3,6)]
        M[i, j] = 0.82
        M[j, i] = 0.82
    end

    # highlight the modified node row
    highlight_row = mod_node

    # ------------------------------------------------------------
    # Data for panel 2: time response
    # ------------------------------------------------------------
    t = range(0, 10, length=900)
    y_orig = gamma_bump(t; t0=1.2,  k=3.2, θ=0.78, A=1.0)
    y_mod  = gamma_bump(t; t0=1.35, k=3.6, θ=0.84, A=0.88)

    ggate = t .>= 0.0

    # 1) left-heavy tail
    c1 = gate .* (
        0.36 .* exp.(-0.5 .* ((t .- 4.0) ./ 0.95).^2) .* (1 .- 0.85 .* tanh.((t .- 2.8) ./ 0.85))
    )
    c1 .-= minimum(c1)
    c1 ./= maximum(c1)
    c1 .*= 0.42

    # 2) same shape, taller, shifted to the right
    # c2 further right
    c2 = gate .* (
        0.36 .* exp.(-0.5 .* ((t .- 4.6) ./ 1.0).^2) .* (1 .+ 0.9 .* tanh.((t .- 4.6) ./ 0.9))
    )

    # c3 further right
    c3 = gate .* (
        0.18 .* exp.(-0.5 .* ((t .- 6.8) ./ 1.55).^2) .+
        0.10 .* exp.(-0.5 .* ((t .- 7.6) ./ 1.10).^2)
    )

    # 4) much taller early sharp peak, then decays
    x4 = max.(t .- 1.35, 0.0)
    c4 = gate .* ((x4 .^ 0.28) .* exp.(-x4 ./ 0.42))
    c4 ./= maximum(c4)
    c4 .*= 0.62
    # ------------------------------------------------------------
    # Data for bridge
    # ------------------------------------------------------------
    u = range(0, 1, length=900)

    f_slow = 1.2
    f_mid  = 4.0
    f_fast = 11.0

    slow_orig = bandwave(u; f=f_slow, A=0.05, phase=0.15)
    mid_orig  = bandwave(u; f=f_mid,  A=0.035, phase=0.35)
    fast_orig = bandwave(u; f=f_fast, A=0.022, phase=0.10)

    slow_mod = bandwave(u; f=f_slow, A=0.035, phase=0.55)
    mid_mod  = bandwave(u; f=f_mid,  A=0.020, phase=0.65)
    fast_mod = bandwave(u; f=f_fast, A=0.010, phase=0.80)

    # ------------------------------------------------------------
    # Data for panel 4: frequency profile
    # ------------------------------------------------------------
    ω = exp.(range(log(1e-2), log(1e2), length=900))
    x = log10.(ω)

    S_orig =
    0.4 .+                                         # lower overall level
    gauss(x, -0.95, 0.25, 0.18) .+                  # smaller low-frequency peak
    gauss(x, -0.18, 0.28, 0.46) .-                  # lower main peak
    0.10 ./ (1 .+ exp.(-4.0 .* (x .- 0.45)))       # gentle high-frequency suppression
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
    fig = Figure(size = (1800, 1050), backgroundcolor = COL_BG)

    top = fig[1, 1] = GridLayout()
    bottom = fig[2, 1] = GridLayout()

    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 24)

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

    xlims!(ax_net, -0.08, 1.18)
    ylims!(ax_net, -0.05, 1.02)

    for (i, j) in base_edges
        p1, p2 = pts[i], pts[j]
        lines!(ax_net, [p1[1], p2[1]], [p1[2], p2[2]], color = COL_EDGE, linewidth = 2.4)
    end

    for (i, j) in mod_edges
        p1, p2 = pts[i], pts[j]
        lines!(ax_net, [p1[1], p2[1]], [p1[2], p2[2]], color = COL_MOD, linewidth = 5.0)
    end

    xs = first.(pts)
    ys_pts = last.(pts)

    scatter!(ax_net, xs, ys_pts,
        color = COL_NODE,
        markersize = 24
    )

    scatter!(ax_net, [xs[mod_node]], [ys_pts[mod_node]],
        color = COL_MOD,
        markersize = 28
    )

    x_focal = xs[mod_node]
    y_focal = ys_pts[mod_node]

    text!(
        ax_net,
        x_focal + 0.25, y_focal -0.09,
        text = "modifying\nstructure",
        fontsize = 20,
        color = COL_TEXT,
        align = (:left, :center)
    )

    arrows!(
        ax_net,
        [x_focal+0.03], [y_focal-0.01],
        [0.19], [-0.07],
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

    xlims!(ax_mat, -1.2, n + 1.2)
    ylims!(ax_mat, -1.2, n + 1.4)

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
            poly!(ax_mat,
                cellrect((j - 1) + pad, (i - 1) + pad, 1 - 2pad, 1 - 2pad),
                color = c, strokecolor = :transparent)
        end
    end

    lw = 3.0
    lines!(ax_mat, [-0.35, -0.35], [0, n], color = COL_TEXT, linewidth = lw)
    lines!(ax_mat, [-0.35, 0.15], [0, 0], color = COL_TEXT, linewidth = lw)
    lines!(ax_mat, [-0.35, 0.15], [n, n], color = COL_TEXT, linewidth = lw)
    lines!(ax_mat, [n+0.35, n+0.35], [0, n], color = COL_TEXT, linewidth = lw)
    lines!(ax_mat, [n-0.15, n+0.35], [0, 0], color = COL_TEXT, linewidth = lw)
    lines!(ax_mat, [n-0.15, n+0.35], [n, n], color = COL_TEXT, linewidth = lw)

    text!(ax_mat, n/2, -0.72,
        text = "Community matrix",
        fontsize = 24, color = COL_TEXT,
        align = (:center, :center))

    # ============================================================
    # BOTTOM LEFT — TIME RESPONSE
    # ============================================================
    ax_time = Axis(
        bottom[1, 1],
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

    xlims!(ax_time, 0, 10)
    ylims!(ax_time, 0, 1.18)

    ax_time.yticksvisible = false
    ax_time.yticklabelsvisible = false
    ax_time.xticksvisible = false
    ax_time.xticklabelsvisible = false

    hidedecorations!(ax_time, ticks = false, ticklabels = false, label = false)

    lines!(ax_time, t, c1, color = COL_FAINT, linewidth = 3)
    lines!(ax_time, t, c2, color = COL_FAINT, linewidth = 3)
    lines!(ax_time, t, c3, color = COL_FAINT, linewidth = 3)
    lines!(ax_time, t, c4, color = COL_FAINT, linewidth = 3)

    lbl_orig = "Original community"

    lines!(
        ax_time, t, y_orig,
        color = COL_ORIG,
        linewidth = 4,
        label = lbl_orig
    )
    lines!(ax_time, t, y_mod,  color = COL_MOD,  linewidth = 4, label = "Modified community")
    lbl_test = L"R_{\omega} = (i\omega I - A)^{-1}"

    arrows!(ax_time, [1.1], [1.02], [0.0], [-0.78],
        color = :black, linewidth = 2.5, arrowsize = 18, lengthscale = 1.0)

    text!(ax_time, 1.15, 1.04,
        text = "Pulse perturbation",
        fontsize = 15, color = COL_TEXT,
        align = (:center, :bottom))

    axislegend(ax_time;
        position = :rt,
        framevisible = false,
        labelsize = 18,
        patchsize = (34, 18))

    hidespines!(ax_time, :t, :r)

    # ============================================================
    # BOTTOM MIDDLE — SCALE BRIDGE
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

    x_strip0 = 0.2
    x_strip1 = 0.8
    
    x_label = 0.12
    
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

    # text!(ax_bridge, 0.52, 0.94,
    #     text = "Re-express\nby scale",
    #     fontsize = 18, color = COL_TEXT,
    #     align = (:center, :top))

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

    xlims!(ax_freq, minimum(ω), maximum(ω))
    ylims!(ax_freq, 0, 1.05)

    ax_freq.yticksvisible = false
    ax_freq.yticklabelsvisible = false
    ax_freq.xticksvisible = false
    ax_freq.xticklabelsvisible = false

    vspan!(ax_freq, 1e-2, 1e-1, color = BAND_SLOW)
    vspan!(ax_freq, 1e-1, 1e0, color = BAND_MID)
    vspan!(ax_freq, 1e0, 1e2, color = BAND_FAST)

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
    # --- callout target point in data coordinates ---
    px, py = 1.2e1, 0.22

    # --- box position in data coordinates ---
    bx0, bx1 = 1.8, 8.5e1
    by0, by1 = 1.3, 0.18

    textlabel!(
        ax_freq,
        [sqrt(bx0 * bx1)], [0.75],
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
    # ------------------------------------------------------------
    # Now size columns after they exist
    # ------------------------------------------------------------
    colsize!(top, 1, Relative(0.42))
    colsize!(top, 2, Relative(0.58))

    colsize!(bottom, 1, Relative(0.36))
    colsize!(bottom, 2, Relative(0.28))
    colsize!(bottom, 3, Relative(0.36))

    # ------------------------------------------------------------
    # Row headers
    # ------------------------------------------------------------
    Label(fig[0, 1], "Structure", fontsize = 28, color = COL_TEXT, tellwidth = false)
    Label(fig[2, 1, Top()], "Dynamics", fontsize = 28, color = COL_TEXT, tellwidth = false)

    display(fig)

    # save("main_illustration.png", fig, px_per_unit = 2)
end