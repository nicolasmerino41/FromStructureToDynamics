begin
    fig = Figure(size = (1600, 1000))

    allvals = vcat([vec(A) for (_, A) in cases]...)
    mx = maximum(abs.(allvals))

    # Row 1: small network drawings
    for (j, (name, A)) in enumerate(cases)
        ax_net = Axis(fig[1, j])
        draw_colored_network!(ax_net, A, name, mx; node_size=14, line_width=2.0)
    end

    # Row 2: matrix heatmaps
    for (j, (name, A)) in enumerate(cases)
        ax = Axis(
            fig[2, j],
            title = name,
            xlabel = "Species j",
            ylabel = "Species i",
            aspect = DataAspect()
        )

        heatmap!(
            ax, 1:size(A,2), 1:size(A,1), A;
            colorrange = (-mx, mx),
            colormap = :balance
        )

        ylims!(ax, size(A,1) + 0.5, 0.5)
        xlims!(ax, 0.5, size(A,2) + 0.5)

        ax.xticksvisible = false
        ax.yticksvisible = false
        ax.xticklabelsvisible = false
        ax.yticklabelsvisible = false
        ax.xminorticksvisible = false
        ax.yminorticksvisible = false
        ax.xgridvisible = false
        ax.ygridvisible = false
    end

    # Colorbar for networks + matrices
    Colorbar(
        fig[2, 5],
        limits = (-mx, mx),
        colormap = :balance,
        label = "Interaction strength"
    )

    # Row 3: intrinsic sensitivity
    ax_intr = Axis(
        fig[3, 1:4],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Intrinsic sensitivity",
        xscale = log10,
    )

    for (name, _) in cases
        lines!(ax_intr, ωs, intr_profiles[name], linewidth=3, label=name)
    end
    axislegend(ax_intr, position=:rt)

    # rowsize!(fig.layout, 1, Relative(0.45))
    # rowsize!(fig.layout, 2, Relative(1.0))
    # rowsize!(fig.layout, 3, Relative(0.9))

    # rowgap!(fig.layout, 10)
    # colgap!(fig.layout, 12)

    display(fig)
end