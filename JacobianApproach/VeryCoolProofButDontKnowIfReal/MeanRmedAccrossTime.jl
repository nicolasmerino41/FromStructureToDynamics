function mean_curve(curves::Vector{Vector{Float64}})
    S = length(curves[1])
    avg = zeros(S)
    for k in 1:S
        avg[k] = mean(c[k] for c in curves)
    end
    return avg
end

function plot_rmed_mean_grid(results, t_vals;
        q_targets = sort(collect(keys(results))),
        figsize = (1600, 1400)
    )

    fig = Figure(size=figsize)

    rows, cols = 3, 3
    idx = 1

    for q in q_targets
        curves = results[q]   # 30 curves

        mean_c = mean_curve(curves)

        r = (idx-1) ÷ cols + 1
        c = (idx-1) % cols + 1

        ax = Axis(fig[r, c];
            title = "q = $(round(q, digits=3))",
            xlabel = "t",
            ylabel = "mean rmed(t)",
            xscale = log10,
        )

        lines!(ax, t_vals, mean_c; color=:black, linewidth=3)

        idx += 1
    end

    display(fig)
end

plot_rmed_mean_grid(results, t_vals)

function plot_rmed_mean_grid_with_reference(results, t_vals;
        q_targets = sort(collect(keys(results))),
        q_ref = 0.0,
        figsize = (1100,720),
        title = ""
    )

    # Compute reference mean curve
    ref_mean = mean_curve(results[q_ref])

    fig = Figure(size=figsize)
    Label(fig[0, 1:3], title, fontsize=18, font=:bold, halign=:center)

    rows, cols = 3, 3
    idx = 1

    for q in q_targets
        curves = results[q]
        mean_c = mean_curve(curves)

        r = (idx-1) ÷ cols + 1
        c = (idx-1) % cols + 1

        ax = Axis(fig[r, c];
            title = "q = $(round(q, digits=3))",
            xlabel = "t",
            ylabel = "mean rmed(t)",
            xscale = log10,
        )

        # ----- reference mean (red) -----
        lines!(ax, t_vals, ref_mean; color=:red, linewidth=3)

        # ----- mean of current q (black) -----
        lines!(ax, t_vals, mean_c; color=:black, linewidth=3)

        idx += 1
    end

    display(fig)
end

plot_rmed_mean_grid_with_reference(results, t_vals)