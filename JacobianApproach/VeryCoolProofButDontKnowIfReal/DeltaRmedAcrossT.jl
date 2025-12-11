function delta_curve(mean_q, mean_ref)
    return mean_q .- mean_ref
end

function plot_rmed_delta_grid(results, t_vals;
        q_targets = sort(collect(keys(results))),
        # q_ref = 0.0,
        figsize = (1100,720),
        title = ""
    )
    kis = keys(results)
    q_ref = minimum(kis)
    # reference mean curve
    ref_mean = mean_curve(results[q_ref])

    # --- 1) Compute global max Δ across all q ---
    global_max = 0.0
    global_min = 0.0
    for q in q_targets
        mean_q = mean_curve(results[q])
        Δ = delta_curve(mean_q, ref_mean)
        m = maximum(Δ)
        lso = minimum(Δ)
        if m > global_max
            global_max = m
        end
        if lso < global_min
            global_min = lso
        end
    end

    fig = Figure(size=figsize)
    Label(fig[0, 1:3], title, fontsize=18, font=:bold, halign=:center)
    rows, cols = 3, 3
    idx = 1

    for q in q_targets
        curves_q = results[q]
        mean_q = mean_curve(curves_q)
        Δ = delta_curve(mean_q, ref_mean)

        r = (idx-1) ÷ cols + 1
        c = (idx-1) % cols + 1

        ax = Axis(fig[r, c];
            title = "q = $(round(q, digits=3))",
            xlabel = "t",
            ylabel = "|Δ rmed(t)|",
            xscale = log10,
        )

        lines!(ax, t_vals, Δ; color=:blue, linewidth=3)
        ylims!(ax, global_min-abs(global_max*0.1), global_max+global_max*0.1)
        idx += 1
    end

    display(fig)
end

plot_rmed_delta_grid(results, t_vals)