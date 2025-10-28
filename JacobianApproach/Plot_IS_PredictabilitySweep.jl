# -------------------------------
# Plotting helpers
# -------------------------------
function plot_predictability_vs_IS(summary; metric::String, title::String="Predictability vs IS", modes=[:TR])
    steps = ["reshuf", "uni", "rarer", "rew"] #unique(summary.step)
    fig = Figure(size=(1000, 420))
    cols = length(steps)
    for (j, mode) in enumerate(modes)
        ax = Axis(
            fig[j,1]; 
            xlabel="IS_target (mean |A|)", ylabel="R² to y=x", title="$title — $mode — $metric",
            # ylimits=(-0.5, 1.05)
        )
        for step in stepsn
            d = filter(row -> row.mode==mode && row.metric==metric && !row.stable_only && row.step==step, summary)
            isempty(d) && continue
            p = sortperm(d.IS_target)
            lines!(ax, d.IS_target[p], d.r2_corrected[p]; label=step)
            scatter!(ax, d.IS_target[p], d.r2_corrected[p])
        end
        axislegend(ax; position=:rb)
        limits!(ax, (0.02, 1.2), (-0.1, 1.1))
    end
    display(fig)
end

"""
plot_r2_grid(summary; metric=:res_full, steps=[:rew, :reshuf, :rarer, :uni], modes=[:TR], title="Predictability grid")

Creates a 4×4 (or smaller) grid of R² vs IS_target plots, one per step.
Each subplot shows how well the chosen metric (e.g. :res_full) correlates with each step.
"""
function plot_r2_grid(summary; metric::Symbol=:res_full, steps::Vector{Symbol}=[:rew, :reshuf, :rarer, :uni],
                      modes::Vector{Symbol}=[:TR], title::String="Predictability grid")

    # colors per mode (if you want to compare modes in one plot)
    palette = Makie.wong_colors()[1:length(modes)]
    step_labels = string.(steps)

    fig = Figure(size=(950, 650))
    Label(fig[0, 1:3], title; fontsize=18, font=:bold, halign=:center)

    for (i, step) in enumerate(steps)
        ax = Axis(
            fig[(i-1) ÷ 3 + 1, (i-1) % 3 + 1];
            xlabel="IS_target", ylabel="R²",
            title=uppercase(string(step)),
            limits=(nothing, (-0.2, 1.05)),
            ygridvisible=true, xgridvisible=false
        )

        for (mi, mode) in enumerate(modes)
            df_sub = filter(row ->
                row.metric == string(metric) &&
                row.step == string(step) &&
                row.mode == mode &&
                !row.stable_only,
                summary)

            isempty(df_sub) && continue
            p = sortperm(df_sub.IS_target)
            lines!(ax, df_sub.IS_target[p], df_sub.r2_corrected[p]; color=palette[mi], label=string(mode))
            scatter!(ax, df_sub.IS_target[p], df_sub.r2_corrected[p]; color=palette[mi], markersize=6)
        end

        if i == 1
            axislegend(ax; position=:rb)
        end
    end

    display(fig)
end