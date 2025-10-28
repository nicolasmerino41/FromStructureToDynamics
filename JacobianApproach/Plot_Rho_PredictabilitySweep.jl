# -------------------------------
# Plotting helpers (rho as x-axis)
# -------------------------------
function plot_predictability_vs_rho(summary; metric::String, title::String="Predictability vs ρ", modes=[:TR])
    steps = ["reshuf", "uni", "rarer", "rew"]  # focus subset; change as needed
    fig = Figure(size=(1000, 420))
    for (j, mode) in enumerate(modes)
        ax = Axis(
            fig[j,1];
            xlabel="ρ (trophic antisymmetry of magnitudes)", ylabel="R² to y=x", title="$title — $mode — $metric",
            limits=((0.0, 1.0), (-0.1, 1.1))
        )
        for step in steps
            d = filter(row -> row.mode==mode && row.metric==metric && !row.stable_only && row.step==step, summary)
            isempty(d) && continue
            d.r2_corrected = copy(d.r2)
            d.r2_corrected[d.r2_corrected .< 0] .= 0.0
            p = sortperm(d.rho_sym)
            lines!(ax, d.rho_sym[p], d.r2_corrected[p]; label=step)
            scatter!(ax, d.rho_sym[p], d.r2_corrected[p])
        end
        axislegend(ax; position=:rb)
    end
    display(fig)
end

"""
plot_r2_grid_rho(summary; metric=:res_full, steps=[:rew, :reshuf, :rarer, :uni], modes=[:TR], title="Predictability grid vs ρ")

Creates a grid of R² vs rho_sym plots, one per step.
Each subplot shows how well the chosen metric (e.g. :res_full) correlates with each step.
"""
function plot_r2_grid_rho(summary; metric::Symbol=:res_full, steps::Vector{Symbol}=[:rew, :reshuf, :rarer, :uni],
                          modes::Vector{Symbol}=[:TR], title::String="Predictability grid vs ρ")

    palette = Makie.wong_colors()[1:length(modes)]
    fig = Figure(size=(950, 650))
    Label(fig[0, 1:3], title; fontsize=18, font=:bold, halign=:center)

    for (i, step) in enumerate(steps)
        ax = Axis(
            fig[(i-1) ÷ 3 + 1, (i-1) % 3 + 1];
            xlabel="ρ", ylabel="R²",
            title=uppercase(string(step)),
            limits=((0.0, 1.0), (-0.2, 1.05)),
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
            p = sortperm(df_sub.rho_sym)
            lines!(ax, df_sub.rho_sym[p], df_sub.r2[p]; color=palette[mi], label=string(mode))
            scatter!(ax, df_sub.rho_sym[p], df_sub.r2[p]; color=palette[mi], markersize=6)
        end

        if i == 1
            axislegend(ax; position=:rb)
        end
    end

    display(fig)
end