################################################################################
# SYSTEMATIC STABILITY SWEEPS
# - explores abundance heterogeneity
# - explores degree heterogeneity
# - explores trophic symmetry correlation
################################################################################
using Random, DataFrames, Statistics, CairoMakie, Printf, CategoricalArrays

################################################################################
# Helper: multi-metric plotting for one sweep
################################################################################
function plot_metrics(df; xvar::Symbol, title::String)
    metrics = [:res_full, :rea_full, :rmed_full]
    # metrics = [:res_rel_to_min_u_full, :rea_rel_to_min_u_full, :rmed_rel_to_min_u_full]
    labels  = ["Resilience", "Reactivity", "Median Return Rate"]
    fig = Figure(size=(1000, 400))
    for (i, metric) in enumerate(metrics)
        ax = Axis(fig[1,i], xlabel=string(xvar), ylabel=labels[i], title=labels[i])
        scatter!(ax, df[!, xvar], df[!, metric]; color=:dodgerblue, alpha=0.5)
        lines!(ax, sort(df[!, xvar]), sort(df[!, metric]); color=:black, linewidth=1)
    end
    Label(fig[0, 1:3], title; fontsize=18, halign=:center)
    display(fig)
end

function plot_metrics_box(df; xvar::Symbol, title::String)
    # metrics = [:res_rel_to_min_u_full, :rea_rel_to_min_u_full, :rmed_rel_to_min_u_full]
    metrics = [:res_full, :rea_full, :rmed_full]
    labels  = ["Resilience", "Reactivity", "Median Return Rate"]
    fig = Figure(size = (1000, 400))

    for (i, (m, label)) in enumerate(zip(metrics, labels))
        ax = Axis(
            fig[1, i]; xlabel=string(xvar), ylabel=label, title=label,
            xticklabelsize=10, yticklabelsize=10,
            xlabelsize=12, ylabelsize=12
        )

        x, y = df[!, xvar], df[!, m]
        if eltype(x) <: Number
            edges = range(extrema(x)...; length=11)
            bin_labels = [@sprintf("%.2f–%.2f", edges[i], edges[i+1]) for i in 1:10]
            bin_idx = map(xi -> findfirst(xi .<= edges[2:end] .&& xi .> edges[1:end-1]) |> x->x===nothing ? 1 : x, x)
            boxplot!(ax, bin_idx, y; color=:dodgerblue)
            ax.xticks = (1:10, bin_labels)
        else
            cats = unique(string.(x))
            idx = [findfirst(==(string(xi)), cats) for xi in x]
            boxplot!(ax, idx, y; color=:dodgerblue)
            ax.xticks = (1:length(cats), cats)
        end
    end

    Label(fig[0, 1:3], title; fontsize=18, halign=:center)
    display(fig)
end
################################################################################
# Sweep 1 — Abundance heterogeneity
################################################################################
println("Running abundance heterogeneity sweep...")
df_abund = run_sweep_stable(
    ; modes=[:TR],
      S_vals=[120],
      u_cv_vals=[0.0, 0.3, 0.5, 1.0, 2.0, 3.0],
      degree_families=[:lognormal],
      rho_sym_vals=[1.0],
      number_of_combinations=400,
      reps_per_combo=2,
      seed=11
)
df_abund = filter(row -> row.shrink_alpha > 0.5, df_abund)
plot_metrics(df_abund; xvar=:u_cv, title="Effect of abundance heterogeneity (lognormal abundance & rho=1)")
plot_metrics_box(df_abund; xvar=:u_cv, title="Effect of abundance heterogeneity (lognormal abundance & rho=1)")
################################################################################
# Sweep 2 — Degree heterogeneity
################################################################################
println("Running degree heterogeneity sweep...")
df_degree = run_sweep_stable(
    ; modes=[:TR],
      S_vals=[120],
      u_cv_vals=[1.0],
      degree_families=[:uniform, :lognormal, :pareto],
      deg_cv_vals=[0.0, 0.5, 1.0, 2.0],
      rho_sym_vals=[1.0],
      number_of_combinations=200,
      reps_per_combo=2,
      seed=22
)
plot_metrics(df_degree; xvar=:deg_cv_all, title="Effect of degree heterogeneity (rho=1)")
plot_metrics_box(df_degree; xvar=:deg_cv_all, title="Effect of degree heterogeneity (rho=1)")
################################################################################
# Sweep 3 — Trophic symmetry (pairwise correlation)
################################################################################
println("Running symmetry sweep...")
df_sym = run_sweep_stable(
    ; modes=[:TR],
      S_vals=[120],
      u_cv_vals=[0.1],
      degree_families=[:uniform],
      rho_sym_vals=range(0,1,length=10),
      number_of_combinations=200,
      reps_per_combo=2,
      seed=33
)
plot_metrics(df_sym; xvar=:rho_sym, title="Effect of trophic symmetry")
plot_metrics_box(df_sym; xvar=:rho_sym, title="Effect of trophic symmetry")
################################################################################
# Sweep 4 — Mean interaction strength (IS)
################################################################################
println("Running interaction strength (IS) sweep...")
df_IS = run_sweep_stable(
    ; modes=[:TR],
      S_vals=[120],
      u_cv_vals=[1.0],
      degree_families=[:lognormal],
      rho_sym_vals=[1.0],
      mean_abs_vals=range(0.01, 1.3, length=12),
      number_of_combinations=200,
      reps_per_combo=2,
      seed=44
)
# optional filtering, same as in abundance case
df_IS = filter(row -> row.shrink_alpha > 0.5, df_IS)
plot_metrics(df_IS; xvar=:mean_abs, title="Effect of mean interaction strength (ρ=1.0, lognormal degrees)")
plot_metrics_box(df_IS; xvar=:mean_abs, title="Effect of mean interaction strength (ρ=1.0, lognormal degrees)")
################################################################################
# Combined summary — mean ± SD across sweeps
################################################################################
function summarize_sweep(df, xvar::Symbol)
    g = groupby(df, xvar)
    summary = combine(g, [:res_full, :rea_full, :rmed_full] .=> mean .=> [:res_m, :rea_m, :rmed_m],
                         [:res_full, :rea_full, :rmed_full] .=> std .=> [:res_sd, :rea_sd, :rmed_sd])
    return summary
end

summary_abund  = summarize_sweep(df_abund, :u_cv)
summary_degree = summarize_sweep(df_degree, :deg_cv_all)
summary_sym    = summarize_sweep(df_sym, :rho_sym)
summary_IS = summarize_sweep(df_IS, :mean_abs)

save("summary_abundance.csv", summary_abund)
save("summary_degree.csv", summary_degree)
save("summary_symmetry.csv", summary_sym)

println("✅ All sweeps complete. CSV summaries and plots generated.")

using CairoMakie
using DataFrames

function plot_shrink_alpha(df::DataFrame, x_var::Symbol; title::Union{String, Nothing}=nothing)
    # 1. Validate input
    if in(x_var, names(df))
        throw(ArgumentError("Column $(x_var) not found in DataFrame. Available columns: $(names(df))"))
    end
    if !in("shrink_alpha", names(df))
        throw(ArgumentError("Column :shrink_alpha not found in DataFrame."))
    end

    # 2. Drop missing data for clean plotting
    clean_df = dropmissing(df[:, [x_var, :shrink_alpha]])

    # 3. Create the scatter plot
    fig = Figure(; size=(800, 600))
    ax = Axis(
        fig[1, 1];
        xlabel = String(x_var),
        ylabel = "shrink_alpha",
        title = isnothing(title) ? "shrink_alpha vs $(x_var)" : title,
        # grid = (visible = true,)
    )

    scatter!(ax, clean_df[:, x_var], clean_df[:, :shrink_alpha];
             markersize = 8, color = :blue, strokewidth = 0.5, alpha = 0.8)

    display(fig)
end
plot_shrink_alpha(df_degree, :deg_cv_all; title="Shrinkage effect of degree heterogeneity")
plot_shrink_alpha(df_abund, :u_cv; title="Shrinkage effect of abundance heterogeneity")
plot_shrink_alpha(df_sym, :rho_sym; title="Shrinkage effect of trophic symmetry")