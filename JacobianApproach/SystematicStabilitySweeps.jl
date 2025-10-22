################################################################################
# SYSTEMATIC STABILITY SWEEPS
# - explores abundance heterogeneity
# - explores degree heterogeneity
# - explores trophic symmetry correlation
# Author: [Your Name]
################################################################################

using Random, DataFrames, Statistics, CairoMakie

################################################################################
# Helper: multi-metric plotting for one sweep
################################################################################
function plot_metrics(df; xvar::Symbol, title::String)
    metrics = [:res_full, :rea_full, :rmed_full]
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

################################################################################
# Sweep 1 — Abundance heterogeneity
################################################################################
println("Running abundance heterogeneity sweep...")
df_abund = run_sweep_stable(
    ; modes=[:TR],
      S_vals=[120],
      u_cv_vals=[0.0, 0.3, 0.5, 1.0, 2.0, 3.0],
      degree_families=[:uniform],
      rho_sym_vals=[0.0],
      number_of_combinations=200,
      reps_per_combo=2,
      seed=11
)
plot_metrics(df_abund; xvar=:u_cv, title="Effect of abundance heterogeneity")

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
      rho_sym_vals=[0.0],
      number_of_combinations=200,
      reps_per_combo=2,
      seed=22
)
plot_metrics(df_degree; xvar=:deg_cv_all, title="Effect of degree heterogeneity")

################################################################################
# Sweep 3 — Trophic symmetry (pairwise correlation)
################################################################################
println("Running symmetry sweep...")
df_sym = run_sweep_stable(
    ; modes=[:TR],
      S_vals=[120],
      u_cv_vals=[1.0],
      degree_families=[:uniform],
      rho_sym_vals=range(0,1,length=10),
      number_of_combinations=200,
      reps_per_combo=2,
      seed=33
)
plot_metrics(df_sym; xvar=:rho_sym, title="Effect of trophic symmetry")

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

save("summary_abundance.csv", summary_abund)
save("summary_degree.csv", summary_degree)
save("summary_symmetry.csv", summary_sym)

println("✅ All sweeps complete. CSV summaries and plots generated.")
