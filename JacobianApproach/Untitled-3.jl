
using CairoMakie, Statistics

# ---------- helpers ----------
# absolute relative error with a small clip to avoid blowups near zero
relerr(y, f; clip=0.10) = begin
    if !(isfinite(y) && isfinite(f)); return NaN; end
    τ = max(abs(f)*clip, 1e-8)
    abs(y - f) / max(abs(f), τ)
end

# ordinary least squares y = a + b x
function simple_linfit(x::AbstractVector, y::AbstractVector)
    n = length(x)
    μx, μy = mean(x), mean(y)
    Sxx = sum((xi-μx)^2 for xi in x)
    Sxy = sum((xi-μx)*(yi-μy) for (xi,yi) in zip(x,y))
    b = Sxx == 0 ? 0.0 : Sxy / Sxx
    a = μy - b*μx
    return a, b
end

# map step keyword -> column suffixes
const STEP_KEYS = Dict(
    :row      => :row,        # Step 1: Row mean
    :thr      => :thr,        # Step 2: Threshold
    :uni      => :uni,        # Step 3: Uniform u*
    :thr_row  => :thr_row,    # Step 4: Threshold -> Row mean
    :row_uni  => :row_uni,    # Step 5: Row mean + Uniform
    :thr_uni  => :thr_uni     # Step 6: Threshold + Uniform
)

# nice, stable color set
const STEP_COLORS = Dict(
    :row     => (:steelblue,    "Row mean"),
    :thr     => (:darkorange,   "Threshold"),
    :uni     => (:seagreen,     "Uniform u*"),
    :thr_row => (:firebrick,    "Threshold → Row"),
    :row_uni => (:orchid4,      "Row + Uniform"),
    :thr_uni => (:goldenrod,    "Threshold + Uniform")
)

"""
plot_error_vs_structure(
    df; xvars=[:conn_real,:deg_cv_all,:u_cv,:IS_real],
    metric=:resilience, steps=[:row,:thr,:uni,:thr_row,:row_uni,:thr_uni],
    resolution=(1100, 700), sample_frac=1.0, max_points=200_000
)

- `metric` ∈ (:resilience, :reactivity)
- `xvars` choose among :conn_real, :deg_cv_in, :deg_cv_out, :deg_cv_all, :u_cv, :IS_real
- `steps` choose which simplified models to plot
- Scatters of |relative error| and per-step OLS regression lines
"""
function plot_error_vs_structure(df::DataFrame;
    xvars = [:conn_real, :deg_cv_all, :u_cv, :IS_real],
    metric::Symbol = :resilience,
    steps = [:row,:thr,:uni,:thr_row,:row_uni,:thr_uni],
    resolution = (1100, 700),
    sample_frac::Float64 = 1.0,
    max_points::Int = 200_000
)
    # pick base and step column name templates
    base_col = metric === :resilience ? :res_full : :rea_full
    step_prefix = metric === :resilience ? :res_ : :rea_

    # subsample for speed/clarity if requested
    idxs = collect(1:nrow(df))
    if sample_frac < 1.0
        nkeep = max(1, min(length(idxs), round(Int, sample_frac*length(idxs))))
        idxs = rand(idxs, nkeep)
    end

    fig = Figure(size=resolution)

    for (j, xcol) in enumerate(xvars)
        ax = Axis(fig[1, j];
            title = string(xcol),
            xlabel = string(xcol),
            ylabel = "|relative error|",
            xgridvisible=false, ygridvisible=false
        )

        # draw per step
        for st in steps
            suffix = STEP_KEYS[st]
            ycol = Symbol(step_prefix, suffix)
            color, label = STEP_COLORS[st]

            # collect finite pairs (x, |relerr|)
            xs = Float64[]; ys = Float64[]
            for i in idxs
                x = df[i, xcol]
                f = df[i, base_col]
                s = df[i, ycol]
                if x isa Real && f isa Real && s isa Real
                    e = min(1.0,relerr(s, f; clip=0.10))
                    if isfinite(x) && isfinite(e)
                        push!(xs, float(x)); push!(ys, float(e))
                    end
                end
            end

            if isempty(xs); continue; end

            # cap points to reduce overplot
            if length(xs) > max_points
                keep = rand(1:length(xs), max_points; replace=false)
                xs = xs[keep]; ys = ys[keep]
            end

            scatter!(ax, xs, ys; markersize=4, alpha=0.25, color=color, label=label)

            # regression line
            a, b = simple_linfit(xs, ys)
            xmin, xmax = minimum(xs), maximum(xs)
            xline = range(xmin, xmax; length=200)
            yline = a .+ b .* xline
            lines!(ax, xline, yline; color=color, linewidth=2)
        end

        axislegend(ax; position=:rt, framevisible=false)
    end

    display(fig)
end

# Resilience errors vs structure (all steps)
plot_error_vs_structure(
    df_tr; metric=:resilience,
    xvars=[:u_cv,:IS_real],
    steps=[:row,:thr,:uni,:thr_row,:row_uni,:thr_uni]
)

# Reactivity, only a subset of steps
plot_error_vs_structure(
    df; metric=:reactivity,
    xvars=[:conn_real,:deg_cv_in,:deg_cv_out,:IS_real],
    steps=[:row,:thr,:thr_uni]
)
