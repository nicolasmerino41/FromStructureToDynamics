# This file contains functions for plotting results from MainScript.jl,
# basically to generate Figure 2 and 3 of the paper

# ------------------------------------------------------------------------------
# Plot Correlations (Figure 2)
# ------------------------------------------------------------------------------
function plot_correlations(
    df::DataFrame; # DataFrame obtained from RunSimulations in MainScript.jl
    scenarios = [:ER, :PL, :MOD], # Choose which scenarios to plot
    steps = [1, 2, 3, 5], # Choose which steps to plot
    metrics = [
        (:resilience, "Resilience"), 
        (:reactivity, "Reactivity"),
        (:rt_pulse, "Return Time"),
        (:after_press, "Persistence"),
        
        # All these other metrics can also be explored if desired,
        # uncommenting them will add them to the plot automatically
        
        # (:mean_SL, "Mean SL"), 
        # (:rt_press, "RT_press"),
        # (:after_pulse, "after_pulse"),
        # (:rmed, "rmed"),
        # (:after_persistence, "Persistence"),
        # (:collectivity, "Collectivity"),
        # (:sigma_over_min_d, "σ/min(d)")
    ],
    fit_to_1_1_line::Bool = true, # Whether to express correlation as r^2 or r
    save_plot::Bool = true,
    resolution = (1100, 1000),
    pixels_per_unit = 6.0
)
    # Display names for modification steps
    step_names = ["Rewiring", "Rewiring + ↻C", "Rewiring + ↻IS", "Rewiring + ↻C + ↻IS", "Changing groups"]

    # Define color palette
    red_colors = [:firebrick, :orangered, :crimson, :darkred]

    fig = Figure(; size=resolution)

    for (i, (sym, label)) in enumerate(metrics)
        dot_color = red_colors[mod1(i, length(red_colors))]

        for (j, step) in enumerate(steps)
            col_full = Symbol(string(sym, "_full"))
            col_step = Symbol(string(sym, "_S", step))

            x_raw = df[!, col_full]
            y_raw = df[!, col_step]

            # Optional: Subsample for resilience/reactivity plots
            if sym == :resilience || sym == :reactivity
                idxf = sample(1:length(x_raw), Int(round(0.6 * length(x_raw))); replace=false)
                x_raw = x_raw[idxf]
                y_raw = y_raw[idxf]
            end

            # Filter finite, non-missing values
            x_finite = Float64[]
            y_finite = Float64[]

            for k in 1:min(length(x_raw), length(y_raw))
                xi = x_raw[k]
                yi = y_raw[k]
                if !(ismissing(xi) || ismissing(yi)) && isfinite(xi) && isfinite(yi)
                    push!(x_finite, xi)
                    push!(y_finite, yi)
                end
            end

            # Skip plot if data is invalid or empty
            if isempty(x_finite) || isempty(y_finite) || any(!isfinite(v) for v in (minimum(x_finite), maximum(x_finite), minimum(y_finite), maximum(y_finite)))
                @warn "Skipping $sym at step $step: invalid or empty data."
                continue
            end

            # Determine axis limits
            mn = min(minimum(x_finite), minimum(y_finite))
            mx = max(maximum(x_finite), maximum(y_finite))

            # Create subplot
            ax = Axis(
                fig[i, j];
                title = "$label: $(step_names[step])",
                titlesize = 12,
                xlabelsize = 11,
                ylabelsize = 11,
                xticklabelsize = 11,
                yticklabelsize = 11,
                limits = ((mn, mx), (mn, mx)),
                xgridvisible = false,
                ygridvisible = false
            )

            # Plot scatter points
            scatter!(ax, x_finite, y_finite; color=dot_color, alpha=0.3)

            # Plot 1:1 line
            lines!(ax, [mn, mx], [mn, mx]; color = :black, linestyle = :dash)

            # Compute and display fit or correlation
            if fit_to_1_1_line
                y_hat = x_finite
                ss_tot = sum((y_finite .- mean(y_finite)).^2)
                ss_res = sum((y_finite .- y_hat).^2)
                r2_1to1 = ss_tot == 0 ? NaN : 1 - ss_res / ss_tot

                if isfinite(r2_1to1) && isfinite(mx) && isfinite(mn)
                    x_pos = sym == :after_press && j == 3 ? mx - 0.32 * (mx - mn) : mx
                    align = sym == :after_press && j == 3 ? (:left, :bottom) : (:right, :bottom)
                    text!(
                        ax, 
                        "R²=$(round(r2_1to1, digits=3))";
                        position = (x_pos, mn),
                        align = align,
                        fontsize = 12,
                        color = :black
                    )
                end
            else
                r_val = cor(x_finite, y_finite)
                if isfinite(r_val) && isfinite(mx) && isfinite(mn)
                    text!(
                        ax, 
                        "r=$(round(r_val, digits=3))";
                        position = (mx, mn),
                        align = (:right, :bottom),
                        fontsize = 10,
                        color = :black
                    )
                end
            end
        end
    end

    # Save the figure
    if save_plot
        filename = "../Figures/Correlation_results_for_scenarios_$(join(scenarios, "_")).png"
        save(filename, fig; px_per_unit=pixels_per_unit)
    end

    display(fig)
end

# ------------------------------------------------------------------------
# Plot error vs structural properties (Figure 3)
# ------------------------------------------------------------------------
"""
For each of the four system-level metrics (Resilience, Reactivity,
Return Time, Persistence), plots the absolute or relative error at each
step against each of the three structural properties
(connectance, modularity, degree_cv). The result is a 3x4 grid:
rows = structural properties, columns = metrics.
"""
function plot_error_vs_structural_properties(
    df::DataFrame;
    steps = [1, 2, 3, 5],
    remove_unstable = false,
    n_bins::Int = 50,
    save_plot = false,
    error_bars = true,
    outlier_quantile = nothing,
    outlier_quantile_x = 1.0,
    relative_error = true,
    resolution = (1100, 1000),
    pixels_per_unit = 6
)
    # 1) Optionally filter out unstable runs (resilience > 0)
    if remove_unstable
        rescols = Symbol.(string(:resilience) .* "_S" .* string.(steps))
        df = filter(row -> all(row[c] < 0 for c in rescols), df)
    end

    # 2) Compute structural properties for each system
    N = nrow(df)
    conn      = Vector{Float64}(undef, N)
    mdeg      = Vector{Float64}(undef, N)
    modu      = Vector{Float64}(undef, N)
    degree_cv = Vector{Float64}(undef, N)
    nest      = Vector{Float64}(undef, N)  # optional nestedness

    for (i, row) in enumerate(eachrow(df))
        A   = row.p_final[2]
        Adj = A .!= 0.0
        S   = size(A, 1)

        g = SimpleGraph(A .!= 0)
        degs = degree(g)
        degree_cv[i] = std(degs) / mean(degs)

        conn[i] = sum(Adj) / (S * (S - 1))
        mdeg[i] = mean(sum(Adj, dims=2))

        # Modularity using Newman's leading eigenvector method
        k = sum(Adj, dims=2)[:]
        m = sum(k) / 2
        B = zeros(Float64, S, S)
        for u in 1:S, v in 1:S
            B[u, v] = (Adj[u, v] ? 1.0 : 0.0) - (k[u] * k[v]) / (2m)
        end
        vals, vecs = eigen(Symmetric(B))
        v1 = vecs[:, argmax(vals)]
        svec = map(x -> x >= 0 ? 1.0 : -1.0, v1)
        modu[i] = (svec' * (B * svec)) / (4m)

        # (Optional) nestedness
        nested_sum = 0.0
        nested_count = 0
        for u in 1:S, v in u+1:S
            du, dv = sum(Adj[u, :]), sum(Adj[v, :])
            denom = max(du, dv)
            if denom > 0
                nested_sum  += sum(Adj[u, :] .& Adj[v, :]) / denom
                nested_count += 1
            end
        end
        nest[i] = nested_count > 0 ? nested_sum / nested_count : NaN
    end

    df.connectance = conn
    df.mean_degree = mdeg
    df.modularity  = modu
    df.nestedness  = nest
    df.degree_cv   = degree_cv

    # 3) Define structural properties and dynamical metrics
    props = [:connectance, :modularity, :degree_cv]
    props_names = ["Connectance", "Modularity", "Degree CV"]
    metrics = [:resilience, :reactivity, :rt_pulse, :after_press]
    titles = Dict(
        :resilience  => "Resilience",
        :reactivity  => "Reactivity",
        :rt_pulse    => "Return Time",
        :after_press => "Persistence"
    )

    # 4) Create figure layout
    nP = length(props)
    nM = length(metrics)
    fig = Figure(; size = resolution)
    colors = [:red, :blue, :green, :orange, :purple]
    step_names = ["Rewiring", "Rewiring + ↻C", "Rewiring + ↻IS", "Changing groups"]

    for (pi, p) in enumerate(props), (mi, m) in enumerate(metrics)
        ax = Axis(
            fig[mi, pi];
            title  = "$(titles[m]) Vs $(props_names[pi])",
            xlabel = "$(props_names[pi])",
            ylabel = relative_error ? "Relative error" : "Absolute error",
            xgridvisible = false,
            ygridvisible = false
        )

        # Plot one line per simplification step
        for (si, s) in enumerate(steps)
            scol = Symbol("$(m)_S$(s)")
            fullco = Symbol("$(m)_full")

            # Compute error based on user input
            e = relative_error ?
                abs.(df[!, scol] .- df[!, fullco]) ./ (abs.(df[!, fullco]) .+ 1e-6) :
                abs.(df[!, scol] .- df[!, fullco])
            errs = (e .+ 1e-6) ./ (1 + 2e-6)

            xs = df[!, p]
            xs[xs .<= 0.0] .= 0.0

            # Remove outliers if enabled
            if outlier_quantile !== nothing
                thresh = quantile(errs, outlier_quantile)
                keep   = errs .<= thresh
                xs     = xs[keep]
                errs   = errs[keep]
                thresh = quantile(xs, outlier_quantile_x)
                keep   = xs .<= thresh
                xs     = xs[keep]
                errs   = errs[keep]
            end

            # println("max metric value is: for metric $(p) and step $(s): $(maximum(xs))")

            # Bin points into intervals
            xmin, xmax = minimum(xs), maximum(xs)
            edges = range(xmin, xmax, length = n_bins + 1)
            bix = searchsortedlast.(Ref(edges), xs)

            mx, my, sy = Float64[], Float64[], Float64[]
            for b in 1:n_bins
                idxs = findall(bix .== b)
                if !isempty(idxs)
                    push!(mx, mean(xs[idxs]))
                    push!(my, mean(errs[idxs]))
                    push!(sy, std(errs[idxs]))
                end
            end

            lines!(ax, mx, my;
                color     = colors[si],
                linewidth = 2,
                label     = step_names[si]
            )

            if error_bars
                errorbars!(ax, mx, my, sy; color = colors[si])
            end
        end

        # Show legend only on the top-left panel
        if pi == 1 && mi == 1
            axislegend(ax; position = :rt)
        end
    end

    display(fig)

    if save_plot
        save("../Figures/error_vs_structure.png", fig; px_per_unit=pixels_per_unit)
    end

    return df
end
