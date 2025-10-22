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
        (:rmed, "rmed"),
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
                ygridvisible = false,
                titlefont = Makie.to_font("Book Antiqua"),
                yticklabelfont = Makie.to_font("Book Antiqua"), xticklabelfont = Makie.to_font("Book Antiqua"),
                ylabelfont = Makie.to_font("Book Antiqua"), xlabelfont = Makie.to_font("Book Antiqua"),
                subtitlefont = Makie.to_font("Book Antiqua")
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
                        color = :black,
                        font = Makie.to_font("Book Antiqua")
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
                        color = :black,
                        font = Makie.to_font("Book Antiqua")
                    )
                end
            end
        end
    end

    # Save the figure
    if save_plot
        filename = "Figures/Correlation_results_for_scenarios_$(join(scenarios, "_")).png"
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
    pixels_per_unit = 6,
    # --- NEW smoothing/robustness controls ---
    binning::Symbol = :equal,       # :equal (equal-width) or :quantile (equal-count)
    trim_frac::Float64 = 0.10,      # 10% trimmed mean per bin
    smooth_window::Int = 5          # moving-average window (odd is nicer). 1 = no smoothing
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

        # Modularity (leading eigenvector, Newman)
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

        # (Optional) nestedness (very coarse)
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

    # --- helpers ---
    trimmed_mean(v::AbstractVector{<:Real}, trim::Float64) = begin
        n = length(v)
        n == 0 && return NaN
        t = clamp(floor(Int, trim*n), 0, max(0, div(n-1,2)))
        vs = sort(collect(skipmissing(v)))
        vs = vs[(t+1):(n-t)]
        return mean(vs)
    end
    function moving_avg(v::AbstractVector{<:Real}, k::Int)
        k ≤ 1 && return collect(v)
        n = length(v); out = similar(collect(v))
        h = k ÷ 2
        for i in 1:n
            lo = max(1, i-h); hi = min(n, i+h)
            out[i] = mean(@view v[lo:hi])
        end
        out
    end

    # 4) Create figure layout
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

            xs = copy(df[!, p])
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

            # ---- BINNING (equal-width or quantile) ----
            mx, my, sy = Float64[], Float64[], Float64[]
            if binning == :quantile
                # split into ~equal-count bins to stabilize variance
                n = length(xs)
                ord = sortperm(xs)
                chunk = max(1, fld(n, n_bins))
                for b in 1:n_bins
                    lo = (b-1)*chunk + 1
                    hi = b==n_bins ? n : min(b*chunk, n)
                    if lo > hi; continue; end
                    idxs = ord[lo:hi]
                    push!(mx, mean(xs[idxs]))
                    push!(my, trimmed_mean(errs[idxs], trim_frac))
                    push!(sy, std(errs[idxs]))
                end
            else
                # equal-width bins (original)
                xmin, xmax = minimum(xs), maximum(xs)
                edges = range(xmin, xmax, length = n_bins + 1)
                bix = searchsortedlast.(Ref(edges), xs)
                for b in 1:n_bins
                    idxs = findall(bix .== b)
                    if !isempty(idxs)
                        push!(mx, mean(xs[idxs]))
                        push!(my, trimmed_mean(errs[idxs], trim_frac))
                        push!(sy, std(errs[idxs]))
                    end
                end
            end

            # ---- SMOOTHING ----
            smx = moving_avg(mx, smooth_window)
            smy = moving_avg(my, smooth_window)
            ssy = moving_avg(sy, smooth_window)

            lines!(ax, smx, smy;
                color     = colors[si],
                linewidth = 2.5,
                label     = step_names[si]
            )

            if error_bars
                errorbars!(ax, smx, smy, ssy; color = colors[si])
            end
        end

        # legend once
        if pi == 1 && mi == 1
            axislegend(ax; position = :rt)
        end
    end

    display(fig)
    if save_plot
        save("Figures/error_vs_structure.png", fig; px_per_unit=pixels_per_unit)
    end
    return df
end

"""
    plot_random_SAD_grid(df::DataFrame;
        n::Int=9, bins::Int=30, log10::Bool=false,
        seed::Union{Nothing,Int}=nothing,
        save_plot::Bool=false,
        filename::String="../Figures/random_SAD_grid.png",
        resolution=(1100, 1100), pixels_per_unit=6
    )

Pick 9 random systems from `df` and plot their Species Abundance Distributions (SADs)
as 3×3 histograms of the equilibrium abundances `B_eq` (extant species only, > 1e-6).
- If `log10=true`, it plots log10-abundances.
- Re-run to get a new random sample (set `seed` for reproducibility).
"""
function plot_random_SAD_grid(
    df::DataFrame;
    n::Int=9, bins::Int=30, log10::Bool=false,
    which::Symbol=:all,                # :all, :R (resources), :C (consumers)
    k_consumer_cutoff::Float64=0.015,  # classify consumers as K ≤ this (full-model K has 0.01)
    ext_threshold::Float64=1e-6,       # only plot extant spp.
    seed::Union{Nothing,Int}=nothing,
    save_plot::Bool=false,
    filename::String="../Figures/random_SAD_grid.png",
    resolution=(1100, 800), pixels_per_unit=6
)
    nrow(df) == 0 && error("DataFrame is empty.")
    nsel = min(n, nrow(df))
    seed === nothing || Random.seed!(seed)

    idxs = sample(1:nrow(df), nsel; replace=false)

    fig = Figure(; size=resolution)
    nrows, ncols = 3, 3
    lab = which === :R ? " (R only)" : which === :C ? " (C only)" : ""

    for (k, idx) in enumerate(idxs)
        i = fldmod1(k, ncols)[1]
        j = fldmod1(k, ncols)[2]
        ax = Axis(fig[i, j];
            title = "SAD #$idx$lab",
            xlabel = log10 ? "log10 abundance" : "abundance",
            ylabel = "count",
            xgridvisible=false, ygridvisible=false
        )

        # abundances (full model) and group mask from full-model K
        abund = copy(df[idx, :B_eq])::Vector{Float64}
        K, _A = df[idx, :p_final]
        @assert length(K) == length(abund)

        # choose species subset
        mask = trues(length(K))
        if which === :R
            mask .= K .> k_consumer_cutoff        # resources: larger K
        elseif which === :C
            mask .= K .<= k_consumer_cutoff       # consumers: ~0.01
        elseif which === :all
            # keep mask = trues
        else
            error("Argument `which` must be :all, :R, or :C")
        end

        abund = abund[mask]
        abund = abund[abund .> ext_threshold]     # extant only

        if isempty(abund)
            text!(ax, "No extant spp.", position=(0.5, 0.5), align=(:center,:center))
            continue
        end

        vals = abund # log10 ? log10.(abund .+ 1e-12) : abund
        hist!(ax, vals; bins=bins, normalization=:none)
    end

    display(fig)
    if save_plot
        save(filename, fig; px_per_unit=pixels_per_unit)
    end
    return nothing
end

"""
    plot_species_level_SL_correlations(
        df::DataFrame;
        steps = [1, 2, 3, 5],
        fit_to_1_1_line::Bool = true,
        subsample_frac::Float64 = 0.5,   # 0–1; lower to thin points
        max_points::Int = 200_000,       # hard cap for performance
        alpha::Float64 = 0.15,
        save_plot::Bool = false,
        filename::String = "../Figures/species_level_SL_alignment.png",
        resolution = (1100, 320),
        pixels_per_unit = 6.0,
        seed::Union{Nothing,Int} = nothing
    )

Species-level alignment of SL_time (−1 / J_ii): each panel scatters
SL_full (x) vs SL_step (y) across ALL species in ALL runs.
A 1:1 line is shown; text reports R² to the 1:1 line.
"""
function plot_species_level_SL_correlations(
    df::DataFrame;
    steps = [1, 2, 3, 5],
    which::Symbol = :all,          # :all, :R (resources), :C (consumers)
    k_consumer_cutoff::Float64 = 0.015,  # consumers: K ≤ cutoff (full-model consumers ≈ 0.01)
    fit_to_1_1_line::Bool = true,
    subsample_frac::Float64 = 0.5,
    max_points::Int = 200_000,
    alpha::Float64 = 0.15,
    save_plot::Bool = false,
    filename::String = "../Figures/species_level_SL_alignment.png",
    resolution = (1100, 320),
    pixels_per_unit = 6.0,
    seed::Union{Nothing,Int} = nothing,
    sl_max::Real = 100,            # per-point cutoff for SL on either axis
    remove_unstable::Bool = true
)
    seed === nothing || Random.seed!(seed)
    
    if remove_unstable
        rescols = Symbol.(string(:resilience) .* "_S" .* string.(steps))
        push!(rescols, Symbol(:resilience_full))
        df = filter(row -> all(row[c] < 0 for c in rescols), df)
    end

    step_names = Dict(
        1 => "Rewiring",
        2 => "Rewiring + ↻C",
        3 => "Rewiring + ↻IS",
        4 => "Rewiring + ↻C + ↻IS",
        5 => "Changing groups"
    )

    _r2_to_1to1 = function(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
        if isempty(x) || isempty(y); return NaN; end
        yhat = x; μy = mean(y)
        sst = sum((y .- μy).^2); ssr = sum((y .- yhat).^2)
        return sst == 0 ? NaN : 1 - ssr/sst
    end

    fig = Figure(; size = resolution)
    dot_colors = [:firebrick, :orangered, :crimson, :darkred]

    for (j, step) in enumerate(steps)
        xs = Float64[]; ys = Float64[]
        s_col = Symbol("SL_S$(step)")

        for row in eachrow(df)
            sfull = row.SL_full
            sstep = row[s_col]
            K, _A = row.p_final  # K is first element of the tuple
            m = min(minimum((length(sfull), length(sstep), length(K))))

            if m == 0; continue; end

            @inbounds for k in 1:m
                # species filter by group
                is_consumer = K[k] <= k_consumer_cutoff
                keep_species = which === :all || (which === :C && is_consumer) || (which === :R && !is_consumer)
                if !keep_species; continue; end

                xi = sfull[k]; yi = sstep[k]
                # per-point filters: finite and under threshold on BOTH axes
                if isfinite(xi) && isfinite(yi) && xi <= sl_max && yi <= sl_max
                    push!(xs, xi); push!(ys, yi)
                end
            end
        end

        N = length(xs)
        if N == 0
            Axis(fig[1, j]; title="No data", xgridvisible=false, ygridvisible=false)
            continue
        end

        # subsample / cap
        nkeep = Int(clamp(round(N * subsample_frac), 1, N))
        nkeep = min(nkeep, max_points)
        keep_idx = (nkeep == N) ? collect(1:N) : sample(1:N, nkeep; replace=false)
        xuse = @view xs[keep_idx]; yuse = @view ys[keep_idx]

        # axis limits
        mn = min(minimum(xuse), minimum(yuse))
        mx = max(maximum(xuse), maximum(yuse))
        if !isfinite(mn) || !isfinite(mx) || mn == mx; mn, mx = -1.0, 1.0; end

        grp = which === :R ? " (R)" : which === :C ? " (C)" : ""
        ax = Axis(fig[1, j];
            title = "SL: $(get(step_names, step, "Step $step"))$grp",
            xlabel = "Full model",
            ylabel = j == 1 ? "Simplified model" : "",
            limits = ((mn, mx), (mn, mx)),
            xgridvisible=false, ygridvisible=false,
            xlabelsize=11, ylabelsize=11, titlesize=12,
            xticklabelsize=10, yticklabelsize=10
        )

        scatter!(ax, xuse, yuse;
            color = dot_colors[mod1(j, length(dot_colors))],
            markersize = 3.5, transparency = true, alpha = alpha
        )
        lines!(ax, [mn, mx], [mn, mx]; color=:black, linestyle=:dash)

        if fit_to_1_1_line
            r2 = _r2_to_1to1(xuse, yuse)
            if isfinite(r2)
                text!(ax, "R²=$(round(r2, digits=3))";
                     position=(mx, mn), align=(:right,:bottom), fontsize=12, color=:black)
            end
        else
            r = cor(xuse, yuse)
            if isfinite(r)
                text!(ax, "r=$(round(r, digits=3))";
                     position=(mx, mn), align=(:right,:bottom), fontsize=12, color=:black)
            end
        end
    end

    display(fig)
    if save_plot
        save(filename, fig; px_per_unit = pixels_per_unit)
    end
    return nothing
end
