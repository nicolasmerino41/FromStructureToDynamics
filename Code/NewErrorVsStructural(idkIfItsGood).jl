using CairoMakie, Statistics, Random, Graphs, LinearAlgebra

function plot_error_vs_structural_properties(
    df::DataFrame;
    steps = [1, 2, 3, 5],
    remove_unstable = false,
    n_bins::Int = 40,
    save_plot = false,
    error_bars = false,            # replaced by IQR ribbon
    outlier_quantile = nothing,    # optional global y-trim (e.g., 0.995)
    outlier_quantile_x = 1.0,      # keep x-range intact by default
    relative_error = true,         # kept for compatibility
    resolution = (1100, 1000),
    pixels_per_unit = 6,
    # --- robustness controls (new) ---
    binning::Symbol = :quantile,   # :quantile (equal-count) or :equal (equal-width)
    min_per_bin::Int = 1,         # drop bins with fewer data
    use_smape_for_spectral::Bool = true,  # SMAPE for resilience/reactivity
    iqr_alpha::Float64 = 0.15,     # ribbon transparency
    clip_y_q::Union{Nothing,Float64} = nothing # winsorize panel y at this quantile
)
    # 1) Filter unstable (if asked)
    if remove_unstable
        rescols = Symbol.(string(:resilience) .* "_S" .* string.(steps))
        df = filter(row -> all(row[c] < 0 for c in rescols), df)
    end
    nrow(df) == 0 && error("No rows to plot.")

    # 2) Structural properties (same as your original, with a couple of safety checks)
    N = nrow(df)
    conn      = Vector{Float64}(undef, N)
    mdeg      = Vector{Float64}(undef, N)
    modu      = Vector{Float64}(undef, N)
    degree_cv = Vector{Float64}(undef, N)
    nest      = Vector{Float64}(undef, N)

    for (i, row) in enumerate(eachrow(df))
        A   = row.p_final[2]
        Adj = A .!= 0.0
        S   = size(A, 1)

        g = SimpleGraph(Adj)
        degs = degree(g)
        μd = mean(degs)
        degree_cv[i] = μd == 0 ? 0.0 : std(degs) / μd

        conn[i] = sum(Adj) / (S * (S - 1))
        mdeg[i] = mean(sum(Adj, dims=2))

        # Modularity (Newman leading eigenvector)
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

        # Coarse nestedness
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

    # 3) Setup
    props = [:connectance, :modularity, :degree_cv]
    prop_names = ["Connectance","Modularity","Degree CV"]
    metrics = [:resilience, :reactivity, :rt_pulse, :after_press]
    titles = Dict(
        :resilience  => "Resilience",
        :reactivity  => "Reactivity",
        :rt_pulse    => "Return Time",
        :after_press => "Persistence"
    )

    # Error definitions
    compute_err = function(full::AbstractVector, step::AbstractVector, m::Symbol)
        if use_smape_for_spectral && (m === :resilience || m === :reactivity)
            # SMAPE in [0,2]
            return 2 .* abs.(step .- full) ./ (abs.(full) .+ abs.(step) .+ 1e-8)
        else
            return relative_error ?
                abs.(step .- full) ./ (abs.(full) .+ 1e-6) :
                abs.(step .- full)
        end
    end

    # helpers
    q25(v) = quantile(v, 0.25)
    q75(v) = quantile(v, 0.75)
    med(v) = median(v)

    fig = Figure(size=resolution)
    colors = [:red, :blue, :green, :orange]
    step_names = ["Rewiring", "Rewiring + ↻C", "Rewiring + ↻IS", "Changing groups"]

    for (pi, p) in enumerate(props), (mi, m) in enumerate(metrics)
        ax = Axis(fig[mi, pi];
            title  = "$(titles[m]) Vs $(prop_names[pi])",
            xlabel = prop_names[pi],
            ylabel = "Relative error",
            xgridvisible=false, ygridvisible=false)

        xall = df[!, p]

        # Optional global y clipping per panel (computed after we have all y, below)
        panel_y_values = Float64[]

        # One colored line per step
        for (si, s) in enumerate(steps)
            step_col = Symbol("$(m)_S$(s)")
            full_col = Symbol("$(m)_full")
            errs = compute_err(df[!, full_col], df[!, step_col], m)

            xs = copy(xall)

            # Optional global trimming (keeps x intact unless you set outlier_quantile_x < 1)
            if outlier_quantile !== nothing
                thr = quantile(errs, outlier_quantile)
                keep = errs .<= thr
                xs   = xs[keep]
                errs = errs[keep]

                thrx = quantile(xs, outlier_quantile_x)
                keep = xs .<= thrx
                xs   = xs[keep]
                errs = errs[keep]
            end

            # Bin edges
            mx, qlo, qmd, qhi = Float64[], Float64[], Float64[], Float64[]
            if binning == :quantile
                n = length(xs)
                if n > 0
                    ord = sortperm(xs)
                    per = max(1, fld(n, n_bins))
                    for b in 1:n_bins
                        lo = (b-1)*per + 1
                        hi = b==n_bins ? n : min(b*per, n)
                        lo > hi && continue
                        idxs = ord[lo:hi]
                        if length(idxs) < min_per_bin; continue; end
                        push!(mx, mean(@view xs[idxs]))
                        _y = @view errs[idxs]
                        push!(qlo, q25(_y)); push!(qmd, med(_y)); push!(qhi, q75(_y))
                    end
                end
            else
                xmin, xmax = minimum(xs), maximum(xs)
                edges = range(xmin, xmax, length = n_bins + 1)
                bix = searchsortedlast.(Ref(edges), xs)
                for b in 1:n_bins
                    idxs = findall(bix .== b)
                    if length(idxs) < min_per_bin; continue; end
                    push!(mx, mean(@view xs[idxs]))
                    _y = @view errs[idxs]
                    push!(qlo, q25(_y)); push!(qmd, med(_y)); push!(qhi, q75(_y))
                end
            end

            # Optional panel-level clipping
            if clip_y_q !== nothing && !isempty(qhi)
                all_y = vcat(qlo, qmd, qhi)
                ymax = quantile(all_y, clip_y_q)
                @inbounds begin
                    for i in eachindex(qlo); qlo[i] = min(qlo[i], ymax); end
                    for i in eachindex(qmd); qmd[i] = min(qmd[i], ymax); end
                    for i in eachindex(qhi); qhi[i] = min(qhi[i], ymax); end
                end
            end

            append!(panel_y_values, qlo); append!(panel_y_values, qhi)

            # Draw IQR ribbon + median line
            if !isempty(mx)
                band!(ax, mx, qlo, qhi; color=(colors[si], iqr_alpha))
                lines!(ax, mx, qmd; color=colors[si], linewidth=2.5, label=step_names[si])
            end
        end

        if pi == 1 && mi == 1
            axislegend(ax; position=:rt)
        end
    end

    display(fig)
    if save_plot
        save("Figures/error_vs_structure.png", fig; px_per_unit=pixels_per_unit)
    end
    return df
end

plot_error_vs_structural_properties(
    sim_results;
    save_plot=false
)
