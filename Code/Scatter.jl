
using CairoMakie, Statistics, Random, Graphs, LinearAlgebra, Printf

function plot_struct_scatter_with_regression(
    df::DataFrame;
    steps = [1,2,3,5],
    props = [:connectance, :modularity, :degree_cv],
    props_names = ["Connectance","Modularity","Degree CV"],
    metrics = [:resilience, :reactivity, :rt_pulse, :after_press],
    metric_names = Dict(:resilience=>"Resilience", :reactivity=>"Reactivity",
                        :rt_pulse=>"Return Time", :after_press=>"Persistence"),
    relative_error::Bool = true,
    remove_unstable::Bool = false,
    subsample::Int = 6000,
    seed::Union{Nothing,Int} = 1234,
    trim_q::Union{Nothing,Float64} = 0.99,
    alpha::Float64 = 0.15,
    resolution=(1100,1000), px_per_unit=6.0,
    save_plot=false, filename="Figures/struct_scatter_regression.png"
)
    # --- (same filtering / structural computation as in your last message) ---
    if remove_unstable
        rescols = Symbol.(string(:resilience) .* "_S" .* string.(steps))
        df = filter(row -> all(row[c] < 0 for c in rescols), df)
    end
    nrow(df) == 0 && error("No rows to plot after filtering.")

    for col in (:connectance, :modularity, :degree_cv)
        if !(col in names(df))
            N = nrow(df)
            conn = Vector{Float64}(undef, N)
            modu = Vector{Float64}(undef, N)
            degcv = Vector{Float64}(undef, N)
            for (i,row) in enumerate(eachrow(df))
                A = row.p_final[2]; Adj = A .!= 0.0; S = size(A,1)
                conn[i] = sum(Adj) / (S*(S-1))
                degs = vec(sum(Adj, dims=2)); mdeg = mean(degs)
                degcv[i] = mdeg == 0 ? 0.0 : std(degs)/mdeg
                k = sum(Adj, dims=2)[:]; m = sum(k)/2
                B = zeros(Float64,S,S)
                @inbounds for u in 1:S, v in 1:S
                    B[u,v] = (Adj[u,v] ? 1.0 : 0.0) - (k[u]*k[v])/(2m)
                end
                vals, vecs = eigen(Symmetric(B))
                v1 = vecs[:, argmax(vals)]
                svec = map(x -> x >= 0 ? 1.0 : -1.0, v1)
                modu[i] = (svec' * (B * svec)) / (4m)
            end
            df.connectance = conn; df.modularity = modu; df.degree_cv = degcv
            break
        end
    end

    err_of = function(m, s)
        fullco = Symbol("$(m)_full"); stepco = Symbol("$(m)_S$(s)")
        e = relative_error ?
            abs.(df[!, stepco] .- df[!, fullco]) ./ (abs.(df[!, fullco]) .+ 1e-6) :
            abs.(df[!, stepco] .- df[!, fullco])
        (e .+ 1e-6) ./ (1 + 2e-6)
    end

    ols_fit = function(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
        μx, μy = mean(x), mean(y); vx = var(x)
        if vx ≤ 0; return (intercept=μy, slope=0.0, r2=0.0); end
        slope = cov(x,y) / vx
        intercept = μy - slope*μx
        yhat = intercept .+ slope .* x
        sst = sum((y .- μy).^2); ssr = sum((y .- yhat).^2)
        (intercept=intercept, slope=slope, r2 = sst == 0 ? 0.0 : 1 - ssr/sst)
    end

    seed === nothing || Random.seed!(seed)
    colors = [:red, :blue, :green, :orange]
    step_names = ["Rewiring", "Rewiring + ↻C", "Rewiring + ↻IS", "Changing groups"]

    fig = Figure(size=resolution)

    for (mi, m) in enumerate(metrics), (pi, p) in enumerate(props)
        ax = Axis(fig[mi, pi];
            title = "$(metric_names[m]) Vs $(props_names[pi])",
            xlabel = "$(props_names[pi])",
            ylabel = mi==1 ? "Relative error" : "",
            xgridvisible=false, ygridvisible=false)

        xall = df[!, p]
        finite_x = isfinite.(xall)

        stats_per_step = Vector{NamedTuple}(undef, length(steps))
        global_ymin, global_ymax = Inf, -Inf

        for (si,s) in enumerate(steps)
            yall = err_of(m, s)
            y = copy(yall)
            if trim_q !== nothing
                q = quantile(y[isfinite.(y)], trim_q)
                @inbounds for i in eachindex(y)
                    yi = y[i]; if isfinite(yi) && yi > q; y[i] = q; end
                end
            end

            mask = finite_x .& isfinite.(y)
            x = xall[mask]; y = y[mask]
            if isempty(x)
                stats_per_step[si] = (fit=(intercept=NaN,slope=NaN,r2=NaN), color=colors[si], name=step_names[si])
                continue
            end

            # scatter (subsample for clarity)
            n = length(x)
            keep = n > subsample ? sample(1:n, subsample; replace=false) : 1:n
            xs = x[keep]; ys = y[keep]
            scatter!(ax, xs, ys; color=(colors[si], alpha), markersize=3,
                     label=(mi==1 && pi==1 && si==1) ? step_names[si] : "")

            # regression
            fit = ols_fit(x, y)
            xline = range(minimum(x), maximum(x); length=120)
            yline = fit.intercept .+ fit.slope .* xline
            lines!(ax, xline, yline; color=colors[si], linewidth=2)

            global_ymin = min(global_ymin, minimum(ys), minimum(yline))
            global_ymax = max(global_ymax, maximum(ys), maximum(yline))
            stats_per_step[si] = (fit=fit, color=colors[si], name=step_names[si])
        end

        # ---- annotate WITHOUT touching axis internals ----
        # use data ranges we just computed
        ypad = isfinite(global_ymin) && isfinite(global_ymax) ? 0.05*(global_ymax - global_ymin) : 0.0
        ymin = global_ymin - ypad
        # x position: near max x of this panel
        x_pos = maximum(xall[finite_x])
        # start a bit above ymin to avoid clipping
        y_pos = ymin + 0.02 * (global_ymax - global_ymin)
        y_step = 0.05 * (global_ymax - global_ymin)

        for st in stats_per_step
            fit = st.fit
            if !isnan(fit.r2)
                txt = @sprintf("%s: slope=%.3g, R²=%.3g", st.name, fit.slope, fit.r2)
                text!(ax, txt; position=(x_pos, y_pos), align=(:right,:top),
                      color=st.color, fontsize=10)
                y_pos += y_step
            end
        end

        # if mi==1 && pi==1
        #     axislegend(ax; position=:rt)
        # end
    end

    display(fig)
    save_plot && save(filename, fig; px_per_unit=px_per_unit)
    return fig
end

plot_struct_scatter_with_regression(
    sim_results;
    steps=[1],
    relative_error=true,
    remove_unstable=false,   # or true for stability-restricted
    subsample=5000,
    trim_q=0.99,             # trims spikes but keeps trends
    save_plot=false,
    filename="Figures/struct_scatter_regression.png"
)
