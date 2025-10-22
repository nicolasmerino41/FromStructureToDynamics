
"""
Quantile-binned median error vs structure with IQR ribbons.
One grid: rows=metrics, cols=properties; one colored line per step.
"""
function plot_struct_effect_quantile(
    df::DataFrame;
    steps = [1,2,3,5],
    props = [:connectance, :modularity, :degree_cv],
    props_names = ["Connectance","Modularity","Degree CV"],
    metrics = [:resilience, :reactivity, :rt_pulse, :after_press],
    metric_names = Dict(:resilience=>"Resilience", :reactivity=>"Reactivity",
                        :rt_pulse=>"Return Time", :after_press=>"Persistence"),
    nbins::Int = 20,                             # quantile bins
    relative_error::Bool = true,
    remove_unstable::Bool = false,
    resolution=(1100,1000), px_per_unit=6.0,
    save_plot=false, filename="Figures/struct_quantile_binned.png"
)
    # Optional stability filter
    if remove_unstable
        rescols = Symbol.(string(:resilience) .* "_S" .* string.(steps))
        df = filter(row -> all(row[c] < 0 for c in rescols), df)
    end

    # Helper to compute structural columns (if not there yet)
    if !(:connectance in names(df))
        N = nrow(df)
        conn = Vector{Float64}(undef, N)
        modu = Vector{Float64}(undef, N)
        degcv = Vector{Float64}(undef, N)
        for (i,row) in enumerate(eachrow(df))
            A = row.p_final[2]; Adj = A .!= 0.0; S = size(A,1)
            # connectance
            conn[i] = sum(Adj) / (S*(S-1))
            # modularity (very simple Newman split as in your code)
            k = sum(Adj, dims=2)[:]; m = sum(k)/2
            B = zeros(Float64,S,S)
            @inbounds for u in 1:S, v in 1:S
                B[u,v] = (Adj[u,v] ? 1.0 : 0.0) - (k[u]*k[v])/(2m)
            end
            vals, vecs = eigen(Symmetric(B))
            v1 = vecs[:, argmax(vals)]
            svec = map(x -> x>=0 ? 1.0 : -1.0, v1)
            modu[i] = (svec' * (B * svec)) / (4m)
            # degree CV
            degs = vec(sum(Adj, dims=2))
            degcv[i] = std(degs) / max(mean(degs), 1e-9)
        end
        df.connectance = conn; df.modularity = modu; df.degree_cv = degcv
    end

    # error helper
    err_of = function(m, s)
        fullco = Symbol("$(m)_full")
        stepco = Symbol("$(m)_S$(s)")
        e = relative_error ?
            abs.(df[!, stepco] .- df[!, fullco]) ./ (abs.(df[!, fullco]) .+ 1e-6) :
            abs.(df[!, stepco] .- df[!, fullco])
        (e .+ 1e-6) ./ (1 + 2e-6)
    end

    colors = [:red, :blue, :green, :orange]
    step_names = ["Rewiring", "Rewiring + ↻C", "Rewiring + ↻IS", "Changing groups"]

    fig = Figure(size=resolution)

    for (mi, m) in enumerate(metrics), (pi, p) in enumerate(props)
        ax = Axis(fig[mi, pi];
                  title="$(metric_names[m]) Vs $(props_names[pi])",
                  xlabel="$(props_names[pi])",
                  ylabel=mi==1 ? "Median relative error" : "",
                  xgridvisible=false, ygridvisible=false)

        xvals = df[!, p]
        order = sortperm(xvals)
        qbreaks = round.(Int, range(0, length(order), length=nbins+1))
        qbreaks[end] = length(order)

        for (si,s) in enumerate(steps)
            errs = err_of(m, s)
            mx = Float64[]; med = Float64[]; q25 = Float64[]; q75 = Float64[]
            for b in 1:nbins
                lo = qbreaks[b] + 1
                hi = qbreaks[b+1]
                if lo > hi; continue; end
                idx = view(order, lo:hi)
                push!(mx, mean(xvals[idx]))
                ebin = errs[idx]
                push!(med, median(ebin))
                push!(q25, quantile(ebin, 0.25))
                push!(q75, quantile(ebin, 0.75))
            end
            # light smoothing: moving average over 3 bins
            smooth3(v) = [mean(v[max(1,i-1):min(length(v),i+1)]) for i in 1:length(v)]
            smx, smmed, sm25, sm75 = smooth3(mx), smooth3(med), smooth3(q25), smooth3(q75)

            band!(ax, smx, sm25, sm75, color=(colors[si], 0.15))
            lines!(ax, smx, smmed, color=colors[si], linewidth=2, label=step_names[si])
        end

        if mi==1 && pi==1
            axislegend(ax; position=:rt)
        end
    end

    display(fig)
    save_plot && save(filename, fig; px_per_unit=px_per_unit)
    return fig
end
plot_struct_effect_quantile(
    sim_results;
    steps = [1],
    save_plot=false
)