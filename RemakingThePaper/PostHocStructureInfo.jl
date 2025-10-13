"""
    plot_structure_range_summary(
        df::DataFrame;
        metrics = [
            (:conn, "Connectance (realized)"),
            (:realized_mod, "Within-fraction (realized)"),
            (:deg_cv_cons_out_realized, "Consumer degree CV (realized)")
        ],
        steps = [1,2,3,5],
        remove_unstable::Bool = false,
        save_plot::Bool = false,
        save_path::String = "Figures/structure_range_summary.png",
        resolution = (1200, 650),
        pixels_per_unit = 6.0,
        bins::Int = 30
    ) -> NamedTuple

Post-hoc summary of **realized structure** across communities.
- Filters unstable rows if `remove_unstable=true` (any `resilience_S{step} > 0` or `resilience_full > 0`).
- Plots a 2xN grid: **histograms** (top) and **ECDFs** (bottom) per metric.
- Prints and returns a summary per metric: N, min/median/max, IQR, CV, skewness.
"""
function plot_structure_range_summary(
    df::DataFrame;
    metrics = [
        (:conn, "Connectance (realized)"),
        (:realized_mod, "Within-fraction (realized)"),
        (:deg_cv_cons_out_realized, "Consumer degree CV (realized)")
    ],
    steps = [1,2,3,5],
    remove_unstable::Bool = false,
    save_plot::Bool = false,
    save_path::String = "Figures/structure_range_summary.png",
    resolution = (1200, 650),
    pixels_per_unit = 6.0,
    bins::Int = 30
)
    # --- helpers ---
    names_set = Set(names(df))
    _findcol(sym::Symbol) = (sym in names_set ? sym :
                             (String(sym) in names_set ? String(sym) : nothing))
    _finite(v) = !(ismissing(v) || !isfinite(v))
    _iqr(x) = quantile(x, 0.75) - quantile(x, 0.25)
    _cv(x)  = (mean(x) == 0 ? NaN : std(x) / mean(x))
    _skew(x) = begin
        μ = mean(x); σ = std(x)
        σ == 0 ? 0.0 : mean(((x .- μ) ./ σ).^3)
    end
    function _ecdf(x::Vector{Float64})
        xs = sort(x)
        n = length(xs)
        ys = (1:n) ./ n
        return xs, ys
    end

    work = df

    # optional: remove unstable rows (full or any selected step)
    if remove_unstable
        res_cols = Symbol[:resilience_full; (Symbol(:resilience, :_S, s) for s in steps)...]
        keep = trues(nrow(work))
        for (i,row) in enumerate(eachrow(work))
            bad = false
            for rc in res_cols
                col = _findcol(rc); col === nothing && continue
                v = row[col]
                if _finite(v) && v > 0
                    bad = true; break
                end
            end
            keep[i] = !bad
        end
        kept = count(keep)
        println("Removed unstable rows: kept $kept / $(nrow(work)).")
        work = work[keep, :]
        if kept == 0
            @warn "No rows to plot after filtering."
            return (kept=0,)
        end
    end

    # pull each metric as Float64 vector (skip missing/NaN)
    metric_data = Dict{Symbol,Vector{Float64}}()
    labels = Dict{Symbol,String}()
    for (sym,label) in metrics
        col = _findcol(sym)
        if col === nothing
            @warn "Missing column $(sym); skipping."
            continue
        end
        vals = Float64[]
        for v in work[!, col]
            if _finite(v)
                push!(vals, float(v))
            end
        end
        if isempty(vals)
            @warn "All values missing/invalid for $(sym); skipping."
            continue
        end
        metric_data[sym] = vals
        labels[sym] = label
    end
    isempty(metric_data) && (@warn "No metrics to plot."; return (kept=nrow(work),))

    ncols = length(metric_data)
    fig = Figure(size = (max(resolution[1], 360*ncols), resolution[2]))
    Label(fig[0, 1:ncols], "Remove unstable: $(remove_unstable)", fontsize = 20, font = :bold, halign = :left)
    # consistent y-limits for ECDFs
    for (j, (sym, vals)) in enumerate(metric_data)
        # --- histograms (top row) ---
        axh = Axis(fig[1, j];
            title = labels[sym],
            xlabel = j == 1 ? "value" : "",
            ylabel = "count",
            xgridvisible=false, ygridvisible=false,
            titlesize=13
        )
        hist!(axh, vals; bins=bins)

        # --- ECDF (bottom row) ---
        axe = Axis(fig[2, j];
            xlabel = "value",
            ylabel = j == 1 ? "ECDF" : "",
            xgridvisible=false, ygridvisible=false
        )
        xs, ys = _ecdf(vals)
        lines!(axe, xs, ys)
    end

    # compute and print summary
    println("=== Realized structure summary over $(nrow(work)) communities ===")
    out_rows = Vector{NamedTuple}()
    for (sym, vals) in metric_data
        mn = minimum(vals); md = median(vals); mx = maximum(vals)
        iqr = _iqr(vals); cv = _cv(vals); sk = _skew(vals)
        println(rpad(string(sym), 26), ": min=", round(mn,digits=3),
                "  med=", round(md,digits=3),
                "  max=", round(mx,digits=3),
                "  IQR=", round(iqr,digits=3),
                "  CV=", round(cv,digits=3),
                "  skew=", round(sk,digits=3))
        push!(out_rows, (metric=sym, N=length(vals), min=mn, median=md, max=mx,
                         IQR=iqr, CV=cv, skew=sk))
    end

    if save_plot
        save(save_path, fig; px_per_unit=pixels_per_unit)
    end
    display(fig)

    return (kept=nrow(work), stats=out_rows, figure=fig)
end

plot_structure_range_summary(
    sim_results;
    metrics = [
        (:conn, "Connectance (realized)"),
        (:realized_mod, "Within-fraction (realized)"),
        (:deg_cv_cons_out_realized, "Consumer degree CV (realized)")
    ],
    steps = [1,2,3,5],
    remove_unstable = true,
    save_plot = false,
    save_path = "RemakingThePaper/Figures/structure_range_summary.png",
    resolution = (1200, 650),
    pixels_per_unit = 6.0,
    bins = 50
)
