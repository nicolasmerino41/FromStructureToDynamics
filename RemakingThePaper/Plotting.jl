# ---------------------------------------------------------
# Plot correlations (Makie) — robust Symbol handling
# ---------------------------------------------------------
"""
    plot_correlations(
        df; scenarios=[:CL], steps=[1,2,3,5],
        metrics=[(:resilience,"Resilience"), (:reactivity,"Reactivity"),
                 (:rt_pulse,"Return Time"), (:after_press,"Persistence")],
        fit_to_1_1_line=true, remove_unstable=false,
        save_plot=true, save_path="Figures/Correlation_results.png",
        resolution=(1100,1000), pixels_per_unit=6.0, rng=MersenneTwister(7)
    )

Compares each `metric_full` vs `metric_S{step}` in a grid.
"""
function plot_correlations(
    df::DataFrame;
    scenarios = [:CL],
    steps = [1, 2, 3, 5],
    metrics = [
        (:resilience, "Resilience"),
        (:reactivity, "Reactivity"),
        (:rt_pulse, "Return Time"),
        (:after_press, "Persistence")
    ],
    fit_to_1_1_line::Bool = true,
    remove_unstable::Bool = false,
    save_plot::Bool = true,
    save_path::String = "Figures/Correlation_results.png",
    resolution = (1000, 720),
    pixels_per_unit = 6.0,
    rng::AbstractRNG = MersenneTwister(7)
)
    # --- helpers ---
    names_set = Set(names(df))                    # handle Symbol or String names
    _findcol(name_sym::Symbol) = begin
        if name_sym in names_set
            name_sym
        else
            name_str = String(name_sym)
            name_str in names_set ? name_str : nothing
        end
    end
    _warn_missing(col) = @warn "Missing column $(col); skipping."

    # filter scenarios if column exists
    work = if _findcol(:scen) !== nothing
        filter(row -> row[_findcol(:scen)] in scenarios, df)
    else
        df
    end
    total0 = nrow(work)

    # remove rows with positive resilience in ANY selected step
    if remove_unstable
        keep = trues(nrow(work))
        for (i, row) in enumerate(eachrow(work))
            for s in steps
                col = _findcol(Symbol(:resilience, :_S, s))
                if col !== nothing
                    v = row[col]
                    if !(ismissing(v)) && isfinite(v) && v > 0
                        keep[i] = false; break
                    end
                end
            end
        end
        work = work[keep, :]
        println("Removed unstable (any selected step with positive resilience): kept $(nrow(work)) / $total0 rows.")
    end
    nrow(work) == 0 && (@warn "No rows to plot after filtering."; return nothing)

    # step labels
    step_names = ["Rewiring", "Rewiring + ↻C", "Rewiring + ↻IS", "Rewiring + ↻C + ↻IS", "Changing groups"]
    colors = [:firebrick, :orangered, :crimson, :darkred]
    fig = Figure(size = resolution)
    Label(fig[0, 2:3], "Remove unstable: $(remove_unstable)", fontsize = 20, font = :bold, halign = :left)
    for (i, (sym, label)) in enumerate(metrics)
        col_full_sym = Symbol(sym, :_full)
        col_full = _findcol(col_full_sym)
        if col_full === nothing
            _warn_missing(col_full_sym); continue
        end
        dot_color = colors[mod1(i, length(colors))]

        for (j, step) in enumerate(steps)
            col_step_sym = Symbol(sym, :_S, step)
            col_step = _findcol(col_step_sym)
            if col_step === nothing
                _warn_missing(col_step_sym); continue
            end

            x_raw = work[!, col_full]
            y_raw = work[!, col_step]

            # optional thinning for heavy clouds
            if sym == :resilience || sym == :reactivity
                n = length(x_raw); ns = max(100, Int(round(0.6 * n)))
                idx = sort!(rand(rng, 1:n, min(ns, n)))
                x_raw = x_raw[idx]; y_raw = y_raw[idx]
            end

            x = Float64[]; y = Float64[]
            for k in 1:min(length(x_raw), length(y_raw))
                xi = x_raw[k]; yi = y_raw[k]
                if !(ismissing(xi) || ismissing(yi)) && isfinite(xi) && isfinite(yi)
                    push!(x, xi); push!(y, yi)
                end
            end
            isempty(x) && continue

            mn = min(minimum(x), minimum(y))
            mx = max(maximum(x), maximum(y))
            (!isfinite(mn) || !isfinite(mx) || mn == mx) && ( @warn "Degenerate limits for $(sym) step $(step)"; continue )

            ax = Axis(fig[i, j];
                title = "$label: $(step_names[step])",
                xlabel = string(sym, "_full"),
                ylabel = string(sym, "_S", step),
                titlesize = 12, xlabelsize = 11, ylabelsize = 11,
                xticklabelsize = 11, yticklabelsize = 11,
                limits = ((mn, mx), (mn, mx)),
                xgridvisible = false, ygridvisible = false
            )

            scatter!(ax, x, y; color=dot_color, alpha=0.35, markersize=5)
            lines!(ax, [mn, mx], [mn, mx]; color=:black, linestyle=:dash)

            if fit_to_1_1_line
                sst = sum((y .- mean(y)).^2)
                ssr = sum((y .- x).^2)
                r2 = sst == 0 ? NaN : 1 - ssr/sst
                isfinite(r2) && text!(ax, "R²=$(round(r2, digits=3))";
                    position=(mx, mn), align=(:right,:bottom), fontsize=12, color=:black)
            else
                r = cor(x, y)
                isfinite(r) && text!(ax, "r=$(round(r, digits=3))";
                    position=(mx, mn), align=(:right,:bottom), fontsize=12, color=:black)
            end
        end
    end

    if save_plot
        save(save_path, fig; px_per_unit = pixels_per_unit)
    end
    display(fig)
end

# ----------------------
# Example use:
# ----------------------
plot_correlations(
    sim_results;
    scenarios = [:CL],
    steps = [1, 2, 3, 5],
    metrics = [
        (:resilience, "Resilience"),
        (:reactivity, "Reactivity"),
        (:rt_pulse, "Return Time"),
        (:after_press, "Persistence")
    ],
    fit_to_1_1_line = true,
    remove_unstable = false,
    save_plot = true,
    save_path = "Figures/Correlation_results.png",
    resolution = (1000, 720),
    pixels_per_unit = 6.0,
    rng = MersenneTwister(7)
)

"""
    plot_structure_vs_error(
        df;
        steps = [1,2,3,5],
        metrics = [
            (:resilience, "Resilience"),
            (:reactivity, "Reactivity"),
            (:rt_pulse, "Return Time"),
            (:after_press, "Persistence"),
        ],
        remove_unstable::Bool = false,
        save_plot::Bool = true,
        save_path::String = "Figures/Structure_vs_Error.png",
        resolution = (1200, 900),
        pixels_per_unit = 6.0,
        rng::AbstractRNG = MersenneTwister(7)
    )

Columns (x): realized connectance, realized within-fraction (modularity proxy),
             realized consumer-degree CV (CV of consumer out-degree).
Rows (y):   relative error for each selected metric vs steps:
             err = (metric_Sstep - metric_full) / abs(metric_full)

Notes:
- Uses A from `:p_final` to compute realized structure for the **original** community.
- Within-fraction uses deterministic block assignment via `balanced_blocks(Side, :blocks)`.
- If `remove_unstable=true`, drops rows where any selected step has positive `resilience_S{step}`.
"""
function plot_structure_vs_error(
    df;
    steps = [1,2,3,5],
    metrics = [
        (:resilience, "Resilience"),
        (:reactivity, "Reactivity"),
        (:rt_pulse, "Return Time"),
        (:after_press, "Persistence"),
    ],
    remove_unstable::Bool = false,
    save_plot::Bool = true,
    save_path::String = "RemakingThePaper/Figures/Structure_vs_Error.png",
    resolution = (1100, 720),
    pixels_per_unit = 6.0,
    rng::AbstractRNG = MersenneTwister(7)
)
    # --- helpers ---
    _findcol(df, sym::Symbol) = begin
        ns = Set(names(df))
        sym in ns ? sym : (String(sym) in ns ? String(sym) : nothing)
    end
    _cv(v) = (isempty(v) || mean(v) == 0) ? NaN : std(v) / mean(v)
    balanced_blocks(n::Int, blocks::Int) = repeat(1:blocks, inner=ceil(Int, n/blocks))[1:n]
    function _connectance(A, R)
        S = size(A,1); C = S - R; E = 0
        @inbounds for ic in 1:C, jr in 1:R
            E += (A[R+ic, jr] != 0.0)
        end
        return (R*C == 0) ? NaN : E/(R*C)
    end
    function _within_fraction(A, R, blocks)
        S = size(A,1); C = S - R
        r_block = balanced_blocks(R, blocks); c_block = balanced_blocks(C, blocks)
        tot = 0; win = 0
        @inbounds for ic in 1:C, jr in 1:R
            if A[R+ic, jr] != 0.0
                tot += 1; win += (c_block[ic] == r_block[jr]) ? 1 : 0
            end
        end
        return tot == 0 ? NaN : win / tot
    end
    function _degcv_consumer(A, R)
        S = size(A,1); C = S - R
        deg = Vector{Int}(undef, C)
        @inbounds for ic in 1:C
            deg[ic] = count(!iszero, A[R+ic, 1:R])
        end
        return _cv(deg)
    end
    function _ols(x::Vector{Float64}, y::Vector{Float64})
        n = length(x); mx = mean(x); my = mean(y)
        sxx = sum((x .- mx).^2); syy = sum((y .- my).^2)
        sxy = sum((x .- mx) .* (y .- my))
        sxx == 0 && return (my, 0.0, NaN)
        b = sxy / sxx; a = my - b*mx
        yhat = a .+ b .* x
        ssr = sum((y .- yhat).^2)
        r2 = syy == 0 ? NaN : 1 - ssr/syy
        return (a, b, r2)
    end

    work = df

    # remove unstable rows?
    if remove_unstable
        keep = trues(nrow(work))
        for (i,row) in enumerate(eachrow(work))
            for s in steps
                col = _findcol(work, Symbol(:resilience, :_S, s))
                if col !== nothing
                    v = row[col]
                    if !(ismissing(v)) && isfinite(v) && v > 0
                        keep[i] = false; break
                    end
                end
            end
        end
        total0 = nrow(work)
        work = work[keep, :]
        println("Removed unstable (any selected step with positive resilience): kept $(nrow(work)) / $total0 rows.")
    end
    nrow(work) == 0 && (@warn "No rows to plot after filtering."; return nothing)

    # columns we need
    col_p      = :p_final
    col_Sfull  = :S_full
    col_blocks = :blocks

    # realized structure (base community)
    x_conn  = Float64[]; x_within = Float64[]; x_degcv = Float64[]; keep_idx = Int[]
    for (i,row) in enumerate(eachrow(work))
        K,A = row[col_p]; n = size(A,1)
        # heuristic to infer R: maximize negatives in (resources rows, consumer cols)
        bestR, bestScore = 1, -1
        for Rt in 1:(n-1)
            score = count(<(0.0), @view A[1:Rt, (Rt+1):n])
            if score > bestScore; bestScore = score; bestR = Rt; end
        end
        R = bestR
        push!(x_conn,   _connectance(A, R))
        push!(x_within, _within_fraction(A, R, row[col_blocks]))
        push!(x_degcv,  _degcv_consumer(A, R))
        push!(keep_idx, i)
    end
    Xcols = (conn = x_conn, within = x_within, degcv_cons = x_degcv)
    xlabels = ("Connectance (realized)", "Within-fraction (realized)", "Consumer degree CV (realized)")

    colors_steps = [:steelblue, :darkorange, :seagreen, :purple, :brown]
    fig = Figure(size = resolution)
    Label(fig[0, 2:3], "Remove unstable: $(remove_unstable)", fontsize = 20, font = :bold, halign = :left)

    for (ri, (sym, label)) in enumerate(metrics)
        col_full = _findcol(work, Symbol(sym, :_full))
        col_full === nothing && (@warn "Missing $(Symbol(sym,:_full)), skipping $(sym)."; continue)
        full_vals = work[!, col_full][keep_idx]

        for (cj, (xname, xdata)) in enumerate(pairs(Xcols))
            ax = Axis(fig[ri, cj];
                title = "$label vs structure",
                xlabel = xlabels[cj],
                ylabel = "absolute relative error",
                xgridvisible=false, ygridvisible=false
            )

            for (si, step) in enumerate(steps)
                col_step = _findcol(work, Symbol(sym, :_S, step))
                col_step === nothing && ( @warn "Missing $(Symbol(sym,:_S,step)), skipping"; continue )
                step_vals = work[!, col_step][keep_idx]

                xs, errs = Float64[], Float64[]
                @inbounds for k in eachindex(full_vals)
                    f = full_vals[k]; s = step_vals[k]; x = xdata[k]
                    if any(ismissing, (f,s,x)); continue; end
                    if !(isfinite(f) && isfinite(s) && isfinite(x)); continue; end
                    denom = abs(f) + 1e-12
                    push!(errs, abs(s - f) / denom)   # <<< absolute relative error
                    push!(xs, x)
                end
                isempty(xs) && continue

                scatter!(ax, xs, errs;
                         color=colors_steps[mod1(si,length(colors_steps))],
                         alpha=0.35, markersize=5, label="S$step")

                a,b,r2 = _ols(xs, errs)
                xmn, xmx = minimum(xs), maximum(xs)
                lines!(ax, [xmn,xmx], [a + b*xmn, a + b*xmx];
                       color=colors_steps[mod1(si,length(colors_steps))], linewidth=2)

                # text!(ax, "S$step: slope=$(round(b,digits=3))  R²=$(round(r2,digits=3))";
                #      position=(xmx, ax.finallimits[].origin[2] + ax.finallimits[].widths[2]),
                #      align=(:right,:top), fontsize=10)
            end
        end
    end

    if save_plot
        save(save_path, fig; px_per_unit=pixels_per_unit)
    end
    display(fig)
end

plot_structure_vs_error(
    sim_results;
    steps = [1,2,3,5],
    metrics = [
        (:resilience, "Resilience"),
        (:reactivity, "Reactivity"),
        (:rt_pulse, "Return Time"),
        (:after_press, "Persistence"),
    ],
    remove_unstable = false,
    save_plot = true,
    save_path = "RemakingThePaper/Figures/Structure_vs_Error.png",
    resolution = (1100, 720),
    pixels_per_unit = 6.0,
    rng = MersenneTwister(7)
)

#########################################################
#########################################################
#################### Figure 3: ##########################
#########################################################
#########################################################
"""
plot_species_level_SL_correlations(
    df::DataFrame;
    steps = [1, 2, 3, 5],
    which::Symbol = :all,        # :all, :R (resources), :C (consumers)
    fit_to_1_1_line::Bool = true,
    subsample_frac::Float64 = 0.5,
    max_points::Int = 200_000,
    alpha::Float64 = 0.15,
    sl_max::Real = 100,          # per-point cutoff on either axis
    remove_unstable::Bool = true,
    save_plot::Bool = false,
    filename::String = "Figures/species_level_SL_alignment.png",
    resolution = (1100, 320),
    pixels_per_unit = 6.0,
    seed::Union{Nothing,Int} = nothing
)

Species-level alignment of self-regulation loss:
scatter **SL_full (x)** vs **SL_S{step} (y)** pooling *all species* across *all communities*.
Shows 1:1 line and reports R² to that line.
"""
function plot_species_level_SL_correlations(
    df::DataFrame;
    steps = [1, 2, 3, 5],
    which::Symbol = :all,        # :all | :R | :C
    fit_to_1_1_line::Bool = true,
    subsample_frac::Float64 = 0.5,
    max_points::Int = 200_000,
    alpha::Float64 = 0.15,
    sl_max::Real = 100,
    remove_unstable::Bool = false,
    save_plot::Bool = false,
    filename::String = "RemakingThePaper/Figures/species_level_SL_alignment.png",
    resolution = (1100, 350),
    pixels_per_unit = 6.0,
    seed::Union{Nothing,Int} = nothing
)
    # ---------- helpers ----------
    names_set = Set(names(df))
    _findcol(sym::Symbol) = (sym in names_set ? sym :
                             (String(sym) in names_set ? String(sym) : nothing))

    _r2_to_1to1(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = begin
        if isempty(x) || isempty(y); return NaN; end
        μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
        sst == 0 ? NaN : 1 - ssr/sst
    end

    seed === nothing || Random.seed!(seed)

    # required columns
    col_SL_full = :SL_full
    col_R       = :R
    col_S       = :S

    # optional filter: remove rows with positive resilience (full or in any chosen step)
    if remove_unstable
        res_cols = Symbol[:resilience_full; (Symbol(:resilience, :_S, s) for s in steps)...]
        keep = trues(nrow(df))
        for (i,row) in enumerate(eachrow(df))
            bad = false
            for rc in res_cols
                col = _findcol(rc)
                col === nothing && continue
                v = row[col]
                if !(ismissing(v)) && isfinite(v) && v > 0
                    bad = true; break
                end
            end
            keep[i] = !bad
        end
        kept = count(keep)
        println("Removed unstable rows: kept $kept / $(nrow(df)).")
        df = df[keep, :]
        if kept == 0
            @warn "No rows to plot after filtering."
            return nothing
        end
    end

    # plotting
    step_names = Dict(
        1 => "Rewiring",
        2 => "Rewiring + ↻C",
        3 => "Rewiring + ↻IS",
        4 => "Rewiring + ↻C + ↻IS",
        5 => "Changing groups"
    )
    colors = [:firebrick, :orangered, :crimson, :darkred, :teal]

    fig = Figure(size = resolution)
    Label(fig[0, 2:3], "Remove unstable: $(remove_unstable)", fontsize = 20, font = :bold, halign = :left)

    for (j, step) in enumerate(steps)
        col_step = _findcol(Symbol(:SL, :_S, step))
        if col_step === nothing
            @warn "Missing SL_S$step; skipping panel."
            Axis(fig[1, j]; title="Missing SL_S$step", xgridvisible=false, ygridvisible=false)
            continue
        end

        xs = Float64[]
        ys = Float64[]
        append!(xs, Float64[])  # ensure concretely typed
        append!(ys, Float64[])

        # gather pairs across all rows/species
        for row in eachrow(df)
            SLfull = row[col_SL_full]        
            SLstep = row[col_step]           
            Rval   = Int(row[col_R])
            Sval   = Int(row[col_S])
            m = min(length(SLfull), length(SLstep), Sval)
            m == 0 && continue

            @inbounds for k in 1:m
                is_cons = k > Rval          # consumers are indices (R+1..S)
                keep_species = which === :all || (which === :C && is_cons) || (which === :R && !is_cons)
                keep_species || continue

                xi = SLfull[k]; yi = SLstep[k]
                if isfinite(xi) && isfinite(yi) && abs(xi) <= sl_max && abs(yi) <= sl_max
                    push!(xs, float(xi)); push!(ys, float(yi))
                end
            end
        end

        N = length(xs)
        if N == 0
            Axis(fig[1, j]; title="No data", xgridvisible=false, ygridvisible=false)
            continue
        end

        # subsample / cap
        nkeep = min(Int(clamp(round(N*subsample_frac), 1, N)), max_points)
        if nkeep < N
            idx = sort!(sample(1:N, nkeep; replace=false))
            xs = xs[idx]; ys = ys[idx]
        end

        mn = min(minimum(xs), minimum(ys))
        mx = max(maximum(xs), maximum(ys))
        if !(isfinite(mn) && isfinite(mx)) || mn == mx
            mn, mx = -1.0, 1.0
        end

        grp = which === :R ? " (R)" : which === :C ? " (C)" : ""
        ax = Axis(fig[1, j];
            title = "SL: $(get(step_names, step, "Step $step"))$grp",
            xlabel = "SL_full",
            ylabel = (j == 1 ? "SL_S$step" : ""),
            limits = ((mn, mx), (mn, mx)),
            xgridvisible=false, ygridvisible=false,
            xlabelsize=11, ylabelsize=11, titlesize=12,
            xticklabelsize=10, yticklabelsize=10
        )

        scatter!(ax, xs, ys; color=colors[mod1(j, length(colors))], markersize=3.5,
                 transparency=true, alpha=alpha)
        lines!(ax, [mn, mx], [mn, mx]; color=:black, linestyle=:dash)

        if fit_to_1_1_line
            r2 = _r2_to_1to1(xs, ys)
            isfinite(r2) && text!(ax, "R²=$(round(r2, digits=3))";
                position=(mx, mn), align=(:right,:bottom), fontsize=12, color=:black)
        else
            r = cor(xs, ys)
            isfinite(r) && text!(ax, "r=$(round(r, digits=3))";
                position=(mx, mn), align=(:right,:bottom), fontsize=12, color=:black)
        end
    end

    display(fig)
    if save_plot
        save(filename, fig; px_per_unit=pixels_per_unit)
    end
end

plot_species_level_SL_correlations(
    sim_results;
    steps = [1, 2, 3, 5],
    which = :all,        # :all | :R | :C
    fit_to_1_1_line = true,
    subsample_frac = 0.5,
    max_points = 200_000,
    alpha = 0.15,
    sl_max = 100,
    remove_unstable = false,
    save_plot = true,
    filename = "RemakingThePaper/Figures/species_level_SL_alignment.png",
    resolution = (1100, 320),
    pixels_per_unit = 6.0,
    seed = nothing
)

######################################################
######################################################
################### Figure 4 #########################
######################################################
######################################################
function plot_structure_vs_error_binned(
    df;
    steps = [1,2,3,5],
    metrics = [
        (:resilience, "Resilience"),
        (:reactivity, "Reactivity"),
        (:rt_pulse, "Return Time"),
        (:after_press, "Persistence"),
    ],
    n_bins::Int = 12,
    remove_unstable::Bool = false,
    save_plot::Bool = true,
    save_path::String = "RemakingThePaper/Figures/Structure_vs_Error_binned.png",
    resolution = (1100, 720),
    pixels_per_unit = 6.0
)
    _findcol(df, sym::Symbol) = begin
        ns = Set(names(df))
        sym in ns ? sym : (String(sym) in ns ? String(sym) : nothing)
    end
    _cv(v) = (isempty(v) || mean(v) == 0) ? NaN : std(v) / mean(v)
    balanced_blocks(n::Int, blocks::Int) = repeat(1:blocks, inner=ceil(Int, n/blocks))[1:n]
    function _connectance(A, R)
        S = size(A,1); C = S - R; E = 0
        @inbounds for ic in 1:C, jr in 1:R
            E += (A[R+ic, jr] != 0.0)
        end
        (R*C == 0) ? NaN : E/(R*C)
    end
    function _within_fraction(A, R, blocks)
        S = size(A,1); C = S - R
        r_block = balanced_blocks(R, blocks); c_block = balanced_blocks(C, blocks)
        tot = 0; win = 0
        @inbounds for ic in 1:C, jr in 1:R
            if A[R+ic, jr] != 0.0
                tot += 1; win += (c_block[ic] == r_block[jr]) ? 1 : 0
            end
        end
        tot == 0 ? NaN : win/tot
    end
    function _degcv_consumer(A, R)
        S = size(A,1); C = S - R
        deg = Vector{Int}(undef, C)
        @inbounds for ic in 1:C
            deg[ic] = count(!iszero, A[R+ic, 1:R])
        end
        _cv(deg)
    end

    work = df
    if remove_unstable
        keep = trues(nrow(work))
        for (i,row) in enumerate(eachrow(work))
            for s in steps
                col = _findcol(work, Symbol(:resilience, :_S, s))
                if col !== nothing
                    v = row[col]
                    if !(ismissing(v)) && isfinite(v) && v > 0
                        keep[i] = false; break
                    end
                end
            end
        end
        total0 = nrow(work)
        work = work[keep, :]
        println("Removed unstable: kept $(nrow(work)) / $total0")
    end
    nrow(work) == 0 && (@warn "No rows to plot after filtering."; return nothing)

    col_p      = :p_final
    col_Sfull  = :S_full
    col_blocks = :blocks

    # realized structure for base community
    x_conn  = Float64[]; x_within = Float64[]; x_degcv = Float64[]; keep_idx = Int[]
    for (i,row) in enumerate(eachrow(work))
        K,A = row[col_p]; n = size(A,1)
        bestR, bestScore = 1, -1
        for Rt in 1:(n-1)
            score = count(<(0.0), @view A[1:Rt, (Rt+1):n])
            if score > bestScore; bestScore = score; bestR = Rt; end
        end
        R = bestR
        push!(x_conn,   _connectance(A, R))
        push!(x_within, _within_fraction(A, R, row[col_blocks]))
        push!(x_degcv,  _degcv_consumer(A, R))
        push!(keep_idx, i)
    end
    Xs = (x_conn, x_within, x_degcv)
    xlabels = ("Connectance (realized)", "Within-fraction (realized)", "Consumer degree CV (realized)")

    colors_steps = [:steelblue, :darkorange, :seagreen, :purple, :brown]
    fig = Figure(size = resolution)

    for (ri, (sym, label)) in enumerate(metrics)
        col_full = _findcol(work, Symbol(sym, :_full))
        col_full === nothing && (@warn "Missing $(Symbol(sym,:_full)), skipping $(sym)."; continue)
        full_vals = work[!, col_full][keep_idx]

        for (cj, X) in enumerate(Xs)
            ax = Axis(fig[ri, cj];
                title = "$label: mean abs. relative error vs structure",
                xlabel = xlabels[cj],
                ylabel = "mean abs. relative error",
                xgridvisible=false, ygridvisible=false)

            x = X
            # define bins
            xmin, xmax = minimum(x), maximum(x)
            if !isfinite(xmin) || !isfinite(xmax) || xmin == xmax
                @warn "Degenerate structure axis for $(xlabels[cj]); skipping."
                continue
            end
            edges = range(xmin, xmax; length=n_bins+1) |> collect
            centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])

            for (si, step) in enumerate(steps)
                col_step = _findcol(work, Symbol(sym, :_S, step))
                col_step === nothing && ( @warn "Missing $(Symbol(sym,:_S,step)), skipping"; continue )
                step_vals = work[!, col_step][keep_idx]

                # bin & average
                means = fill(NaN, n_bins); counts = zeros(Int, n_bins)
                for k in eachindex(x)
                    f = full_vals[k]; s = step_vals[k]; xv = x[k]
                    if any(ismissing,(f,s,xv)) || !(isfinite(f)&&isfinite(s)&&isfinite(xv)); continue; end
                    denom = abs(f) + 1e-12
                    err = abs(s - f) / denom
                    # bin index
                    j = clamp(searchsortedlast(edges, xv), 1, n_bins)
                    if counts[j] == 0
                        means[j] = err; counts[j] = 1
                    else
                        means[j] = (means[j]*counts[j] + err) / (counts[j]+1)
                        counts[j] += 1
                    end
                end
                # draw line (skip NaN gaps)
                lines!(ax, centers, means; color=colors_steps[mod1(si,length(colors_steps))], label="S$step")
            end
            if ri == 2 && cj == 1
                axislegend(ax; position=:rt, framevisible=false, labelsize=8)
            end
        end
    end

    if save_plot
        save(save_path, fig; px_per_unit=pixels_per_unit)
    end
    display(fig)
end

plot_structure_vs_error_binned(
    sim_results;
     steps = [1,2,3,5],
    metrics = [
        (:resilience, "Resilience"),
        (:reactivity, "Reactivity"),
        (:rt_pulse, "Return Time"),
        (:after_press, "Persistence"),
    ],
    n_bins = 30,
    remove_unstable = false,
    save_plot = true,
    save_path = "RemakingThePaper/Figures/Structure_vs_Error_binned.png",
    resolution = (1100, 720),
    pixels_per_unit = 6.0
)