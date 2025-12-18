function plot_rmed_grid_with_reference(results, t_vals, params;
        q_targets = sort(collect(keys(results))),
        reference = :lowest,                 # :lowest, :highest, or numeric q
        S, B, L, η, u,
        replicates_ref = 30,
        figsize = (1100,720),
        title = "",
        rng = Random.default_rng()
    )

    kis = sort(collect(keys(results)))
    q_ref = if reference === :lowest
        first(kis)
    elseif reference === :highest
        last(kis)
    elseif reference isa Real
        kis[argmin(abs.(kis .- reference))]
    else
        error("reference must be :lowest, :highest, or a numeric q")
    end

    # --- build NEW baseline curves (not copied from results) ---
    ref_curves = build_reference_rmed!(
        q_ref, S, B, L, η, u, t_vals, params;
        replicates_ref=replicates_ref,
        rng=rng
    )

    fig = Figure(size=figsize)
    Label(fig[0, 1:3], title, fontsize=18, font=:bold, halign=:center)

    rows, cols = 3, 3
    idx = 1

    for q in q_targets
        curves = results[q]

        r = (idx-1) ÷ cols + 1
        c = (idx-1) % cols + 1

        ax = Axis(fig[r, c];
            title = "q = $(round(q, digits=3)) (ref=$(round(q_ref,digits=3)))",
            xlabel = "t",
            ylabel = "rmed(t)",
            xscale = log10,
        )

        # baseline (fresh)
        for curve in ref_curves
            lines!(ax, t_vals, curve; color=(:red, 0.5))
        end

        # curves for this q
        for curve in curves
            lines!(ax, t_vals, curve; color=(:black, 0.3))
        end

        idx += 1
    end

    display(fig)
end

function plot_rmed_mean_grid_with_reference(results, t_vals, params;
        q_targets = sort(collect(keys(results))),
        reference = :lowest,
        S, B, L, η, u,
        replicates_ref = 30,
        figsize = (1100,720),
        title = "",
        rng = Random.default_rng()
    )

    kis = sort(collect(keys(results)))
    q_ref = if reference === :lowest
        first(kis)
    elseif reference === :highest
        last(kis)
    elseif reference isa Real
        kis[argmin(abs.(kis .- reference))]
    else
        error("reference must be :lowest, :highest, or a numeric q")
    end

    # --- build NEW baseline curves ---
    ref_curves, _τ_ref = build_reference_rmed!(
        q_ref, S, B, L, η, u, t_vals, params;
        replicates_ref=replicates_ref,
        rng=rng
    )
    ref_mean = mean_curve(ref_curves)

    fig = Figure(size=figsize)
    Label(fig[0, 1:3], title, fontsize=18, font=:bold, halign=:center)

    rows, cols = 3, 3
    idx = 1

    for q in q_targets
        mean_q = mean_curve(results[q])

        r = (idx-1) ÷ cols + 1
        c = (idx-1) % cols + 1

        ax = Axis(fig[r, c];
            title = "q = $(round(q, digits=3)) (ref=$(round(q_ref,digits=3)))",
            xlabel = "t",
            ylabel = "mean rmed(t)",
            xscale = log10,
        )

        lines!(ax, t_vals, ref_mean; color=:red, linewidth=3)
        lines!(ax, t_vals, mean_q;   color=:black, linewidth=3)

        idx += 1
    end

    display(fig)
end

function plot_rmed_delta_grid(results, t_vals, params;
        q_targets = sort(collect(keys(results))),
        reference = :lowest,
        S, B, L, η, u,
        replicates_ref = 30,
        figsize = (1100,720),
        title = "",
        rng = Random.default_rng()
    )

    kis = sort(collect(keys(results)))
    q_ref = if reference === :lowest
        first(kis)
    elseif reference === :highest
        last(kis)
    elseif reference isa Real
        kis[argmin(abs.(kis .- reference))]
    else
        error("reference must be :lowest, :highest, or a numeric q")
    end

    # --- build NEW baseline curves ---
    ref_curves = build_reference_rmed!(
        q_ref, S, B, L, η, u, t_vals, params;
        replicates_ref=replicates_ref,
        rng=rng
    )
    ref_mean = mean_curve(ref_curves)

    # --- global y-limits across all q ---
    all_deltas = Float64[]
    for q in q_targets
        mean_q = mean_curve(results[q])
        Δ = delta_curve(mean_q, ref_mean)
        append!(all_deltas, Δ)
    end
    global_min = minimum(all_deltas)
    global_max = maximum(all_deltas)
    pad = 0.1 * max(abs(global_min), abs(global_max))
    ylims_global = (global_min - pad, global_max + pad)

    fig = Figure(size=figsize)
    Label(fig[0, 1:3], title, fontsize=18, font=:bold, halign=:center)

    rows, cols = 3, 3
    idx = 1

    for q in q_targets
        mean_q = mean_curve(results[q])
        Δ = delta_curve(mean_q, ref_mean)

        r = (idx-1) ÷ cols + 1
        c = (idx-1) % cols + 1

        ax = Axis(fig[r, c];
            title = "q = $(round(q, digits=3)) (ref=$(round(q_ref,digits=3)))",
            xlabel = "t",
            ylabel = "|Δ rmed(t)|",
            xscale = log10,
        )

        lines!(ax, t_vals, Δ; color=:blue, linewidth=3)
        ylims!(ax, ylims_global...)
        idx += 1
    end

    display(fig)
end
