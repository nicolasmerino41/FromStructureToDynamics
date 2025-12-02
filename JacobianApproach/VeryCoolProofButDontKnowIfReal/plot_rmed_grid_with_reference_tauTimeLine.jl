function plot_rmed_grid_with_reference_tauTimeLine(
        results, τ_axes, t_vals;
        q_targets = sort(collect(keys(results))),
        figsize = (2000,1400),
        title = ""
    )

    n = length(q_targets)
    cols = 3
    rows = ceil(Int, n / cols)

    fig = Figure(size = figsize)
    Label(fig[0,1:3], title, fontsize=18, font=:bold, halign=:center)

    # reference
    q_ref = q_targets[1]
    ref_curves = results[q_ref]
    τ_ref      = τ_axes[q_ref]

    idx_ref = argmin(abs.(τ_ref .- 1))
    tP_ref = t_vals[idx_ref]

    for (idx, q) in enumerate(q_targets)
        curves = results[q]
        τ = τ_axes[q]

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1
        @info "Placing q=$q at r=$r, c=$c"

        ax = Axis(fig[r, c];
            title = "q = $(round(q,digits=3))",
            xlabel = "t",
            ylabel = "rmed(t)",
            xscale = log10
        )

        # reference line
        vlines!(ax, tP_ref; color=(:red, 0.45), linewidth=2)

        # reference curves
        for curve in ref_curves
            lines!(ax, t_vals, curve; color=(:red, 0.3))
        end

        # τ(q)=1 line
        idx_q = argmin(abs.(τ .- 1))
        tP_q = t_vals[idx_q]
        vlines!(ax, tP_q; color=(:black, 0.35))

        # curves
        for curve in curves
            lines!(ax, t_vals, curve; color=(:black, 0.3))
        end
    end

    display(fig)
end

plot_rmed_grid_with_reference_tauTimeLine(results, τ_axes, t_vals; title = "MODULARITY")
function plot_rmed_grid_with_reference_t95(
    results, τ_axes, t_vals;
    q_targets = sort(collect(keys(results))),
    q_ref = first(sort(collect(keys(results)))),
    t_or_tau = :t,
    figsize = (1600,1400),
    title = ""
)
    fig = Figure(size = figsize)
    Label(fig[0,1:3], title, fontsize = 20, font = :bold, halign = :center)

    rows, cols = 3, 3
    idx = 1

    # --- reference data ---
    ref_curves = results[q_ref]
    τ_ref      = τ_axes[q_ref]

    # ----------------------------------------------------
    # function to compute t95 directly from the curve
    # ----------------------------------------------------
    true_t95 = function(t_vals, rcurve)
        idx = findfirst(rcurve .<= 0.05)
        return isnothing(idx) ? Inf : t_vals[idx]
    end

    # compute reference t95 (take mean of ref curves)
    t95_ref = begin
        vals = [true_t95(t_vals, c) for c in ref_curves]
        minimum(vals)   # most conservative, like your plot
    end

    for q in q_targets

        curves = results[q]
        τ_q    = τ_axes[q]

        # compute t95 for this q (again: minimum is most conservative)
        t95_q = begin
            vals = [true_t95(t_vals, c) for c in curves]
            minimum(vals)
        end

        # grid position
        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        xlabel = t_or_tau == :t ? "t" : "τ"

        if t_or_tau == :t
            ax = Axis(fig[r, c];
                title = "q = $(round(q,digits=3))",
                xlabel = xlabel,
                ylabel = "rmed(t)",
                xscale = log10
            )
        else
            ax = Axis(fig[r, c];
                title = "q = $(round(q,digits=3))",
                xlabel = xlabel,
                ylabel = "rmed(τ)"
            )
            xlims!(ax, -0.01, 1.1)
        end

        # ==========================================================
        # MODE 1: PLOT VS REAL TIME t
        # ==========================================================
        if t_or_tau == :t

            # --- red vertical reference t95 ---
            if isfinite(t95_ref)
                vlines!(ax, t95_ref; color=(:red,0.5), linewidth=2)
            end

            # --- red reference curves ---
            for curve in ref_curves
                lines!(ax, t_vals, curve; color=(:red,0.35))
            end

            # --- black vertical lines for q ---
            if isfinite(t95_q)
                vlines!(ax, t95_q; color=(:black,0.35))
            end

            # --- black curves ---
            for curve in curves
                lines!(ax, t_vals, curve; color=(:black,0.3))
            end

        # ==========================================================
        # MODE 2: TAU MODE → reparametrize curves by τ
        # ==========================================================
        else
            # reference curves (red) with τ_ref
            if !all(isnan, τ_ref)
                for curve in ref_curves
                    lines!(ax, τ_ref, curve; color=(:red,0.35))
                end
            end

            # curves for the current q (black)
            if !all(isnan, τ_q)
                for curve in curves
                    lines!(ax, τ_q, curve; color=(:black,0.3))
                end
            end
        end

        idx += 1
    end

    display(fig)
end

plot_rmed_grid_with_reference_t95(results, τ_axes, t_vals; t_or_tau=:tau)