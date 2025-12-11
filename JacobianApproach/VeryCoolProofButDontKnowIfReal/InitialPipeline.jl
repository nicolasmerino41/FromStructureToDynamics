function rmed_curve(J, u;
        t_vals = 10 .^ range(log10(0.01), log10(100.0), length=30)
    )

    rvals = similar(t_vals)
    for (k, t) in enumerate(t_vals)
        rvals[k] = median_return_rate(J, u; t=t, perturbation=:biomass)
    end
    return rvals
end

function links_for_connectance(S, C)
    return round(Int, C * S * (S - 1))
end

links_for_connectance(120, 0.15)

function run_pipeline_rmed!(
    B::Int, S::Int, L::Int;
    q_targets = range(0.01, 0.4, length=9),
    replicates = 30,
    mag_abs = 0.5,
    mag_cv = 0.3,
    u_cv = 0.5,
    corr = 0.0,
    rng = Random.default_rng()
)

    t_vals = 10 .^ range(log10(0.01), log10(100.0), length=30)

    # results[q] = vector of 30 curves (each curve is length 30)
    results = Dict{Float64, Vector{Vector{Float64}}}()

    # Same u for everyone
    u = random_u(S; mean=1.0, cv=u_cv, rng=rng)

    for qtarget in q_targets
        println("Running q = ", qtarget)

        curves = Vector{Vector{Float64}}()

        for rep in 1:replicates

            # Build PPM
            b = PPMBuilder()
            η_value = 0.2
            set!(b; S=S, B=B, L=L, T=qtarget, η=η_value)
            
            net = build(b)

            A = net.A

            # Interaction matrix
            W = build_interaction_matrix(A;
                mag_abs=mag_abs,
                mag_cv=mag_cv,
                corr_aij_aji=corr,
                rng=rng
            )

            # Jacobian
            J = jacobian(W, u)

            # rmed(t) curve
            curve = rmed_curve(J, u; t_vals=t_vals)

            push!(curves, curve)
        end

        results[qtarget] = curves
    end
    
    return results, t_vals
end

function plot_rmed_grid(results, t_vals;
        q_targets = sort(collect(keys(results))),
        figsize=(1600,1400)
    )

    fig = Figure(size=figsize)

    rows, cols = 3, 3
    idx = 1

    for q in q_targets
        rcurves = results[q]

        r = (idx-1) ÷ cols + 1
        c = (idx-1) % cols + 1

        ax = Axis(fig[r, c];
            title = "q = $(round(q, digits=3))",
            xlabel = "t",
            ylabel = "rmed(t)",
            xscale = log10,
        )

        for curve in rcurves
            lines!(ax, t_vals, curve; color=(:black, 0.3))  # faint replicates
        end

        idx += 1
    end

    display(fig)
end
# plot_rmed_grid(results, t_vals)

function plot_rmed_grid_with_reference(results, t_vals;
        q_targets = sort(collect(keys(results))),
        # q_ref = 0.784587,
        figsize = (1100,720),
        title = ""
    )

    # reference curves (30 curves)
    kis = keys(results)
    q_ref = minimum(kis)
    ref_curves = results[q_ref]

    fig = Figure(size=figsize)
    Label(fig[0, 1:3], title, fontsize=18, font=:bold, halign=:center)

    rows, cols = 3, 3
    idx = 1

    for q in q_targets
        curves = results[q]

        r = (idx-1) ÷ cols + 1
        c = (idx-1) % cols + 1

        ax = Axis(fig[r, c];
            title = "q = $(round(q, digits=3))",
            xlabel = "t",
            ylabel = "rmed(t)",
            xscale = log10,
        )

        # ----- plot all reference lines (red) -----
        for curve in ref_curves
            lines!(ax, t_vals, curve; color=(:red, 0.5))
        end

        # ----- plot all lines for this q (black/transparent) -----
        for curve in curves
            lines!(ax, t_vals, curve; color=(:black, 0.3))
        end

        idx += 1
    end

    display(fig)
end

results, t_vals = run_pipeline_rmed!(
    10, 120, 2142;
    q_targets = range(0.01, 1.5, length=9),
    replicates = 30,
    mag_abs = 1.0,
    mag_cv = 0.5,
    corr = 0.0,
    u_cv = 0.5,
    rng = Random.default_rng()
)

plot_rmed_grid_with_reference(results, t_vals; title="SIMPLE TROPHIC COEHERENCE simpler network")
plot_rmed_mean_grid_with_reference(results, t_vals, title = "SIMPLE TROPHIC COEHERENCE simpler network")
plot_rmed_delta_grid(results, t_vals; title = "SIMPLE TROPHIC COEHERENCE simpler network")

for u in [0.5, 2.0], mag_abs in [0.3, 1.0], corr in [0.0, 0.99]
    results, t_vals = run_pipeline_rmed!(
        10, 50, 120;
        q_targets = range(0.01, 1.5, length=9),
        replicates = 30,
        mag_abs = mag_abs,
        mag_cv = 0.5,
        u_cv = u,
        corr = corr,
        rng = Random.default_rng()
    )

    plot_rmed_grid_with_reference(results, t_vals; title="U_cv = $u, mag_abs = $mag_abs, corr = $corr")
    plot_rmed_mean_grid_with_reference(results, t_vals, title = "U_cv = $u, mag_abs = $mag_abs, corr = $corr")
    plot_rmed_delta_grid(results, t_vals; title = "U_cv = $u, mag_abs = $mag_abs, corr = $corr")
end

for i in 1:3
    for corr in [0.0, 0.99], u in [0.5, 2.0], mag_abs in [0.3, 1.0]
        results, t_vals = run_pipeline_rmed!(
            24, 120, 2142;
            q_targets = range(0.01, 1.5, length=9),
            replicates = 30,
            mag_abs = mag_abs,
            mag_cv = 0.5,
            u_cv = u,
            corr = corr,
            rng = Random.default_rng()
        )

        if i == 1
            plot_rmed_grid_with_reference(results, t_vals; q_ref = 1.5, title="Q_ref = 1.5, U_cv = $u, mag_abs = $mag_abs, corr = $corr")
        elseif i == 2
            plot_rmed_mean_grid_with_reference(results, t_vals; q_ref = 1.5, title = "Q_ref = 1.5, U_cv = $u, mag_abs = $mag_abs, corr = $corr")
        elseif i == 3
            plot_rmed_delta_grid(results, t_vals; q_ref = 1.5, title = "Q_ref = 1.5, U_cv = $u, mag_abs = $mag_abs, corr = $corr")
        end
    end
end