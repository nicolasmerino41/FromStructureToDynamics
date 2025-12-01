"""
    modularity_network(S, L; m=0.0, K=4, rng=Random.default_rng())

Generate a directed trophic network with modularity m ∈ [0,1].

m = 0 → random network (Erdős–Rényi)
m = 1 → strongly modular (mostly intra-module edges)
K = number of modules
"""
function modularity_network(S, L; m=0.0, K=4, rng=Random.default_rng())
    # assign modules
    modules = rand(rng, 1:K, S)

    # baseline link probability for overall connectance
    p0 = L / (S*(S-1))

    # modularity-controlled probabilities
    p_in  = p0 + m*(1 - p0)     # increases with modularity
    p_out = p0*(1 - m)          # decreases with modularity

    A = zeros(Int, S, S)

    # fill adjacency matrix probabilistically
    for i in 1:S
        for j in 1:S
            if i == j; continue; end
            if modules[i] == modules[j]
                # intra-module
                if rand(rng) < p_in
                    A[i,j] = 1
                end
            else
                # inter-module
                if rand(rng) < p_out
                    A[i,j] = 1
                end
            end
        end
    end

    # adjust exact link count to L by random sampling
    current_L = sum(A)
    if current_L > L
        # remove extra edges
        edges = findall(A .== 1)
        to_remove = rand(rng, edges, current_L - L)
        for I in to_remove
            A[I] = 0
        end

    elseif current_L < L
        # add missing edges
        nonedges = findall((A .== 0) .& .!I(S))   # This actually expands to a full Bool matrix
        to_add = rand(rng, nonedges, L - current_L)
        for I in to_add
            i, j = Tuple(I)
            A[i, j] = 1
        end
    end

    return A
end

function run_pipeline_modularity_rmed!(
        S, L;
        m_vals = range(0, 1, length=9),
        replicates = 30,
        mag_abs = 1.0,
        mag_cv = 0.5,
        u_cv = 0.5,
        corr = 1.0,
        u = u,
        t_vals = 10 .^ range(log10(0.01), log10(100.0), length=30),
        rng = Random.default_rng()
    )

    results = Dict{Float64, Vector{Vector{Float64}}}()
    τ_axes  = Dict{Float64, Vector{Float64}}()
    Js      = Dict{Float64, Matrix{Float64}}()

    for m in m_vals
        curves = Vector{Vector{Float64}}()

        for rep in 1:replicates
            A = modularity_network(S, L; m=m, rng=rng)
            W = random_trophic_interaction_matrix(A; mag_abs=mag_abs, mag_cv=mag_cv, corr=corr, rng=rng)

            J = jacobian(W, u)
            Js[m] = J

            rcurve = compute_rmed_curve(J, u, t_vals)
            push!(curves, rcurve)
        end

        τ_vals, t95, Rinf = standardized_time_axis(t_vals, Js[m])
        τ_axes[m] = τ_vals

        println("Finished m = $m   (R∞=$(round(Rinf,digits=3)), t95=$(round(t95,digits=3)))")

        results[m] = curves
    end

    return results, τ_axes, Js, t_vals
end

function build_reference_rmed_modularity!(
        m_ref, S, L, η, u, t_vals;
        replicates_ref = 30,
        mag_abs = 1.0,
        mag_cv  = 0.5,
        corr    = 1.0,
        rng     = Random.default_rng()
    )

    curves = Vector{Vector{Float64}}()

    for rep in 1:replicates_ref
        A = modularity_network(S, L; m=m_ref, rng=rng)
        W = random_trophic_interaction_matrix(A; mag_abs=mag_abs, mag_cv=mag_cv,
                                              corr=corr, rng=rng)
        J = jacobian(W, u)
        push!(curves, compute_rmed_curve(J, u, t_vals))
    end

    # standardized τ using one ref
    A0 = modularity_network(S, L; m=m_ref, rng=rng)
    W0 = random_trophic_interaction_matrix(A0; mag_abs=mag_abs, mag_cv=mag_cv,
                                           corr=corr, rng=rng)
    J0 = jacobian(W0, u)
    τ_ref, _, _ = standardized_time_axis(t_vals, J0)

    return curves, τ_ref
end

function plot_rmed_grid_modularity(
        results, τ_axes;
        m_vals,
        reference = :lowest,
        S::Int, L::Int, η,
        u, t_vals,
        replicates_ref = 30,
        mag_abs = 1.0,
        mag_cv  = 0.5,
        corr    = 1.0,
        figsize = (1600,1400),
        rng = Random.default_rng()
    )

    m_ref = (reference == :lowest)  ? minimum(m_vals) :
            (reference == :highest) ? maximum(m_vals) :
            error("reference must be :lowest or :highest")

    ref_curves, τ_ref = build_reference_rmed_modularity!(
        m_ref, S, L, η, u, t_vals; replicates_ref, mag_abs, mag_cv, corr, rng
    )

    fig = Figure(size=figsize)
    rows, cols = 3, 3
    idx = 1

    for m in m_vals
        curves = results[m]
        τ_vals = τ_axes[m]

        r = div(idx-1, cols)+1
        c = mod(idx-1, cols)+1

        ax = Axis(fig[r,c]; title="m = $(round(m,digits=3))",
                  xlabel="τ", ylabel="rₘₑd(τ)")

        for curve in ref_curves
            lines!(ax, τ_ref, curve; color=(:red,0.25))
        end

        for curve in curves
            lines!(ax, τ_vals, curve; color=(:black,0.25))
        end

        idx += 1
        xlims!(ax, -0.01, 1.1)
    end

    display(fig)
end

function plot_rmed_mean_grid_modularity(
        results, τ_axes;
        m_vals,
        reference = :lowest,
        S::Int, L::Int, η,
        u, t_vals,
        replicates_ref = 30,
        mag_abs = 1.0,
        mag_cv  = 0.5,
        corr    = 1.0,
        figsize = (1600,1400),
        rng = Random.default_rng()
    )

    m_ref = (reference == :lowest)  ? minimum(m_vals) :
            (reference == :highest) ? maximum(m_vals) :
            error("reference must be :lowest or :highest")

    ref_curves, τ_ref = build_reference_rmed_modularity!(
        m_ref, S, L, η, u, t_vals;
        replicates_ref=replicates_ref,
        mag_abs=mag_abs, mag_cv=mag_cv,
        corr=corr, rng=rng
    )

    ref_mean = mean_curve(ref_curves)

    fig = Figure(size=figsize)
    rows, cols = 3, 3
    idx = 1

    for m in m_vals
        curves = results[m]
        mean_m = mean_curve(curves)
        τ_vals = τ_axes[m]

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        ax = Axis(fig[r,c];
            title="m = $(round(m,digits=3))",
            xlabel="τ",
            ylabel="mean rₘₑd(τ)"
        )

        lines!(ax, τ_ref, ref_mean; color=:red, linewidth=3)
        lines!(ax, τ_vals, mean_m;  color=:black, linewidth=3)

        idx += 1
        xlims!(ax, -0.01, 1.1)
    end

    display(fig)
end

function plot_rmed_delta_grid_modularity(
        results, τ_axes;
        m_vals,
        reference = :lowest,
        S::Int, L::Int, η,
        u, t_vals,
        replicates_ref = 30,
        mag_abs = 1.0,
        mag_cv  = 0.5,
        corr    = 1.0,
        figsize = (1600,1400),
        rng = Random.default_rng()
    )

    # reference
    m_ref = (reference == :lowest)  ? minimum(m_vals) :
            (reference == :highest) ? maximum(m_vals) :
            error("reference must be :lowest or :highest")

    ref_curves, τ_ref = build_reference_rmed_modularity!(
        m_ref, S, L, η, u, t_vals;
        replicates_ref=replicates_ref,
        mag_abs=mag_abs, mag_cv=mag_cv,
        corr=corr, rng=rng
    )
    ref_mean = mean_curve(ref_curves)

    # ---------------------------------------------------------
    # GLOBAL y-limits — SAME FOR ALL SUBPLOTS
    # ---------------------------------------------------------
    all_deltas = Float64[]
    for m in m_vals
        Δ = abs.(delta_curve(mean_curve(results[m]), ref_mean))
        append!(all_deltas, Δ)
    end

    global_ymin = minimum(all_deltas)
    global_ymax = maximum(all_deltas)

    # Add some margin
    y_min = global_ymin - 0.1 * abs(global_ymax)
    y_max = global_ymax + 0.1 * abs(global_ymax)
    # ---------------------------------------------------------

    fig = Figure(size=figsize)
    rows, cols = 3, 3
    idx = 1

    for m in m_vals

        mean_m = mean_curve(results[m])
        Δ = abs.(delta_curve(mean_m, ref_mean))
        τ_vals = τ_axes[m]

        # -------- FIND THE TRUE PEAK FOR THIS SUBPLOT --------
        peak_idx = argmax(Δ)
        Δ_peak   = Δ[peak_idx]         # y-value of the peak (correct)
        τ_peak   = τ_vals[peak_idx]    # x-value used for label
        # ------------------------------------------------------

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        ax = Axis(fig[r,c];
            title="m = $(round(m, digits=3))",
            xlabel="τ",
            ylabel="|Δ rₘₑd(τ)|",
        )

        # Δ curve
        lines!(ax, τ_vals, Δ; color=:blue, linewidth=3)

        # Horizontal line at peak — must match EXACTLY
        hlines!(ax, Δ_peak; color=:red, linestyle=:dash, linewidth=2)

        # τ label
        text!(
            ax,
            τ_peak, Δ_peak;
            text = "τ = $(round(τ_peak, digits=3))",
            align = (:center, :bottom),
            offset = (0,5),
            color = :red
        )

        # APPLY **GLOBAL** Y-LIMITS HERE (same for all panels)
        ylims!(ax, y_min, y_max)
        xlims!(ax, -0.01, 1.1)

        idx += 1
    end

    display(fig)
end

S = 120
connectance = 0.15
L = round(Int, connectance * S * (S-1))

m_vals = range(0, 1, length=9)  # modularity parameter
u = random_u(S, mean=1.0, cv=0.5)

results, τ_axes, Js, t_vals = run_pipeline_modularity_rmed!(S, L; u=u)

plot_rmed_grid_modularity(
    results, τ_axes;
    m_vals=m_vals, reference=:lowest,
    S=S, L=L, η=0.0, u=u, t_vals=t_vals
)

plot_rmed_mean_grid_modularity(
    results, τ_axes;
    m_vals=m_vals, reference=:lowest,
    S=S, L=L, η=0.0, u=u, t_vals=t_vals
)

plot_rmed_delta_grid_modularity(
    results, τ_axes;
    m_vals=m_vals, reference=:highest,
    S=S, L=L, η=0.0, u=u, t_vals=t_vals
)
