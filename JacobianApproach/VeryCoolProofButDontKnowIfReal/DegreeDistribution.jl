"""
    degree_distribution_network(S, L; alpha=0.0, rng=Random.default_rng())

Generate a directed trophic network of size S, total links L,
with degree heterogeneity controlled by α.

α = 0   → nearly uniform
α = 1   → moderately skewed
α = 2   → heavy-tailed / scale-free-like
"""
function degree_distribution_network(S, L; alpha=0.0, rng=Random.default_rng())

    # Step 1: sample out-degree propensities
    # mean=1, sigma=alpha determines skewness
    sigma = alpha
    mu = -sigma^2 / 2        # so exp(mu + sigma^2/2) = 1
    props = rand(rng, LogNormal(mu, sigma), S)

    # Step 2: convert propensities → expected degrees summing to L
    props_norm = props ./ sum(props)
    out_degrees = round.(Int, props_norm .* L)

    # adjust to ensure sum(out_degrees) = L
    diff = L - sum(out_degrees)
    if diff != 0
        # randomly distribute leftover ±1 adjustments
        idxs = rand(rng, 1:S, abs(diff))
        for i in idxs
            out_degrees[i] += sign(diff)
        end
    end

    # Step 3: build adjacency matrix
    A = zeros(Int, S, S)

    for i in 1:S
        k = out_degrees[i]
        if k > 0
            # choose k distinct targets (excluding i)
            targets = shuffle(rng, filter(j -> j != i, 1:S))[1:min(k, S-1)]
            A[i, targets] .= 1
        end
    end

    return A
end

function random_trophic_interaction_matrix(A; mag_abs=1.0, mag_cv=0.5, corr=1.0, rng=Random.default_rng())
    S = size(A,1)
    W = zeros(Float64, S, S)

    sigma = sqrt(log(1 + mag_cv^2))
    mu = log(mag_abs) - sigma^2/2

    for i in 1:S, j in 1:S
        if A[i,j] == 1
            # predator j → prey i
            a = rand(rng, LogNormal(mu, sigma))        # positive
            if corr == 1
                b = a
            elseif corr == 0
                b = rand(rng, LogNormal(mu, sigma))
            else
                # correlated magnitudes
                a2 = rand(rng, LogNormal(mu, sigma))
                b = corr * a + (1-corr) * a2
            end

            W[j,i] =  a      # predator benefits
            W[i,j] = -b      # prey loses
        end
    end
    return W
end

function run_pipeline_degree_rmed!(
        S, L;
        alpha_vals = range(0, 2, length=9),
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

    # u = random_u(S; mean=1.0, cv=u_cv, rng=rng)

    for α in alpha_vals
        curves = Vector{Vector{Float64}}()

        for rep in 1:replicates
            A = degree_distribution_network(S, L; alpha=α, rng=rng)
            W = random_trophic_interaction_matrix(A; mag_abs=mag_abs, mag_cv=mag_cv, corr=corr, rng=rng)

            J = jacobian(W, u)
            Js[α] = J

            rcurve = compute_rmed_curve(J, u, t_vals)
            push!(curves, rcurve)
        end

        τ_vals, t95, Rinf = standardized_time_axis(t_vals, Js[α])
        τ_axes[α] = τ_vals

        println("Finished α = $α   (R∞=$(round(Rinf,digits=3)), t95=$(round(t95,digits=3)))")

        results[α] = curves
    end

    return results, τ_axes, Js, t_vals
end

function build_reference_rmed_degree!(
        α_ref, S, L, η, u, t_vals;
        replicates_ref = 30,
        mag_abs = 1.0,
        mag_cv  = 0.5,
        corr    = 1.0,
        rng     = Random.default_rng()
    )

    curves = Vector{Vector{Float64}}()

    for rep in 1:replicates_ref
        A = degree_distribution_network(S, L; alpha=α_ref, rng=rng)
        W = random_trophic_interaction_matrix(A; mag_abs=mag_abs, mag_cv=mag_cv,
                                              corr=corr, rng=rng)
        J = jacobian(W, u)

        rcurve = compute_rmed_curve(J, u, t_vals)
        push!(curves, rcurve)
    end

    # τ-axis from one reference J (small differences do not matter)
    A0 = degree_distribution_network(S, L; alpha=α_ref, rng=rng)
    W0 = random_trophic_interaction_matrix(A0; mag_abs=mag_abs, mag_cv=mag_cv,
                                           corr=corr, rng=rng)
    J0 = jacobian(W0, u)

    τ_ref, _, _ = standardized_time_axis(t_vals, J0)

    return curves, τ_ref
end

function plot_rmed_grid_degree(
        results, τ_axes;
        alpha_vals,
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

    α_ref = (reference == :lowest)  ? minimum(alpha_vals) :
            (reference == :highest) ? maximum(alpha_vals) :
            error("reference must be :lowest or :highest")

    # regenerate reference curves
    ref_curves, τ_ref = build_reference_rmed_degree!(
        α_ref, S, L, η, u, t_vals;
        replicates_ref=replicates_ref,
        mag_abs=mag_abs, mag_cv=mag_cv,
        corr=corr, rng=rng
    )

    fig = Figure(size=figsize)
    rows, cols = 3, 3
    idx = 1

    for α in alpha_vals
        curves = results[α]
        τ_vals = τ_axes[α]

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        ax = Axis(fig[r,c];
            title="α = $(round(α,digits=3))",
            xlabel="τ",
            ylabel="rₘₑd(τ)"
        )

        # reference replicates (in red)
        for curve in ref_curves
            lines!(ax, τ_ref, curve; color=(:red,0.25))
        end

        # community curves (in black)
        for curve in curves
            lines!(ax, τ_vals, curve; color=(:black,0.25))
        end

        idx += 1
    end

    display(fig)
end

function plot_rmed_mean_grid_degree(
        results, τ_axes;
        alpha_vals,
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

    α_ref = (reference == :lowest)  ? minimum(alpha_vals) :
            (reference == :highest) ? maximum(alpha_vals) :
            error("reference must be :lowest or :highest")

    ref_curves, τ_ref = build_reference_rmed_degree!(
        α_ref, S, L, η, u, t_vals;
        replicates_ref=replicates_ref,
        mag_abs=mag_abs, mag_cv=mag_cv,
        corr=corr, rng=rng
    )

    ref_mean = mean_curve(ref_curves)

    fig = Figure(size=figsize)
    rows, cols = 3, 3
    idx = 1

    for α in alpha_vals
        curves = results[α]
        mean_α = mean_curve(curves)
        τ_vals = τ_axes[α]

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        ax = Axis(fig[r,c];
            title="α = $(round(α,digits=3))",
            xlabel="τ",
            ylabel="mean rₘₑd(τ)"
        )

        lines!(ax, τ_ref, ref_mean; color=:red, linewidth=3)
        lines!(ax, τ_vals, mean_α;  color=:black, linewidth=3)

        idx += 1
    end

    display(fig)
end

function plot_rmed_delta_grid_degree(
        results, τ_axes;
        alpha_vals,
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

    α_ref = (reference == :lowest)  ? minimum(alpha_vals) :
            (reference == :highest) ? maximum(alpha_vals) :
            error("reference must be :lowest or :highest")

    ref_curves, τ_ref = build_reference_rmed_degree!(
        α_ref, S, L, η, u, t_vals;
        replicates_ref=replicates_ref,
        mag_abs=mag_abs, mag_cv=mag_cv,
        corr=corr, rng=rng
    )
    ref_mean = mean_curve(ref_curves)

    # global y-limits
    all_deltas = Float64[]
    for α in alpha_vals
        Δ = delta_curve(mean_curve(results[α]), ref_mean)
        append!(all_deltas, Δ)
    end
    global_min = minimum(all_deltas)
    global_max = maximum(all_deltas)
    y_min = global_min - 0.1 * abs(global_max)
    y_max = global_max + 0.1 * abs(global_max)

    fig = Figure(size=figsize)
    rows, cols = 3, 3
    idx = 1

    for α in alpha_vals
        mean_α = mean_curve(results[α])
        Δ = delta_curve(mean_α, ref_mean)

        τ_vals = τ_axes[α]

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        ax = Axis(fig[r,c];
            title="α = $(round(α,digits=3))",
            xlabel="τ",
            ylabel="|Δ rₘₑd(τ)|",
        )

        lines!(ax, τ_vals, Δ; color=:blue, linewidth=3)
        ylims!(ax, y_min, y_max)

        idx += 1
    end

    display(fig)
end

S = 120
connectance = 0.15
L = round(Int, connectance * S * (S-1))
alpha_vals = range(0, 2, length=9)
u = random_u(S, mean=1.0, cv=0.5)
results, τ_axes, Js, t_vals = run_pipeline_degree_rmed!(
    S, L; u = u,
    alpha_vals = range(0, 2, length=9),
    replicates = 30,
    mag_abs = 0.5,
    mag_cv = 0.5,
    u_cv = 0.5,
    corr = 0.0
)

plot_rmed_grid_degree(results, τ_axes;
    alpha_vals=alpha_vals,
    reference=:lowest,
    S=S, L=L, η=0.0, u=u, t_vals=t_vals)

plot_rmed_mean_grid_degree(results, τ_axes;
    alpha_vals=alpha_vals,
    reference=:lowest,
    S=S, L=L, η=0.0, u=u, t_vals=t_vals)

plot_rmed_delta_grid_degree(results, τ_axes;
    alpha_vals=alpha_vals,
    reference=:lowest,
    S=S, L=L, η=0.0, u=u, t_vals=t_vals)
