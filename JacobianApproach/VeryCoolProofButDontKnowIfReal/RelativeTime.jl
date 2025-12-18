using Interpolations
# --- Asymptotic resilience: R∞ = -Re(λ_dom) ---
function asymptotic_resilience(J)
    λmax = maximum(real.(eigvals(J)))
    return -λmax
end

# --- Standardised recovery time τ(t) = t / tP ---
function standardized_time_axis(t_vals, J; P = 0.05)
    Rinf = asymptotic_resilience(J)
    # @assert Rinf > 0 "System not asymptotically stable (R∞ ≤ 0)."

    tP = -log(P) / Rinf        # time to reach P fraction of displacement
    τ = t_vals ./ tP
    return τ, tP, Rinf
end

function compute_rmed_curve(J, u, t_vals)
    r = similar(t_vals)
    for i in eachindex(t_vals)
        r[i] = median_return_rate(J, u; t=t_vals[i], perturbation=:biomass)
    end
    return r
end

mean_curve(curves) = [mean(c[i] for c in curves) for i in 1:length(curves)]
delta_curve(mean_q, mean_ref) = abs.(mean_q .- mean_ref)

############################################################
# 1. Compute D(t) = || exp(J t) u ||
############################################################
function compute_distance_curve(J, u, t_vals)
    D = similar(t_vals)
    for (i, t) in enumerate(t_vals)
        x_t = exp(J * t) * u        # state after time t
        D[i] = norm(x_t)            # distance to equilibrium
    end
    return D
end

############################################################
# 2. Compute t95: time when D(t)/D(0) <= 0.05
############################################################
function compute_t95(D, t_vals; threshold = 0.05)
    D0 = D[1]
    ratio = D ./ D0
    idx = findfirst(ratio .<= threshold)
    return isnothing(idx) ? Inf : t_vals[idx]
end

############################################################
# 3. Compute τ-axis = t_vals / t95
############################################################
function compute_tau_axis(t_vals, t95)
    if isinf(t95)
        return fill(NaN, length(t_vals))
    else
        return t_vals ./ t95
    end
end

############################################################
# 4. --- MAIN PIPELINE USING ARNOLDI CORRECT t95 DEFINITION ---
############################################################
# Implicit t95 from an rmed(t) curve: exp(-rmed(t)*t) = 0.05
function t95_from_rmed_curve(t_vals::AbstractVector, rmed::AbstractVector; target::Real=0.05)
    @assert length(t_vals) == length(rmed)
    y = @. exp(-rmed * t_vals)                 # predicted remaining fraction
    idx = findfirst(y .<= target)
    isnothing(idx) && return Inf
    idx == 1 && return float(t_vals[1])

    # linear interpolation between grid points
    t1, t2 = float(t_vals[idx-1]), float(t_vals[idx])
    y1, y2 = float(y[idx-1]), float(y[idx])
    y2 == y1 && return t2
    return t1 + (target - y1) * (t2 - t1) / (y2 - y1)
end

function run_pipeline_rmed_standardized!(
        B, S, L;
        # q_targets = 10 .^ range(log10(0.01), log10(10.0), length=9),
        q_targets = range(0.01, 1.5, length=9),
        replicates = 30,
        mag_abs = 1.0,
        mag_cv = 0.5,
        corr = 0.0,
        u_cv = 0.5,
        t_vals = 10 .^ range(log10(0.01), log10(100.0), length=60),
        u,
        rng = Random.default_rng()
    )

    results = Dict{Float64, Vector{Vector{Float64}}}()     # rmed curves (unchanged)
    Js      = Dict{Float64, Matrix{Float64}}()             # one Jacobian per q
    τ_axes  = Dict{Float64, Vector{Float64}}()             # new tau axes (method 2)
    t95_vals = Dict{Float64, Float64}()                    # store t95 per q

    for q in q_targets
        curves_q = Vector{Vector{Float64}}()

        # -------------------------------
        # Build interaction + Jacobian
        # -------------------------------
        @info "Target q = $q"
        avg_q_vec = Float64[]
        for rep in 1:replicates

            # Build PPM
            b = PPMBuilder()
            # set!(b; S=S, B=B, L=L, T=q, η=0.0)
            set!(b; S=120, B=24, L=2142, T=q)
            net = build(b)
            A = net.A
            realized_q = net.q
            # @info "q = $q"
            push!(avg_q_vec, realized_q)
            # Interaction matrix
            W = build_interaction_matrix(A;
                mag_abs=mag_abs,
                mag_cv=mag_cv,
                corr_aij_aji=corr,
                rng=rng
            )

            # Jacobian
            J = jacobian(W, u)
            Js[q] = J       # store any one of them

            # rmed curve (unchanged)
            push!(curves_q, compute_rmed_curve(J, u, t_vals))
        end

        avg_q = mean(avg_q_vec)
        @info "Average q = $avg_q"

        # ----------------------------------------------------------
        # Compute t95 using Arnoldi/rmed implicit definition:
        # exp(-Ravg(t)*t) = 0.05  with Ravg(t) ≈ rmed(t)
        # ----------------------------------------------------------
        t95 = begin
            vals = [t95_from_rmed_curve(t_vals, c) for c in curves_q]
            finite_vals = filter(isfinite, vals)
            isempty(finite_vals) ? Inf : median(finite_vals)
            # if you want your old “most conservative” behavior, use:
            # isempty(finite_vals) ? Inf : minimum(finite_vals)
        end
        t95_vals[q] = t95

        # τ-axis (if system never reaches 5%, τ = NaN)
        τ_axes[q] = compute_tau_axis(t_vals, t95)

        results[q] = curves_q

        # println("Finished q = $q   (t95 = $t95)")
    end

    params = (mag_abs=mag_abs, mag_cv=mag_cv, corr=corr)
    return results, τ_axes, Js, t_vals, t95_vals, params
end

function plot_rmed_grid_with_reference_tau(
        results, τ_axes, params;
        q_targets = sort(collect(keys(results))),
        reference = :lowest,
        S, B, L, η, u, t_vals,
        replicates_ref = 30,
        figsize = (1600,1400)
    )

    q_ref = (reference == :lowest)  ? first(q_targets) :
            (reference == :highest) ? last(q_targets)  :
            error("reference must be :lowest or :highest")

    # rebuild reference replicates
    ref_curves, τ_ref = build_reference_rmed!(
        q_ref, S, B, L, η, u, t_vals, params;
        replicates_ref = replicates_ref
    )

    # transform τ → log10(1+τ)
    # τ_ref_plot = tau_to_logtau(τ_ref)
    τ_ref_plot = τ_ref

    fig = Figure(size = figsize)
    rows, cols = 3, 3
    idx = 1

    for q in q_targets
        curves = results[q]
        τ_vals    = τ_axes[q]
        # τ_plot    = tau_to_logtau(τ_vals)
        τ_plot    = τ_vals

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        ax = Axis(fig[r,c];
            title="q=$(round(q, digits=3))",
            xlabel="τ",
            ylabel="rₘₑd(τ)")

        # reference curves (red)
        for curve in ref_curves
            lines!(ax, τ_ref_plot, curve; color=(:red, 0.35))
        end

        # curves for this q (black)
        for curve in curves
            lines!(ax, τ_plot, curve; color=(:black, 0.25))
        end

        xlims!(ax, (-0.1, 1.1))

        idx += 1
    end

    display(fig)
end

function plot_rmed_mean_grid_with_reference_tau(
        results, τ_axes, params;
        q_targets = sort(collect(keys(results))),
        reference = :lowest,
        S, B, L, η, u, t_vals,
        replicates_ref = 30,
        figsize = (1600,1400)
    )

    q_ref = (reference == :lowest)  ? first(q_targets) :
            (reference == :highest) ? last(q_targets)  :
            error("reference must be :lowest or :highest")

    ref_curves, τ_ref = build_reference_rmed!(
        q_ref, S, B, L, η, u, t_vals, params;
        replicates_ref = replicates_ref
    )

    ref_mean = mean_curve(ref_curves)
    # τ_ref_plot = tau_to_logtau(τ_ref)
    τ_ref_plot = τ_ref

    fig = Figure(size = figsize)
    rows, cols = 3, 3
    idx = 1

    for q in q_targets
        curves = results[q]
        mean_q = mean_curve(curves)

        τ_vals     = τ_axes[q]
        # τ_plot     = tau_to_logtau(τ_vals)
        τ_plot     = τ_vals

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        ax = Axis(fig[r,c];
            title="q=$(round(q,digits=3))",
            xlabel="τ",
            ylabel="mean rₘₑd(τ)"
        )

        lines!(ax, τ_ref_plot, ref_mean; color=:red, linewidth=3)
        lines!(ax, τ_plot, mean_q;    color=:black, linewidth=3)
        xlims!(ax, (-0.1, 1.1))

        idx += 1
    end

    display(fig)
end

function plot_rmed_delta_grid_tau(
        results, τ_axes, params;
        q_vals=sort(collect(keys(results))),
        reference = :lowest,
        S::Int, B::Int, L::Int, η,
        u, t_vals,
        replicates_ref = 30,
        figsize = (1600,1400),
        rng = Random.default_rng()
    )

    # --- choose reference q ---
    q_ref = reference == :lowest  ? minimum(q_vals) :
            reference == :highest ? maximum(q_vals) :
            error("reference must be :lowest or :highest")

    # --- build fresh reference curves ---
    ref_curves, τ_ref = build_reference_rmed!(
        q_ref, S, B, L, η, u, t_vals, params;
        replicates_ref=replicates_ref,
        rng=rng
    )
    ref_mean = mean_curve(ref_curves)

    # --- Build a common τ-grid ---
    τ_max = maximum(vcat([maximum(τ_axes[q]) for q in q_vals]...))
    τ_common = range(0, τ_max; length=200)

    # --- Interpolate reference onto τ_common ---
    f_ref = LinearInterpolation(τ_ref, ref_mean, extrapolation_bc=Line())
    ref_common = f_ref.(τ_common)

    # --- compute global y-limits for Δ ---
    all_deltas = Float64[]
    for q in q_vals
        mean_q = mean_curve(results[q])
        τ_q    = τ_axes[q]

        f_q = LinearInterpolation(τ_q, mean_q, extrapolation_bc=Line())
        Δ_common = abs.(f_q.(τ_common) .- ref_common)

        append!(all_deltas, Δ_common)
    end

    y_min = minimum(all_deltas)
    y_max = maximum(all_deltas)
    pad   = 0.1 * abs(y_max)
    ylims = (y_min - pad, y_max + pad)

    # --- plotting ---
    fig = Figure(size=figsize)
    rows, cols = 3, 3
    idx = 1

    for q in q_vals
        mean_q = mean_curve(results[q])
        τ_q    = τ_axes[q]

        f_q = LinearInterpolation(τ_q, mean_q, extrapolation_bc=Line())
        Δ_common = abs.(f_q.(τ_common) .- ref_common)

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        ax = Axis(fig[r,c];
            title="q = $(round(q,digits=3))",
            xlabel="τ",
            ylabel="|Δ rₘₑd(τ)|"
        )

        lines!(ax, τ_common, Δ_common; color=:blue, linewidth=3)
        ylims!(ax, ylims...)  # apply global y limits
        xlims!(ax, (-0.1, 1.1))

        idx += 1
    end

    display(fig)
end

function build_reference_rmed!(
        q_ref, S, B, L, η, u, t_vals, params;
        replicates_ref = 30,
        rng = Random.default_rng()
    )

    curves = Vector{Vector{Float64}}()
    mag_abs = params.mag_abs
    mag_cv  = params.mag_cv
    corr    = params.corr

    for rep in 1:replicates_ref
        b = PPMBuilder()
        η_value = 0.2
        set!(b; S=S, B=B, L=L, T=q_ref, η=η_value)
        
        net = build(b)

        A = net.A

        # Interaction matrix
        W = build_interaction_matrix(A;
            mag_abs=mag_abs,
            mag_cv=mag_cv,
            corr_aij_aji=corr,
            rng=rng
        )

        J = jacobian(W, u)

        rcurve = compute_rmed_curve(J, u, t_vals)
        push!(curves, rcurve)
    end

    # # compute standardized τ-axis using the *first* reference J
    # Jref = jacobian(build(b).W, u)
    # τ_ref, _, _ = standardized_time_axis(t_vals, Jref)

    return curves
end

# transform tau → tau_plot = log10(1 + tau)
tau_to_logtau(tau) = log10.(1 .+ tau)

S, B, L = 120, 24, 2142 
u = random_u(S, mean=1.0, cv=0.5)
results, τ_axes, Js, t_vals, t95_vals, params = run_pipeline_rmed_standardized!(
    B, S, L;
    mag_abs = 0.5,
    mag_cv = 0.5,
    corr = 0.0,
    u_cv = 2.0,
    t_vals = 10 .^ range(log10(0.01), log10(100.0), length=30),
    u = u,
    rng = Random.default_rng()
);

plot_rmed_grid_with_reference(
    results, t_vals, params; 
    title="TROPHIC COEHERENCE",
    q_targets = sort(collect(keys(results))),
    reference = :lowest,                 # :lowest, :highest, or numeric q
    S=S, B=B, L=L, u=u,
    η=0.2
)

plot_rmed_delta_grid(
    results, t_vals, params;
    q_targets = sort(collect(keys(results))),
    reference = :lowest,
    S, B, L, η=0.2, u,
    title = "TROPHIC COEHERENCE",
    rng = Random.default_rng()
)

plot_rmed_grid_with_reference_tau(results, τ_axes, params;
    reference = :lowest,
    S=S, B=B, L=L, η=0.2, u=u, t_vals=t_vals)

plot_rmed_mean_grid_with_reference_tau(results, τ_axes, params;
    reference = :lowest,
    S=S, B=B, L=L, η=0.2, u=u, t_vals=t_vals)

plot_rmed_delta_grid_tau(results, τ_axes, params;
    reference = :lowest,
    S=S, B=B, L=L, η=0.2, u=u, t_vals=t_vals)
