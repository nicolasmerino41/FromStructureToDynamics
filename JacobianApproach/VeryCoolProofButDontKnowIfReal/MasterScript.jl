############################################################
# UNIFIED STRUCTURAL PIPELINE — REAL NETWORK FUNCTIONS ONLY
############################################################

using Random
using LinearAlgebra
using Distributions
using Interpolations
using CairoMakie

############################################################
# 2. JACOBIAN + RMED (your real functions must exist)
############################################################

# we assume:
# jacobian(W,u)
# median_return_rate(J,u; t=..., perturbation=:biomass)

function compute_rmed_curve(J, u, t_vals)
    r = similar(t_vals)
    for i in eachindex(t_vals)
        r[i] = median_return_rate(J, u; t=t_vals[i])
    end
    return r
end


############################################################
# 3. STABILITY + SAFE τ PER NETWORK (Option B4)
############################################################

"""
    safe_tau(t_vals, J)

Compute τ only if stable, otherwise return NaN vector.
"""
function safe_tau(t_vals, J; P=0.05)
    λmax = maximum(real.(eigvals(J)))
    if λmax ≥ 0
        return fill(NaN, length(t_vals)), λmax, nothing
    end
    Rinf = -λmax
    tP = -log(P)/Rinf
    τ = t_vals ./ tP
    return τ, λmax, tP
end


############################################################
# 4. u-vector generator
############################################################

function generate_u_list(S; cv_range=range(0.1,1.0,length=5), rng=Random.default_rng())
    u_list = Vector{Vector{Float64}}()
    for cv in cv_range
        sigma = sqrt(log(1 + cv^2))
        mu = -sigma^2/2
        push!(u_list, rand(rng, LogNormal(mu,sigma), S))
    end
    return u_list
end


############################################################
# 5. UNIFIED STRUCTURAL PIPELINE (REAL FUNCTIONS)
############################################################
function run_unified_structural_pipeline!(
        S, B, L;
        u_list,
        q_vals       = range(0.01, 1.5, length=9),
        alpha_vals   = range(0, 2,   length=9),
        m_vals       = range(0, 1,   length=9),
        replicates   = 20,
        t_vals       = 10 .^ range(log10(0.01), log10(100.0), length=40),
        mag_abs      = 1.0,
        mag_cv       = 0.5,
        corr = 0.0,
        rng          = Random.default_rng()
    )

    ############################################################
    # Output dictionaries
    ############################################################
    results = Dict(
        :coherence => Dict(),     # (u_index, q) => [curves...]
        :degree    => Dict(),     # (u_index, α)
        :modularity=> Dict(),     # (u_index, m)
    )

    τ_axes = Dict(
        :coherence => Dict(),     # (u_index, q) => [τ vectors...]
        :degree    => Dict(),     # per replicate
        :modularity=> Dict(),
    )

    ############################################################
    # Iterate over u vectors
    ############################################################
    for (u_index, u) in enumerate(u_list)

        ##################################################################
        # 1. COHERENCE EXPERIMENT — PPMBuilder + build_interaction_matrix
        ##################################################################
        for q in q_vals
            curves = Vector{Vector{Float64}}()    # store rₘₑd(t)
            taus   = Vector{Vector{Float64}}()    # store τ per replicate

            for rep in 1:replicates

                # --- Build PPM ---
                b = PPMBuilder()
                set!(b; S=S, B=B, L=L, T=q, η=0.2)   # η is irrelevant for A
                net = build(b)
                A = net.A

                # --- Interaction matrix ---
                W = build_interaction_matrix(
                        A;
                        mag_abs      = mag_abs,
                        mag_cv       = mag_cv,
                        corr_aij_aji = corr,
                        rng          = rng
                    )

                # --- Jacobian ---
                J = jacobian(W, u)

                # --- Stability + τ ---
                τ, λmax, _ = safe_tau(t_vals, J)
                if any(isnan, τ)
                    continue  # drop unstable (Option B4)
                end

                # --- rₘₑd(t) ---
                rcurve = compute_rmed_curve(J, u, t_vals)

                push!(curves, rcurve)
                push!(taus,   τ)
            end

            results[:coherence][(u_index, q)] = curves
            τ_axes[:coherence][(u_index, q)]  = taus
        end


        ##################################################################
        # 2. DEGREE HETEROGENEITY — degree_distribution_network
        ##################################################################
        for α in alpha_vals
            curves = Vector{Vector{Float64}}()
            taus   = Vector{Vector{Float64}}()

            for rep in 1:replicates
                # --- Network ---
                A = degree_distribution_network(S, L; alpha=α, rng=rng)

                W = random_trophic_interaction_matrix(
                    A;
                    mag_abs=mag_abs, mag_cv=mag_cv, corr=corr,
                    rng=Random.default_rng()
                )

                # --- Interaction ---
                # W = build_interaction_matrix(
                #         A;
                #         mag_abs      = mag_abs,
                #         mag_cv       = mag_cv,
                #         corr_aij_aji = corr,
                #         rng          = rng
                #     )

                J = jacobian(W, u)

                # --- τ ---
                τ, λmax, _ = safe_tau(t_vals, J)
                if any(isnan, τ)
                    continue
                end

                # --- rₘₑd ---
                rcurve = compute_rmed_curve(J, u, t_vals)

                push!(curves, rcurve)
                push!(taus,   τ)
            end

            results[:degree][(u_index, α)] = curves
            τ_axes[:degree][(u_index, α)]  = taus
        end


        ##################################################################
        # 3. MODULARITY — modularity_network
        ##################################################################
        for m in m_vals
            curves = Vector{Vector{Float64}}()
            taus   = Vector{Vector{Float64}}()

            for rep in 1:replicates
                # --- Network ---
                A = modularity_network(S, L; m=m, rng=rng)

                # --- Interaction ---
                W = build_interaction_matrix(
                        A;
                        mag_abs      = mag_abs,
                        mag_cv       = mag_cv,
                        corr_aij_aji = corr,
                        rng          = rng
                    )

                J = jacobian(W, u)

                # --- τ ---
                τ, λmax, _ = safe_tau(t_vals, J)
                if any(isnan, τ)
                    continue
                end

                # --- rₘₑd ---
                rcurve = compute_rmed_curve(J, u, t_vals)

                push!(curves, rcurve)
                push!(taus,   τ)
            end

            results[:modularity][(u_index, m)] = curves
            τ_axes[:modularity][(u_index, m)]  = taus
        end
    end

    params = (mag_abs=mag_abs, mag_cv=mag_cv, corr=corr)
    return results, τ_axes, t_vals, params
end

############################################################
# 6. PLOTTING — EMPTY PANEL IF ALL UNSTABLE (C1)
############################################################
############################################################
# REFERENCE CURVE BUILDER — REAL NETWORKS, B4, C1
############################################################
function build_reference_rmed!(
        p_ref, S, B, L, η, mode,
        u, t_vals, params;
        replicates_ref = 30,
        rng = Random.default_rng()
    )

    mag_abs      = params.mag_abs
    mag_cv       = params.mag_cv
    corr_aij_aji = params.corr

    ref_curves  = Vector{Vector{Float64}}()
    τ_list      = Vector{Vector{Float64}}()

    for rep in 1:replicates_ref

        ####################################################
        # Build network according to structural MODE
        ####################################################
        if mode == :coherence
            # --- REAL PPM via PPMBuilder ---
            b = PPMBuilder()
            set!(b; S=S, B=B, L=L, T=p_ref, η=η)
            net = build(b)
            A   = net.A

        elseif mode == :degree
            # --- REAL degree distribution ---
            A = degree_distribution_network(S, L; alpha=p_ref, rng=rng)

        elseif mode == :modularity
            # --- REAL modularity network ---
            A = modularity_network(S, L; m=p_ref, rng=rng)

        else
            error("Unknown mode $mode")
        end

        ####################################################
        # Interaction matrix
        ####################################################
        W = build_interaction_matrix(
                A;
                mag_abs      = mag_abs,
                mag_cv       = mag_cv,
                corr_aij_aji = corr_aij_aji,
                rng          = rng
            )

        ####################################################
        # Jacobian
        ####################################################
        J = jacobian(W, u)

        ####################################################
        # τ-axis — drop unstable
        ####################################################
        τ, λmax, _ = safe_tau(t_vals, J)
        if any(isnan, τ)
            continue
        end

        ####################################################
        # rₘₑd(t)
        ####################################################
        rcurve = compute_rmed_curve(J, u, t_vals)

        push!(ref_curves, rcurve)
        push!(τ_list, τ)
    end

    # If ALL replicates unstable → panel becomes empty
    if isempty(ref_curves)
        return Vector{Vector{Float64}}(), Vector{Float64}()
    end

    # Use τ from the FIRST stable replicate
    return ref_curves, τ_list[1]
end

############################################################
# MAIN PLOT FUNCTION — WITH REFERENCE, τ OR t
# Matches old behavior, works for ALL MODES
############################################################
function plot_results(
        master_results, τ_axes, t_vals, params;
        mode = :coherence,
        u_index = 1,
        reference = :lowest,
        t_or_tau = :tau,                   
        S=120, B=24, L=2142, η = 0.2,     
        u,                                
        replicates_ref = 30,
        figsize = (1600, 1400),
        rng = Random.default_rng()
    )

    ############################################################
    # 0. Validate mode
    ############################################################
    if !haskey(master_results, mode)
        error("Unknown mode=$mode. Must be :coherence, :degree, :modularity")
    end

    ############################################################
    # 1. Extract parameters for this mode & u_index
    ############################################################
    all_keys = keys(master_results[mode])
    keys_mode = sort([k for k in all_keys if k[1] == u_index])

    if isempty(keys_mode)
        @warn "No stable networks for mode=$mode and u_index=$u_index"
        return Figure()
    end

    param_vals = [k[2] for k in keys_mode]

    ############################################################
    # 2. Choose reference parameter
    ############################################################
    p_ref =
        reference == :lowest  ? first(param_vals) :
        reference == :highest ? last(param_vals)  :
        error("reference must be :lowest or :highest")

    println("\nReference for $mode = $p_ref\n")

    ############################################################
    # 3. Build reference curves using REAL NETWORK FUNCTIONS
    ############################################################
    ref_curves, τ_ref = build_reference_rmed!(
        p_ref, S, B, L, η, mode,
        u, t_vals, params;
        replicates_ref = replicates_ref,
        rng = rng
    )

    # If all unstable → reference is empty
    τ_ref_plot = (t_or_tau == :tau ? τ_ref : t_vals)

    ############################################################
    # 4. FIGURE
    ############################################################
    fig = Figure(size = figsize)
    rows, cols = 3, 3
    idx = 1

    for p in param_vals
        curves_p = master_results[mode][(u_index, p)]
        τ_list   = τ_axes[String(mode)][(u_index, p)]

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        xlabel = t_or_tau == :tau ? "τ" : "t"
        ylabel = "rₘₑd(" * (t_or_tau == :tau ? "τ" : "t") * ")"

        ax = Axis(fig[r, c];
            title  = "$mode = $(round(p,digits=3))",
            xlabel = xlabel,
            ylabel = ylabel,
            xscale = (t_or_tau == :t ? log10 : identity)
        )

        ############################################################
        # 5. Draw REFERENCE (RED) if available
        ############################################################
        if !isempty(ref_curves)
            for curve in ref_curves
                lines!(ax, τ_ref_plot, curve; color = (:red, 0.35))
            end
        end

        ############################################################
        # 6. Draw CURVES for this parameter (BLACK)
        ############################################################
        if !isempty(curves_p)
            for (curve, τ_raw) in zip(curves_p, τ_list)
                τ_plot = t_or_tau == :tau ? τ_raw : t_vals
                lines!(ax, τ_plot, curve; color = (:black, 0.25))
            end

            # τ axis limits only in τ-mode
            if t_or_tau == :tau
                xlims!(ax, -0.1, 1.1)
            end
        else
            # Empty panel (C1 behavior): do nothing
            @warn "All replicates unstable for p=$p (mode=$mode, u_index=$u_index)"
        end

        idx += 1
    end

    display(fig)
    return fig
end


############################################################
# 7. Example driver code
############################################################
S = 120
B = 24
connectance = 0.15
L = round(Int, connectance * S * (S-1))

q_vals = range(0.01,1.5,length=9)
alpha_vals = range(0.01,1.5,length=9)
m_vals = range(0.01,1.5,length=9)

u_CVs = range(0.1,2.0,length=10)
rng = Random.default_rng()

u_list = generate_u_list(S; cv_range=u_CVs, rng=rng)

master_results, τ_axes, t_vals, params = run_unified_structural_pipeline!(
    S,B,L;
    u_list=u_list,
    q_vals=q_vals,
    alpha_vals=alpha_vals,
    m_vals=m_vals,
    mag_abs=0.5,
    mag_cv=0.5,
    corr=0.0,
    rng=rng
)

for i in 1:10
    u = u_list[i]
    plot_results(
        master_results, τ_axes, t_vals, params;
        mode=:coherence,
        u_index=3,
        reference=:lowest,
        t_or_tau=:tau,
        S=S, B=B, L=L, u=u
    )
end

"""
    compute_delta_rmed(master_results, τ_axes, t_vals; mode=:coherence, u_index=1, t_or_tau=:tau)

Returns a dictionary mapping structural parameter → Δ-rₘₑd curve
for the given u_index.

Δ is computed as the absolute difference between the mean curve
and the reference mean curve (lowest param value).
"""
function compute_delta_rmed(master_results, τ_axes, t_vals;
        mode = :coherence,
        u_index = 1,
        t_or_tau = :tau
    )

    # Extract all parameter values for this mode and u
    keys_mode = sort([k for k in keys(master_results[mode]) if k[1] == u_index])
    param_vals = [k[2] for k in keys_mode]

    ref_param = param_vals[1]  # lowest = reference

    # Compute reference mean curve
    ref_curves = master_results[mode][(u_index, ref_param)]
    ref_mean = mean(reduce(hcat, ref_curves), dims=2)[:]

    if t_or_tau == :tau
        τ_ref = τ_axes[mode][(u_index, ref_param)]
    else
        τ_ref = t_vals
    end

    # Create dict of Δ curves
    deltas = Dict{Float64, Vector{Float64}}()

    for p in param_vals
        curves = master_results[mode][(u_index, p)]
        mean_curve_p = mean(reduce(hcat, curves), dims=2)[:]

        if t_or_tau == :tau
            τ_p = τ_axes[mode][(u_index, p)]
        else
            τ_p = t_vals
        end

        # INTERPOLATE onto reference grid
        f_p   = LinearInterpolation(τ_p, mean_curve_p, extrapolation_bc=Line())
        f_ref = LinearInterpolation(τ_ref, ref_mean,   extrapolation_bc=Line())

        Δ = abs.(f_p.(τ_ref) .- f_ref.(τ_ref))

        deltas[p] = Δ
    end

    return deltas, τ_ref
end

function plot_delta_rmed(master_results, τ_axes, t_vals;
        mode = :coherence,
        u_index = 1,
        figsize=(1600,1200),
        t_or_tau = :tau
    )

    # Compute Δ curves (dictionary p → Δ_p)
    deltas, τ = compute_delta_rmed(master_results, τ_axes, t_vals;
                                   mode=mode, u_index=u_index, t_or_tau=t_or_tau)

    param_vals = sort(collect(keys(deltas)))

    # ---------------------------------------------
    # GLOBAL y-axis limits computed from deltas
    # ---------------------------------------------
    all_vals = vcat([deltas[p] for p in param_vals]...)
    global_min = minimum(all_vals)
    global_max = maximum(all_vals)

    ypad = 0.1 * abs(global_max)
    ylo = global_min - ypad
    yhi = global_max + ypad

    # ---------------------------------------------
    # FIG setup
    # ---------------------------------------------
    fig = Figure(size=figsize)
    rows, cols = 3, 3
    idx = 1

    for p in param_vals
        Δ = deltas[p]

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        xlabel = t_or_tau == :tau ? "τ" : "t"

        ax = if t_or_tau == :tau
            Axis(fig[r,c];
                title="$(mode) = $(round(p, digits=3))",
                xlabel=xlabel, ylabel="|Δ rₘₑd|"
            )
        else
            Axis(fig[r,c];
                title="$(mode) = $(round(p, digits=3))",
                xlabel=xlabel, ylabel="|Δ rₘₑd(t)|",
                xscale=log10
            )
        end

        lines!(ax, τ, Δ; color=:blue, linewidth=2)

        # x-limits
        if t_or_tau == :tau
            xlims!(ax, -0.01, 1.1)
        else
            xlims!(ax, minimum(t_vals), maximum(t_vals))
        end

        # y-limits
        ylims!(ax, ylo, yhi)

        idx += 1
    end

    display(fig)
end

"""
    plot_delta_rmed_all_u(master_results, τ_axes, t_vals;
        mode=:coherence, t_or_tau=:tau)

Computes mean Δ-rₘₑd across all u’s and plots them in a 3×3 grid.
"""
function plot_delta_rmed_all_u(master_results, τ_axes, t_vals;
        mode = :coherence,
        t_or_tau = :tau,
        figsize=(1600,1200)
    )

    # Extract all u indices
    u_indices = sort(unique(k[1] for k in keys(master_results[mode])))

    # Extract parameter values (same for all u)
    param_vals = sort(unique(k[2] for k in keys(master_results[mode])))

    # param → collection of Δ curves across u
    Δ_per_param = Dict(p => Vector{Vector{Float64}}() for p in param_vals)

    # ---------------------------------------------
    # Compute deltas for each u and store
    # ---------------------------------------------
    for u_index in u_indices
        deltas, τ = compute_delta_rmed(master_results, τ_axes, t_vals;
                        mode=mode, u_index=u_index, t_or_tau=t_or_tau)

        for p in param_vals
            push!(Δ_per_param[p], deltas[p])
        end
    end

    # Mean Δ across all u
    Δ_mean = Dict(p => mean(reduce(hcat, Δ_per_param[p]), dims=2)[:] for p in param_vals)

    # ---------------------------------------------
    # GLOBAL y-limits
    # ---------------------------------------------
    all_vals = vcat([Δ_mean[p] for p in param_vals]...)
    global_min = minimum(all_vals)
    global_max = maximum(all_vals)

    ypad = 0.1 * abs(global_max)
    ylo = global_min - ypad
    yhi = global_max + ypad

    # ---------------------------------------------
    # FIGURE
    # ---------------------------------------------
    fig = Figure(size=figsize)
    rows, cols = 3, 3
    idx = 1

    xlabel = t_or_tau == :tau ? "τ" : "t"

    for p in param_vals
        Δavg = Δ_mean[p]

        r = div(idx-1, cols) + 1
        c = mod(idx-1, cols) + 1

        ax = if t_or_tau == :tau
            Axis(fig[r,c];
                title="$(mode) = $(round(p,digits=3))",
                xlabel=xlabel, ylabel="mean |Δ rₘₑd|"
            )
        else
            Axis(fig[r,c];
                title="$(mode) = $(round(p,digits=3))",
                xlabel=xlabel, ylabel="mean |Δ rₘₑd(t)|",
                xscale=log10
            )
        end

        lines!(ax, τ, Δavg; color=:red, linewidth=3)

        if t_or_tau == :tau
            xlims!(ax, -0.01, 1.1)
        else
            xlims!(ax, minimum(t_vals), maximum(t_vals))
        end

        ylims!(ax, ylo, yhi)

        idx += 1
    end

    display(fig)
end

plot_delta_rmed(master_results, τ_axes, t_vals; mode=:coherence, u_index=5, t_or_tau=:tau)
plot_delta_rmed_all_u(master_results, τ_axes, t_vals; mode=:coherence, t_or_tau=:t)

deltas, τ = compute_delta_rmed(master_results, τ_axes, t_vals; mode=:modularity, u_index=3)
