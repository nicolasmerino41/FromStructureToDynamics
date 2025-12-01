using Random
using LinearAlgebra
using Interpolations
using CairoMakie

# ================================================================
# 1. GENERATE u-VECTORS WITH VARIABLE HETEROGENEITY
# ================================================================

"""
    generate_u_list(S; cv_range=range(0.1,1.0,length=5))

Generates a list of u vectors of size S, each with a different CV.
"""
function generate_u_list(S; cv_range=range(0.1, 1.0, length=5), rng=Random.default_rng())
    u_list = Vector{Vector{Float64}}()
    for cv in cv_range
        sigma = sqrt(log(1 + cv^2))
        mu = -sigma^2/2
        push!(u_list, rand(rng, LogNormal(mu, sigma), S))
    end
    return u_list
end


# ================================================================
# 2. TROPHIC COHERENCE NETWORK GENERATION
# ================================================================

# ppm must exist externally
# interaction_matrix must exist externally

# We assume:
#   ppm(S,B,L,T) -> A, s
#   trophic_levels(A)
#   trophic_incoherence(A,s)

function build_ppm_network(S,B,L,T)
    A, _ = ppm(S,B,L,T)
    return A
end


# ================================================================
# 5. INTERACTION MATRIX (same as before)
# ================================================================

function random_trophic_interaction_matrix(A; mag_abs=1.0, mag_cv=0.5, corr=1.0, rng=Random.default_rng())
    S = size(A,1)
    W = zeros(Float64, S, S)

    sigma = sqrt(log(1 + mag_cv^2))
    mu = log(mag_abs) - sigma^2/2

    for i in 1:S, j in 1:S
        if A[i,j] == 1
            a = rand(rng, LogNormal(mu, sigma))
            if corr == 1
                b = a
            elseif corr == 0
                b = rand(rng, LogNormal(mu, sigma))
            else
                a2 = rand(rng, LogNormal(mu, sigma))
                b = corr*a + (1-corr)*a2
            end

            W[j,i] =  a
            W[i,j] = -b
        end
    end

    return W
end


# ================================================================
# 6. R-MED, STANDARDIZED TIME, INTERPOLATION
# ================================================================

# median_return_rate, compute_rmed_curve, standardized_time_axis
# must be defined previously in your code.
# We will assume they exist.

function interpolate_curve(τ, r_vals, τ_common)
    f = LinearInterpolation(τ, r_vals, extrapolation_bc=Line())
    return f.(τ_common)
end


# ================================================================
# 7. UNIFIED PIPELINE FUNCTION
# ================================================================

"""
    run_unified_structural_pipeline!(S, B, L; u_list, q_vals, alpha_vals, m_vals)

Runs all three structural experiments (coherence, degree, modularity)
using the SAME u_list.
"""
function run_unified_structural_pipeline!(
        S, B, L;
        u_list,
        q_vals = range(0.01,1.5,length=9),
        alpha_vals = range(0,2,length=9),
        m_vals = range(0,1,length=9),
        replicates = 20,
        t_vals = 10 .^ range(log10(0.01), log10(100.0), length=40),
        rng = Random.default_rng()
    )

    results = Dict(
        :coherence => Dict(),
        :degree    => Dict(),
        :modularity=> Dict(),
    )

    τ_axes = Dict(
        :coherence => Dict(),
        :degree    => Dict(),
        :modularity=> Dict(),
    )

    # For each u vector...
    for (u_index, u) in enumerate(u_list)

        # -----------------------------------------------------------
        # TROPHIC COHERENCE
        # -----------------------------------------------------------
        for q in q_vals
            curves = Vector{Vector{Float64}}()
            for rep in 1:replicates
                A = build_ppm_network(S,B,L,q)
                W = random_trophic_interaction_matrix(A; rng=rng)
                J = jacobian(W,u)
                push!(curves, compute_rmed_curve(J,u,t_vals))
            end
            # standardized τ
            τ, _, _ = standardized_time_axis(t_vals, jacobian(random_trophic_interaction_matrix(build_ppm_network(S,B,L,first(q_vals));rng=rng),u))
            results[:coherence][(u_index,q)] = curves
            τ_axes[:coherence][(u_index,q)] = τ
        end

        # -----------------------------------------------------------
        # DEGREE DISTRIBUTION
        # -----------------------------------------------------------
        for α in alpha_vals
            curves = Vector{Vector{Float64}}()
            for rep in 1:replicates
                A = degree_distribution_network(S,L; alpha=α)
                W = random_trophic_interaction_matrix(A)
                J = jacobian(W,u)
                push!(curves, compute_rmed_curve(J,u,t_vals))
            end
            τ, _, _ = standardized_time_axis(t_vals, jacobian(random_trophic_interaction_matrix(degree_distribution_network(S,L;alpha=first(alpha_vals))),u))
            results[:degree][(u_index,α)] = curves
            τ_axes[:degree][(u_index,α)] = τ
        end

        # -----------------------------------------------------------
        # MODULARITY
        # -----------------------------------------------------------
        for m in m_vals
            curves = Vector{Vector{Float64}}()
            for rep in 1:replicates
                A = modularity_network(S,L; m=m)
                W = random_trophic_interaction_matrix(A)
                J = jacobian(W,u)
                push!(curves, compute_rmed_curve(J,u,t_vals))
            end
            τ, _, _ = standardized_time_axis(t_vals, jacobian(random_trophic_interaction_matrix(modularity_network(S,L;m=first(m_vals))),u))
            results[:modularity][(u_index,m)] = curves
            τ_axes[:modularity][(u_index,m)] = τ
        end
    end

    return results, τ_axes, t_vals
end


# ================================================================
# 8. UNIFIED PLOTTING FUNCTION
# ================================================================

"""
    plot_results(results, τ_axes; mode=:coherence, u_index=1)

Plots results for a given structural mode and u index.
"""
function plot_results(results, τ_axes, t_vals;
        mode = :coherence,
        u_index = 1,
        figsize=(1600,1200),
        t_or_tau = :tau
    )

    keys_mode = sort([k for k in keys(results[mode]) if k[1] == u_index])
    param_vals = [k[2] for k in keys_mode]

    fig = Figure(size=figsize)
    rows, cols = 3, 3
    idx = 1

    for p in param_vals
        curves = results[mode][(u_index,p)]
        if t_or_tau == :tau
            τ    = τ_axes[mode][(u_index,p)]
        else
            τ    = t_vals
        end

        r = div(idx-1, cols)+1
        c = mod(idx-1, cols)+1

        if t_or_tau == :tau
            ax = Axis(fig[r,c];
                title="$(mode) = $(round(p,digits=3))",
                xlabel="τ", ylabel="rₘₑd(τ)"
            )
        else
            ax = Axis(fig[r,c];
                title="$(mode) = $(round(p,digits=3))",
                xlabel="t", ylabel="rₘₑd(t)",
                xscale=log10
            )
        end

        for curve in curves
            lines!(ax, τ, curve; color=(:black,0.25))
        end
        if t_or_tau == :tau
            xlims!(ax, -0.01, 1.1)
        end

        idx += 1
    end

    display(fig)
end

using Random

# --- community parameters ---
S = 120                      # species
B = 24                       # for PPM (only coherence pipeline uses this)
connectance = 0.15
L = round(Int, connectance * S * (S - 1))

# --- structural parameter values ---
q_vals = range(0.01, 1.5; length=9)
alpha_vals = range(0, 2; length=9)
m_vals = range(0, 1; length=9)

# --- u heterogeneity range ---
u_CVs = range(0.1, 2.0; length=10)   # 5 u-vectors from homogeneous → heterogeneous

# --- other settings ---
replicates = 20
rng = Random.default_rng()

u_list = generate_u_list(S; cv_range=u_CVs, rng=rng)

results, τ_axes, t_vals = run_unified_structural_pipeline!(
    S, B, L;
    u_list = u_list,
    q_vals = q_vals,
    alpha_vals = alpha_vals,
    m_vals = m_vals,
    replicates = replicates,
    rng = rng
)

for i in 1:10
    plot_results(results, τ_axes, t_vals; mode=:coherence, u_index=i, t_or_tau = :tau)
    plot_results(results, τ_axes, t_vals; mode=:coherence, u_index=i, t_or_tau = :t)
end
for i in 1:10
    plot_results(results, τ_axes, t_vals; mode=:degree, u_index=i, t_or_tau = :tau)
end
for i in 1:10
    plot_results(results, τ_axes, t_vals; mode=:modularity, u_index=i, t_or_tau = :tau)
end

"""
    compute_delta_rmed(results, τ_axes, t_vals; mode=:coherence, u_index=1, t_or_tau=:tau)

Returns a dictionary mapping structural parameter → Δ-rₘₑd curve
for the given u_index.

Δ is computed as the absolute difference between the mean curve
and the reference mean curve (lowest param value).
"""
function compute_delta_rmed(results, τ_axes, t_vals;
        mode = :coherence,
        u_index = 1,
        t_or_tau = :tau
    )

    # Extract all parameter values for this mode and u
    keys_mode = sort([k for k in keys(results[mode]) if k[1] == u_index])
    param_vals = [k[2] for k in keys_mode]

    ref_param = param_vals[1]  # lowest = reference

    # Compute reference mean curve
    ref_curves = results[mode][(u_index, ref_param)]
    ref_mean = mean(reduce(hcat, ref_curves), dims=2)[:]

    if t_or_tau == :tau
        τ_ref = τ_axes[mode][(u_index, ref_param)]
    else
        τ_ref = t_vals
    end

    # Create dict of Δ curves
    deltas = Dict{Float64, Vector{Float64}}()

    for p in param_vals
        curves = results[mode][(u_index, p)]
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

function plot_delta_rmed(results, τ_axes, t_vals;
        mode = :coherence,
        u_index = 1,
        figsize=(1600,1200),
        t_or_tau = :tau
    )

    # Compute Δ curves (dictionary p → Δ_p)
    deltas, τ = compute_delta_rmed(results, τ_axes, t_vals;
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
    plot_delta_rmed_all_u(results, τ_axes, t_vals;
        mode=:coherence, t_or_tau=:tau)

Computes mean Δ-rₘₑd across all u’s and plots them in a 3×3 grid.
"""
function plot_delta_rmed_all_u(results, τ_axes, t_vals;
        mode = :coherence,
        t_or_tau = :tau,
        figsize=(1600,1200)
    )

    # Extract all u indices
    u_indices = sort(unique(k[1] for k in keys(results[mode])))

    # Extract parameter values (same for all u)
    param_vals = sort(unique(k[2] for k in keys(results[mode])))

    # param → collection of Δ curves across u
    Δ_per_param = Dict(p => Vector{Vector{Float64}}() for p in param_vals)

    # ---------------------------------------------
    # Compute deltas for each u and store
    # ---------------------------------------------
    for u_index in u_indices
        deltas, τ = compute_delta_rmed(results, τ_axes, t_vals;
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

plot_delta_rmed(results, τ_axes, t_vals; mode=:coherence, u_index=5, t_or_tau=:tau)
plot_delta_rmed_all_u(results, τ_axes, t_vals; mode=:coherence, t_or_tau=:t)

deltas, τ = compute_delta_rmed(results, τ_axes, t_vals; mode=:modularity, u_index=3)
