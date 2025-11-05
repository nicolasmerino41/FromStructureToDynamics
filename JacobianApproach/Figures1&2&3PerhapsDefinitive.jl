###########################################################
# MainFigures.jl
#   Core visual story: how structural heterogeneity & antisymmetry
#   govern predictability across time for the three structural branches
#
#   Requires: your existing ecosystem functions:
#     build_niche_trophic / build_random_trophic_ER
#     jacobian, alpha_off_from, build_J_from, realized_IS,
#     realized_connectance, random_u, median_return_rate
#
#   Output: three figures saved to ./JacobianApproach/FinalFigures/
###########################################################

using Random, Statistics, LinearAlgebra, DataFrames
using CairoMakie

# ------------------------------------------------------------
# 0.  tiny helpers (consistent with your current code base)
# ------------------------------------------------------------

_r2_to_identity(x::AbstractVector, y::AbstractVector) = begin
    n = length(x)
    n == 0 && return 0.0
    μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
    sst == 0 ? (ssr == 0 ? 1.0 : 0.0) : max(1 - ssr/sst, 0.0)
end

logspace10(a::Real, b::Real, n::Int) = 10 .^ range(a, b; length=n)

# build_A picks whatever builder you have
function _build_A(S; conn, mean_abs, mag_cv, degree_family, deg_param, rho_sym, rng)
    if isdefined(@__MODULE__, :build_niche_trophic)
        return build_niche_trophic(S; conn, mean_abs, mag_cv,
                                   degree_family, deg_param, rho_sym, rng)
    else
        return build_random_trophic(S; conn, mean_abs, mag_cv,
                                    degree_family, deg_param, rho_sym, rng)
    end
end

# paired reshuffle preserving antisymmetry level
function op_reshuffle_alpha_paired(alpha; rng=Random.default_rng())
    S = size(alpha,1)
    used = falses(S,S)
    pairs = Tuple{Int,Int}[]
    for i in 1:S, j in 1:S
        (i==j || used[i,j]) && continue
        push!(pairs, (i,j))
        used[i,j] = true; used[j,i] = true
    end
    vals = [(alpha[i,j], alpha[j,i]) for (i,j) in pairs]
    perm = randperm(rng, length(vals))
    αnew = zeros(eltype(alpha), S, S)
    for (k, (i,j)) in enumerate(pairs)
        v1, v2 = vals[perm[k]]
        αnew[i,j] = v1; αnew[j,i] = v2
    end
    αnew
end

# metrics per t for 3 branches
function _metrics_for_times(A, u, tgrid; rng)
    Jfull = jacobian(A, u)
    α = alpha_off_from(Jfull, u)
    u_sh = copy(u); shuffle!(rng, u_sh)
    α_rp = op_reshuffle_alpha_paired(α; rng)
    A_er = build_random_trophic_ER(size(A,1);
        conn=realized_connectance(A),
        mean_abs=realized_IS(A), mag_cv=0.60,
        rho_sym=1.0, rng)
    β = realized_IS(A) / max(realized_IS(A_er), eps())
    J_ush = build_J_from(α, u_sh)
    J_rp  = build_J_from(α_rp, u)
    J_rew = jacobian(β .* A_er, u)
    full  = [median_return_rate(Jfull, u; t=t, perturbation=:biomass) for t in tgrid]
    ush   = [median_return_rate(J_ush,  u_sh; t=t, perturbation=:biomass) for t in tgrid]
    rsh   = [median_return_rate(J_rp,   u; t=t, perturbation=:biomass) for t in tgrid]
    rew   = [median_return_rate(J_rew,  u; t=t, perturbation=:biomass) for t in tgrid]
    return full, ush, rsh, rew
end

# scenario simulator: returns R²(t) curves
function simulate_scenario(; S=120, conn=0.10, mean_abs=0.5, mag_cv=0.6, rho_sym=1.0,
        u_mean=1.0, u_cv=0.6, degree_family=:lognormal, deg_param=0.5,
        IS_target=0.5, reps=24, tgrid=logspace10(-2, 2, 40), seed=42)

    rng0 = Random.Xoshiro(seed)
    F = [Float64[] for _ in tgrid]
    U = [Float64[] for _ in tgrid]
    R = [Float64[] for _ in tgrid]
    W = [Float64[] for _ in tgrid]

    for rep in 1:reps
        rng = Random.Xoshiro(rand(rng0, UInt64))
        A0 = _build_A(S; conn, mean_abs, mag_cv, degree_family, deg_param, rho_sym, rng)
        baseIS = realized_IS(A0); baseIS ≤ 0 && continue
        A = (IS_target / baseIS) .* A0
        u = random_u(S; mean=u_mean, cv=u_cv, rng)
        full, ush, rsh, rew = _metrics_for_times(A, u, tgrid; rng)
        for i in eachindex(tgrid)
            push!(F[i], full[i]); push!(U[i], ush[i]); push!(R[i], rsh[i]); push!(W[i], rew[i])
        end
    end
    (; tgrid, R2_ush=[_r2_to_identity(F[i],U[i]) for i in eachindex(tgrid)],
        R2_rsh=[_r2_to_identity(F[i],R[i]) for i in eachindex(tgrid)],
        R2_rew=[_r2_to_identity(F[i],W[i]) for i in eachindex(tgrid)])
end

# ------------------------------------------------------------
# 1. FIGURE 1: Archetypal communities (3×3 grid)
# ------------------------------------------------------------
function figure1_archetypes()
    tgrid = logspace10(-2,2,48)
    # fix rho high
    scenarios = [
        ("u_cv", [(0.05,0.8,0.6),(1.0,0.8,0.6),(3.0,0.8,0.6)]),   # vary u_cv
        ("mag_cv", [(0.6,0.8,0.1),(0.6,0.8,0.6),(0.6,0.8,1.5)]), # vary mag_cv
        ("degcv", [(0.6,0.8,0.1),(0.6,0.8,0.6),(0.6,0.8,1.5)])   # vary degcv (for show)
    ]

    fig = Figure(size=(1100, 950))
    Label(fig[0,1:3], "Fig 1. Predictability over time for archetypal communities"; fontsize=22, font=:bold)

    rowlabels = ["Abundance heterogeneity (u_cv)", "Interaction heterogeneity (mag_cv)", "Topology heterogeneity (deg_cv)"]
    collabels = ["Low","Medium","High"]

    for (r,(axis,triples)) in enumerate(scenarios)
        for (c,(u_cv,rho,mag_cv)) in enumerate(triples)
            sim = simulate_scenario(; u_cv=u_cv, rho_sym=rho, mag_cv=mag_cv, deg_param=(axis=="degcv" ? [0.01,0.8,1.8][c] : 0.8))
            ax = Axis(
                fig[r,c]; xscale=log10, xlabel="time t", ylabel="R²", title="$(collabels[c])",
                limits=((nothing,nothing), (-0.1, 1.1))
            )
            lines!(ax, sim.tgrid, sim.R2_ush; color=:blue, label="ushuf")
            lines!(ax, sim.tgrid, sim.R2_rsh; color=:orange, label="reshuf(pair)")
            lines!(ax, sim.tgrid, sim.R2_rew; color=:green, label="rew")
            if c==1; Label(fig[r,0], rowlabels[r]; rotation=pi/2, fontsize=14); end
            if r==1 && c==3; axislegend(ax; position=:lb, framevisible=false); end
        end
    end
    save("JacobianApproach/FinalFigures/Fig1_archetypes.png", fig); display(fig)
end

# ------------------------------------------------------------
# 2. FIGURE 2: Dominance maps (ρ × heterogeneity)
# ------------------------------------------------------------
"""
figure2_maps(; mode=:deterministic)

Plot dominance maps across pairs of structural parameters and time
slices (short, mid, long). Each pixel = branch that most reduces R².

Arguments
----------
mode::Symbol  - either :deterministic or :stochastic
"""
function figure2_maps(; mode::Symbol = :deterministic)
    # === shared setup ===
    tgrid = logspace10(-2, 2, 40)
    timeslices = [0.01, 0.5, 10.0]  # short / mid / long times

    # parameter pairs to explore
    param_pairs = [
        (:degcv, :u_cv,  [0.1, 0.8, 1.8],  [0.2, 1.0, 3.0]),
        (:mag_cv, :u_cv, [0.1, 0.6, 1.5],  [0.2, 1.0, 3.0]),
        (:mag_cv, :degcv, [0.1, 0.6, 1.5], [0.1, 0.8, 1.8]),
        (:mag_cv, :rho,   [0.1, 0.6, 1.5], [0.01, 0.25, 0.5, 0.75, 1.0])
    ]

    colors = Dict(
        :U => RGBf(0.2,0.4,1.0),   # time-scales
        :IS => RGBf(1.0,0.6,0.2),  # interaction strength distribution
        :TOP => RGBf(0.2,0.8,0.2)  # topology
    )

    mkpath("JacobianApproach/FinalFigures")

    # === baseline parameter configuration ===
    base_config = Dict(
        :u_cv => 0.6,
        :mag_cv => 0.56,
        :deg_param => 0.5,
        :rho_sym => 1.0,
        :reps => 12,
        :tgrid => tgrid
    )

    # === deterministic RNG ===
    base_rng = MersenneTwister(2025)
    seeds = Dict()  # will store per-cell seeds for deterministic mode

    for (p1, p2, vals1, vals2) in param_pairs
        fig = Figure(size=(1200, 850))
        Label(fig[0, 1:3],
            "Fig. 2 — Dominant branch across ($(p1), $(p2)) and time ($(mode) mode)";
            fontsize=22, font=:bold)

        for (j, tsel) in enumerate(timeslices)
            matcol = zeros(RGBf, length(vals2), length(vals1))

            for (ci, v1) in enumerate(vals1)
                for (ri, v2) in enumerate(vals2)
                    # --- build kwargs
                    kwargs = deepcopy(base_config)

                    # apply p1 and p2 values
                    if p1 == :degcv
                        kwargs[:deg_param] = v1
                    elseif p1 == :rho
                        kwargs[:rho_sym] = v1
                    else
                        kwargs[p1] = v1
                    end

                    if p2 == :degcv
                        kwargs[:deg_param] = v2
                    elseif p2 == :rho
                        kwargs[:rho_sym] = v2
                    else
                        kwargs[p2] = v2
                    end

                    # --- control randomness
                    if mode == :deterministic
                        # always same seed pattern → identical base network except for varied params
                        sid = UInt64(1000 * ci + ri)
                        if !haskey(seeds, sid)
                            seeds[sid] = rand(base_rng, UInt64)
                        end
                        kwargs[:seed] = seeds[sid]
                    elseif mode == :stochastic
                        kwargs[:seed] = rand(UInt64)  # fully random each cell
                    else
                        error("mode must be :deterministic or :stochastic")
                    end

                    # --- simulate scenario
                    sim = simulate_scenario(; kwargs...)
                    tidx = findmin(abs.(log10.(tgrid) .- log10(tsel)))[2]

                    vals = Dict(
                        :U => sim.R2_ush[tidx],
                        :IS => sim.R2_rsh[tidx],
                        :TOP => sim.R2_rew[tidx]
                    )

                    dominant = argmax(vals)
                    matcol[ri, ci] = colors[dominant]
                end
            end

            # --- plotting
            ax = Axis(fig[1, j];
                title = "t ≈ $(tsel)",
                xlabel = string(p1),
                ylabel = string(p2),
                xticks = (1:length(vals1), string.(vals1)),
                yticks = (1:length(vals2), string.(vals2))
            )
            image!(ax, permutedims(matcol))
        end

        outname = "JacobianApproach/FinalFigures/Fig2_$(p1)_vs_$(p2)_dominance_$(String(mode)).png"
        save(outname, fig)
        display(fig)
        println("✅ Saved: ", outname)
    end
end

# ------------------------------------------------------------
# 3. FIGURE 3: Lever plots (sweep one axis)
# ------------------------------------------------------------
function figure3_levers()
    tgrid = logspace10(-2,2,48)
    sweeps = Dict(
        :u_cv => [0.05,0.4,0.8,1.2,2.0,3.0],
        :mag_cv => [0.1,0.3,0.6,1.0,1.5,2.0],
        :degcv => [0.01,0.25,0.5,1.0,1.5,2.0]
    )

    for (kind, vals) in sweeps
        fig = Figure(size=(1100,650))
        Label(fig[0,1:3], "Fig 3. Predictability vs $(kind) sweep"; fontsize=22, font=:bold)
        for (i, val) in enumerate(vals)
            sim = simulate_scenario(; u_cv=(kind==:u_cv ? val : 0.6),
                                      mag_cv=(kind==:mag_cv ? val : 0.6),
                                      deg_param=(kind==:degcv ? val : 0.8),
                                      rho_sym=1.0, reps=16, tgrid)
            ax = Axis(
                fig[(i-1)÷3+1,(i-1)%3+1];
                title="$(kind)=$(round(val,digits=2))", xscale=log10,
                limits=((nothing, nothing), (-0.05, 1.05)),
                xlabel="time t", ylabel="R²"
            )
            lines!(ax, sim.tgrid, sim.R2_ush; color=:blue, label="ushuf")
            lines!(ax, sim.tgrid, sim.R2_rsh; color=:orange, label="reshuf(pair)")
            lines!(ax, sim.tgrid, sim.R2_rew; color=:green, label="rew")

            if i==1; axislegend(ax; position=:lb, framevisible=false); end
        end
        save("JacobianApproach/FinalFigures/Fig3_lever_$(kind).png", fig)
        display(fig)
    end
end

# ------------------------------------------------------------
# 4. Run all and export
# ------------------------------------------------------------
function main()
    mkpath("JacobianApproach/FinalFigures")
    figure1_archetypes()
    figure2_maps(; mode=:deterministic)
    figure3_levers()
    println("Figures saved under ./JacobianApproach/FinalFigures/")
end

# Uncomment to run all
main()
