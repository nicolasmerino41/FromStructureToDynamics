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
r2_to_identity(x::AbstractVector, y::AbstractVector) = begin
    n = length(x)
    n == 0 && return 0.0
    μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
    sst == 0 ? (ssr == 0 ? 1.0 : 0.0) : max(1 - ssr/sst, 0.0)
end
logspace10(a::Real, b::Real, n::Int) = 10 .^ range(a, b; length=n)

# metrics per t for 3 branches
function _metrics_for_times(A, u, tgrid; rng)
    Jfull = jacobian(A, u)
    α = alpha_off_from(Jfull, u)
    u_sh = copy(u); shuffle!(rng, u_sh)
    α_rp = op_reshuffle_preserve_pairs(α; rng)
    # α_rp = op_reshuffle_alpha(α; rng)
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
        A0 = build_niche_trophic(S; conn, mean_abs, mag_cv, degree_family, deg_param, rho_sym, rng)
        baseIS = realized_IS(A0); baseIS ≤ 0 && continue
        A = (IS_target / baseIS) .* A0
        u = random_u(S; mean=u_mean, cv=u_cv, rng)
        full, ush, rsh, rew = _metrics_for_times(A, u, tgrid; rng)
        for i in eachindex(tgrid)
            push!(F[i], full[i]); push!(U[i], ush[i]); push!(R[i], rsh[i]); push!(W[i], rew[i])
        end
    end
    (; tgrid, R2_ush=[r2_to_identity(F[i],U[i]) for i in eachindex(tgrid)],
        R2_rsh=[r2_to_identity(F[i],R[i]) for i in eachindex(tgrid)],
        R2_rew=[r2_to_identity(F[i],W[i]) for i in eachindex(tgrid)])
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
    save("JacobianApproach/FinalFigures/Fig1_archetypes_reshuf.png", fig); display(fig)
end

# ------------------------------------------------------------
# 2. FIGURE 2: Dominance maps (ρ × heterogeneity)
# ------------------------------------------------------------
"""
figure2_maps(; mode=:deterministic, tgrid=logspace10(-2, 2, 40), reps=4)

Generates Fig. 2-like dominance maps with:
- Threaded parameter sweep (analysis stage)
- Separate plotting stage
- Dense grids for smooth structure
- Saturation proportional to dominance strength (ΔR²)
- Dominance defined as *lowest R²* (most important branch)

Arguments
---------
mode :: Symbol  — `:deterministic` or `:random`
tgrid :: Vector — times to simulate
reps  :: Int    — repetitions per (p1,p2) pair (ignored for deterministic)

Returns
-------
Nothing; saves one figure per parameter pair to FinalFigures/
"""
function figure2_maps(
    ; mode::Symbol = :deterministic,
    #    tgrid = logspace10(-2, 2, 40),
    tgrid = logspace10(-2, 2, 48),
    reps::Int = 3, granularity::Int = 20
)
    # === PARAMETERS ==========================================================
    # timeslices = [0.01, 0.5, 10.0]
    timeslices = [20.0, 50.0, 100.0]
    param_pairs = [
        (:degcv, :u_cv,  range(0.05, 1.8; length=granularity), range(0.2, 3.0; length=granularity)),
        (:mag_cv, :u_cv, range(0.1, 1.5; length=granularity), range(0.2, 3.0; length=granularity)),
        (:mag_cv, :degcv, range(0.1, 1.5; length=granularity), range(0.05, 1.8; length=granularity)),
        (:mag_cv, :rho,   range(0.1, 1.5; length=granularity), range(0.0, 1.0; length=granularity))
    ]

    colors = Dict(:U => RGBf(0.2,0.4,1.0),   # blue
                  :IS => RGBf(1.0,0.6,0.2),  # orange
                  :TOP => RGBf(0.2,0.8,0.2)) # green

    mkpath("JacobianApproach/FinalFigures")

    # === HELPER FUNCTIONS ====================================================
    @inline function _splitmix64(x::UInt64)
        x += 0x9E3779B97F4A7C15
        z = x
        z ⊻= z >>> 30; z *= 0xBF58476D1CE4E5B9
        z ⊻= z >>> 27; z *= 0x94D049BB133111EB
        return z ⊻ (z >>> 31)
    end

    # map symbolic names to simulate_scenario kwargs
    _map_param(sym::Symbol) = sym === :degcv ? :deg_param :
                              sym === :rho   ? :rho_sym : sym

    # === MAIN LOOP OVER PARAMETER PAIRS =====================================
    for (p1, p2, vals1, vals2) in param_pairs
        println("🧮 Running parameter sweep for ($p1, $p2)... grid $(length(vals1))×$(length(vals2))")
        results = Dict{Float64, Matrix{RGBf}}()

        # precompute all (v1,v2) combinations
        combos = [(v1,v2) for v1 in vals1 for v2 in vals2]

        # === THREADED ANALYSIS ==============================================
        all_results = Dict(ts => Matrix{RGBf}(undef, length(vals2), length(vals1)) for ts in timeslices)
        Threads.@threads for idx in eachindex(combos)
            v1, v2 = combos[idx]

            # reproducible RNG seed
            seed = if mode == :deterministic
                UInt64(42)
            else
                _splitmix64(UInt64(idx + 1000)) ⊻ UInt64(threadid())
            end

            # run simulation once per grid cell
            sim = simulate_scenario(
                ; u_cv = 0.6, mag_cv = 0.6,
                  deg_param = 0.8, rho_sym = 0.8,
                  reps = reps, tgrid,
                  (_map_param(p1)) => v1,
                  (_map_param(p2)) => v2,
                  seed = Int(seed)
            )

            # evaluate dominance at each time slice
            for tsel in timeslices
                tidx = findmin(abs.(log10.(tgrid) .- log10(tsel)))[2]
                vals = Dict(
                    :U   => sim.R2_ush[tidx],
                    :IS  => sim.R2_rsh[tidx],
                    :TOP => sim.R2_rew[tidx]
                )

                arr = collect(values(vals))
                keys_arr = collect(keys(vals))
                order = sortperm(arr)           # ascending (lowest = most important)
                dom = keys_arr[order[1]]
                Δ = arr[order[2]] - arr[order[1]]  # separation vs next

                basecol = colors[dom]
                α = clamp(Δ / 0.3, 0, 1)           # saturation scaling
                dimmed = RGBf(basecol.r*α + 0.2*(1-α),
                              basecol.g*α + 0.2*(1-α),
                              basecol.b*α + 0.2*(1-α))

                # find corresponding matrix indices
                ci = findfirst(==(v1), vals1)
                ri = findfirst(==(v2), vals2)
                all_results[tsel][ri, ci] = dimmed
            end
        end # threads

        # === PLOTTING (single-threaded) =====================================
        fig = Figure(size=(1200, 850))
        Label(fig[0, 1:3],
            "Fig. 2 — Dominant branch across ($(p1), $(p2)) and time ($(mode) mode)";
            fontsize=22, font=:bold)

        for (j, tsel) in enumerate(timeslices)
            mat = all_results[tsel]
            ax = Axis(fig[1, j];
                title = "t ≈ $(tsel)",
                xlabel = string(p1),
                ylabel = string(p2),
                xticks = (1:length(vals1), string.(round.(collect(vals1), digits=2))),
                yticks = (1:length(vals2), string.(round.(collect(vals2), digits=2))))
            image!(ax, permutedims(mat))
        end

        outname = "JacobianApproach/FinalFigures/Fig2_$(p1)_vs_$(p2)_dominance_$(mode)_highTs.png"
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
    # figure1_archetypes()
    figure2_maps(; mode=:deterministic, granularity=5)
    # figure3_levers()
    # println("Figures saved under ./JacobianApproach/FinalFigures/")
end

# Uncomment to run all
main()