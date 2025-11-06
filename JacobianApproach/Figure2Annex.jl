"""
figure2_r2grids(; mode=:deterministic, tgrid, reps, granularity)

Generates 3×3 heatmap grids of actual R² values for each step (U, IS, TOP)
across parameter pairs and time scales, instead of dominance hue.

- Rows = steps (U, IS, TOP)
- Columns = times (short, mid, long)
- Each panel = R²(step) for that (p₁,p₂) parameter pair
"""
function figure2_r2grids(; mode::Symbol = :deterministic,
                           tgrid = logspace10(-2, 2, 40),
                           reps::Int = 4, granularity::Int = 20)

    # === PARAMETERS ==========================================================
    # timeslices = [0.01, 0.5, 10.0]
    timeslices = [20.0, 50.0, 100.0]
    param_pairs = [
        (:degcv, :u_cv,  range(0.05, 1.8; length=granularity), range(0.2, 3.0; length=granularity)),
        (:mag_cv, :u_cv, range(0.1, 1.5; length=granularity), range(0.2, 3.0; length=granularity)),
        (:mag_cv, :degcv, range(0.1, 1.5; length=granularity), range(0.05, 1.8; length=granularity)),
        (:mag_cv, :rho,   range(0.1, 1.5; length=granularity), range(0.0, 1.0; length=granularity))
    ]

    steps = [:U, :IS, :TOP]
    colorscale = :viridis
    mkpath("JacobianApproach/FinalFigures")

    # === HELPERS =============================================================
    @inline function _splitmix64(x::UInt64)
        x += 0x9E3779B97F4A7C15
        z = x
        z ⊻= z >>> 30; z *= 0xBF58476D1CE4E5B9
        z ⊻= z >>> 27; z *= 0x94D049BB133111EB
        return z ⊻ (z >>> 31)
    end

    _map_param(sym::Symbol) = sym === :degcv ? :deg_param :
                              sym === :rho   ? :rho_sym : sym

    # === MAIN LOOP ===========================================================
    for (p1, p2, vals1, vals2) in param_pairs
        println("🧮 Running parameter sweep for ($p1, $p2)... grid $(length(vals1))×$(length(vals2))")

        # store results as Dict[t][step] => matrix
        results = Dict{Float64,Dict{Symbol,Matrix{Float64}}}()
        for tsel in timeslices
            results[tsel] = Dict(s => Matrix{Float64}(undef, length(vals2), length(vals1)) for s in steps)
        end

        # all parameter combinations
        combos = [(v1, v2) for v1 in vals1 for v2 in vals2]

        Threads.@threads for idx in eachindex(combos)
            v1, v2 = combos[idx]

            seed = if mode == :deterministic
                UInt64(42)
            else
                _splitmix64(UInt64(idx + 1234)) ⊻ UInt64(threadid())
            end

            sim = simulate_scenario(
                ; u_cv = 0.6, mag_cv = 0.6,
                  deg_param = 0.8, rho_sym = 0.8,
                  reps = reps, tgrid,
                  (_map_param(p1)) => v1,
                  (_map_param(p2)) => v2,
                  seed = Int(seed)
            )

            for tsel in timeslices
                tidx = findmin(abs.(log10.(tgrid) .- log10(tsel)))[2]
                ci = findfirst(==(v1), vals1)
                ri = findfirst(==(v2), vals2)

                results[tsel][:U][ri, ci]   = sim.R2_ush[tidx]
                results[tsel][:IS][ri, ci]  = sim.R2_rsh[tidx]
                results[tsel][:TOP][ri, ci] = sim.R2_rew[tidx]
            end
        end # threads

        # === PLOTTING ========================================================
        fig = Figure(size=(1200, 1000))
        Label(fig[0, 1:3],
            "Fig. 2-R² — Branch predictability across ($(p1), $(p2)) and time ($(mode) mode)";
            fontsize=22, font=:bold)

        for (i, step) in enumerate(steps)
            for (j, tsel) in enumerate(timeslices)
                ax = Axis(fig[i, j];
                    title = "t ≈ $(tsel)",
                    xlabel = string(p1),
                    ylabel = string(p2),
                    xticks = (1:length(vals1), string.(round.(collect(vals1), digits=2))),
                    yticks = (1:length(vals2), string.(round.(collect(vals2), digits=2))),
                    titlealign = :left)
                mat = results[tsel][step]
                hm = heatmap!(ax, permutedims(mat); colormap=colorscale)
                # Colorbar(fig[i, j+1], hm; width=10, vertical=false, label="R²($(step))")
            end
        end

        outname = "JacobianApproach/FinalFigures/Fig2_R2grid_$(p1)_vs_$(p2)_$(mode)_highTs.png"
        save(outname, fig)  
        display(fig)
        println("✅ Saved: ", outname)
    end
end

figure2_r2grids(; mode=:deterministic, reps = 4, granularity = 5)