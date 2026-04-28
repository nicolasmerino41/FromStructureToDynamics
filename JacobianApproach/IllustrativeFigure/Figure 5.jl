using LinearAlgebra
using Statistics
using CairoMakie

begin

    # ============================================================
    # Multifrequency forcing demo:
    # intrinsic sensitivity selects the dominant response frequency
    # ============================================================

    # -------------------------
    # Helpers
    # -------------------------
    function resolvent(A::AbstractMatrix{<:Real}, ω::Real)
        n = size(A, 1)
        Icomplex = Matrix{ComplexF64}(I, n, n)
        Ac = ComplexF64.(A)
        F = factorize(im * ω .* Icomplex - Ac)
        return F \ Icomplex
    end

    function intrinsic_profile(
        A::AbstractMatrix{<:Real},
        ωs
    )
        vals = zeros(Float64, length(ωs))
        for (k, ω) in pairs(ωs)
            R = resolvent(A, ω)
            vals[k] = opnorm(R, 2)
        end
        return vals
    end

    function simulate_forced_system(
        A::AbstractMatrix{<:Real},
        forcing_fun::Function;
        dt::Float64,
        tmax::Float64
    )
        ts = collect(0:dt:tmax)
        n = size(A, 1)
        X = zeros(Float64, n, length(ts))

        f(x, t) = A * x + forcing_fun(t)

        x = zeros(Float64, n)
        X[:, 1] .= x

        for k in 1:length(ts)-1
            t = ts[k]
            k1 = f(x, t)
            k2 = f(x .+ 0.5 * dt .* k1, t + 0.5 * dt)
            k3 = f(x .+ 0.5 * dt .* k2, t + 0.5 * dt)
            k4 = f(x .+ dt .* k3, t + dt)
            x = x .+ (dt / 6.0) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
            X[:, k+1] .= x
        end

        return ts, X
    end

    # -------------------------
    # Target frequencies
    # -------------------------
    ω_low_target  = 0.08
    ω_mid_target  = 0.62
    ω_high_target = 2.8

    # -------------------------
    # Resonant blocks
    #
    # Three almost independent oscillatory blocks.
    # The intermediate block has much weaker damping,
    # so the intrinsic resolvent profile is dominated by it.
    # -------------------------
    d_low  = 2.0
    d_mid  = 0.00005
    d_high = 2.0

    ϵ12 = 2e-4
    ϵ23 = 2e-4

    A = [
        -d_low          -ω_low_target    ϵ12            0.0            0.0             0.0
         ω_low_target   -d_low           0.0            ϵ12            0.0             0.0

         ϵ12             0.0            -d_mid         -ω_mid_target   ϵ23             0.0
         0.0             ϵ12             ω_mid_target  -d_mid          0.0             ϵ23

         0.0             0.0             ϵ23            0.0           -d_high         -ω_high_target
         0.0             0.0             0.0            ϵ23            ω_high_target  -d_high
    ]

    # -------------------------
    # Frequency profile
    # -------------------------
    ωs = exp.(range(log(0.03), log(10.0), length=12000))
    S = intrinsic_profile(A, ωs)

    idx_low  = argmin(abs.(ωs .- ω_low_target))
    idx_mid  = argmin(abs.(ωs .- ω_mid_target))
    idx_high = argmin(abs.(ωs .- ω_high_target))

    ω_low  = ωs[idx_low]
    ω_mid  = ωs[idx_mid]
    ω_high = ωs[idx_high]

    # -------------------------
    # Time grid
    # -------------------------
    dt = 0.02
    tmax = 220.0

    # -------------------------
    # Multifrequency forcing
    #
    # Equal-amplitude low, intermediate, and high inputs.
    # Direction b is chosen so all three blocks are forced,
    # but the system response is still dominated by the intrinsically
    # sensitive intermediate block.
    # -------------------------
    b = [0.35, -0.25, 1.00, -0.85, 0.30, -0.20]
    b ./= norm(b)

    a_low  = 0.12
    a_mid  = 0.12
    a_high = 0.12

    ϕ_low  = 0.4
    ϕ_mid  = 1.1
    ϕ_high = -0.7

    low_component(t)  = a_low  * sin(ω_low  * t + ϕ_low)
    mid_component(t)  = a_mid  * sin(ω_mid  * t + ϕ_mid)
    high_component(t) = a_high * sin(ω_high * t + ϕ_high)

    forcing_scalar(t) =
        low_component(t) + mid_component(t) + high_component(t)

    forcing_fun(t) = forcing_scalar(t) .* b

    # -------------------------
    # Simulate
    # -------------------------
    ts, X = simulate_forced_system(A, forcing_fun; dt=dt, tmax=tmax)

    # Aggregate biomass/readout.
    # Mostly reads the intermediate block so that the community-level
    # response visibly locks onto the sensitive frequency.
    c = [0.15, 0.10, 2.4, 2.0, 0.12, 0.08]
    y = vec(c' * X)

    low_trace     = [low_component(t) for t in ts]
    mid_trace     = [mid_component(t) for t in ts]
    high_trace    = [high_component(t) for t in ts]
    forcing_trace = [forcing_scalar(t) for t in ts]

    # -------------------------
    # Limits
    # -------------------------
    yprof_min = max(minimum(S[S .> 0]), 1e-8)
    yprof_max = maximum(S)

    ycomp = maximum(abs, vcat(low_trace, mid_trace, high_trace))
    yf = maximum(abs, forcing_trace)
    yr = maximum(abs, y)
    yspecies = maximum(abs, X)

    # -------------------------
    # Colors
    # -------------------------
    low_col   = :dodgerblue3
    mid_col   = :orangered2
    high_col  = :seagreen4
    multi_col = :black
    prof_col  = :black

    species_cols = [
        :dodgerblue3,
        :deepskyblue4,
        :orangered2,
        :tomato3,
        :seagreen4,
        :darkolivegreen4
    ]

    # -------------------------
    # Figure
    # -------------------------
    fig = Figure(size=(1450, 1100), fontsize=16)

    Label(
        fig[0, 1:2],
        "A multifrequency perturbation enters the system, but the community mainly responds at the intrinsically sensitive intermediate band",
        fontsize = 18
    )

    # A. Intrinsic profile
    axA = Axis(
        fig[1, 1:2],
        title = "A. Intrinsic sensitivity profile",
        xscale = log10,
        xticks = ([-0.6, -0.3, 0.0], ["-0.6", "-0.3", "0.0"]),
        xlabel = "log(ω)",
        ylabel = "resolvent norm",
        # xscale = log10,
        yscale = log10,
        xgridvisible = false,
        ygridvisible = false,
        xminorticksvisible = false,
        yminorticksvisible = false
    )

    lines!(axA, ωs, S, color = prof_col, linewidth = 4)

    vlines!(axA, [ω_low],  color = low_col,  linestyle = :dash, linewidth = 3)
    vlines!(axA, [ω_mid],  color = mid_col,  linestyle = :dash, linewidth = 3)
    vlines!(axA, [ω_high], color = high_col, linestyle = :dash, linewidth = 3)

    scatter!(axA, [ω_low],  [S[idx_low]],  color = low_col,  markersize = 10)
    scatter!(axA, [ω_mid],  [S[idx_mid]],  color = mid_col,  markersize = 12)
    scatter!(axA, [ω_high], [S[idx_high]], color = high_col, markersize = 10)

    # text!(axA, ω_low,  1.25 * S[idx_low],
    #     text = "low", color = low_col, align = (:center, :bottom), fontsize = 13)

    # text!(axA, ω_mid,  1.08 * S[idx_mid],
    #     text = "huge sensitive band", color = mid_col, align = (:center, :bottom), fontsize = 13)

    # text!(axA, ω_high, 1.25 * S[idx_high],
    #     text = "high", color = high_col, align = (:center, :bottom), fontsize = 13)

    ylims!(axA, 0.8 * yprof_min, 5.0 * yprof_max)
    xlims!(axA, 0.05, 4.0)
    # B1. Low component
    axB1 = Axis(
        fig[2, 1],
        title = "B1. Low-frequency component",
        xlabel = "time",
        ylabel = "amplitude",
        xgridvisible = false,
        ygridvisible = false
    )
    lines!(axB1, ts, low_trace, color = low_col, linewidth = 2.5)
    ylims!(axB1, -1.1 * ycomp, 1.1 * ycomp)

    # B2. Intermediate component
    axB2 = Axis(
        fig[3, 1],
        title = "B2. Intermediate-frequency component",
        xlabel = "time",
        ylabel = "amplitude",
        xgridvisible = false,
        ygridvisible = false
    )
    lines!(axB2, ts, mid_trace, color = mid_col, linewidth = 2.5)
    ylims!(axB2, -1.1 * ycomp, 1.1 * ycomp)

    # B3. High component
    axB3 = Axis(
        fig[4, 1],
        title = "B3. High-frequency component",
        xlabel = "time",
        ylabel = "amplitude",
        xgridvisible = false,
        ygridvisible = false
    )
    lines!(axB3, ts, high_trace, color = high_col, linewidth = 2.5)
    ylims!(axB3, -1.1 * ycomp, 1.1 * ycomp)

    # C. Combined forcing
    axC = Axis(
        fig[2:4, 2],
        title = "C. Single multifrequency perturbation",
        xlabel = "time",
        ylabel = "forcing",
        xgridvisible = false,
        ygridvisible = false
    )
    lines!(axC, ts, forcing_trace, color = multi_col, linewidth = 3)
    ylims!(axC, -1.1 * yf, 1.1 * yf)

    # # D. Individual species responses
    # axD = Axis(
    #     fig[5, 1:2],
    #     title = "D. Individual species responses",
    #     xlabel = "time",
    #     ylabel = "species state",
    #     xgridvisible = false,
    #     ygridvisible = false
    # )
    # # after simulation
    # burn = ts .>= 80.0

    # ts_plot = ts[burn]
    # X_plot = X[:, burn]

    # # biomass readout: ONLY intermediate species
    # c = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    # y = vec(c' * X)

    # y_plot = y[burn]
    # for i in 1:6
    #     lines!(axD, ts_plot, X_plot[i, :],
    #         color = species_cols[i],
    #         linewidth = 2,
    #         label = "species $i"
    #     )
    # end

    # axislegend(
    #     axD,
    #     ["species $i" for i in 1:6],
    #     position = :rt,
    #     framevisible = false,
    #     labelsize = 11
    # )

    ylims!(axD, -1.1 * yspecies, 1.1 * yspecies)

    # E. Aggregate community response
    axE = Axis(
        fig[5, 1],
        title = "E. Aggregate community response",
        xlabel = "time",
        ylabel = "biomass response",
        xgridvisible = false,
        ygridvisible = false
    )

    lines!(axE, ts_plot, y_plot, color = multi_col, linewidth = 3)
    yr = maximum(abs, y_plot)
    yspecies = maximum(abs, X_plot)
    ylims!(axE, -1.1 * yr, 1.1 * yr)

    rowgap!(fig.layout, 10)
    colgap!(fig.layout, 14)

    display(fig)
end