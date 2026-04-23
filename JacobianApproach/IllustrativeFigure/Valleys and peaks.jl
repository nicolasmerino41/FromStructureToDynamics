begin
    using LinearAlgebra
    using CairoMakie
    using Printf

    # ============================================================
    # 3-species demonstration:
    # valley in RPR -> small trajectory change
    # peak in RPR   -> large trajectory change
    #
    # Layout (3 x 2):
    #   row 1: forcing (low-RPR regime) | forcing (high-RPR regime)
    #   row 2: community dynamics        | community dynamics
    #   row 3: intrinsic profile         | RPR profile
    #
    # Same forcing direction in both regimes.
    # The system starts at equilibrium and forcing begins at 10% of total time.
    # ============================================================

    # -------------------------
    # Parameters
    # -------------------------
    const OMEGAS = exp.(range(log(0.07), log(3.6), length = 2200))
    const DT = 0.03
    const TMAX = 95.0
    const FORCE_START = 0.15 * TMAX
    const FORCE_AMPLITUDE = 0.14

    # Structural change magnitude
    const EPS_P = 0.2

    # ============================================================
    # Linear algebra helpers
    # ============================================================
    function resolvent(A::AbstractMatrix{<:Real}, ω::Real)
        n = size(A, 1)
        Icomplex = Matrix{ComplexF64}(I, n, n)
        Ac = ComplexF64.(A)
        F = factorize(im * ω .* Icomplex - Ac)
        return F \ Icomplex
    end

    function intrinsic_profile(A::AbstractMatrix{<:Real}, ωs::AbstractVector{<:Real})
        vals = zeros(Float64, length(ωs))
        for (k, ω) in pairs(ωs)
            vals[k] = opnorm(resolvent(A, ω), 2)
        end
        return vals
    end

    function rpr_profile(A::AbstractMatrix{<:Real}, P::AbstractMatrix{<:Real}, ωs::AbstractVector{<:Real})
        vals = zeros(Float64, length(ωs))
        for (k, ω) in pairs(ωs)
            R = resolvent(A, ω)
            vals[k] = opnorm(R * P * R, 2)
        end
        return vals
    end

    # ============================================================
    # Frequency selection helpers
    # ============================================================
    function local_maxima(y::AbstractVector{<:Real})
        idx = Int[]
        for i in 2:length(y)-1
            if y[i] > y[i-1] && y[i] > y[i+1]
                push!(idx, i)
            end
        end
        return idx
    end

    function local_minima(y::AbstractVector{<:Real})
        idx = Int[]
        for i in 2:length(y)-1
            if y[i] < y[i-1] && y[i] < y[i+1]
                push!(idx, i)
            end
        end
        return idx
    end

    function choose_peak_in_band(S::AbstractVector, ωs::AbstractVector, band::Tuple{Float64,Float64})
        idxs = findall((ωs .>= band[1]) .& (ωs .<= band[2]))
        isempty(idxs) && error("No frequencies in band $(band)")
        return idxs[argmax(S[idxs])]
    end

    function choose_valley_in_band(S::AbstractVector, ωs::AbstractVector, band::Tuple{Float64,Float64})
        idxs = findall((ωs .>= band[1]) .& (ωs .<= band[2]))
        isempty(idxs) && error("No frequencies in band $(band)")
        return idxs[argmin(S[idxs])]
    end

    # ============================================================
    # Dynamics
    # ============================================================
    # dx/dt = A*x + u(t)*b
    function simulate_forced_system(
        A::AbstractMatrix{<:Real},
        b::AbstractVector{<:Real},
        ω::Float64;
        dt::Float64 = DT,
        tmax::Float64 = TMAX,
        forcing_amplitude::Float64 = FORCE_AMPLITUDE,
        forcing_start::Float64 = FORCE_START
    )
        ts = collect(0:dt:tmax)

        n = size(A, 1)
        X = zeros(Float64, n, length(ts))

        forcing_scalar(t) = t < forcing_start ? 0.0 :
                            forcing_amplitude * sin(ω * (t - forcing_start))

        forcing_at_time(t) = forcing_scalar(t) .* b

        f(x, t) = A * x + forcing_at_time(t)

        x = zeros(Float64, n)  # starts at equilibrium displacement 0
        for k in 1:length(ts)-1
            t = ts[k]
            k1 = f(x, t)
            k2 = f(x .+ 0.5dt .* k1, t + 0.5dt)
            k3 = f(x .+ 0.5dt .* k2, t + 0.5dt)
            k4 = f(x .+ dt .* k3, t + dt)
            x = x .+ (dt / 6.0) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
            X[:, k+1] .= x
        end

        forcing_signal = [forcing_scalar(t) for t in ts]
        return ts, X, forcing_signal
    end

    # ============================================================
    # Model
    # ============================================================
    # 3-species community:
    # a damped oscillatory 1-2 pair plus a third species weakly coupled in.
    A = [
        -0.16  -0.92   0.04   0.00   0.00   0.00
        0.90  -0.15  -0.06   0.00   0.00   0.00
        -0.02   0.08  -1.05   0.10   0.00   0.00
        0.00   0.00   0.06  -0.34  -1.22   0.00
        0.00   0.00   0.00   1.10  -0.30   0.12
        0.00   0.00   0.00   0.00   0.08  -1.40
    ]
    # Structural modification P:
    # changing the 2<->3 pathway
    P = [
        0.00   0.00   0.00   0.00   0.00   0.00
        0.00   0.00   0.00   0.00   0.00   0.00
        0.00   0.00   0.00   0.45   0.00   0.00
        0.00   0.00  -0.06   0.00  -1.30   0.00
        0.00   0.00   0.00   1.05   0.00   0.28
        0.00   0.00   0.00   0.00  -0.10   0.00
    ]
    A_mod = A + EPS_P .* P

    println("Eigenvalues of A:")
    println(eigvals(A))
    println("\nEigenvalues of A_mod:")
    println(eigvals(A_mod))

    # Frequency-domain profiles
    S_intr = intrinsic_profile(A, OMEGAS)
    S_rpr  = rpr_profile(A, P, OMEGAS)

    # Choose one valley and one peak from the RPR profile
    # valley in a lower-frequency band, peak at the dominant RPR feature
    idx_intr_valley = choose_valley_in_band(S_intr, OMEGAS, (0.20, 0.38))
    idx_intr_peak   = choose_peak_in_band(S_intr, OMEGAS, (0.62, 0.95))

    ω_intr_valley = OMEGAS[idx_intr_valley]
    ω_intr_peak   = OMEGAS[idx_intr_peak]
    idx_peak   = choose_peak_in_band(S_rpr, OMEGAS, (1.10, 1.90))
    idx_valley = choose_valley_in_band(S_rpr, OMEGAS, (0.28, 0.55))
    ω_valley = OMEGAS[idx_valley]
    ω_peak   = OMEGAS[idx_peak]

    println()
    println(@sprintf("Chosen valley frequency: ω = %.4f | RPR = %.4f", ω_valley, S_rpr[idx_valley]))
    println(@sprintf("Chosen peak frequency:   ω = %.4f | RPR = %.4f", ω_peak,   S_rpr[idx_peak]))
    println(@sprintf("RPR peak/valley ratio:   %.2f", S_rpr[idx_peak] / S_rpr[idx_valley]))

    # Same forcing direction in both regimes
    b = [0.65, -0.55, 0.22, 0.00, 0.00, 0.00]
    b ./= norm(b)

    # Simulations
    ts, X_valley_base, f_valley = simulate_forced_system(A,     b, ω_valley)
    _,  X_valley_mod,  _        = simulate_forced_system(A_mod, b, ω_valley)

    _,  X_peak_base,   f_peak   = simulate_forced_system(A,     b, ω_peak)
    _,  X_peak_mod,    _        = simulate_forced_system(A_mod, b, ω_peak)

    _, X_mod_intr_valley, f_intr_valley = simulate_forced_system(A_mod, b, ω_intr_valley)
    _, X_mod_intr_peak,   f_intr_peak   = simulate_forced_system(A_mod, b, ω_intr_peak)

    Δ_valley = [norm(X_valley_mod[:, i] - X_valley_base[:, i]) for i in eachindex(ts)]
    Δ_peak   = [norm(X_peak_mod[:, i]   - X_peak_base[:, i])   for i in eachindex(ts)]

    println(@sprintf("Max trajectory difference at valley: %.4f", maximum(Δ_valley)))
    println(@sprintf("Max trajectory difference at peak:   %.4f", maximum(Δ_peak)))
    println(@sprintf("Peak/valley trajectory-difference ratio: %.2f", maximum(Δ_peak) / maximum(Δ_valley)))

    # ============================================================
    # Axis helpers
    # ============================================================
    function forcing_ylim(f::AbstractVector)
        m = maximum(abs, f)
        return (-1.08max(m, 1e-3), 1.08max(m, 1e-3))
    end

    function robust_state_ylim(X1::AbstractMatrix, X2::AbstractMatrix; q::Float64 = 0.985)
        vals = abs.(vcat(vec(X1), vec(X2)))
        m = quantile(vals, q)
        m = max(m, 0.08)
        return (-1.3m, 1.3m)
    end

    function shared_profile_ylim(S1::AbstractVector, S2::AbstractVector)
        ymin = min(minimum(S1), minimum(S2))
        ymax = max(maximum(S1), maximum(S2))
        ymin = max(ymin, 1e-4)
        ymax = max(ymax, 10ymin)
        return (0.95ymin, 1.08ymax)
    end

    # ============================================================
    # Plot
    # ============================================================
    begin
        fig = Figure(size = (1600, 1040))
        Label(
            fig[0, 1:2],
            "Valleys and peaks in the RPR profile predict the time-domain impact of structural change",
            fontsize = 26
        )

        # -------------
        # Top row: forcing
        # -------------
        y_force_valley = forcing_ylim(f_valley)
        y_force_peak   = forcing_ylim(f_peak)

        ax_force_valley = Axis(
            fig[1, 1],
            title = @sprintf("Low-RPR regime (ω = %.3f)", ω_valley),
            xlabel = "time",
            ylabel = "u(t)"
        )
        xlims!(ax_force_valley, first(ts), last(ts))
        ylims!(ax_force_valley, y_force_valley...)
        lines!(ax_force_valley, ts, f_valley, linewidth = 3, color = :firebrick)
        vlines!(ax_force_valley, [FORCE_START], color = :gray35, linestyle = :dash, linewidth = 2)
        # text!(ax_force_valley, FORCE_START, y_force_valley[2]*0.92, text = "forcing starts", align = (:left, :top), fontsize = 16)

        ax_force_peak = Axis(
            fig[1, 2],
            title = @sprintf("High-RPR regime (ω = %.3f)", ω_peak),
            xlabel = "time",
            ylabel = "u(t)"
        )
        xlims!(ax_force_peak, first(ts), last(ts))
        ylims!(ax_force_peak, y_force_peak...)
        lines!(ax_force_peak, ts, f_peak, linewidth = 3, color = :navy)
        vlines!(ax_force_peak, [FORCE_START], color = :gray35, linestyle = :dash, linewidth = 2)
        # text!(ax_force_peak, FORCE_START, y_force_peak[2]*0.92, text = "forcing starts", align = (:left, :top), fontsize = 16)

        # -------------
        # Middle row: dynamics
        # -------------
        y_state = robust_state_ylim(
        hcat(X_valley_base[1:3, :], X_peak_base[1:3, :]),
        hcat(X_valley_mod[1:3, :],  X_peak_mod[1:3, :])
    )

        species_colors = [:black, :royalblue, :seagreen, :orange, :firebrick, :navy]

        ax_dyn_valley = Axis(
            fig[2, 1],
            title = "Trajectory response under structural change: low-RPR regime",
            xlabel = "time",
            ylabel = "species displacement"
        )
        xlims!(ax_dyn_valley, first(ts), last(ts))
        ylims!(ax_dyn_valley, y_state...)
        vlines!(ax_dyn_valley, [FORCE_START], color = :gray35, linestyle = :dash, linewidth = 2)

        for sp in 1:6
            lines!(ax_dyn_valley, ts, X_valley_base[sp, :], color = species_colors[sp], linewidth = 2.5,
                label = sp == 1 ? "baseline" : nothing)
            # lines!(ax_dyn_valley, ts, X_valley_mod[sp, :], color = species_colors[sp], linewidth = 2.5,
            #        linestyle = :dash, label = sp == 1 ? "modified" : nothing)
        end

        # right axis for trajectory difference
        ax_dyn_valley_r = Axis(
            fig[2, 1],
            yaxisposition = :right,
            ylabel = "‖Δx(t)‖"
        )
        hidespines!(ax_dyn_valley_r)
        hidexdecorations!(ax_dyn_valley_r)
        ylims!(ax_dyn_valley_r, 0, maximum(Δ_peak) * 1.05)
        # lines!(ax_dyn_valley_r, ts, Δ_valley, color = :firebrick, linewidth = 3.6)

        # axislegend(ax_dyn_valley, position = :rt)

        ax_dyn_peak = Axis(
            fig[2, 2],
            title = "Trajectory response under structural change: high-RPR regime",
            xlabel = "time",
            ylabel = "species displacement"
        )
        xlims!(ax_dyn_peak, first(ts), last(ts))
        ylims!(ax_dyn_peak, y_state...)
        vlines!(ax_dyn_peak, [FORCE_START], color = :gray35, linestyle = :dash, linewidth = 2)

        for sp in 1:6
            lines!(ax_dyn_peak, ts, X_peak_base[sp, :], color = species_colors[sp], linewidth = 2.5,
                label = sp == 1 ? "baseline" : nothing)
            # lines!(ax_dyn_peak, ts, X_peak_mod[sp, :], color = species_colors[sp], linewidth = 2.5,
            #        linestyle = :dash, label = sp == 1 ? "modified" : nothing)
        end

        ax_dyn_peak_r = Axis(
            fig[2, 2],
            yaxisposition = :right,
            ylabel = "‖Δx(t)‖"
        )
        hidespines!(ax_dyn_peak_r)
        hidexdecorations!(ax_dyn_peak_r)
        ylims!(ax_dyn_valley_r, 0, maximum(Δ_peak) * 1.25)
        # ylims!(ax_dyn_peak_r, 0, maximum(Δ_peak) * 1.25)
        # lines!(ax_dyn_peak_r, ts, Δ_peak, color = :navy, linewidth = 3.6)

        # axislegend(ax_dyn_peak, position = :rt)

        # -------------
        # Bottom row: profiles
        # -------------
        y_prof = shared_profile_ylim(S_intr, S_rpr)

        ax_intr = Axis(
            fig[3, 1],
            title = "Intrinsic sensitivity profile",
            xlabel = "frequency ω",
            ylabel = "‖R(iω)‖₂",
            xscale = log10,
            yscale = log10
        )
        ylims!(ax_intr, y_prof...)
        lines!(ax_intr, OMEGAS, S_intr, linewidth = 3.2, color = :black)
        vlines!(ax_intr, [ω_valley], color = :firebrick, linestyle = :dash, linewidth = 3)
        vlines!(ax_intr, [ω_peak],   color = :navy,      linestyle = :dash, linewidth = 3)

        ax_rpr = Axis(
            fig[3, 2],
            title = "RPR profile  ‖R(iω) P R(iω)‖₂",
            xlabel = "frequency ω",
            ylabel = "RPR",
            xscale = log10,
            yscale = log10
        )
        ylims!(ax_rpr, y_prof...)
        lines!(ax_rpr, OMEGAS, S_rpr, linewidth = 3.2, color = :purple4)
        vlines!(ax_rpr, [ω_valley], color = :firebrick, linestyle = :dash, linewidth = 3)
        vlines!(ax_rpr, [ω_peak],   color = :navy,      linestyle = :dash, linewidth = 3)
        scatter!(ax_rpr, [ω_valley], [S_rpr[idx_valley]], color = :firebrick, markersize = 13)
        scatter!(ax_rpr, [ω_peak],   [S_rpr[idx_peak]],   color = :navy, markersize = 13)

        text!(ax_rpr, ω_valley, S_rpr[idx_valley] * 1.45,
            text = "valley → small trajectory change", color = :firebrick,
            align = (:center, :bottom), fontsize = 16)

        text!(ax_rpr, ω_peak, S_rpr[idx_peak] * 1.15,
            text = "peak → large trajectory change", color = :navy,
            align = (:center, :bottom), fontsize = 16)

                # -------------
    # New bottom row: modified community at intrinsic valley/peak frequencies
    # -------------
    y_mod_intr = robust_state_ylim(X_mod_intr_valley[1:3, :], X_mod_intr_peak[1:3, :])

    ax_mod_intr_valley = Axis(
        fig[4, 1],
        title = @sprintf("Modified community at intrinsic valley (ω = %.3f)", ω_intr_valley),
        xlabel = "time",
        ylabel = "species displacement"
    )
    xlims!(ax_mod_intr_valley, first(ts), last(ts))
    ylims!(ax_mod_intr_valley, y_mod_intr...)
    vlines!(ax_mod_intr_valley, [FORCE_START], color = :gray35, linestyle = :dash, linewidth = 2)

    for sp in 1:6
        lines!(ax_mod_intr_valley, ts, X_mod_intr_valley[sp, :],
            color = species_colors[sp], linewidth = 2.5)
    end

    ax_mod_intr_peak = Axis(
        fig[4, 2],
        title = @sprintf("Modified community at intrinsic peak (ω = %.3f)", ω_intr_peak),
        xlabel = "time",
        ylabel = "species displacement"
    )
    xlims!(ax_mod_intr_peak, first(ts), last(ts))
    ylims!(ax_mod_intr_peak, y_mod_intr...)
    vlines!(ax_mod_intr_peak, [FORCE_START], color = :gray35, linestyle = :dash, linewidth = 2)

    for sp in 1:6
        lines!(ax_mod_intr_peak, ts, X_mod_intr_peak[sp, :],
            color = species_colors[sp], linewidth = 2.5)
    end

        rowgap!(fig.layout, 18)
        colgap!(fig.layout, 22)

        display(fig)
    end
end