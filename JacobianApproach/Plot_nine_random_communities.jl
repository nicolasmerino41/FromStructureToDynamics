function plot_nine_random_communities(
    ; S=120, conn=0.10, mean_abs=0.5, mag_cv=0.60,
    rho_sym=0.5, u_mean=1.0, u_cv=0.6, IS_target=0.5,
    t_vals=10 .^ range(-2, 2; length=20), seed=1234,
    single_community=true, absdiff::Bool=true,
    same_u::Bool=true, reshuffled::Bool=false,
    perturbation_type=:biomass
)
    if single_community && !absdiff
        @warn "SINGLE COMMUNITY CAN ONLY BE PLOTTED WITH ABSDIFF"
    end

    # helpers ------------------------------------------------------------
    finite_extrema(v) = begin
        vals = filter(isfinite, v)
        isempty(vals) ? (0.0, 1.0) : (minimum(vals), maximum(vals))
    end
    pad_range(minv, maxv) = begin
        if !isfinite(minv) || !isfinite(maxv) || minv == maxv
            δ = isfinite(minv) ? abs(minv) : 1.0
            return (minv - 0.05δ, minv + 0.95δ)
        end
        δ = 0.03 * (maxv - minv + eps())
        (minv - δ, maxv + δ)
    end
    # -------------------------------------------------------------------

    rng0 = Random.Xoshiro(UInt(seed))
    fig = Figure(size=(1100, 900))
    if same_u
        Label(
            fig[0, 1:3],
            "Random ER communities — Convergence toward toward resilience; SINGLE COMMUNITY = $single_community & ABSDIFF = $absdiff, same_u = $same_u, perturbation_type = $perturbation_type";
            fontsize=13, font=:bold, halign=:left
        )
    else
        Label(
            fig[0, 1:3],
            "Random ER communities — Convergence toward toward resilience; SINGLE COMMUNITY = $single_community & ABSDIFF = $absdiff, same_u = $same_u, perturbation_type = $perturbation_type";
            fontsize=13, font=:bold, halign=:left
        )
    end

    ncols, nrows = 3, 3
    tmin, tmax = minimum(t_vals), maximum(t_vals)

    for k in 1:9
        ri, ci = ((k-1) ÷ ncols) + 1, ((k-1) % ncols) + 1

        ax = Axis(fig[ri, ci];
                  xscale=log10,
                  xlabel="time t",
                  ylabel=(ci == 1 ? "median return rate" : ""),
                  title="Community $k")

        # Collect finite y-values for limits (left & right axes) -----------
        left_vals  = Float64[]   # r_med curves + resilience guides
        right_vals = Float64[]   # |ΔR_med| or R²

        rngA = Random.Xoshiro(rand(rng0, UInt))
        u = random_u(S; mean=u_mean, cv=u_cv, rng=Random.Xoshiro(rand(rng0, UInt)))

        # --- original community
        A0 = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs,
                                     mag_cv=mag_cv, rho_sym=rho_sym, rng=rngA)
        J0 = jacobian(A0, u)
        resilience0 = -maximum(real(eigvals(J0)))
        rmed0 = [median_return_rate(J0, u; t=t, perturbation=perturbation_type) for t in t_vals]

        append!(left_vals, filter(isfinite, rmed0))
        if isfinite(resilience0); push!(left_vals, resilience0); end
        hlines!(ax, resilience0; color=:dodgerblue, linestyle=:dash, linewidth=1.5)

        if single_community
            # single pair ---------------------------------------------------
            lines!(ax, t_vals, rmed0; color=:dodgerblue, linewidth=2)

            rngB = Random.Xoshiro(rand(rng0, UInt))
            A1 = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs,
                                         mag_cv=mag_cv, rho_sym=rho_sym, rng=rngB)
            J1 = jacobian(A1, u)
            resilience1 = -maximum(real(eigvals(J1)))
            rmed1 = [median_return_rate(J1, u; t=t, perturbation=perturbation_type) for t in t_vals]

            lines!(ax, t_vals, rmed1; color=:orange, linewidth=2)
            hlines!(ax, resilience1; color=:orange, linestyle=:dash, linewidth=1.5)

            append!(left_vals, filter(isfinite, rmed1))
            if isfinite(resilience1); push!(left_vals, resilience1); end

            absdiff_vals = abs.(rmed0 .- rmed1)
            append!(right_vals, filter(isfinite, absdiff_vals))

            ax2 = Axis(fig[ri, ci];
                       yaxisposition=:right, ylabel="|ΔRₘₑd|",
                       yticklabelcolor=:red, ylabelcolor=:red,
                       xscale=log10)
            hidespines!(ax2, :l, :t)
            linkxaxes!(ax, ax2)
            lines!(ax2, t_vals, absdiff_vals; color=:red, linewidth=1.8)

            # limits (finite only)
            lmin, lmax = finite_extrema(left_vals)
            rmin, rmax = finite_extrema(right_vals)
            lmin, lmax = pad_range(lmin, lmax)
            rmin, rmax = pad_range(rmin, rmax)

            xlims!(ax,  tmin, tmax); ylims!(ax,  lmin, lmax)
            xlims!(ax2, tmin, tmax); ylims!(ax2, rmin, rmax)

        else
            # 20×20 paired comparison --------------------------------------
            rngB = Random.Xoshiro(rand(rng0, UInt))
            nrep = 20
            all_rmed0 = Matrix{Float64}(undef, length(t_vals), nrep)
            all_rmed1 = Matrix{Float64}(undef, length(t_vals), nrep)

            for i in 1:nrep
                new_u = random_u(S; mean=u_mean, cv=u_cv, rng=Random.Xoshiro(rand(rng0, UInt)))

                A0i = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs,
                                              mag_cv=mag_cv, rho_sym=rho_sym, rng=Random.Xoshiro(rand(rng0, UInt)))
                J0i = jacobian(A0i, new_u)
                rmed0i = [median_return_rate(J0i, new_u; t=t, perturbation=perturbation_type) for t in t_vals]
                all_rmed0[:, i] .= rmed0i
                lines!(ax, t_vals, rmed0i; color=:dodgerblue, linewidth=1.2)
                append!(left_vals, filter(isfinite, rmed0i))
                if same_u
                    new_u_i = new_u
                else
                    if reshuffled
                        new_u_i = reshuffle_u(new_u; rng=Random.Xoshiro(rand(rng0, UInt)))
                    else
                        new_u_i = random_u(S; mean=u_mean, cv=u_cv, rng=Random.Xoshiro(rand(rng0, UInt)))
                    end
                end
                A1i = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs,
                                              mag_cv=mag_cv, rho_sym=rho_sym, rng=Random.Xoshiro(rand(rng0, UInt)))
                J1i = jacobian(A1i, new_u_i)  # as you had it
                rmed1i = [median_return_rate(J1i, new_u_i; t=t, perturbation=perturbation_type) for t in t_vals]
                all_rmed1[:, i] .= rmed1i
                lines!(ax, t_vals, rmed1i; color=:orange, linewidth=1.2)
                append!(left_vals, filter(isfinite, rmed1i))
            end

            # RHS: |ΔR_med| or R²
            if absdiff
                absdiff_vals = [mean(abs.(all_rmed0[ti, :] .- all_rmed1[ti, :])) for ti in eachindex(t_vals)]
                append!(right_vals, filter(isfinite, absdiff_vals))
            else
                r2_vals = Float64[]
                for ti in eachindex(t_vals)
                    r2_t = r2_to_identity(all_rmed0[ti, :], all_rmed1[ti, :])
                    push!(r2_vals, (isfinite(r2_t) && !isnan(r2_t)) ? r2_t : 0.0)
                end
                append!(right_vals, filter(isfinite, r2_vals))
            end

            ax2 = Axis(fig[ri, ci];
                       yaxisposition=:right,
                       ylabel= absdiff ? "|ΔRₘₑd|" : "R²",
                       yticklabelcolor=:red, ylabelcolor=:red,
                       xscale=log10)
            hidespines!(ax2, :l, :t)
            linkxaxes!(ax, ax2)
            if absdiff
                lines!(ax2, t_vals, absdiff_vals; color=:red, linewidth=1.8)
            else
                lines!(ax2, t_vals, r2_vals;      color=:red, linewidth=1.8)
            end

            # limits (finite only)
            lmin, lmax = finite_extrema(left_vals)
            rmin, rmax = finite_extrema(right_vals)
            lmin, lmax = pad_range(lmin, lmax)
            rmin, rmax = pad_range(rmin, rmax)

            xlims!(ax,  tmin, tmax); ylims!(ax, -0.1, lmax)
            if !absdiff
                xlims!(ax2, tmin, tmax); ylims!(ax2, -0.1/lmax, 1.1)
            else
                xlims!(ax2, tmin, tmax); ylims!(ax2, rmin, rmax)
            end
        end
    end

    display(fig)
end
# ---- run it ----
t_vals = 10 .^ range(-2, 5; length=20)
plot_nine_random_communities(; t_vals=t_vals, seed=Int(rand(1:200)), single_community=true, absdiff=true)
plot_nine_random_communities(
    ; t_vals=t_vals, seed=Int(rand(1:200)), single_community=false,
    absdiff=true, same_u=false, reshuffled=false, perturbation_type=:uniform
)
plot_nine_random_communities(
    ; t_vals=t_vals, seed=Int(rand(1:200)), single_community=false,
    absdiff=false, same_u=false, reshuffled=false, perturbation_type=:uniform
)
