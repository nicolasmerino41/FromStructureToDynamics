function plot_nine_random_communities(
    ; S=120, conn=0.10, mean_abs=0.5, mag_cv=0.60,
    rho_sym=0.5, u_mean=1.0, u_cv=0.6, IS_target=0.5,
    t_vals=10 .^ range(-2, 2; length=20), seed=1234,
    single_community=true, absdiff::Bool=true
    )
    if single_community && !absdiff
        @warn "SINGLE COMMUNITY CAN ONLY BE PLOTTED WITH ABSDIFF"
    end
    rng0 = Random.Xoshiro(UInt(seed))
    fig = Figure(size=(1100, 900))
    Label(
        fig[0, 1:3], 
        "Random ER communities — Convergence toward toward resilience; SINGLE COMMUNITY = $single_community & ABSDIFF = $absdiff";
        fontsize=15, font=:bold, halign=:left
    )

    ncols, nrows = 3, 3

    for k in 1:9
        ri, ci = ((k-1) ÷ ncols) + 1, ((k-1) % ncols) + 1

        ax = Axis(
            fig[ri, ci];
            xscale=log10,
            xlabel="time t",
            ylabel=(ci == 1 ? "median return rate" : ""),
            title="Community $k"
        )

        rngA = Random.Xoshiro(rand(rng0, UInt))
        u = random_u(S; mean=u_mean, cv=u_cv, rng=Random.Xoshiro(rand(rng0, UInt)))

        # --- single original community
        A0 = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs, 
                                     mag_cv=mag_cv, rho_sym=rho_sym, rng=rngA)
        J0 = jacobian(A0, u)
        resilience0 = -maximum(real(eigvals(J0)))
        rmed0 = [median_return_rate(J0, u; t=t, perturbation=:biomass) for t in t_vals]

        # lines!(ax, t_vals, rmed0; color=:dodgerblue, linewidth=2)
        hlines!(ax, resilience0; color=:dodgerblue, linestyle=:dash, linewidth=1.5)

        if single_community
            # -------------------------------------------------- single pair
            lines!(ax, t_vals, rmed0; color=:dodgerblue, linewidth=2)
            hlines!(ax, resilience0; color=:dodgerblue, linestyle=:dash, linewidth=1.5)
            rngB = Random.Xoshiro(rand(rng0, UInt))
            A1 = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs,
                                         mag_cv=mag_cv, rho_sym=rho_sym, rng=rngB)
            J1 = jacobian(A1, u)
            resilience1 = -maximum(real(eigvals(J1)))
            rmed1 = [median_return_rate(J1, u; t=t, perturbation=:biomass) for t in t_vals]
            lines!(ax, t_vals, rmed1; color=:orange, linewidth=2)
            hlines!(ax, resilience1; color=:orange, linestyle=:dash, linewidth=1.5)
            absdiff_vals = abs.(rmed0 .- rmed1)

            ax2 = Axis(fig[ri, ci], 
                       yaxisposition=:right, ylabel="|ΔRₘₑd|",
                       yticklabelcolor=:red, ylabelcolor=:red,
                       xscale=log10)
            hidespines!(ax2, :l, :t)
            linkxaxes!(ax, ax2)
            lines!(ax2, t_vals, absdiff_vals; color=:red, linewidth=1.8)

        else
            # -------------------------------------------------- 20×20 paired comparison
            rngB = Random.Xoshiro(rand(rng0, UInt))
            nrep = 20
            all_rmed0 = Matrix{Float64}(undef, length(t_vals), nrep)
            all_rmed1 = Matrix{Float64}(undef, length(t_vals), nrep)

            for i in 1:nrep
                # original i
                new_u = random_u(S; mean=u_mean, cv=u_cv, rng=Random.Xoshiro(rand(UInt)))
                A0i = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs,
                                              mag_cv=mag_cv, rho_sym=rho_sym, rng=rngA)
                J0i = jacobian(A0i, u)
                rmed0i = [median_return_rate(J0i, new_u; t=t, perturbation=:biomass) for t in t_vals]
                all_rmed0[:, i] .= rmed0i
                lines!(ax, t_vals, rmed0i; color=:dodgerblue, linewidth=1.2)

                # new i
                # A1i = build_random_trophic_ER(
                #     S; conn=conn, mean_abs=mean_abs,
                #     mag_cv=mag_cv, rho_sym=rho_sym, rng=rngB
                # )
                A1i = build_random_ER(
                    S; conn=conn, mean_abs=mean_abs,
                    mag_cv=mag_cv, rho_sym=0.0, rng=rngB
                )
                J1i = jacobian(A1i, u)
                rmed1i = [median_return_rate(J1i, new_u; t=t, perturbation=:biomass) for t in t_vals]
                all_rmed1[:, i] .= rmed1i
                lines!(ax, t_vals, rmed1i; color=:orange, linewidth=1.2)
            end

            # compute mean R²(t) across 20 pairs
            r2_vals = Float64[]
            for ti in eachindex(t_vals)
                r2_t = r2_to_identity(all_rmed0[ti, :], all_rmed1[ti, :])
                r2_t = isnan(r2_t) || !isfinite(r2_t) ? 0.0 : clamp(r2_t, 0, 1)
                push!(r2_vals, r2_t)
            end
            absdiff_vals = Float64[]
            for ti in eachindex(t_vals)
                push!(absdiff_vals, mean(abs.(all_rmed0[ti, :] .- all_rmed1[ti, :])))
            end

            # secondary axis for R²
            ax2 = Axis(fig[ri, ci],
                       yaxisposition=:right,
                       ylabel= absdiff ? "|ΔRₘₑd|" : "R²",
                       yticklabelcolor=:red, ylabelcolor=:red,
                       xscale=log10)
            hidespines!(ax2, :l, :t)
            linkxaxes!(ax, ax2)
            if !absdiff
                lines!(ax2, t_vals, r2_vals; color=:red, linewidth=1.8)
            else
                lines!(ax2, t_vals, absdiff_vals; color=:red, linewidth=1.8)
            end
        end
    end

    display(fig)
end

# ---- run it ----
t_vals = 10 .^ range(-2, 2; length=20)
plot_nine_random_communities(; t_vals=t_vals, seed=Int(rand(1:200)), single_community=true, absdiff=true)
plot_nine_random_communities(; t_vals=t_vals, seed=Int(rand(1:200)), single_community=false, absdiff=true)
plot_nine_random_communities(; t_vals=t_vals, seed=Int(rand(1:200)), single_community=false, absdiff=false)
