using CairoMakie, Random, Statistics

function plot_random_ER_degrees(; S=120, conn=0.10, mean_abs=0.10, mag_cv=0.60,
                                rho_sym=0.5, nplots=9, seed=42)
    rng = MersenneTwister(seed)
    fig1 = Figure(size=(1000,900))
    Label(fig1[0,1:3], "Random ER trophic — In/Out Degree Distributions";
          fontsize=22, font=:bold, halign=:left)

    ncols, nrows = 3, 3
    for k in 1:nplots
        A = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs,
                                    mag_cv=mag_cv, rho_sym=rho_sym, rng=rng)
        prey, preds = trophic_degrees_from_A(A) 
        r, c = divrem(k-1, ncols)
        ax = Axis(fig1[r+1, c+1];
                  xlabel="Degree k", ylabel="count",
                  title=@sprintf("ER sample %d", k))
        maxk = max(maximum(prey; init=0), maximum(preds; init=0))
        bins = 0:(maxk+1)
        hist!(ax, preds; bins=bins, normalization=:none,
              color=(RGBAf(70/255,130/255,180/255,0.6)),
              label="Out-degree (predators)")
        hist!(ax, prey; bins=bins, normalization=:none,
              color=(RGBAf(1.0,0.55,0.0,0.6)),
              label="In-degree (prey)")
        axislegend(ax; position=:rt, framevisible=false)
    end
    display(fig1)

    # ---- total degree grid ----
    fig2 = Figure(size=(1000,900))
    Label(fig2[0,1:3], "Random ER trophic — Total Degree Distributions";
          fontsize=22, font=:bold, halign=:left)
    for k in 1:nplots
        A = build_random_trophic_ER(
            S; conn=conn, mean_abs=mean_abs,
            mag_cv=mag_cv, rho_sym=rho_sym, rng=rng
        )
        
        A =  build_random_nontrophic(
            S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
            rho_sym=rho_sym, rng=rng
        )                                                               
        totdeg = sum(abs.(A) .> 0; dims=2)[:] + sum(abs.(A) .> 0; dims=1)'[:]
        r, c = divrem(k-1, ncols)
        ax = Axis(fig2[r+1,c+1];
                  xlabel="Total degree", ylabel="count",
                  title=@sprintf("ER sample %d", k))
        maxk = maximum(totdeg; init=0)
        bins = 0:(maxk+1)
        hist!(ax, totdeg; bins=bins, normalization=:none,
              color=(RGBAf(0.3,0.7,0.9,0.6)))
    end
    display(fig2)
end

function plot_niche_trophic_degrees(; S=120, conn=0.10, mean_abs=0.10, mag_cv=0.60,
                                   rho_sym=0.5, seed=42)
    rng = MersenneTwister(seed)

    families = [
        (:uniform, 0.0),
        (:lognormal, 0.2),
        (:lognormal, 0.8),
        (:lognormal, 1.5),
        (:pareto, 1.2),
        (:pareto, 2.0),
        (:pareto, 3.0),
        (:uniform, 0.0),
        (:lognormal, 0.5)
    ]

    fig1 = Figure(size=(1000,900))
    Label(fig1[0,1:3], "Niche trophic — In/Out Degree Distributions";
          fontsize=22, font=:bold, halign=:left)

    ncols, nrows = 3, 3
    for (k,(fam,degp)) in enumerate(families)
        A = build_niche_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                degree_family=fam, deg_param=degp,
                                rho_sym=rho_sym, rng=rng)
        prey, preds = trophic_degrees_from_A(A)
        r, c = divrem(k-1,ncols)
        ax = Axis(fig1[r+1,c+1];
                  xlabel="Degree k", ylabel="count",
                  title=@sprintf("%s, deg_cv=%.2f", String(fam), degp))
        maxk = max(maximum(prey; init=0), maximum(preds; init=0))
        bins = 0:(maxk+1)
        hist!(ax, preds; bins=bins, normalization=:none,
              color=(RGBAf(70/255,130/255,180/255,0.6)),
              label="Out-degree (predators)")
        hist!(ax, prey; bins=bins, normalization=:none,
              color=(RGBAf(1.0,0.55,0.0,0.6)),
              label="In-degree (prey)")
        axislegend(ax; position=:rt, framevisible=false)
    end
    display(fig1)

    # ---- total degree grid ----
    fig2 = Figure(size=(1000,900))
    Label(fig2[0,1:3], "Niche trophic — Total Degree Distributions";
          fontsize=22, font=:bold, halign=:left)
    for (k,(fam,degp)) in enumerate(families)
        A = build_niche_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                degree_family=fam, deg_param=degp,
                                rho_sym=rho_sym, rng=rng)
        totdeg = sum(abs.(A) .> 0; dims=2)[:] + sum(abs.(A) .> 0; dims=1)'[:]
        r, c = divrem(k-1,ncols)
        ax = Axis(fig2[r+1,c+1];
                  xlabel="Total degree", ylabel="count",
                  title=@sprintf("%s, deg_cv=%.2f", String(fam), degp))
        maxk = maximum(totdeg; init=0)
        bins = 0:(maxk+1)
        hist!(ax, totdeg; bins=bins, normalization=:none,
              color=(RGBAf(0.3,0.7,0.9,0.6)))
    end
    display(fig2)
end

plot_random_ER_degrees()
plot_niche_trophic_degrees()
