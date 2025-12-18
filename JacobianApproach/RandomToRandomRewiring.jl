# --------------------------------------------------------
# Core computation
# --------------------------------------------------------
"""
run_random_to_random_rewiring(; deg_cv_vals, reps, t_vals)

For each "deg_cv" level, build an ER random trophic network A₀, then
another independent random ER A₁, and compare median return rates over time.
Computes R²(t) = R²(rmed(A₀), rmed(A₁)).

This tests whether "random → random" rewiring intrinsically causes
predictability loss independent of structure.
"""
function run_random_to_random_rewiring(; 
        deg_cv_vals = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
        S = 120, conn = 0.10, mean_abs = 0.5, mag_cv = 0.60,
        rho_sym = 0.99, u_mean = 1.0, u_cv = 0.6,
        IS_target = 0.5, reps = 50,
        t_vals = 10 .^ range(-2, 2; length=20),
        seed = 20251107
)

    base = _splitmix64(UInt64(seed))
    bucket = [Vector{NamedTuple}() for _ in 1:nthreads()]

    Threads.@threads for idx in eachindex(deg_cv_vals)
        dcv = deg_cv_vals[idx]
        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx) ⊻ UInt64(threadid())))
        local_rows = bucket[threadid()]

        for rep in 1:reps
            rng = Random.Xoshiro(rand(rng0, UInt64))

            # --- Build baseline random ER network
            A0 = build_random_trophic_ER(
                S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                rho_sym=rho_sym, rng=rng
            )

            # A0 = build_random_ER(
            #     S; conn=conn, mean_abs=mean_abs,
            #     mag_cv=mag_cv, rho_sym=rho_sym,
            #     rng=rng
            # )

            baseIS = realized_IS(A0)
            baseIS == 0 && continue
            β = IS_target / baseIS
            A0 .*= β

            # --- Build new random ER (rewired version)
            A1 = build_random_trophic_ER(
                S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                rho_sym=rho_sym, rng=rng
            )

            # A1 = build_random_ER(
            #     S; conn=conn, mean_abs=mean_abs,
            #     mag_cv=mag_cv, rho_sym=rho_sym,
            #     rng=rng
            # )

            β1 = IS_target / realized_IS(A1)
            A1 .*= β1

            # --- Same abundances for both
            u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

            # --- Compute median return rates over time
            r_full = [median_return_rate(jacobian(A0, u), u; t=t, perturbation=:biomass) for t in t_vals]
            r_rew  = [median_return_rate(jacobian(A1, u), u; t=t, perturbation=:biomass) for t in t_vals]

            # store each timepoint
            for (i, t) in enumerate(t_vals)
                push!(local_rows, (; deg_cv=dcv, rep, t, r_full=r_full[i], r_rew=r_rew[i]))
            end
        end
    end

    df = DataFrame(vcat(bucket...))

    # --- Summarize R²(t)
    rowsS = NamedTuple[]
    for xv in sort(unique(df.deg_cv)), t in sort(unique(df.t))
        sub = df[(df.deg_cv .== xv) .& (df.t .== t), :]
        isempty(sub) && continue
        r2 = r2_to_identity(sub.r_full, sub.r_rew)
        if isnan(r2) || r2 < 0
            r2 = 0.0
        end
        push!(rowsS, (; deg_cv=xv, t, r2))
    end

    return df, DataFrame(rowsS)
end

# --------------------------------------------------------
# Plotting
# --------------------------------------------------------
function plot_random_to_random(summary::DataFrame)
    deg_vals = sort(unique(summary.deg_cv))
    @assert length(deg_vals) == 6 "Expected 6 deg_cv levels."
    fig = Figure(size=(1100, 700))
    Label(fig[0,1:3], "Predictability vs degree heterogeneity (Random → Random rewiring)";
          fontsize=20, font=:bold, halign=:left)

    for (pi, xv) in enumerate(deg_vals)
        ax = Axis(
            fig[(pi-1) ÷ 3 + 1, (pi-1) % 3 + 1];
            xscale=log10,
            xlabel="time t", ylabel="R²(full, rew)",
            title="deg_cv = $(round(xv,digits=2))",
            limits=((nothing,nothing),(-0.05,1.05))
        )

        sub = summary[summary.deg_cv .== xv, :]
        isempty(sub) && continue
        sort!(sub, :t)
        lines!(ax, sub.t, sub.r2; color=:steelblue, linewidth=2)
        scatter!(ax, sub.t, sub.r2; color=:steelblue)
    end
    display(fig)
end

# --------------------------------------------------------
# Run it
# --------------------------------------------------------
t_vals = 10 .^ range(-2, 2; length=40)
deg_vals = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

df_raw, df_sum = run_random_to_random_rewiring(
    ; deg_cv_vals=deg_vals, t_vals=t_vals, reps=60,
    S = 120, conn = 0.10, mean_abs = 0.5, mag_cv = 0.60,
    rho_sym = 0.0, u_mean = 1.0, u_cv = 0.6,
    IS_target = 0.5,
)
plot_random_to_random(df_sum) 