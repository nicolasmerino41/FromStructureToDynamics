using Random, Statistics, LinearAlgebra, DataFrames, CairoMakie
using Base.Threads

# ---------- tiny helpers ----------
@inline function _splitmix64(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
    z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
    z ⊻ (z >>> 31)
end

@inline function _r2_to_identity(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == 0 && return NaN
    μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
    sst == 0 && return (ssr == 0 ? 1.0 : 0.0)
    max(1 - ssr/sst, 0.0)
end

# --------------------------------------------------------
# Core computation: two-step comparison
# --------------------------------------------------------
"""
run_struct_vs_random_and_random_vs_random(; deg_cv_vals, reps, t_vals, ...)

For each deg_cv in deg_cv_vals:

- Build a *structured* (niche) network A_s with degree_family=:lognormal, deg_param=deg_cv.
- Build two *independent ER* networks A_r0 and A_r1 (same targets).
- Use the *same u* per replicate for all three.
- For each t, compute:
    struct→random: R²( rmed(A_s), rmed(A_r0) )
    random→random: R²( rmed(A_r0), rmed(A_r1) )

Returns:
  df_raw  — row per (deg_cv, rep, t) with rmeds (struct, r0, r1)
  df_sum  — row per (deg_cv, t) with R²_sr (struct→random) and R²_rr (random→random)
"""
function run_struct_vs_random_and_random_vs_random(;
        deg_cv_vals = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
        S = 120, conn = 0.10, mean_abs = 0.10, mag_cv = 0.60,
        rho_sym = 0.5, u_mean = 1.0, u_cv = 0.6,
        IS_target = 0.2, reps = 60,
        t_vals = 10 .^ range(-2, 2; length=20),
        seed = 20251107,
    )

    base = _splitmix64(UInt64(seed))
    buckets = [Vector{NamedTuple}() for _ in 1:nthreads()]

    Threads.@threads for idx in eachindex(deg_cv_vals)
        dcv = deg_cv_vals[idx]
        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx) ⊻ UInt64(threadid())))
        local_rows = buckets[threadid()]

        for rep in 1:reps
            rng = Random.Xoshiro(rand(rng0, UInt64))

            # --- Structured (niche) network
            As = build_niche_trophic(S;
                                     conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                     degree_family=:lognormal, deg_param=dcv,
                                     rho_sym=rho_sym, rng=rng)
            ISs = realized_IS(As); ISs == 0 && continue
            As .*= (IS_target / ISs)

            # --- ER random (r0) and independent ER random (r1)
            # Ar0 = build_random_trophic_ER(
            #     S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
            #     rho_sym=rho_sym, rng=rng
            # )

            Ar0 = build_random_nontrophic(
                S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                rho_sym=rho_sym, rng=rng
            )

            IS0 = realized_IS(Ar0); IS0 == 0 && continue
            Ar0 .*= (IS_target / IS0)

            # Ar1 = build_random_trophic_ER(
            #     S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
            #     rho_sym=rho_sym, rng=rng
            # )
            Ar1 = build_random_nontrophic(
                S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                rho_sym=rho_sym, rng=rng
            )                                        
            IS1 = realized_IS(Ar1); IS1 == 0 && continue
            Ar1 .*= (IS_target / IS1)

            # --- same u for all
            u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

            # precompute Jacobians (cheaper than recomputing per t)
            Js = jacobian(As, u)
            Jr0 = jacobian(Ar0, u)
            Jr1 = jacobian(Ar1, u)

            # evaluate rmed at each t
            r_s  = [median_return_rate(Js,  u; t=t, perturbation=:biomass) for t in t_vals]
            r_r0 = [median_return_rate(Jr0, u; t=t, perturbation=:biomass) for t in t_vals]
            r_r1 = [median_return_rate(Jr1, u; t=t, perturbation=:biomass) for t in t_vals]

            for (i,t) in enumerate(t_vals)
                push!(local_rows, (;
                    deg_cv = dcv, rep, t,
                    r_struct = r_s[i],
                    r_rand0 = r_r0[i],
                    r_rand1 = r_r1[i],
                ))
            end
        end
    end

    df_raw = DataFrame(vcat(buckets...))

    # --- summarise to R² curves
    rowsS = NamedTuple[]
    for dcv in sort(unique(df_raw.deg_cv)), t in sort(unique(df_raw.t))
        sub = df_raw[(df_raw.deg_cv .== dcv) .& (df_raw.t .== t), :]
        isempty(sub) && continue

        # struct → random
        r2_sr = _r2_to_identity(sub.r_struct, sub.r_rand0)

        # random → random
        r2_rr = _r2_to_identity(sub.r_rand0, sub.r_rand1)

        push!(rowsS, (; deg_cv=dcv, t, r2_sr, r2_rr))
    end

    return df_raw, DataFrame(rowsS)
end

# --------------------------------------------------------
# Plotting: two lines per panel
# --------------------------------------------------------
function plot_struct_and_random(summary::DataFrame)
    dvals = sort(unique(summary.deg_cv))
    @assert length(dvals) == 6 "Expected 6 deg_cv levels."

    fig = Figure(size=(1100, 700))
    Label(fig[0,1:3], "Predictability (R²) — struct→random vs random→random";
          fontsize=20, font=:bold, halign=:left)

    for (pi, dcv) in enumerate(dvals)
        ax = Axis(fig[(pi-1) ÷ 3 + 1, (pi-1) % 3 + 1];
                  xscale=log10, xlabel="time t", ylabel="R²",
                  title="deg_cv = $(round(dcv,digits=2))",
                  limits=((nothing,nothing),(-0.05,1.05)),
                  ygridvisible=true)

        sub = summary[summary.deg_cv .== dcv, :]
        isempty(sub) && continue
        sort!(sub, :t)

        lines!(ax, sub.t, sub.r2_sr; color=:dodgerblue,  linewidth=2, label="struct → random")
        lines!(ax, sub.t, sub.r2_rr; color=:seagreen4,  linewidth=2, label="random → random")
        scatter!(ax, sub.t, sub.r2_sr; color=:dodgerblue)
        scatter!(ax, sub.t, sub.r2_rr; color=:seagreen4)

        if pi == 1
            axislegend(ax; position=:rb, framevisible=false)
        end
    end
    display(fig)
end

# --------------------------------------------------------
# Example run
# --------------------------------------------------------
t_vals = 10 .^ range(-2, 2; length=20)
deg_vals = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

df_raw, df_sum = run_struct_vs_random_and_random_vs_random(;
    deg_cv_vals=deg_vals, t_vals=t_vals, reps=60,
    S=120, conn=0.10, mean_abs=0.5, mag_cv=0.60,
    rho_sym=1.0, u_mean=1.0, u_cv=0.8, IS_target=0.5
)

plot_struct_and_random(df_sum)
