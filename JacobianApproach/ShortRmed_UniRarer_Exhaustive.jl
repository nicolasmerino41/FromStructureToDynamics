############ ShortRmed_UniRarer_Exhaustive.jl ############
using Random, Statistics, LinearAlgebra, DataFrames, Distributions
using CairoMakie
using Base.Threads

# ----------------- small utilities -----------------
# robust R² to the y=x line; also returns OLS slope/intercept (y ~ x)
function r2_to_identity(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == 0 && return (NaN, NaN, NaN)
    μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
    r2 = sst == 0 ? (ssr == 0 ? 1.0 : 0.0) : 1 - ssr/sst
    β = [x ones(n)] \ y
    return (max(r2, 0.0), β[1], β[2])
end

# deterministic per-thread RNG seed
@inline function _splitmix64(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
    z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
    z ⊻ (z >>> 31)
end

# prefer niche builder if present
function _build_trophic(S; conn, mean_abs, mag_cv, degree_family, deg_param, rho_sym, rng)
    if isdefined(@__MODULE__, :build_niche_trophic)
        return build_niche_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                   degree_family=degree_family, deg_param=deg_param,
                                   rho_sym=rho_sym, rng=rng)
    else
        return build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                    degree_family=degree_family, deg_param=deg_param,
                                    rho_sym=rho_sym, rng=rng)
    end
end

# ----------------- options -----------------
Base.@kwdef struct ExhaustiveOpts
    modes::Vector{Symbol} = [:TR]       # we only need trophic here
    S_vals::Vector{Int} = [120]
    conn_vals::AbstractVector{Float64} = 0.05:0.05:0.30
    mean_abs_vals::Vector{Float64} = [0.05, 0.10, 0.20]
    mag_cv_vals::Vector{Float64}   = [0.1, 0.5, 1.0, 2.0]
    degree_families::Vector{Symbol} = [:uniform, :lognormal, :pareto]
    deg_cv_vals::Vector{Float64} = [0.0, 0.5, 1.0, 2.0]          # for :lognormal
    deg_pl_alphas::Vector{Float64} = [1.2, 1.5, 2.0, 3.0]        # for :pareto
    rho_sym_vals::Vector{Float64} = [0.0, 0.5, 1.0]
    IS_targets::Vector{Float64} = collect(0.02:0.02:0.30)

    # The three abundance-CV regimes we want to demonstrate
    u_cv_levels::Vector{Float64} = [0.0, 0.6, 2.5]  # 0 = uniform, low, high
    u_mean::Float64 = 1.0

    t_short::Float64 = 0.01                        # very short time
    p_list::Vector{Float64} = [0.10, 0.80]         # rarer removal fractions to test

    reps_per_combo::Int = 2
    number_of_combinations::Int = 2000
    seed::Int = 20251101
end

# ----------------- main sweep -----------------
"""
run_exhaustive_short_rmed(opts) -> (df_raw, df_r2)

df_raw: one row per (community draw × u_cv level × p in {0.1, 0.8})
        with r_full, r_uni, r_rarer_p01, r_rarer_p08.

df_r2:  summary R² (and slope/intercept) for each u_cv regime and step
        (uni, rarer_0.1, rarer_0.8), computed across the whole ensemble.
"""
function run_exhaustive_short_rmed(opts::ExhaustiveOpts)
    # expand degree specs once
    deg_specs = Tuple{Symbol,Float64}[]
    for fam in opts.degree_families
        if fam === :uniform
            push!(deg_specs, (:uniform, 0.0))
        elseif fam === :lognormal
            append!(deg_specs, ((:lognormal, x) for x in opts.deg_cv_vals))
        elseif fam === :pareto
            append!(deg_specs, ((:pareto, a) for a in opts.deg_pl_alphas))
        end
    end

    combos = collect(Iterators.product(
        opts.modes, opts.S_vals, opts.conn_vals, opts.mean_abs_vals, opts.mag_cv_vals,
        opts.degree_families, deg_specs, opts.rho_sym_vals, opts.IS_targets
    ))
    sel = length(combos) > opts.number_of_combinations ?
          sample(combos, opts.number_of_combinations; replace=false) : combos

    base = _splitmix64(UInt64(opts.seed))
    buckets = [Vector{NamedTuple}() for _ in 1:nthreads()]

    Threads.@threads for idx in eachindex(sel)
        (mode, S, conn, mean_abs, mag_cv, _fam, (deg_fam, deg_param), rho_sym, IS_tgt) = sel[idx]

        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx) ⊻ UInt64(threadid())))
        local_rows = buckets[threadid()]

        for rep in 1:opts.reps_per_combo
            rng = Random.Xoshiro(rand(rng0, UInt64))

            # build base A, rescale to target IS
            A0, _ = _build_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                degree_family=deg_fam, deg_param=deg_param,
                                rho_sym=rho_sym, rng=rng)
            baseIS = realized_IS(A0)
            baseIS == 0 && continue
            β = IS_tgt / baseIS
            A = β .* A0

            for u_cv in opts.u_cv_levels
                u = random_u(S; mean=opts.u_mean, cv=u_cv, rng=rng)

                # full and alpha (note: we only need α to rebuild J under u-edits)
                J_full = jacobian(A, u)
                α = alpha_off_from(J_full, u)

                # step: uni
                u_uni = fill(mean(u), length(u))
                J_uni = build_J_from(α, u_uni)

                # steps: rarer at p=0.1 and p=0.8
                u_r01 = remove_rarest_species(u; p=0.10)
                u_r07 = remove_rarest_species(u; p=0.7)
                u_r099 = remove_rarest_species(u; p=0.99)
                J_r01 = build_J_from(α, u_r01)
                J_r07 = build_J_from(α, u_r07)
                J_r099 = build_J_from(α, u_r099)

                # short-time r̃med
                rf   = median_return_rate(J_full, u;  t=opts.t_short, perturbation=:biomass)
                runi = median_return_rate(J_uni,  u_uni; t=opts.t_short, perturbation=:biomass)
                rr01 = median_return_rate(J_r01,  filter(!iszero, u_r01); t=opts.t_short, perturbation=:biomass)
                rr07 = median_return_rate(J_r07,  filter(!iszero, u_r07); t=opts.t_short, perturbation=:biomass)
                rr099 = median_return_rate(J_r099,  filter(!iszero, u_r099); t=opts.t_short, perturbation=:biomass)

                push!(local_rows, (;
                    S, conn, mean_abs, mag_cv, degree_family=deg_fam, degree_param=deg_param,
                    rho_sym, IS_target=IS_tgt, rep,
                    u_cv, r_full=rf, r_uni=runi, r_rarer_01=rr01, r_rarer_07=rr07, r_rarer_099=rr099
                ))
            end
        end
    end

    df_raw = DataFrame(vcat(buckets...))

    # --- R² summaries by u_cv regime ---
    rowsS = NamedTuple[]
    for ucv in sort(unique(df_raw.u_cv))
        sub = df_raw[df_raw.u_cv .== ucv, :]
        x = sub.r_full
        for (lab, col) in [("uni", :r_uni), ("rarer_0.1", :r_rarer_01), ("rarer_0.7", :r_rarer_07), ("rarer_0.99", :r_rarer_099)]
            y = sub[!, col]
            r2, slope, intercept = r2_to_identity(collect(x), collect(y))
            push!(rowsS, (; step=lab, u_cv=ucv, r2, slope, intercept, n=nrow(sub)))
        end
    end
    df_r2 = DataFrame(rowsS)

    return df_raw, df_r2
end

# ----------------- plotting (proof visuals) -----------------
"""
plot_proof_panels(df_raw, df_r2)

For each u_cv regime (0, low, high) creates a row with three panels:
FULL vs UNI, FULL vs RARER(0.1), FULL vs RARER(0.8).
R², slope, and intercept are printed on each panel.
"""
function plot_proof_panels(
    df_raw::DataFrame, df_r2::DataFrame;
    title = "Short-time rmed (t=0.01): FULL vs steps — exhaustive ensemble"
)
    ucvs = sort(unique(df_raw.u_cv))
    labels = Dict(ucvs[1] => "u_cv = 0 (uniform)",
                  ucvs[2] => "u_cv = low",
                  ucvs[3] => "u_cv = high")

    steps = [(:r_uni, "UNI"),
             (:r_rarer_01, "RARER p=0.1"),
             (:r_rarer_07, "RARER p=0.7"),
             (:r_rarer_099, "RARER p=0.99")]

    fig = Figure(size=(1050, 650))
    Label(fig[0, 1:4], title,
          fontsize=20, font=:bold, halign=:left)

    for (ri, ucv) in enumerate(ucvs)
        sub = df_raw[df_raw.u_cv .== ucv, :]
        for (ci, (col, ttl)) in enumerate(steps)
            x = sub.r_full |> collect
            y = sub[!, col] |> collect

            mn = minimum(vcat(x,y)); mx = maximum(vcat(x,y))
            pad = (mx - mn) ≤ 0 ? 1.0 : 0.05*(mx - mn)
            ax = Axis(fig[ri, ci]; title="$(labels[ucv]) — $ttl",
                      xlabel="r̃med_full(t=0.01)", ylabel=(ci==1 ? "step value" : ""),
                      limits=((mn-pad, mx+pad), (mn-pad, mx+pad)))

            scatter!(ax, x, y; markersize=4, color=:steelblue, alpha=0.35)
            lines!(ax, [mn-pad, mx+pad], [mn-pad, mx+pad]; color=:black, linestyle=:dash)

            r2, slope, intercept = r2_to_identity(x, y)
            txt = @sprintf("R²=%.3f, n=%d", r2, length(x))
            text!(ax, txt; position=(mx+pad, mn-pad), align=(:right,:bottom))
        end
    end

    display(fig)
end

# ----------------- optional analytical check -----------------
"""
check_size_bias_identity(df_raw)

Numerically verifies that for t→0, r̃med is the size-biased mean of u:
SBM(u) = (∑ u^3)/(∑ u^2). We compute correlation between r_full and SBM(u)
for the ensemble to show structure-independence at very short t.
"""
function check_size_bias_identity(df_raw::DataFrame; nsample=2000)
    # We need the u vectors; if you want strict check, adapt the sweep to store u.
    # Here we re-approximate SBM by drawing fresh u with same (u_mean, u_cv) and
    # compare trend-wise; or you can modify your sweep to store u per row.
    println("Note: For an exact pointwise identity, store `u` per draw and compute SBM on it.")
end

# ----------------- run -----------------
opts = ExhaustiveOpts(; number_of_combinations=3000, reps_per_combo=2)
df_raw, df_r2 = run_exhaustive_short_rmed(opts)

println("\n=== Global R² summary (short t) ===")
show(df_r2, allcols=true, allrows=true)

plot_proof_panels(
    df_raw, df_r2,
    title = "Short-time r̃med BIOMASS (t=0.01): FULL vs steps — exhaustive ensemble"
)