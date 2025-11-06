####################################################################################
# (2) THR & RARER SWEEP: PREDICTABILITY vs % REMOVED with IS LINES (UPGRADED)
####################################################################################
Base.@kwdef struct RemoveSweepOptions
    modes::Vector{Symbol} = [:TR]
    S_vals::Vector{Int} = [120]
    conn_vals::AbstractVector{Float64} = 0.05:0.05:0.30
    mean_abs_vals::Vector{Float64} = [1.0]
    mag_cv_vals::Vector{Float64} = [0.1, 0.5, 1.0]
    u_mean_vals::Vector{Float64} = [1.0]
    u_cv_vals::Vector{Float64} = [0.3, 0.5, 0.8, 1.0, 2.0]
    degree_families::Vector{Symbol} = [:uniform, :lognormal, :pareto]
    deg_cv_vals::Vector{Float64} = [0.0, 0.5, 1.0, 2.0]
    deg_pl_alphas::Vector{Float64} = [1.2, 1.5, 2.0, 3.0]
    rho_sym_vals::Vector{Float64} = [0.0, 0.5, 1.0]
    IS_lines::Vector{Float64} = [0.05, 0.1, 0.4, 0.8, 1.2]
    reps_per_combo::Int = 2
    number_of_combinations::Int = 200
    q_grid::Vector{Float64} = collect(0.0:0.05:0.9)   # fraction of weakest links removed
    p_grid::Vector{Float64} = collect(0.0:0.05:0.9)   # fraction of rarest species removed
    t_short::Float64 = 0.01
    t_long::Float64  = 50.0
    seed::Int = 20251028
end

_metrics_rmed(J,u; t_short=0.01, t_long=0.50) = (
    rmed_s = median_return_rate(J,u; t=t_short, perturbation=:biomass),
    rmed_l = median_return_rate(J,u; t=t_long,  perturbation=:biomass)
)

function run_remove_sweep(opts::RemoveSweepOptions)
    rngG = Random.Xoshiro(opts.seed)

    # --- Build parameter combinations
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
        opts.u_mean_vals, opts.u_cv_vals, deg_specs, opts.rho_sym_vals,
        1:opts.reps_per_combo, opts.IS_lines
    ))

    sel = (length(combos) > opts.number_of_combinations) ?
          sample(combos, opts.number_of_combinations; replace=false) : combos

    nthreads_used = nthreads()
    println("Computing $(length(sel)) of $(length(combos)) combinations using $(nthreads_used) threads")

    # --- one output bucket per thread
    buckets = [Vector{NamedTuple}() for _ in 1:nthreads()]

    Threads.@threads for idx in eachindex(sel)
        combo = sel[idx]
        (mode, S, conn, mean_abs, mag_cv, u_mean, u_cv, (deg_fam, deg_param), rho_sym, rep, IS) = combo

        # independent base RNG per thread & combo
        base = rand(rngG, UInt64)
        rng_local = Random.Xoshiro(base ⊻ UInt64(threadid()) ⊻ UInt64(idx))

        local_rows = NamedTuple[]

        rng = Random.Xoshiro(rand(rng_local, UInt64))

        A0 = build_niche_trophic(S;
            conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
            degree_family=deg_fam, deg_param=deg_param,
            rho_sym=rho_sym, rng=rng)

        baseIS = realized_IS(A0)
        baseIS <= 0 && continue

        β = IS / baseIS
        A = β .* A0
        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
        Jfull = jacobian(A, u)
        metF = _metrics_rmed(Jfull, u; t_short=opts.t_short, t_long=opts.t_long)
        α = alpha_off_from(Jfull, u)

        # --- THR sweep (weakest q%)
        for q in opts.q_grid
            α_thr = op_threshold_alpha(α; q=q)
            J_thr = build_J_from(α_thr, u)
            metT  = _metrics_rmed(J_thr, u; t_short=opts.t_short, t_long=opts.t_long)
            push!(local_rows, (; kind=:THR, mode, S, conn, mean_abs, mag_cv, u_mean, u_cv,
                                degree_family=deg_fam, degree_param=deg_param, rho_sym,
                                IS, rep, q_or_p=q,
                                rmed_full_s=metF.rmed_s, rmed_step_s=metT.rmed_s,
                                rmed_full_l=metF.rmed_l, rmed_step_l=metT.rmed_l))
        end

        # --- RARER sweep (rarest p%)
        for p in opts.p_grid
            u_drop = remove_rarest_species(u; p=p)
            J_rr   = build_J_from(α, u_drop)
            metR   = _metrics_rmed(J_rr, filter(!iszero, u_drop);
                                    t_short=opts.t_short, t_long=opts.t_long)
            push!(local_rows, (; kind=:RARER, mode, S, conn, mean_abs, mag_cv, u_mean, u_cv,
                                degree_family=deg_fam, degree_param=deg_param, rho_sym,
                                IS, rep, q_or_p=p,
                                rmed_full_s=metF.rmed_s, rmed_step_s=metR.rmed_s,
                                rmed_full_l=metF.rmed_l, rmed_step_l=metR.rmed_l))
        end

        # store into that thread's bucket
        append!(buckets[threadid()], local_rows)
    end

    return DataFrame(vcat(buckets...))
end

# --- Compute mean R² across replicates, community types, and IS
function summarize_r2(df::DataFrame)
    rows = NamedTuple[]
    for sub in groupby(df, [:kind, :IS, :q_or_p])
        n = nrow(sub)
        n < 3 && continue

        # short horizon
        r2s = r2_to_identity(sub.rmed_full_s, sub.rmed_step_s)
        # long horizon
        r2l = r2_to_identity(sub.rmed_full_l, sub.rmed_step_l)

        push!(rows, (; kind=sub.kind[1], IS=sub.IS[1], q_or_p=sub.q_or_p[1],
                      r2_short=max(r2s,0.0), r2_long=max(r2l,0.0), n))
    end
    DataFrame(rows)
end

function plot_remove_sweep(df_sum;
        which::Symbol = :THR,
        horizon::Symbol = :short,
        title::String = "Predictability vs removal fraction",
        cmap = :viridis)

    @assert which in (:THR, :RARER)
    @assert horizon in (:short, :long)

    fig = Figure(size=(950, 500))
    ax = Axis(fig[1, 1];
        xlabel = (which == :THR ? "% links removed (q)" : "% species removed (p)"),
        ylabel = (horizon == :short ? "R² r̃med(t_short)" : "R² r̃med(t_long)"),
        title = title,
        limits = ((0, maximum(df_sum.q_or_p)), (-0.05, 1.05)),
        ygridvisible = true, xgridvisible = false)

    # progressive colormap for IS
    IS_list = sort(unique(df_sum.IS))
    col = cgrad(cmap, length(IS_list), categorical = true)
    leg_labels = [@sprintf("IS=%.2f", v) for v in IS_list]

    for (k, IS) in enumerate(IS_list)
        sub = filter(row -> row.kind == which && row.IS == IS, df_sum)
        isempty(sub) && continue
        sort!(sub, :q_or_p)
        y = horizon == :short ? sub.r2_short : sub.r2_long

        lines!(ax, sub.q_or_p, y;
            color = col[k],
            label = leg_labels[k],
            linewidth = 2)
    end

    axislegend(ax; position = :rb, framevisible = false, nbanks = 2)
    display(fig)
end

####################################################################################
# Example usage
####################################################################################
optsR = RemoveSweepOptions(
    number_of_combinations=1000,
    reps_per_combo=3,
    IS_lines=[0.05, 0.1, 0.4, 0.8, 1.2],
    q_grid=0.0:0.05:0.9,
    p_grid=0.0:0.05:0.9,
    seed=20251028
)

df_raw = run_remove_sweep(optsR)
df_sum = summarize_r2(df_raw)

plot_remove_sweep(df_sum; which=:THR,   horizon=:short, title="THR — R²(short)")
plot_remove_sweep(df_sum; which=:THR,   horizon=:long,  title="THR — R²(long)")
plot_remove_sweep(df_sum; which=:RARER, horizon=:short, title="RARER — R²(short)")
plot_remove_sweep(df_sum; which=:RARER, horizon=:long,  title="RARER — R²(long)")
