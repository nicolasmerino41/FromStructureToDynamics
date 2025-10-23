function sweep_rmeds(;
    # --- simulation parameters ---
    modes = [:TR],
    S_vals = [150],
    conn_vals = 0.05:0.05:0.30,
    mean_abs_vals = [0.05, 0.10, 0.20],
    mag_cv_vals   = [0.4, 0.6, 1.0],
    u_mean_vals   = [1.0],
    u_cv_vals     = [0.5],
    degree_families = [:uniform, :lognormal, :pareto],
    deg_cv_vals   = [0.0, 0.5, 1.0, 2.0],
    deg_pl_alphas = [1.2, 1.5, 2.0, 3.0],
    rho_sym_vals  = [0.0, 0.25, 0.5, 0.75, 1.0],
    margin = 0.05, shrink_factor = 0.9, max_shrink_iter = 200,
    reps_per_combo = 2,
    seed = 1234, number_of_combinations = 10_000,
    q_thresh = 0.20,

    # --- new time range parameters ---
    t_min = 0.01,
    t_max = 2.5,
    t_bins = 10
)

    # construct t-values dynamically
    t_vals = collect(range(t_min, t_max; length=t_bins))
    println("Computing median return rates for t-values: ", round.(t_vals, digits=4))

    # helper mapping from index to word
    idx_to_word = Dict(
        1 => "one", 2 => "two", 3 => "three", 4 => "four", 5 => "five",
        6 => "six", 7 => "seven", 8 => "eight", 9 => "nine", 10 => "ten",
        11 => "eleven", 12 => "twelve", 13 => "thirteen", 14 => "fourteen", 15 => "fifteen"
    )

    genA(mode, rho_sym, rng, conn, mean_abs, mag_cv, deg_fam, deg_param, S) = mode === :NT ? 
        build_random_nontrophic(
            S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
            degree_family=deg_fam, deg_param=deg_param, rng=rng
        ) :
        build_random_trophic(
            S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
            degree_family=deg_fam, deg_param=deg_param,
            rho_sym=rho_sym, rng=rng
        )

    # expand degree specs
    deg_specs = Tuple{Symbol,Float64}[]
    for fam in degree_families
        if fam === :uniform
            push!(deg_specs, (:uniform, 0.0))
        elseif fam === :lognormal
            append!(deg_specs, ((:lognormal, x) for x in deg_cv_vals))
        elseif fam === :pareto
            append!(deg_specs, ((:pareto, a) for a in deg_pl_alphas))
        end
    end

    # full parameter combinations
    combos = collect(Iterators.product(
        modes, S_vals, conn_vals, mean_abs_vals, mag_cv_vals,
        u_mean_vals, u_cv_vals, deg_specs, 1:reps_per_combo, rho_sym_vals
    ))
    println("Computing $(number_of_combinations) of $(length(combos)) combinations")

    sel = (length(combos) > number_of_combinations) ?
          sample(combos, number_of_combinations; replace=false) : combos

    base = _splitmix64(UInt64(seed))
    buckets = [Vector{NamedTuple}() for _ in 1:nthreads()]

    for idx in eachindex(sel)
        (mode, S, conn, mean_abs, mag_cv, u_mean, u_cv, (deg_fam, deg_param), _, rho_sym) = sel[idx]
        rng = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx)))

        # generate matrices and dynamics
        A = genA(mode, rho_sym, rng, conn, mean_abs, mag_cv, deg_fam, deg_param, S)
        u = random_u(S; mean=u_mean, cv=u_cv, rng)
        A, αshrink, λmax = stabilize_shrink!(A, u; margin, factor=shrink_factor)
        J = jacobian(A, u)

        # post-stabilization data
        conn_real = realized_connectance(A)
        IS_real   = realized_IS(A)
        degs      = degree_CVs(A)
        ucv_real  = (mean(u)>0 ? std(u)/mean(u) : NaN)

        # α-transformations
        α = alpha_off_from(J, u)
        α_reshuf = op_reshuffle_alpha(α; rng)
        α_row    = op_rowmean_alpha(α)
        α_thr    = op_threshold_alpha(α; q=q_thresh)
        u_uni     = uniform_u(u)
        u_rarerem = remove_rarest_species(u; p=0.1)

        # rewiring
        A_rew = genA(mode, rho_sym, rng, conn, mean_abs, mag_cv, deg_fam, deg_param, S)
        A_rew .*= αshrink
        J_full, J_reshuf, J_row, J_thr = J, build_J_from(α_reshuf, u), build_J_from(α_row, u), build_J_from(α_thr, u)
        J_uni, J_rarer, J_rew = build_J_from(α, u_uni), build_J_from(α, u_rarerem), jacobian(A_rew, u)

        # compute Rmeds dynamically for each t, preserving true t order
        time_labels = Symbol[]
        time_data   = Dict{Symbol,Any}()

        for (ti, t) in enumerate(t_vals)
            word = get(idx_to_word, ti, "t$(ti)")
            label = Symbol("t_", word)

            # record labels in generation order
            append!(time_labels, [
                Symbol(label, "_rmed_full"),
                Symbol(label, "_rmed_reshuf"),
                Symbol(label, "_rmed_thr"),
                Symbol(label, "_rmed_row"),
                Symbol(label, "_rmed_uni"),
                Symbol(label, "_rmed_rarer"),
                Symbol(label, "_rmed_rew"),
            ])

            # store values
            time_data[Symbol(label, "_rmed_full")]   = median_return_rate(J_full,  u; t, perturbation=:biomass)
            time_data[Symbol(label, "_rmed_reshuf")] = median_return_rate(J_reshuf, u; t)
            time_data[Symbol(label, "_rmed_thr")]    = median_return_rate(J_thr,    u; t)
            time_data[Symbol(label, "_rmed_row")]    = median_return_rate(J_row,    u; t)
            time_data[Symbol(label, "_rmed_uni")]    = median_return_rate(J_uni,    u_uni; t)
            time_data[Symbol(label, "_rmed_rarer")]  = median_return_rate(J_rarer,  filter(!iszero, u_rarerem); t)
            time_data[Symbol(label, "_rmed_rew")]    = median_return_rate(J_rew,    u; t)
        end

        # --- build row with guaranteed ordered columns ---
        meta_vals = Dict{Symbol,Any}(
            :mode => mode, :S => S,
            :conn_target => conn, :mean_abs => mean_abs, :mag_cv => mag_cv,
            :u_mean_target => u_mean, :u_cv_target => u_cv,
            :degree_family => deg_fam, :degree_param => deg_param,
            :conn_real => conn_real, :IS_real => IS_real, :u_cv => ucv_real,
            :deg_cv_in => degs.deg_cv_in, :deg_cv_out => degs.deg_cv_out, :deg_cv_all => degs.deg_cv_all,
            :rho_sym => rho_sym, :shrink_alpha => αshrink, :lambda_max => λmax
        )

        meta_cols = collect(keys(meta_vals))

        # combine metadata + time metrics preserving order
        pairs_ordered = vcat(
            [(k => meta_vals[k]) for k in meta_cols]...,
            [(k => time_data[k]) for k in time_labels]...
        )

        # construct ordered NamedTuple
        ordered = (; pairs_ordered...)

        push!(buckets[threadid()], ordered)

    end

    return DataFrame(vcat(buckets...)), t_vals
end


"""
plot_correlations(df; steps=1:6, metrics=[:res, :rea])

Scatter of Full vs Step k with 1:1 line and R² to y=x.
"""
function plot_rmed_sweep(
    df::DataFrame;
    t_steps=1:10,
    steps=["reshuf","thr","row","uni","rarer","rew"],
    title=""
)
    # mapping for correct order
    ordermap = Dict(
        "one"=>1, "two"=>2, "three"=>3, "four"=>4, "five"=>5,
        "six"=>6, "seven"=>7, "eight"=>8, "nine"=>9, "ten"=>10,
        "eleven"=>11, "twelve"=>12, "thirteen"=>13, "fourteen"=>14, "fifteen"=>15
    )

    # detect t-columns
    tcols = filter(c -> occursin("_rmed_full", string(c)), names(df))

    # extract labels and sort them numerically based on word order
    tlabels = [replace(string(c), "_rmed_full" => "") for c in tcols]
    sort!(tlabels, by = s -> get(ordermap, replace(s, "t_" => ""), 999))

    # select subset of t-values to plot
    tlabels = tlabels[intersect(t_steps, 1:length(tlabels))]

    println("Detected t-labels in numeric order: ", tlabels)

    colors = fill(:red, length(steps))
    fig = Figure(size=(1100, 725))
    Label(fig[0, 2:5], title; fontsize=18, font=:bold, halign=:left)

    for (ti, tlabel) in enumerate(tlabels)
        xname = string(tlabel, "_rmed_full")
        for (si, step) in enumerate(steps)
            yname = string(tlabel, "_rmed_", step)
            if !(xname in names(df) && yname in names(df))
                continue
            end

            xs = collect(skipmissing(df[!, xname]))
            ys = collect(skipmissing(df[!, yname]))
            valid = findall(i -> isfinite(xs[i]) && isfinite(ys[i]), 1:length(xs))
            x, y = xs[valid], ys[valid]
            if isempty(x); continue; end

            ax = Axis(
                fig[ti, si];
                title="t=$(replace(tlabel, "t"=>"")) — $step",
                # limits=:auto,
                xlabel=string(xname),
                ylabel=string(yname),
                # xgridvisible=false, ygridvisible=false
            )

            scatter!(ax, x, y; color=colors[ti], markersize=4, alpha=0.35)
            xx = [minimum(x), maximum(x)]
            lines!(ax, xx, xx; color=:black, linestyle=:dash)

            μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
            r2 = sst == 0 ? NaN : 1 - ssr/sst
            if isfinite(r2)
                text!(ax, "R²=$(round(r2, digits=3))";
                      position=(maximum(x), minimum(y)), align=(:right,:bottom))
            end
        end
    end
    display(fig)
end

df_rmed_sweep, t_vals = sweep_rmeds(;
    modes=[:TR], S_vals=[120], conn_vals=0.05:0.05:0.30,
    mean_abs_vals=[0.5, 1.0, 2.0], mag_cv_vals=[0.01, 0.1, 0.5, 1.0, 2.0],
    u_mean_vals=[1.0], u_cv_vals=[0.3,0.5,0.8,1.0,2.0,3.0],
    degree_families = [:uniform, :lognormal, :pareto],
    deg_cv_vals   = [0.0, 0.5, 1.0, 2.0],
    deg_pl_alphas = [1.2, 1.5, 2.0, 3.0],
    rho_sym_vals  = range(0, 1, length=10),
    reps_per_combo=2, seed=42, number_of_combinations=500,
    margin=0.05, shrink_factor=0.9, max_shrink_iter=200, q_thresh=0.20,
    t_min=0.01, t_max=0.2, t_bins=12
)

plot_rmed_sweep(
    df_rmed_sweep;
    t_steps=8:12, title="Rmed Dynamics Sweep"
)
