using Random, Statistics, DataFrames, CairoMakie

# ----------------------- Pair-wise rewiring (preserve values) -----------------------
"""
rewire_pairs_preserving_values(A; rng, random_targets=true)

Collects all unordered pairs (i<j) where at least one direction is nonzero,
keeps the ordered pair values (Aij,Aji) EXACTLY, and places them onto new unordered
pairs. If random_targets=false, just permutes among the existing occupied pairs;
if true, assigns onto a fresh uniform random set of K unordered pairs.

Returns a new matrix with same S and same multiset of pair values.
"""
function rewire_pairs_preserving_values(A::AbstractMatrix{<:Real};
        rng::AbstractRNG=Random.default_rng(), random_targets::Bool=true)
    S = size(A,1); @assert size(A,2) == S
    # source pairs and their ordered values
    src = Tuple{Int,Int}[]; vals = Tuple{Float64,Float64}[]
    @inbounds for i in 1:S-1, j in i+1:S
        v1 = A[i,j]; v2 = A[j,i]
        if v1 != 0.0 || v2 != 0.0
            push!(src, (i,j)); push!(vals, (float(v1), float(v2)))
        end
    end
    K = length(src); K == 0 && return zeros(Float64, S, S)

    # choose targets
    tgt = if random_targets
        allpairs = [(i,j) for i in 1:S-1 for j in i+1:S]
        sample(rng, allpairs, K; replace=false)
    else
        shuffle(rng, copy(src))
    end

    perm = shuffle(rng, collect(1:K))
    B = zeros(Float64, S, S)
    @inbounds for k in 1:K
        (p,q) = tgt[k]
        (v1,v2) = vals[perm[k]]
        B[p,q] = v1;  B[q,p] = v2    # keep ordered pair values
    end
    return B
end

# ------------------ u–|A| row-load alignment (simple, fast) -------------------
"""
align_u_to_rowload(u, A; rho_align, rng)

Returns a copy of u whose largest entries are assigned preferentially to species
with largest row-load r_i = sum_j |A_ij|. rho_align in [0,1]: fraction of greedy
assignments to the current top-load slot; remaining picks are random.
"""
function align_u_to_rowload(u::AbstractVector{<:Real}, A::AbstractMatrix{<:Real};
        rho_align::Real=0.0, rng::AbstractRNG=Random.default_rng())
    rho = clamp(float(rho_align), 0.0, 1.0)
    v = sort(collect(u); rev=true)              # u values high→low
    S = length(u)
    rload = [sum(abs, @view A[i, :]) - abs(A[i,i]) for i in 1:S]
    order = sortperm(rload; rev=true)           # species high→low load
    taken = falses(S)
    res = zeros(Float64, S)
    ptr = 1
    for val in v
        if rand(rng) < rho
            # take next available top-load slot
            while ptr ≤ S && taken[order[ptr]]; ptr += 1; end
            idx = (ptr ≤ S) ? order[ptr] : (findfirst(!, taken) |> x->(x === nothing ? 1 : x))
        else
            # take a random available slot
            avail = findall(!, taken)
            idx = avail[rand(rng, 1:length(avail))]
        end
        res[idx] = val
        taken[idx] = true
    end
    return res
end

# ------------------------- One-shot R̃med series helper -------------------------
compute_rmed_series(A, u, t_vals; perturb=:biomass) = begin
    J = jacobian(A, u)
    [median_return_rate(J, u; t=t, perturbation=perturb) for t in t_vals]
end

# ------------------------------- Master runner --------------------------------
"""
run_rewire_axis_grid(; axis, levels, t_vals,
    reps=50, S=120, conn=0.10, mean_abs=0.50, mag_cv=0.60,
    u_mean=1.0, u_cv=0.6, degcv0=0.0,
    magcorr_baseline=0.0, seed=20251111)

Builds three baselines per draw:
  NT-ER  : non-trophic ER (build_ER_degcv, rho_mag uses magcorr, rho_sign free)
  TR-ER  : trophic ER (build_random_trophic_ER, rho_sym uses magcorr)
  NICHE  : niche trophic (build_niche_trophic, degree_family=:lognormal, deg_param=degcv)

Then rewires each by pair-permutation, computes R̃med series for Full vs Rewired,
and aggregates R² and |ΔR̃med| across reps, per t.

axis choices:
  :degcv    -> levels are degree CVs; applies to NT-ER (degcv) and NICHE (deg_param)
  :u_cv     -> levels are u CVs
  :uA_corr  -> levels are rho_align in [0,1] for aligning u to |A| row-load
  :magcorr  -> levels are pairwise magnitude correlations (NT: rho_mag, TR/NICHE: rho_sym)

Magnitude correlation baseline is 0.0 unless axis==:magcorr.
"""
# ---- helper: build a "pure random, no-structure" NT-ER with same conn & IS ----
pure_random_NT(S; conn, mean_abs, mag_cv, rng) = begin
    A = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs,
                                mag_cv=mag_cv, degree_family=:uniform,
                                deg_param=0.0, rho_sym=0.0, rng=rng)
    is0 = realized_IS(A); is0 == 0 && return A
    A .* (mean_abs / is0)
end

# ---- main runner with options and 4 lines (NT, TRdeg, NI, TR0) ----
function run_rewire_axis_grid(; axis::Symbol, levels::AbstractVector, t_vals::AbstractVector,
        reps::Int=50, S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.50, mag_cv::Float64=0.60,
        u_mean::Float64=1.0, u_cv::Float64=0.6, degcv0::Float64=0.0,
        magcorr_baseline::Float64=0.0, rewire_option::Symbol=:within, seed::Int=20251111)

    @assert rewire_option in (:within, :to_purerand)
    rng0 = Random.Xoshiro(seed)
    rows = NamedTuple[]

    for (case_idx, lvl) in enumerate(levels)
        # per-level knobs
        degcv_lvl      = (axis == :degcv   ? float(lvl) : degcv0)
        ucv_lvl        = (axis == :u_cv    ? float(lvl) : u_cv)
        rho_align_lvl  = (axis == :uA_corr ? float(lvl) : 0.0)
        magcorr_lvl    = (axis == :magcorr ? float(lvl) : magcorr_baseline)

        # accumulators: 4 lines
        tags = (:NT, :TRdeg, :NI, :TR0)
        acc = Dict{Symbol,Dict{Float64,Tuple{Vector{Float64},Vector{Float64}}}}(
            tag => Dict(t => (Float64[], Float64[]) for t in t_vals) for tag in tags
        )

        for rep in 1:reps
            rng = Random.Xoshiro(rand(rng0, UInt64))

            # ---- build baselines ----
            # NT: non-trophic ER, degcv not used here (uniform)
            A_nt = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                           degree_family=:uniform, deg_param=0.0, rho_sym=magcorr_lvl, rng=rng)
            is_nt = realized_IS(A_nt); is_nt == 0 && continue
            A_nt .*= mean_abs / is_nt

            # TRdeg: trophic ER with degree heterogeneity (use build_random_trophic with degree_family)
            A_trdeg = build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                           degree_family=:lognormal, deg_param=degcv_lvl,
                                           rho_sym=magcorr_lvl, rng=rng)
            is_trd = realized_IS(A_trdeg); is_trd == 0 && continue
            A_trdeg .*= mean_abs / is_trd

            # NI: niche trophic (lognormal degree; same degcv sweep)
            A_ni = build_niche_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                       degree_family=:lognormal, deg_param=degcv_lvl,
                                       rho_sym=magcorr_lvl, rng=rng)
            is_ni = realized_IS(A_ni); is_ni == 0 && continue
            A_ni .*= mean_abs / is_ni

            # TR0: trophic ER with no structure (uniform degrees)
            A_tr0 = build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                         degree_family=:uniform, deg_param=0.0,
                                         rho_sym=magcorr_lvl, rng=rng)
            is_tr0 = realized_IS(A_tr0); is_tr0 == 0 && continue
            A_tr0 .*= mean_abs / is_tr0

            # ---- u draws (per baseline), optional alignment to |A| row-load ----
            u_nt    = random_u(S; mean=u_mean, cv=ucv_lvl, rng=rng)
            u_trdeg = random_u(S; mean=u_mean, cv=ucv_lvl, rng=rng)
            u_ni    = random_u(S; mean=u_mean, cv=ucv_lvl, rng=rng)
            u_tr0   = random_u(S; mean=u_mean, cv=ucv_lvl, rng=rng)

            if rho_align_lvl > 0
                u_nt    = align_u_to_rowload(u_nt,    A_nt;    rho_align=rho_align_lvl, rng=rng)
                u_trdeg = align_u_to_rowload(u_trdeg, A_trdeg; rho_align=rho_align_lvl, rng=rng)
                u_ni    = align_u_to_rowload(u_ni,    A_ni;    rho_align=rho_align_lvl, rng=rng)
                u_tr0   = align_u_to_rowload(u_tr0,   A_tr0;   rho_align=rho_align_lvl, rng=rng)
            end

            # ---- targets, depending on option ----
            if rewire_option == :within
                # pair-preserving rewire onto the stated ensemble
                R_nt    = rewire_pairs_preserving_values(A_nt;    rng=rng, random_targets=true)        # NT→NT
                R_trdeg = rewire_pairs_preserving_values(A_trdeg; rng=rng, random_targets=true)        # TRdeg→TRdeg
                # NI→tER: drop NI pair values onto a random tER topology
                R_ni    = rewire_pairs_preserving_values(A_ni;    rng=rng, random_targets=true)        # topology tER-like via uniform targets
                # TR0→TR0
                R_tr0   = rewire_pairs_preserving_values(A_tr0;   rng=rng, random_targets=true)
            else
                # to_purerand: compare against a freshly generated pure NT (no trophic, uniform degrees)
                R_nt    = pure_random_NT(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
                R_trdeg = pure_random_NT(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
                R_ni    = pure_random_NT(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
                R_tr0   = pure_random_NT(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
            end

            # ---- R̃med series (Full vs "Rewired") ----
            f_nt    = compute_rmed_series(A_nt,    u_nt,    t_vals; perturb=:biomass)
            g_nt    = compute_rmed_series(R_nt,    u_nt,    t_vals; perturb=:biomass)

            f_trdeg = compute_rmed_series(A_trdeg, u_trdeg, t_vals; perturb=:biomass)
            g_trdeg = compute_rmed_series(R_trdeg, u_trdeg, t_vals; perturb=:biomass)

            f_ni    = compute_rmed_series(A_ni,    u_ni,    t_vals; perturb=:biomass)
            g_ni    = compute_rmed_series(R_ni,    u_ni,    t_vals; perturb=:biomass)

            f_tr0   = compute_rmed_series(A_tr0,   u_tr0,   t_vals; perturb=:biomass)
            g_tr0   = compute_rmed_series(R_tr0,   u_tr0,   t_vals; perturb=:biomass)

            @inbounds for (i,t) in enumerate(t_vals)
                push!(acc[:NT][t][1],    f_nt[i]);    push!(acc[:NT][t][2],    g_nt[i])
                push!(acc[:TRdeg][t][1], f_trdeg[i]); push!(acc[:TRdeg][t][2], g_trdeg[i])
                push!(acc[:NI][t][1],    f_ni[i]);    push!(acc[:NI][t][2],    g_ni[i])
                push!(acc[:TR0][t][1],   f_tr0[i]);   push!(acc[:TR0][t][2],   g_tr0[i])
            end
        end

        # collapse to R² and |ΔR̃med|
        for t in t_vals
            for tag in tags
                x, y = acc[tag][t]; isempty(x) && continue
                r2 = r2_to_identity(x, y)
                ad = mean(abs.(y .- x))
                meta = if axis == :degcv
                    (deg_cv=degcv_lvl, u_cv=NaN, uA_corr=NaN, magcorr=magcorr_lvl)
                elseif axis == :u_cv
                    (deg_cv=degcv0, u_cv=ucv_lvl, uA_corr=NaN, magcorr=magcorr_lvl)
                elseif axis == :uA_corr
                    (deg_cv=degcv0, u_cv=ucv_lvl, uA_corr=rho_align_lvl, magcorr=magcorr_lvl)
                else
                    (deg_cv=degcv0, u_cv=ucv_lvl, uA_corr=0.0, magcorr=magcorr_lvl)
                end
                push!(rows, (; axis, case=case_idx, line=String(tag), t, r2, absdiff=ad, meta...))
            end
        end
    end

    return DataFrame(rows)
end

# ------------------------------- Plotting (3×3) -------------------------------
"""
plot_rewire_axis_grid(df, axis; title, absdiff=false)

Expects df from run_rewire_axis_grid. Draws 3 lines per panel:
  NT (non-trophic ER→ER), TR (trophic ER→ER), NI (niche→random).

If absdiff=false -> plot R²; else -> plot mean |ΔR̃med|.
"""
using Printf

function plot_rewire_axis_grid(df::DataFrame, axis::Symbol; title::String, absdiff::Bool=false)
    # 0) Be liberal about column names (strings → symbols)
    df2 = deepcopy(df)
    if eltype(names(df2)) <: AbstractString
        rename!(df2, Symbol.(names(df2)))
    end

    # 1) Validate the minimum columns and pick y-column
    required = ["case", "line", "t"]
    miss = setdiff(required, names(df2))
    if !isempty(miss)
        error("df is missing required columns $(miss). Available: $(names(df2))")
    end
    ycol = absdiff ? "absdiff" : :"r2"
    if !(ycol in names(df2))
        error("df is missing column $(ycol). Available: $(names(df2))")
    end

    levels = sort(unique(df2.case))
    ncases = length(levels)
    ncols, nrows = 3, ceil(Int, ncases / 3)

    tmin = minimum(df2.t); tmax = maximum(df2.t)
    if absdiff
        # global ymax across all panels (ignore missings/NaNs)
        vals = filter(x -> isfinite(x), skipmissing(df[!, "absdiff"]))
        ymax = isempty(vals) ? 1.0 : maximum(vals)
        ylims = (-0.02, ymax)
    else
        ylims = (-0.02, 1.05)
    end

    fig = Figure(size=(1100, 900))
    Label(fig[0, 1:3], title; fontsize=20, font=:bold, halign=:left)

    # Up to four series (skip silently if a given tag isn't present)
    tags  = ["NT","TRdeg","NI","TR0"]
    cols  = [:steelblue, :orangered, :seagreen, :purple]
    labs  = ["NT (ER→ER)", "TRdeg (tER→tER)", "NI (niche→tER)", "TR0 (no-struct tER→tER)"]

    for (k, cidx) in enumerate(levels)
        sub = df2[df2.case .== cidx, :]
        isempty(sub) && continue

        # Panel title by axis value
        lab = if axis == :degcv
            have = filter(!ismissing, sub.deg_cv)
            @sprintf("deg_cv=%.2f", isempty(have) ? NaN : first(unique(have)))
        elseif axis == :u_cv
            have = filter(!ismissing, sub.u_cv)
            @sprintf("u_cv=%.2f", isempty(have) ? NaN : first(unique(have)))
        elseif axis == :uA_corr
            have = filter(!ismissing, sub.uA_corr)
            @sprintf("u–A align=%.2f", isempty(have) ? NaN : first(unique(have)))
        else # :magcorr
            have = filter(!ismissing, sub.magcorr)
            @sprintf("mag corr=%.2f", isempty(have) ? NaN : first(unique(have)))
        end

        r, c = divrem(k-1, ncols)
        ax = Axis(fig[r+1, c+1];
                  xscale=log10,
                  xlabel="t",
                  ylabel=(c == 0 ? (absdiff ? "|ΔR̃med|" : "R²") : ""),
                  title=lab,
                  limits=((tmin, tmax), ylims))

        # Plot whatever series exist in this panel
        any_plotted = false
        for (tag, col, labstr) in zip(tags, cols, labs)
            s = sub[sub.line .== tag, :]
            isempty(s) && continue
            sort!(s, :t)
            ys = s[!, ycol]
            lines!(ax, s.t, ys; color=col, linewidth=2, label=labstr)
            scatter!(ax, s.t, ys; color=col)
            any_plotted = true
        end

        if k == 1 && any_plotted
            axislegend(ax; position=(absdiff ? :lt : :rt), framevisible=false)
        end
    end

    display(fig)
end

# Common knobs
S, conn, mean_abs, mag_cv = 120, 0.10, 0.50, 0.60
t_vals = 10 .^ range(-2, 2; length=40)  # same as before
seed = 20251111

# 1) Degree heterogeneity sweep (applies to ER and niche)
levels_deg = collect(range(0.00, 1.50; length=9))
df_deg = run_rewire_axis_grid(
    ; axis=:degcv, levels=levels_deg, t_vals=t_vals,
    reps=50, S=S, conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, 
    u_mean=1.0, u_cv=0.6, degcv0=0.0,
    magcorr_baseline=0.0, seed=seed
)
plot_rewire_axis_grid(df_deg, :degcv; title="Rewiring predictability — Degree CV (R²)", absdiff=false)
plot_rewire_axis_grid(df_deg, :degcv; title="Rewiring predictability — Degree CV (|ΔR̃med|)", absdiff=true)

# 2) u_cv sweep
levels_u = collect(range(0.1, 2.0; length=9))
df_u = run_rewire_axis_grid(
    ; axis=:u_cv, levels=levels_u, t_vals=t_vals,
    reps=50, S=S, conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
    u_mean=1.0, u_cv=0.6, magcorr_baseline=0.0, seed=seed
)
plot_rewire_axis_grid(df_u, :u_cv; title="Rewiring predictability — u_cv (R²)", absdiff=false)
plot_rewire_axis_grid(df_u, :u_cv; title="Rewiring predictability — u_cv (|ΔR̃med|)", absdiff=true)

# 3) u–A correlation sweep (0..1 alignment of u to |A| row-load)
levels_align = collect(range(0.0, 1.0; length=9))
df_align = run_rewire_axis_grid(; axis=:uA_corr, levels=levels_align, t_vals,
    reps=50, S, conn, mean_abs, mag_cv, u_mean=1.0, u_cv=0.6,
    magcorr_baseline=0.0, seed)
plot_rewire_axis_grid(df_align, :uA_corr; title="Rewiring predictability — u–A alignment (R²)", absdiff=false)
plot_rewire_axis_grid(df_align, :uA_corr; title="Rewiring predictability — u–A alignment (|ΔR̃med|)", absdiff=true)

# 4) Magnitude correlation sweep (baseline elsewhere is 0.0)
levels_mag = collect(range(0.0, 1.0; length=9))
df_mag = run_rewire_axis_grid(; axis=:magcorr, levels=levels_mag, t_vals,
    reps=50, S, conn, mean_abs, mag_cv, u_mean=1.0, u_cv=0.6,
    magcorr_baseline=0.0, seed)
plot_rewire_axis_grid(df_mag, :magcorr; title="Rewiring predictability — magnitude corr (R²)", absdiff=false)
plot_rewire_axis_grid(df_mag, :magcorr; title="Rewiring predictability — magnitude corr (|ΔR̃med|)", absdiff=true)
