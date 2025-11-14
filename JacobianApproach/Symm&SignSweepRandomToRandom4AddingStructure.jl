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
align_u_to_rowload(u, A; rho_align=0.0, mode=:positive, side=:row, rng)

- mode = :positive  -> large u ↔ large load  (your current behavior)
- mode = :negative  -> large u ↔ small load  (abundant species have weaker links)
- side = :row | :col | :both  -> use outgoing, incoming, or sum of both

ρ-align mixes greedy picks (prob ρ) with random picks (1-ρ).
"""
function align_u_to_rowload(u::AbstractVector{<:Real}, A::AbstractMatrix{<:Real};
        rho_align::Real=0.0, mode::Symbol=:positive, side::Symbol=:row,
        rng::AbstractRNG=Random.default_rng())

    rho = clamp(float(rho_align), 0.0, 1.0)
    S   = length(u)
    v   = sort(collect(u); rev=true)  # assign from largest u downward

    # choose load definition
    rowload = [sum(abs, @view A[i, :]) - abs(A[i,i]) for i in 1:S]
    colload = [sum(abs, @view A[:, i]) - abs(A[i,i]) for i in 1:S]
    load = side === :row  ? rowload :
           side === :col  ? colload :
           side === :both ? rowload .+ colload :
           error("side must be :row, :col or :both")

    # order to fill: high→low for :positive, low→high for :negative
    order = mode === :positive ? sortperm(load; rev=true) :
            mode === :negative ? sortperm(load; rev=false) :
            error("mode must be :positive or :negative")

    taken = falses(S)
    res   = zeros(Float64, S)
    ptr   = 1

    for val in v
        if rand(rng) < rho
            while ptr ≤ S && taken[order[ptr]]; ptr += 1; end
            idx = (ptr ≤ S) ? order[ptr] : findfirst(!, taken)
        else
            avail = findall(!, taken)
            idx   = avail[rand(rng, 1:length(avail))]
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

# ------- helper: "pure random, no-structure" NT-ER with same conn & IS -------
pure_random_NT(S; conn, mean_abs, mag_cv, rng) = begin
    A = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs,
                                mag_cv=mag_cv, degree_family=:uniform,
                                deg_param=0.0, rho_sym=0.0, rng=rng)
    is0 = realized_IS(A); is0 == 0 && return A
    A .* (mean_abs / is0)
end

############### SPEED HELPERS — SCHUR-CACHED R̃med ###############
using LinearAlgebra

# Store T as a Matrix; wrap with UpperTriangular only when exponentiating.
struct SchurPack
    Z::Matrix{Float64}         # Schur vectors (orthogonal/unitary)
    T::Matrix{Float64}         # real Schur form (quasi upper-triangular in 1×1/2×2 blocks)
end

schur_pack(J::AbstractMatrix{<:Real}) = begin
    F = schur(Matrix{Float64}(J))   # J = Z*T*Z'
    SchurPack(F.Z, F.T)
end

function _rmed_series_schur(sp::SchurPack, w::AbstractVector{<:Real}, t_vals)
    Z, T = sp.Z, sp.T
    sqrtw = sqrt.(w)
    out = Vector{Float64}(undef, length(t_vals))
    for (k, t) in pairs(t_vals)
        EtT = exp(t .* T)          # <-- block-aware; no UpperTriangular()
        E   = Z * EtT * Z'
        @views E .= E .* reshape(sqrtw, 1, :)
        out[k] = -(log(tr(E*E')) - log(sum(w))) / (2t)
    end
    out
end

function compute_rmed_series(J::AbstractMatrix{<:Real}, u::AbstractVector{<:Real},
                             t_vals::AbstractVector{<:Real}; perturb::Symbol=:biomass)
    sp = schur_pack(J)
    w = perturb === :biomass ? (u .^ 2) :
        perturb === :uniform ? fill(1.0, length(u)) :
        error("Unknown perturbation: $perturb")
    _rmed_series_schur(sp, w, t_vals)
end

# convenience wrapper (unchanged API)
compute_rmed_series(A::AbstractMatrix{<:Real}, u::AbstractVector{<:Real},
                    t_vals::AbstractVector{<:Real}; perturb::Symbol=:biomass) =
    compute_rmed_series(jacobian(A, u), u, t_vals; perturb=perturb)

############### SMALL UTILITIES YOU REFERENCED ################
# Pure non-trophic ER-like "no structure" target (uniform degrees, rho=0), scaled to mean_abs
function pure_random_NT(S; conn::Float64, mean_abs::Float64, mag_cv::Float64, rng)
    A = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                degree_family=:uniform, deg_param=0.0,
                                rho_sym=0.0, rng=rng)
    isA = realized_IS(A)
    if isA > 0
        A .*= mean_abs / isA
    end
    return A
end

############### FASTER, OUTER-THREADED RUNNER ################
"""
run_rewire_axis_grid(; axis, levels, t_vals,
    reps=50, S=120, conn=0.10, mean_abs=0.50, mag_cv=0.60,
    u_mean=1.0, u_cv=0.6, degcv0=0.0,
    magcorr_baseline=0.0, rewire_option=:within,   # :within or :to_purerand
    lines_to_run::Vector{Symbol}=[:NT,:TRdeg,:NI,:TR0,:TRdeg_to_NT,:NI_to_NT],
    seed=20251111)

Lines (skip any you don’t need to save time):
  :NT            => NT ER → NT ER (pair rewiring)
  :TRdeg         => trophic ER (lognormal deg_cv) → trophic ER (pair rewiring)
  :NI            => niche → trophic random (pair rewiring)
  :TR0           => trophic ER (no-structure; uniform degrees) → trophic ER (pair rewiring)
  :TRdeg_to_NT   => trophic ER (deg_cv) → pure NT (ER-like) target
  :NI_to_NT      => niche → pure NT (ER-like) target

axis choices:
  :degcv, :u_cv, :uA_corr, :magcorr
"""
function run_rewire_axis_grid(; axis::Symbol, levels::AbstractVector, t_vals::AbstractVector,
        reps::Int=50, S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.50, mag_cv::Float64=0.60,
        u_mean::Float64=1.0, u_cv::Float64=0.6, degcv0::Float64=0.0,
        magcorr_baseline::Float64=0.0, rewire_option::Symbol=:within,
        lines_to_run::Vector{Symbol}=[:NT,:TRdeg,:NI,:TR0,:TRdeg_to_NT,:NI_to_NT],
        seed::Int=20251111, same_u :: Bool = true)

    @assert rewire_option in (:within, :to_purerand)

    # tag metadata / labels (only used for final "line" string)
    tag_to_name = Dict(
        :NT           => "NT",
        :TRdeg        => "TRdeg",
        :NI           => "NI",
        :TR0          => "TR0",
        :TRdeg_to_NT  => "TRdeg_to_NT",
        :NI_to_NT     => "NI_to_NT",
    )

    # index t once (array accumulators, no Dict{Float64,…})
    nt = length(t_vals)

    # helper: allocate empty (x,y) slots per tag
    make_acc = () -> Dict{Symbol, Vector{Tuple{Vector{Float64},Vector{Float64}}}}(
        tag => [ (Float64[], Float64[]) for _ in 1:nt ] for tag in keys(tag_to_name)
    )

    # stash rows for each level (outer threads own each level → no contention)
    rows_by_case = Vector{Vector{NamedTuple}}(undef, length(levels))
    rng_master = Random.Xoshiro(seed)
    case_seeds = [rand(rng_master, UInt64) for _ in eachindex(levels)]
    Threads.@threads for idx in eachindex(levels)
        lvl = levels[idx]

        # knobs for this case level
        degcv_lvl     = (axis == :degcv   ? float(lvl) : degcv0)
        ucv_lvl       = (axis == :u_cv    ? float(lvl) : u_cv)
        rho_align_lvl = (axis == :uA_corr ? float(lvl) : 0.0)
        magcorr_lvl   = (axis == :magcorr ? float(lvl) : magcorr_baseline)

        rng_case = Random.Xoshiro(case_seeds[idx])
        rep_seeds = [rand(rng_case, UInt64) for _ in 1:reps]
        acc = make_acc()

        # sequential reps inside the case thread
        for rep in 1:reps
            rng = Random.Xoshiro(rep_seeds[rep])  # deterministic per (case,rep)

            # ---- build source As (only what we need, based on lines_to_run) ----
            need_NT    = :NT in lines_to_run || :TRdeg_to_NT in lines_to_run || :NI_to_NT in lines_to_run
            need_TRdeg = :TRdeg in lines_to_run || :TRdeg_to_NT in lines_to_run
            need_NI    = :NI in lines_to_run || :NI_to_NT in lines_to_run
            need_TR0   = :TR0 in lines_to_run

            A_nt = nothing; A_trdeg = nothing; A_ni = nothing; A_tr0 = nothing

            if need_NT
                A_nt = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                               degree_family=:uniform, deg_param=0.0,
                                               rho_sym=magcorr_lvl, rng=rng)
                is_nt = realized_IS(A_nt); is_nt == 0 && continue
                A_nt .*= mean_abs / is_nt
            end

            if need_TRdeg
                # A_trdeg = build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                #                                degree_family=:lognormal, deg_param=degcv_lvl,
                #                                rho_sym=magcorr_lvl, rng=rng)
                A_trdeg = build_ER_degcv(S, conn, mean_abs, mag_cv, 0.0, 1.0, degcv_lvl; rng=rng)       

                is_trd = realized_IS(A_trdeg); is_trd == 0 && continue
                A_trdeg .*= mean_abs / is_trd
            end

            if need_NI
                A_ni = build_niche_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                           degree_family=:lognormal, deg_param=degcv_lvl,
                                           rho_sym=magcorr_lvl, rng=rng)
                is_ni = realized_IS(A_ni); is_ni == 0 && continue
                A_ni .*= mean_abs / is_ni
            end

            if need_TR0
                A_tr0 = build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                             degree_family=:uniform, deg_param=0.0,
                                             rho_sym=magcorr_lvl, rng=rng)
                is_tr0 = realized_IS(A_tr0); is_tr0 == 0 && continue
                A_tr0 .*= mean_abs / is_tr0
            end

            # ---- u draws (+ optional alignment) for only what we use ----
            function draw_u_for(A)
                u = random_u(S; mean=u_mean, cv=ucv_lvl, rng=rng)
                if rho_align_lvl > 0
                    u = align_u_to_rowload(u, A; rho_align=rho_align_lvl, rng=rng, mode=:negative, side=:row)
                end
                u
            end
            u_nt    = (need_NT    ? draw_u_for(A_nt)    : nothing)
            u_trdeg = (need_TRdeg ? draw_u_for(A_trdeg) : nothing)
            u_ni    = (need_NI    ? draw_u_for(A_ni)    : nothing)
            u_tr0   = (need_TR0   ? draw_u_for(A_tr0)   : nothing)

            u_nt_i = (need_NT    ? draw_u_for(A_nt)    : nothing)
            u_trdeg_i = (need_TRdeg ? draw_u_for(A_trdeg) : nothing)
            u_ni_i = (need_NI    ? draw_u_for(A_ni)    : nothing)
            u_tr0_i = (need_TR0   ? draw_u_for(A_tr0)   : nothing)

            # ---- targets (rewire/purerand) for each requested line ----
            # helper to produce (sourceA, targetA, u) for a tag
            function source_target(tag::Symbol)
                if tag === :NT
                    # NT: independent NT draw as target
                    R = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                               degree_family=:uniform, deg_param=0.0,
                                               rho_sym=magcorr_lvl, rng=rng)
                    isR = realized_IS(R); isR == 0 && return (A_nt, R, u_nt, u_nt_i)
                    R  .*= mean_abs / isR
                    return (A_nt, R, u_nt, u_nt_i)

                elseif tag === :TRdeg
                    # TRdeg: independent trophic-ER draw as target (same deg_cv etc.)
                    R = build_ER_degcv(S, conn, mean_abs, mag_cv, 0.0, 1.0, degcv_lvl; rng=rng)
                    isR = realized_IS(R); isR == 0 && return (A_trdeg, R, u_trdeg, u_trdeg_i)
                    R  .*= mean_abs / isR
                    return (A_trdeg, R, u_trdeg, u_trdeg_i)

                elseif tag === :TR0
                    # TR0: independent uniform-degree trophic ER as target
                    R = build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                             degree_family=:uniform, deg_param=0.0,
                                             rho_sym=magcorr_lvl, rng=rng)
                    isR = realized_IS(R); isR == 0 && return (A_tr0, R, u_tr0, u_tr0_i)
                    R  .*= mean_abs / isR
                    return (A_tr0, R, u_tr0, u_tr0_i)

                elseif tag === :NI
                    # NI: niche → trophic ER (different families stays different)
                    R = build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                             degree_family=:lognormal, deg_param=degcv_lvl,
                                             rho_sym=magcorr_lvl, rng=rng)
                    isR = realized_IS(R); isR == 0 && return (A_ni, R, u_ni, u_ni_i)
                    R  .*= mean_abs / isR
                    return (A_ni, R, u_ni, u_ni_i)

                elseif tag === :TRdeg_to_NT
                    R = pure_random_NT(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
                    return (A_trdeg, R, u_trdeg, u_trdeg_i)

                else # :NI_to_NT
                    R = pure_random_NT(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
                    return (A_ni, R, u_ni, u_ni_i)
                end
            end

            # If :to_purerand, force all targets to pure NT
            function force_to_purerand(A, _R, u)
                R = pure_random_NT(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
                return (A, R, u, u_i)
            end

            # ---- compute R̃med series with Schur cache & accumulate ----
            for tag in lines_to_run
                A, R, u, u_i = source_target(tag)
                if rewire_option == :to_purerand && tag ∉ (:TRdeg_to_NT, :NI_to_NT)
                    A, R, u, u_i = force_to_purerand(A, R, u)
                end

                # cache Schur once per J
                JA = jacobian(A, u)
                spA = schur_pack(JA)
                
                if same_u
                    JR = jacobian(R, u)
                    spR = schur_pack(JR)
                else
                    JR = jacobian(R, u_i)
                    spR = schur_pack(JR)
                end
                # --- minimal, correct block ---
                if same_u
                    w = u .^ 2
                    f = _rmed_series_schur(spA, w, t_vals)
                    g = _rmed_series_schur(spR, w, t_vals)   # use the same weights
                else
                    w  = u   .^ 2
                    w_i = u_i.^ 2
                    f = _rmed_series_schur(spA, w,   t_vals)
                    g = _rmed_series_schur(spR, w_i, t_vals)
                end

                as = acc[tag]
                @inbounds for i in 1:nt
                    push!(as[i][1], f[i]);    # x
                    push!(as[i][2], g[i]);    # y
                end
            end
        end # reps

        # ---- collapse this case to rows ----
        rows_case = NamedTuple[]
        for tag in lines_to_run
            line_name = tag_to_name[tag]
            as = acc[tag]
            for (i, t) in pairs(t_vals)
                x = as[i][1];  y = as[i][2]
                isempty(x) && continue
                r2 = r2_to_identity(x, y)
                ad = mean(abs.(y .- x))
                meta = if axis == :degcv
                    (deg_cv=degcv_lvl, u_cv=NaN,    uA_corr=NaN,           magcorr=magcorr_lvl)
                elseif axis == :u_cv
                    (deg_cv=degcv0,   u_cv=ucv_lvl, uA_corr=NaN,           magcorr=magcorr_lvl)
                elseif axis == :uA_corr
                    (deg_cv=degcv0,   u_cv=ucv_lvl, uA_corr=rho_align_lvl, magcorr=magcorr_lvl)
                else
                    (deg_cv=degcv0,   u_cv=ucv_lvl, uA_corr=0.0,           magcorr=magcorr_lvl)
                end
                push!(rows_case, (; axis=String(axis), case=idx, line=line_name,
                                   t=t, r2=r2, absdiff=ad, meta...))
            end
        end
        rows_by_case[idx] = rows_case
    end # threaded cases

    df = DataFrame(vcat(rows_by_case...))
    rename!(df, Dict(n => String(n) for n in names(df)))  # keep string names
    return df
end

# ------------------------------- Plotting (3×3) -------------------------------
"""
plot_rewire_axis_grid(df, axis; title, absdiff=false)

Expects df from run_rewire_axis_grid. Draws 3 lines per panel:
  NT (non-trophic ER→ER), TR (trophic ER→ER), NI (niche→random).

If absdiff=false -> plot R²; else -> plot mean |ΔR̃med|.
"""

using Printf

function plot_rewire_axis_grid(
    df::DataFrame, axis::Symbol; title::String, absdiff::Bool=false,
    lines_to_plot::Union{Nothing,Vector{String}}=nothing
)
    # required columns (string names)
    required = ["case", "line", "t"]
    miss = setdiff(required, names(df))
    !isempty(miss) && error("df is missing required columns $(miss). Available: $(names(df))")

    # y column and global limits
    ycol = absdiff ? "absdiff" : "r2"
    !(ycol in names(df)) && error("df is missing column $(ycol). Available: $(names(df))")

    tmin = minimum(df[!, "t"]);  tmax = maximum(df[!, "t"])
    # restrict df if lines_to_plot is given
    df_plot = isnothing(lines_to_plot) ? df : df[in.(df[!, "line"], Ref(lines_to_plot)), :]

    if absdiff
        vals = filter(isfinite, skipmissing(df_plot[!, "absdiff"]))
        ymax = isempty(vals) ? 1.0 : maximum(vals)
        ylims = (-0.05, ymax * 1.05)  # small 5% margin for clarity
    else
        vals = filter(isfinite, skipmissing(df_plot[!, "r2"]))
        ymax = isempty(vals) ? 1.0 : maximum(vals)
        ylims = (-0.05, ymax * 1.05)
    end

    # all possible series (tag => (label, color))
    series_def = Dict(
        "NT"           => ("NT (ER→ER)",               :steelblue),
        "TRdeg"        => ("TRdeg (tER→tER)",          :orangered),
        "NI"           => ("NI (niche→tER)",           :seagreen),
        "TR0"          => ("TR0 (no-struct tER→tER)",  :purple),
        "TRdeg_to_NT"  => ("TRdeg→NT",                 :gray40),
        "NI_to_NT"     => ("NI→NT",                    :goldenrod),
    )

    # decide which lines to plot (preserve user order)
    tags_all = collect(keys(series_def))
    tags = lines_to_plot === nothing ? tags_all :
           [t for t in lines_to_plot if haskey(series_def, t)]

    levels = sort(unique(df[!, "case"]))
    ncases = length(levels)
    ncols, nrows = 3, ceil(Int, ncases / 3)

    fig = Figure(size=(1100, 750))
    Label(fig[0, 1:3], title; fontsize=20, font=:bold, halign=:left)

    for (k, cidx) in enumerate(levels)
        sub = df[df[!, "case"] .== cidx, :]
        isempty(sub) && continue

        # panel label by axis
        colname = axis == :degcv   ? "deg_cv"  :
                  axis == :u_cv    ? "u_cv"    :
                  axis == :uA_corr ? "uA_corr" : "magcorr"

        v = colname in names(sub) ? sub[!, colname] : fill(NaN, nrow(sub))
        v = filter(isfinite, skipmissing(v))
        label_val = isempty(v) ? NaN : first(unique(v))

        panel_title = axis == :degcv   ? @sprintf("deg_cv=%.2f", label_val) :
                      axis == :u_cv    ? @sprintf("u_cv=%.2f", label_val)   :
                      axis == :uA_corr ? @sprintf("u–A align=%.2f", label_val) :
                                         @sprintf("mag corr=%.2f", label_val)

        r, c = divrem(k-1, ncols)
        ax = Axis(fig[r+1, c+1];
                  xscale=log10, xlabel="t",
                  ylabel=(c == 0 ? (absdiff ? "|ΔR̃med|" : "R²") : ""),
                  title=panel_title, limits=((tmin, tmax), ylims))

        any_plotted = false
        for tag in tags
            (labstr, col) = series_def[tag]
            s = sub[sub[!, "line"] .== tag, :]
            isempty(s) && continue
            sort!(s, "t")
            ys = s[!, ycol]
            lines!(ax, s[!, "t"], ys; color=col, linewidth=2, label=labstr)
            scatter!(ax, s[!, "t"], ys; color=col)
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
seed = 250463

# 1) Degree heterogeneity sweep (applies to ER and niche)
levels_deg = collect(range(0.00, 1.50; length=9))
@time df_deg_different_u = run_rewire_axis_grid(
    ; axis=:degcv, levels=levels_deg, t_vals=t_vals,
    reps=10, S=S, conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, 
    u_mean=1.0, u_cv=0.6, degcv0=0.0,
    magcorr_baseline=1.0, seed=seed,
    same_u=false
)
df_deg_different_u = df_deg
# Rmed
plot_rewire_axis_grid(df_deg_different_u, :degcv; title="Rewiring predictability Degree CV (|ΔR̃med|) Different u", absdiff=true, lines_to_plot=["NT", "TRdeg", "NI", "TR0", "TRdeg_to_NT", "NI_to_NT"])
plot_rewire_axis_grid(df_deg_same_u, :degcv; title="Rewiring predictability Degree CV (|ΔR̃med|) Different u", absdiff=true, lines_to_plot=["NT", "TRdeg", "NI", "TR0"])
# R2
plot_rewire_axis_grid(df_deg_different_u, :degcv; title="Rewiring predictability Degree CV (R²) Different u", absdiff=false, lines_to_plot=["NT", "TRdeg", "NI", "TR0", "TRdeg_to_NT", "NI_to_NT"])
plot_rewire_axis_grid(df_deg_same_u, :degcv; title="Rewiring predictability Degree CV (R²) Same u", absdiff=false, lines_to_plot=["NT", "TRdeg", "NI", "TR0"])

# 2) u_cv sweep
levels_u = collect(range(0.1, 2.0; length=9))
df_u_different_u = run_rewire_axis_grid(
    ; axis=:u_cv, levels=levels_u, t_vals=t_vals,
    reps=10, S=S, conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
    u_mean=1.0, u_cv=0.6, magcorr_baseline=0.0, seed=seed,
    same_u=false
)
df_u_different_u = df_u
# Rmed
plot_rewire_axis_grid(df_u_different_u, :u_cv; title="Rewiring predictability u CV (|ΔR̃med|) Different u", absdiff=true, lines_to_plot=["NT", "TRdeg", "NI", "TR0", "TRdeg_to_NT", "NI_to_NT"])
plot_rewire_axis_grid(df_u_same_u, :u_cv; title="Rewiring predictability u CV (|ΔR̃med|) Same u", absdiff=true, lines_to_plot=["NT", "TRdeg", "NI", "TR0"])
# R2
plot_rewire_axis_grid(df_u_different_u, :u_cv; title="Rewiring predictability u CV (R²) Different u", absdiff=false, lines_to_plot=["NT", "TRdeg", "NI", "TR0", "TRdeg_to_NT", "NI_to_NT"])
plot_rewire_axis_grid(df_u_same_u, :u_cv; title="Rewiring predictability u CV (R²) Same u", absdiff=false, lines_to_plot=["NT", "TRdeg", "NI", "TR0"])

# 3) u–A correlation sweep (0..1 alignment of u to |A| row-load)
levels_align = collect(range(0.0, 1.0; length=9))
df_align_different_u = run_rewire_axis_grid(
    ; axis=:uA_corr, levels=levels_align, t_vals,
    reps=10, S, conn, mean_abs, mag_cv, u_mean=1.0, u_cv=0.6,
    magcorr_baseline=0.0, seed,
    same_u=false
)
# Rmed
plot_rewire_axis_grid(df_align_different_u, :uA_corr; title="Rewiring predictability u-A corr (|ΔR̃med|) Different u", absdiff=true, lines_to_plot=["NT", "TRdeg", "NI", "TR0", "TRdeg_to_NT", "NI_to_NT"])
plot_rewire_axis_grid(df_align_same_u, :uA_corr; title="Rewiring predictability u-A corr (|ΔR̃med|) Negative Corr Same u", absdiff=true, lines_to_plot=["NT", "TRdeg", "NI", "TR0"])
# r2
plot_rewire_axis_grid(df_align_different_u, :uA_corr; title="Rewiring predictability u-A corr (R²) Different u", absdiff=false, lines_to_plot=["NT", "TRdeg", "NI", "TR0", "TRdeg_to_NT", "NI_to_NT"])
plot_rewire_axis_grid(df_align_same_u, :uA_corr; title="Rewiring predictability u-A corr (R²) Negative Corr Same u", absdiff=false, lines_to_plot=["NT", "TRdeg", "NI", "TR0"])

# 4) Magnitude correlation sweep (baseline elsewhere is 0.0)
levels_mag = collect(range(0.0, 1.0; length=9))
df_mag_different_u = run_rewire_axis_grid(
    ; axis=:magcorr, levels=levels_mag, t_vals,
    reps=10, S, conn, mean_abs, mag_cv, u_mean=1.0, u_cv=0.6,
    magcorr_baseline=0.0, seed,
    same_u=false
)
# Rmed
plot_rewire_axis_grid(df_mag_different_u, :magcorr; title="Rewiring predictability mag corr (|ΔR̃med|) Different u", absdiff=true, lines_to_plot=["NT", "TRdeg", "NI", "TR0", "TRdeg_to_NT", "NI_to_NT"])
plot_rewire_axis_grid(df_mag, :magcorr; title="Rewiring predictability mag corr (|ΔR̃med|) Different u", absdiff=true, lines_to_plot=["NT", "TRdeg", "NI", "TR0"])
# R2
plot_rewire_axis_grid(df_mag_different_u, :magcorr; title="Rewiring predictability mag corr (R²) Different u", absdiff=false, lines_to_plot=["NT", "TRdeg", "NI", "TR0", "TRdeg_to_NT", "NI_to_NT"])
plot_rewire_axis_grid(df_mag, :magcorr; title="Rewiring predictability mag corr (R²) Different u", absdiff=false, lines_to_plot=["NT", "TRdeg", "NI", "TR0"])


