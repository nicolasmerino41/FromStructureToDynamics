############################
# 1) Helper functions
############################
using Random, Statistics, LinearAlgebra, DataFrames, Distributions
using Base.Threads

# ===================== Degree-controlled A builders =====================
# degree_family ∈ (:uniform, :lognormal, :pareto)
# :uniform    -> ER-like
# :lognormal  -> node propensities with given CV
# :pareto     -> propensities Pareto(x_min=1, α=deg_param)  (smaller α => heavier tail)

# -- helper: node weights per family
function _node_weights(S; degree_family::Symbol, deg_param::Float64, rng)
    if degree_family === :uniform
        return ones(Float64, S)
    elseif degree_family === :lognormal
        σ = deg_param <= 1e-12 ? 0.0 : sqrt(log(1 + deg_param^2))
        μ = -σ^2/2
        return rand(rng, LogNormal(μ, σ), S)  # mean ≈ 1
    elseif degree_family === :pareto
        α = max(deg_param, 1.01)             # finite mean
        return rand(rng, Pareto(1.0, α), S)  # mean = α/(α-1)
    else
        error("Unknown degree_family = $degree_family")
    end
end

# -- directed, non-trophic: P[i,j] ∝ w_out[i] * w_in[j], i≠j
function build_random_nontrophic(
    S; conn=0.10, mean_abs=0.10, mag_cv=0.60,
    degree_family::Symbol=:uniform, deg_param::Float64=0.0,
    rng=Random.default_rng()
)
    A = zeros(Float64, S, S)
    E_target = clamp(round(Int, conn * S*(S-1)), 0, S*(S-1))

    w_out = _node_weights(S; degree_family=degree_family, deg_param=deg_param, rng=rng)
    w_in  = _node_weights(S; degree_family=degree_family, deg_param=deg_param, rng=rng)

    # unnormalized probs (zero diagonal)
    P = w_out .* transpose(w_in)
    @inbounds for i in 1:S; P[i,i] = 0.0; end
    s = (sum(P) > 0) ? (E_target / sum(P)) : 0.0
    @. P = min(1.0, s*P)

    # sample edges (Bernoulli), assign magnitudes and random signs
    σm = sqrt(log(1 + mag_cv^2)); μm = log(mean_abs) - σm^2/2
    @inbounds for i in 1:S, j in 1:S
        if i != j && rand(rng) < P[i,j]
            m = rand(rng, LogNormal(μm, σm))
            A[i,j] = (rand(rng) < 0.5 ? +m : -m)
        end
    end
    return A
end

# -- trophic (antisymmetric on unordered pairs): Pr(edge {i,j}) ∝ w[i]*w[j]
function build_random_predation(
    S; conn=0.10, mean_abs=0.10, mag_cv=0.60,
    degree_family::Symbol=:uniform, deg_param::Float64=0.0,
    rng=Random.default_rng()
)
    A = zeros(Float64, S, S)
    pairs = [(i,j) for i in 1:S for j in (i+1):S]
    E_target = clamp(round(Int, conn * length(pairs)), 0, length(pairs))

    w = _node_weights(S; degree_family=degree_family, deg_param=deg_param, rng=rng)

    # pair weights
    W = similar(pairs, Float64)
    @inbounds for k in eachindex(pairs)
        i,j = pairs[k]; W[k] = w[i]*w[j]
    end
    Z = sum(W)
    Z == 0 && return A
    s = E_target / Z

    σm = sqrt(log(1 + mag_cv^2)); μm = log(mean_abs) - σm^2/2
    @inbounds for k in eachindex(pairs)
        i,j = pairs[k]
        p   = min(1.0, s*W[k])
        if rand(rng) < p
            m = rand(rng, LogNormal(μm, σm))
            if rand(rng) < 0.5
                A[i,j] = +m; A[j,i] = -m
            else
                A[i,j] = -m; A[j,i] = +m
            end
        end
    end
    return A
end

# ----- u* generator -----
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    σ = sqrt(log(1 + cv^2)); μ = log(mean) - σ^2/2
    rand(rng, LogNormal(μ,σ), S)
end

# ----- Jacobian & metrics -----
jacobian(A,u) = Diagonal(u) * (A - I)
resilience(J) = maximum(real, eigvals(J))
reactivity(J) = maximum(real, eigvals((J + J')/2))

# ----- Stabilize by shrinking A (keep u* fixed) -----
"""
stabilize_shrink!(A,u; margin=0.05, factor=0.9, max_iter=200)

Scales A ← factor*A until λmax(Diag(u)*(A-I)) ≤ -margin (or iterations exhausted).
Returns (A_stab, shrink_alpha, lambda_max).
"""
function stabilize_shrink!(A::AbstractMatrix, u::AbstractVector; margin=0.05, factor=0.9, max_iter=200)
    @assert 0 < factor < 1
    S = size(A,1); @assert length(u) == S
    α = 1.0
    for _ in 1:max_iter
        J = jacobian(A, u)
        λ = maximum(real, eigvals(J))
        if λ <= -margin
            return (A, α, λ)
        end
        A .*= factor
        α *= factor
    end
    # final check/return even if not reached margin
    J = jacobian(A, u)
    λ = maximum(real, eigvals(J))
    return (A, α, λ)
end

# ----- α_off from J, build J from α & u -----
function alpha_off_from(J,u)
    S = length(u); α = zeros(Float64, S, S)
    @inbounds for i in 1:S, j in 1:S
        if i!=j && J[i,j] != 0.0
            α[i,j] = J[i,j] / u[i]
        end
    end
    α
end

function build_J_from(α::AbstractMatrix, u::AbstractVector)
    S = length(u); J = zeros(Float64, S, S)
    @inbounds for i in 1:S
        J[i,i] = -u[i]
        for j in 1:S
            if i!=j && α[i,j] != 0.0
                J[i,j] = u[i]*α[i,j]
            end
        end
    end
    J
end

# ----- α operators & uniform u -----
function op_rowmean_alpha(α::AbstractMatrix)
    S = size(α,1); out = zeros(Float64, S, S)
    @inbounds for i in 1:S
        mags = Float64[]
        for j in 1:S
            if i!=j && α[i,j]!=0.0; push!(mags, abs(α[i,j])); end
        end
        mi = isempty(mags) ? 0.0 : mean(mags)
        for j in 1:S
            if i!=j && α[i,j]!=0.0
                out[i,j] = sign(α[i,j]) * mi
            end
        end
    end
    out
end

function op_threshold_alpha(α; q=0.20)
    mags = [abs(α[i,j]) for i in 1:size(α,1), j in 1:size(α,2) if i!=j && α[i,j]!=0.0]
    τ = isempty(mags) ? 0.0 : quantile(mags, q)
    S = size(α,1); out = zeros(Float64, S, S)
    @inbounds for i in 1:S, j in 1:S
        if i!=j && abs(α[i,j]) >= τ
            out[i,j] = α[i,j]
        end
    end
    out
end

uniform_u(u) = fill(mean(u), length(u))

# ----- realized structure (post-stabilization) -----
function realized_connectance(A)
    S = size(A,1)
    nz = count(!iszero, A) - count(!iszero, diag(A))
    nz / (S*(S-1))
end
function realized_IS(A)
    mags = [abs(A[i,j]) for i in 1:size(A,1), j in 1:size(A,2) if i!=j && A[i,j]!=0.0]
    isempty(mags) ? 0.0 : mean(mags)
end
function degree_CVs(A)
    S = size(A,1)
    outdeg = [count(j->(j!=i && A[i,j]!=0.0), 1:S) for i in 1:S]
    indeg  = [count(i->(i!=j && A[i,j]!=0.0), 1:S) for j in 1:S]
    und    = falses(S,S)
    @inbounds for i in 1:S, j in 1:S
        if i!=j && (A[i,j]!=0.0 || A[j,i]!=0.0); und[i,j] = true; end
    end
    undeg = [count(und[i,:]) for i in 1:S]
    cv(v) = (m=mean(v); m>0 ? std(v)/m : NaN)
    (deg_cv_out=cv(outdeg), deg_cv_in=cv(indeg), deg_cv_all=cv(undeg))
end

# RNG splitter
@inline function _splitmix64(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
    z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
    z ⊻ (z >>> 31)
end

############################
# 2) Threaded sweep (stable communities)
############################
"""
run_sweep_stable(; ...)

Build A,u*, then **stabilize_shrink!** to ensure λmax(J) ≤ -margin.
Returns a DataFrame with realized structure (post-stabilization),
shrink_alpha, lambda_max, and res/rea for full + 6 steps.
"""
function run_sweep_stable(
    ; modes = [:NT, :TR],
      S_vals = [150],
      conn_vals = 0.05:0.05:0.30,
      mean_abs_vals = [0.05, 0.10, 0.20],
      mag_cv_vals   = [0.4, 0.6, 1.0],
      u_mean_vals   = [0.8, 1.0, 1.2],
      u_cv_vals     = [0.3, 0.5, 0.8],
      # degree heterogeneity
      degree_families = [:uniform, :lognormal, :pareto],
      deg_cv_vals   = [0.0, 0.5, 1.0, 2.0],    # for :lognormal
      deg_pl_alphas = [1.2, 1.5, 2.0, 3.0],    # for :pareto
      # stability controls
      margin = 0.05, shrink_factor = 0.9, max_shrink_iter = 200,
      # sampling
      reps_per_combo = 2,
      seed = 1234, number_of_combinations = 10_000,
      q_thresh = 0.20
)
    genA(mode) = mode === :NT ? build_random_nontrophic : build_random_predation

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

    combos = collect(Iterators.product(
        modes, S_vals, conn_vals, mean_abs_vals, mag_cv_vals,
        u_mean_vals, u_cv_vals, deg_specs, 1:reps_per_combo
    ))
    println("Computing $(number_of_combinations) of $(length(combos)) combinations")

    sel = (length(combos) > number_of_combinations) ?
          sample(combos, number_of_combinations; replace=false) : combos
    println("Using $(length(sel)) combinations")

    base = _splitmix64(UInt64(seed))
    buckets = [Vector{NamedTuple}() for _ in 1:nthreads()]

    Threads.@threads for idx in eachindex(sel)
        (mode, S, conn, mean_abs, mag_cv, u_mean, u_cv, (deg_fam, deg_param), _) = sel[idx]
        rng0 = Random.Xoshiro(_splitmix64(base ⊻ UInt64(idx)))

        rng = Random.Xoshiro(_splitmix64(rand(rng0, UInt64)))

        # ---- build base A & u ----
        A = genA(mode)(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                        degree_family=deg_fam, deg_param=deg_param, rng=rng)
        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
        # u = fill(u_mean, S)

        # ---- stabilize (shrink A; keep u fixed) ----
        A, αshrink, λmax = stabilize_shrink!(A, u; margin=margin, factor=shrink_factor, max_iter=max_shrink_iter)

        # realized structure AFTER stabilization
        conn_real = realized_connectance(A)
        IS_real   = realized_IS(A)
        degs      = degree_CVs(A)
        ucv_real  = (mean(u)>0 ? std(u)/mean(u) : NaN)

        # full J (stable by construction)
        J = jacobian(A, u)

        # α_off and steps
        α      = alpha_off_from(J, u)
        α_row  = op_rowmean_alpha(α)
        α_thr  = op_threshold_alpha(α; q=q_thresh)
        # u_uni  = uniform_u(u)
        u_uni  = uniform_u(fill(u_mean, S))

        J_row      = build_J_from(α_row, u)
        J_thr      = build_J_from(α_thr, u)
        J_uni      = build_J_from(α, u_uni)
        J_thr_row  = build_J_from(op_rowmean_alpha(α_thr), u)
        J_row_uni  = build_J_from(α_row, u_uni)
        J_thr_uni  = build_J_from(α_thr, u_uni)

        push!(buckets[threadid()], (;
            mode, S,
            conn_target=conn, mean_abs, mag_cv,
            u_mean_target=u_mean, u_cv_target=u_cv,
            degree_family = deg_fam, degree_param = deg_param,
            # realized post-stabilization
            conn_real, IS_real, u_cv=ucv_real,
            deg_cv_in=degs.deg_cv_in, deg_cv_out=degs.deg_cv_out, deg_cv_all=degs.deg_cv_all,
            shrink_alpha = αshrink, lambda_max = λmax,
            # full (stable)
            res_full = resilience(J), min_u = minimum(u), diff_res_min_u = -resilience(J) - minimum(u),
            rea_full = reactivity(J),
            # 6 steps
            res_row  = resilience(J_row),      rea_row  = reactivity(J_row),
            res_thr  = resilience(J_thr),      rea_thr  = reactivity(J_thr),
            res_uni  = resilience(J_uni),      rea_uni  = reactivity(J_uni),
            res_thr_row = resilience(J_thr_row),  rea_thr_row = reactivity(J_thr_row),
            res_row_uni = resilience(J_row_uni),  rea_row_uni = reactivity(J_row_uni),
            res_thr_uni = resilience(J_thr_uni),  rea_thr_uni = reactivity(J_thr_uni)
        ))
    end

    DataFrame(vcat(buckets...))
end

############################
# 3) Structural summary (unchanged)
############################
function print_structure_summary(df::DataFrame)
    cols = [
        (:conn_real,  "connectance"),
        (:IS_real,    "IS(mean|A|)"),
        (:u_cv,       "abundance CV"),
        (:deg_cv_in,  "degree CV (in)"),
        (:deg_cv_out, "degree CV (out)"),
        (:deg_cv_all, "degree CV (undirected)")
    ]
    groups = (:mode in names(df)) ? groupby(df, :mode) : [df]
    for g in groups
        hdr = (g isa SubDataFrame) ? "mode=$(only(unique(g.mode)))" : "All"
        println("\n--- ", hdr, " ---")
        for (c,label) in cols
            x = collect(skipmissing(g[!, c])) |> x->filter(isfinite, x)
            if isempty(x)
                println(rpad(label, 26), ": (no data)")
            else
                q = quantile(x, (0.10,0.50,0.90))
                println(rpad(label, 26), ": mean=$(round(mean(x),sigdigits=5))  ",
                        "sd=$(round(std(x),sigdigits=5))  ",
                        "p10=$(round(q[1],sigdigits=5))  med=$(round(q[2],sigdigits=5))  p90=$(round(q[3],sigdigits=5))")
            end
        end
    end
end

# --------------------------
# Minimal example run
# --------------------------
dff = run_sweep_stable(
    ; modes=[:TR], S_vals=[120], conn_vals=0.05:0.05:0.30,
      mean_abs_vals=[0.05,0.15, 0.5, 1.0], mag_cv_vals=[0.4,0.6,1.0,2.0],
      u_mean_vals=[1.0], u_cv_vals=[0.3,0.5,0.8,1.0,2.0,3.0],
      degree_families = [:uniform, :lognormal, :pareto],
      deg_cv_vals   = [0.0, 0.5, 1.0, 2.0],
      deg_pl_alphas = [1.2, 1.5, 2.0, 3.0],
      reps_per_combo=2, seed=42, number_of_combinations=500,
      margin=0.05, shrink_factor=0.9, max_shrink_iter=200, q_thresh=0.20
)
print_structure_summary(dff)

# ----------------------------- plotting: correlations ----------------------------
"""
plot_correlations(df; steps=1:6, metrics=[:res, :rea])

Scatter of Full vs Step k with 1:1 line and R² to y=x.
"""
function plot_correlations(df::DataFrame; steps=["row", "thr", "uni", "thr_row", "row_uni", "thr_uni"], metrics=[:res, :rea], title="")
    labels = Dict(:res=>"Resilience", :rea=>"Reactivity")
    colors = [:steelblue, :orangered]

    fig = Figure(size=(1100, 520))
    Label(fig[0, 1:6], title; fontsize=18, font=:bold, halign=:left)

    for (mi, m) in enumerate(metrics)
        xname = Symbol(m, :_full)
        for (si, s) in enumerate(steps)
            yname = Symbol(m, :_, s)

            xs = df[!, xname]   |> collect
            ys = df[!, yname]   |> collect
            x  = Float64[]; y = Float64[]
            @inbounds for i in eachindex(xs)
                xi=xs[i]
                yi=ys[i]
                if xi isa Real && yi isa Real && isfinite(xi) && isfinite(yi)
                    push!(x, float(xi)); push!(y, float(yi))
                end
            end
            if isempty(x)
                Axis(fig[mi, si]; title="$(labels[m]) — S$s", xgridvisible=false, ygridvisible=false)
                continue
            end

            mn = min(minimum(x), minimum(y))
            mx = max(maximum(x), maximum(y))
            if !(isfinite(mn) && isfinite(mx)) || mn == mx
                c = isfinite(mn) ? mn : 0.0
                pad = max(abs(c)*0.1, 1.0)
                mn, mx = c - pad, c + pad
            end

            ax = Axis(
                fig[mi, si];
                title="$(labels[m]) — Step $s",
                xlabel=string(xname), ylabel=string(yname),
                limits=((mn, mx), (mn, mx)),
                xgridvisible=false, ygridvisible=false,
                xticklabelsize=11, yticklabelsize=11,
                xlabelsize=12, ylabelsize=12,
                titlesize=12
            )

            scatter!(ax, x, y; color=colors[mi], markersize=4, alpha=0.35)
            lines!(ax, [mn, mx], [mn, mx]; color=:black, linestyle=:dash)

            μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
            r2 = sst == 0 ? NaN : 1 - ssr/sst
            isfinite(r2) && text!(ax, "R²=$(round(r2, digits=3))";
                                  position=(mx, mn), align=(:right,:bottom))
        end
    end
    display(fig)
end

# ----------------------------- run both modes, show plots ------------------------
df_stable = filter(row -> row.shrink_alpha > 0.5, dff)
# Non-trophic sweep
df_nt = filter(row -> row.mode == :NT, df_stable)
plot_correlations(df_nt; metrics=[:res, :rea],
                  title="Non-trophic — Full vs 6-steps Jacobian simplifications")

# Trophic sweep
df_tr = filter(row -> row.mode == :TR, df_stable)
plot_correlations(df_tr; metrics=[:res, :rea],
                  title="Trophic — Full vs 6-steps Jacobian simplifications")
###################################################################################
