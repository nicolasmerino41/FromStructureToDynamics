###############################
# ALIGNMENT + IS CONTROL TOOLKIT
###############################
using Random, Statistics, LinearAlgebra, DataFrames

# --- 0) Utilities ---
# R² to the identity line y = x (your earlier convention)
r2_to_identity(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = begin
    @assert length(x) == length(y) "x/y length mismatch"
    n = length(x); n < 3 && return 0.0
    ȳ = mean(y); sst = sum((y .- ȳ).^2)
    sst <= eps() && return 0.0
    ssr = sum((y .- x).^2)
    r2 = max(0.0, 1.0 - ssr/sst)
    return isfinite(r2) && !isnan(r2) ? r2 : 0.0
end

# Average-rank Spearman correlation (no deps)
function _ranks(v::AbstractVector{<:Real})
    idx = sortperm(v; alg=QuickSort)
    r = similar(v, Float64)
    i = 1
    while i ≤ length(v)
        j = i
        while j < length(v) && v[idx[j+1]] == v[idx[j]]; j += 1; end
        ρ = (i + j) / 2.0
        for k in i:j; r[idx[k]] = ρ; end
        i = j + 1
    end
    return r
end
spearman(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = cor(_ranks(x), _ranks(y))

# --- Topology-sensitive spectral cosine + resilience R² panel ---

# Spectral cosine: alignment of u with the leading mode of |α|
spectral_cosine(α::AbstractMatrix{<:Real}, u::AbstractVector{<:Real}) = begin
    S = size(α,1); @assert size(α,2) == S && length(u) == S
    U1 = svd(abs.(α)).U[:, 1]
    num = abs(dot(u, U1))
    den = (norm(u) * norm(U1)) + eps()
    num / den
end

"""
resilience_R2_panel(; makeA, makeu, reps=64, rho_sym=0.5, mag_cv=0.60, seed=42)

Computes resilience predictability vs Full for steps {ushuf, reshuf, rew}, with:
  - R2_id        : R² to the identity line y=x
  - slope, intercept, R2_fit : least-squares affine fit y ≈ a + b x
  - corr_dSpecCos_vs_error   : Spearman correlation between Δcosθ and squared error (y-x)^2

Also returns per-draw spectral cosine (|α| lead singular vector vs u) for Full and each step.

Inputs:
  - makeA(rng) -> A  (your generator)
  - makeu(rng) -> u  (your generator)

Returns:
  - df_r2::DataFrame with columns:
      step, R2_id, slope, intercept, R2_fit, corr_dSpecCos_vs_error
  - df_align::DataFrame with columns:
      draw, spec_cos_full, spec_cos_ushuf, spec_cos_reshuf, spec_cos_rew
"""
function resilience_R2_panel(; makeA::Function, makeu::Function,
                              reps::Int=64, rho_sym::Real=0.5, mag_cv::Real=0.60, seed::Int=42)

    # --- tiny locals (no external deps) ---
    _linfit_stats = function (x::Vector{<:Real}, y::Vector{<:Real})
        @assert length(x) == length(y)
        n = length(x)
        n < 3 && return (NaN, NaN, 0.0)
        μx, μy = mean(x), mean(y)
        vx = sum((x .- μx).^2)
        sst = sum((y .- μy).^2)
        if vx ≤ eps() || sst ≤ eps()
            return (NaN, NaN, 0.0)
        end
        covxy = sum((x .- μx) .* (y .- μy))
        b = covxy / vx
        a = μy - b*μx
        sse = sum((y .- (a .+ b .* x)).^2)
        R2_fit = max(0.0, 1.0 - sse/sst)
        return (b, a, isfinite(R2_fit) ? R2_fit : 0.0)
    end

    _spearman = function (x::Vector{<:Real}, y::Vector{<:Real})
        @assert length(x) == length(y)
        n = length(x)
        n < 3 && return 0.0
        # ranks with average ties
        function _ranks(v)
            idx = sortperm(v)
            r = similar(v, Float64)
            i = 1
            while i ≤ n
                j = i
                while j < n && v[idx[j+1]] == v[idx[j]]; j += 1; end
                ρ = (i + j) / 2.0
                for k in i:j; r[idx[k]] = ρ; end
                i = j + 1
            end
            r
        end
        rx, ry = _ranks(x), _ranks(y)
        vx, vy = var(rx), var(ry)
        (vx ≤ eps() || vy ≤ eps()) && return 0.0
        cor(rx, ry)
    end

    rng0 = Random.Xoshiro(seed)

    # arrays across draws
    x_full = Float64[]   # resilience(Full)
    y_ush  = Float64[]   # resilience(ushuf)
    y_rp   = Float64[]   # resilience(reshuf)
    y_rew  = Float64[]   # resilience(rew)

    cos_full = Float64[]; cos_ush = Float64[]; cos_rp = Float64[]; cos_rew = Float64[]

    for _ in 1:reps
        rng = Random.Xoshiro(rand(rng0, UInt64))
        A = makeA(rng); u = makeu(rng)

        IS0 = realized_IS(A)
        Jf  = jacobian(A, u)
        αf  = alpha_off_from(Jf, u)

        # Full
        push!(x_full, resilience(Jf))
        push!(cos_full, spectral_cosine(αf, u))

        # ushuf (u only)
        u_sh = reshuffle_u(u; rng=rng)
        push!(y_ush, resilience(build_J_from(αf, u_sh)))
        push!(cos_ush, spectral_cosine(αf, u_sh))

        # reshuf (preserve pairs) + IS-match
        αrp = op_reshuffle_preserve_pairs(αf; rng=rng)
        match_IS!(αrp, IS0)
        push!(y_rp, resilience(build_J_from(αrp, u)))
        push!(cos_rp, spectral_cosine(αrp, u))

        # rew (trophic ER) + IS-match
        Arew = build_random_trophic_ER(size(A,1);
                   conn=realized_connectance(A), mean_abs=IS0,
                   mag_cv=mag_cv, rho_sym=rho_sym, rng=rng)
        αrw = copy(Arew); match_IS!(αrw, IS0)
        push!(y_rew, resilience(build_J_from(αrw, u)))
        push!(cos_rew, spectral_cosine(αrw, u))
    end

    # identity R²
    R2_id_ush  = r2_to_identity(x_full, y_ush)
    R2_id_rp   = r2_to_identity(x_full, y_rp)
    R2_id_rew  = r2_to_identity(x_full, y_rew)

    # affine fits
    b_ush, a_ush, R2_fit_ush = _linfit_stats(x_full, y_ush)
    b_rp,  a_rp,  R2_fit_rp  = _linfit_stats(x_full, y_rp)
    b_rew, a_rew, R2_fit_rew = _linfit_stats(x_full, y_rew)

    # Δcosθ vs squared error correlations (per step)
    dcos_ush = cos_ush .- cos_full
    dcos_rp  = cos_rp  .- cos_full
    dcos_rew = cos_rew .- cos_full

    err_ush = (y_ush .- x_full) .^ 2
    err_rp  = (y_rp  .- x_full) .^ 2
    err_rew = (y_rew .- x_full) .^ 2

    corr_ush = _spearman(dcos_ush, err_ush)
    corr_rp  = _spearman(dcos_rp,  err_rp)
    corr_rew = _spearman(dcos_rew, err_rew)

    df_r2 = DataFrame([
        (step="ushuf",  R2_id=R2_id_ush,  slope=b_ush, intercept=a_ush, R2_fit=R2_fit_ush, corr_dSpecCos_vs_error=corr_ush),
        (step="reshuf", R2_id=R2_id_rp,   slope=b_rp,  intercept=a_rp,  R2_fit=R2_fit_rp,  corr_dSpecCos_vs_error=corr_rp),
        (step="rew",    R2_id=R2_id_rew,  slope=b_rew, intercept=a_rew, R2_fit=R2_fit_rew, corr_dSpecCos_vs_error=corr_rew),
    ])

    df_align = DataFrame(
        draw = collect(1:reps),
        spec_cos_full   = cos_full,
        spec_cos_ushuf  = cos_ush,
        spec_cos_reshuf = cos_rp,
        spec_cos_rew    = cos_rew,
    )

    return df_r2, df_align
end

# --- 1) Alignment metrics ---
"""
commutator_kappa(alpha, u) -> κ in [0,∞), scale-free index of u–α misalignment.
κ = ||[Diag(u), α]||_F^2 / (||u||_2^2 * ||α||_F^2) = sum_{i≠j} (u_i - u_j)^2 α_ij^2 / (sum u_i^2 * sum_{i≠j} α_ij^2)
"""
function commutator_kappa(alpha::AbstractMatrix{<:Real}, u::AbstractVector{<:Real})
    S = size(alpha,1); @assert size(alpha,2) == S; @assert length(u) == S
    den_u = sum(abs2, u); den_u ≤ eps() && return 0.0
    den_a = sum(abs2, alpha) - sum(abs2, diag(alpha))
    den_a ≤ eps() && return 0.0
    num = 0.0
    @inbounds for i in 1:S, j in 1:S
        if i != j
            ui_uj = (u[i] - u[j])
            num += (ui_uj*ui_uj) * (alpha[i,j]^2)
        end
    end
    return num / (den_u * den_a)
end

"Row/col absolute-load vs u alignment (Pearson on logs; Spearman alternative shown)."
function load_alignments(alpha::AbstractMatrix{<:Real}, u::AbstractVector{<:Real}; use_spearman::Bool=false)
    S = size(alpha,1)
    rload = [sum(abs, @view alpha[i, :]) - abs(alpha[i,i]) for i in 1:S]
    cload = [sum(abs, @view alpha[:, j]) - abs(alpha[j,j]) for j in 1:S]
    # avoid zeros for log
    ϵ = eps(Float64)
    x = log.(u .+ ϵ); y1 = log.(rload .+ ϵ); y2 = log.(cload .+ ϵ)
    if use_spearman
        return (row=spearman(x, y1), col=spearman(x, y2))
    else
        return (row=cor(x, y1), col=cor(x, y2))
    end
end

"Edge-wise magnitude vs timescale product alignment."
function pair_alignment(alpha::AbstractMatrix{<:Real}, u::AbstractVector{<:Real}; use_spearman::Bool=false)
    xs = Float64[]; ys = Float64[]
    S = size(alpha,1)
    @inbounds for i in 1:S, j in 1:S
        (i == j || alpha[i,j] == 0.0) && continue
        push!(xs, log(abs(alpha[i,j]) + eps()))
        push!(ys, log(u[i]*u[j] + eps()))
    end
    isempty(xs) && return NaN
    use_spearman ? spearman(xs, ys) : cor(xs, ys)
end

# --- 2) IS matching (realized mean |A|) ---
"Scale α in-place to achieve target_IS on A_off (since α == A_off in your definitions)."
function match_IS!(alpha::AbstractMatrix{<:Real}, target_IS::Real)
    cur = mean(abs, alpha[findall(!iszero, alpha)]); cur ≤ eps() && return alpha
    scale = target_IS / cur
    @. alpha = alpha * scale
    return alpha
end

# --- 3) One-draw evaluation across three steps ---
"""
evaluate_steps(A, u; rho_sym, mag_cv, t_short, t_long, weightings)
Returns a NamedTuple with per-step metrics and alignments for:
  :full, :ushuf, :reshuf, :rew
All steps are IS-matched to the Full network's realized IS.
"""
function evaluate_steps(A::AbstractMatrix{<:Real}, u::AbstractVector{<:Real};
                        rho_sym::Real=0.5, mag_cv::Real=0.60,
                        t_short::Real=0.01, t_long::Real=5.0,
                        weightings::Vector{Symbol}=[:biomass, :uniform],
                        rng::AbstractRNG=Random.default_rng())

    S = size(A,1); @assert size(A,2) == S; @assert length(u) == S

    # Full
    IS0 = realized_IS(A)
    J_full = jacobian(A, u)
    α_full = alpha_off_from(J_full, u) # equals A_off

    # Steps
    # 1) ushuffle (u only)
    u_ush = reshuffle_u(u; rng=rng)
    α_ush = copy(α_full)                     # IS unchanged by u-shuffle
    J_ush = build_J_from(α_ush, u_ush)

    # 2) α-reshuffle (preserve pairs), IS-match
    α_rp = op_reshuffle_preserve_pairs(α_full; rng=rng)
    match_IS!(α_rp, IS0)
    J_rp = build_J_from(α_rp, u)

    # 3) rewiring to trophic ER with same conn, then IS-match
    A_rew = build_random_trophic_ER(S;
        conn=realized_connectance(A), mean_abs=IS0, mag_cv=mag_cv, rho_sym=rho_sym, rng=rng)
    # ensure realized IS matches IS0 exactly
    α_rew = copy(A_rew); match_IS!(α_rew, IS0)
    J_rew = build_J_from(α_rew, u)

    # Alignment metrics
    align = Dict{Symbol,NamedTuple}()
    for (lab, α, uu) in ((:full, α_full, u), (:ushuf, α_ush, u_ush),
                         (:reshuf, α_rp, u), (:rew, α_rew, u))
        κ = commutator_kappa(α, uu)
        la = load_alignments(α, uu; use_spearman=true)
        ρp = pair_alignment(α, uu; use_spearman=true)
        align[lab] = (; kappa=κ, row_align=la.row, col_align=la.col, pair_align=ρp)
    end

    # R̃med at two anchors, both weightings
    function _rmed_both(J, uu)
        Dict(w => median_return_rate(J, uu; t=t_short, perturbation=w) for w in weightings),
        Dict(w => median_return_rate(J, uu; t=t_long,  perturbation=w) for w in weightings)
    end
    rS_full, rL_full = _rmed_both(J_full, u)
    rS_ush,  rL_ush  = _rmed_both(J_ush,  u_ush)
    rS_rp,   rL_rp   = _rmed_both(J_rp,   u)
    rS_rew,  rL_rew  = _rmed_both(J_rew,  u)

    return (IS0=IS0,
            align=align,
            rshort = Dict(:full=>rS_full,  :ushuf=>rS_ush,  :reshuf=>rS_rp,  :rew=>rS_rew),
            rlong  = Dict(:full=>rL_full,  :ushuf=>rL_ush,  :reshuf=>rL_rp,  :rew=>rL_rew))
end

# --- 4) Batch runner over many draws + R² vs Full + alignment deltas ---
"""
run_alignment_battery(; makeA, makeu, reps, rho_sym, mag_cv, t_short, t_long, weightings, seed)
- makeA(rng) -> A (your generator for a single draw)
- makeu(rng) -> u
Returns a DataFrame with per-draw metrics + a summary table of R² by step/anchor/weighting.
"""
function run_alignment_battery(; makeA::Function, makeu::Function,
        reps::Int=64, rho_sym::Real=0.5, mag_cv::Real=0.60,
        t_short::Real=0.01, t_long::Real=5.0,
        weightings::Vector{Symbol}=[:biomass, :uniform],
        seed::Int=42)

    rng0 = Random.Xoshiro(seed)
    rows = NamedTuple[]
    # store r̃med to compute R² across draws
    acc_short = Dict{Symbol, Dict{Symbol, Vector{Float64}}}()  # step => weighting => values
    acc_long  = Dict{Symbol, Dict{Symbol, Vector{Float64}}}()
    for step in (:full, :ushuf, :reshuf, :rew)
        acc_short[step] = Dict(w => Float64[] for w in weightings)
        acc_long[step]  = Dict(w => Float64[] for w in weightings)
    end

    for r in 1:reps
        rng = Random.Xoshiro(rand(rng0, UInt64))
        A = makeA(rng); u = makeu(rng)

        out = evaluate_steps(A, u; rho_sym=rho_sym, mag_cv=mag_cv,
                             t_short=t_short, t_long=t_long, weightings=weightings, rng=rng)

        # collect per-draw alignments and r̃med
        for step in (:full, :ushuf, :reshuf, :rew)
            al = out.align[step]
            for w in weightings
                push!(acc_short[step][w], out.rshort[step][w])
                push!(acc_long[step][w],  out.rlong[step][w])
            end
            push!(rows, (; draw=r,
                          step=String(step),
                          IS=out.IS0,
                          kappa=al.kappa,
                          row_align=al.row_align,
                          col_align=al.col_align,
                          pair_align=al.pair_align))
        end
    end

    df_align = DataFrame(rows)

    # Build R² summary vs Full, and Δκ vs predictability-drop correlations
    summ_rows = NamedTuple[]
    for (anchor, acc) in ((:short, acc_short), (:long, acc_long))
        for w in weightings
            x = acc[:full][w]
            for step in (:ushuf, :reshuf, :rew)
                y = acc[step][w]
                r2 = r2_to_identity(x, y)
                # alignment deltas across the same draws
                κ_full = df_align[df_align.step .== "full", :kappa]
                κ_step = df_align[df_align.step .== String(step), :kappa]
                Δκ = κ_step .- κ_full
                # predictability "drop" per draw: squared error to identity (y-x)^2; summarize by mean SE
                drop = (y .- x).^2
                ρs = spearman(Δκ, drop)  # positive means larger misalignment -> larger drop
                push!(summ_rows, (; anchor=String(anchor), weighting=String(w), step=String(step),
                                   R2=r2, spearman_deltaKappa_vs_drop=ρs))
            end
        end
    end
    df_summary = DataFrame(summ_rows)

    return df_align, df_summary
end

# --- choose your ensemble ---
S, conn, mean_abs, mag_cv = 120, 0.10, 0.50, 0.60
deg_fam, deg_param, rho_sym = :lognormal, 0.8, 0.5

makeA = rng -> begin
    A0 = build_niche_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                             degree_family=deg_fam, deg_param=deg_param,
                             rho_sym=rho_sym, rng=rng)
    β = mean_abs / max(realized_IS(A0), eps())   # match IS to mean_abs
    β .* A0
end

makeu = rng -> random_u(S; mean=1.0, cv=0.8, rng=rng)

# --- run the resilience predictability panel ---
df_r2, df_spec =
    resilience_R2_panel(; makeA, makeu, reps=96, rho_sym=rho_sym, mag_cv=mag_cv, seed=20251111)

# --- inspect results ---
show(df_r2; allrows=true, allcols=true)
println()
println(first(df_spec, 6))

df_align, df_summary = run_alignment_battery(; makeA, makeu,
    reps=96, rho_sym=rho_sym, mag_cv=mag_cv, t_short=0.01, t_long=1000.0,
    weightings=[:biomass, :uniform], seed=20251111)

println(first(df_align, 6))
println(df_summary)


df_r2_res, df_spec = resilience_R2_panel(; makeA, makeu, reps=96, rho_sym=0.5, mag_cv=0.60, seed=20251111)
println(df_r2_res)

# Run the same loop once more to keep arrays (copy-paste from resilience_R2_panel’s body)
function _resilience_arrays(; makeA, makeu, reps=96, rho_sym=0.5, mag_cv=0.60, seed=42)
    rng0 = Random.Xoshiro(seed)
    x_full = Float64[]; y_ush = Float64[]; y_rp = Float64[]; y_rew = Float64[]
    for _ in 1:reps
        rng = Random.Xoshiro(rand(rng0, UInt64)); A = makeA(rng); u = makeu(rng)
        IS0 = realized_IS(A); Jf = jacobian(A,u); αf = alpha_off_from(Jf,u)
        push!(x_full, resilience(Jf))
        u_sh = reshuffle_u(u; rng=rng); push!(y_ush, resilience(build_J_from(αf, u_sh)))
        αrp = op_reshuffle_preserve_pairs(αf; rng=rng); match_IS!(αrp, IS0)
        push!(y_rp, resilience(build_J_from(αrp,u)))
        Arew = build_random_trophic_ER(size(A,1); conn=realized_connectance(A), mean_abs=IS0,
                                       mag_cv=mag_cv, rho_sym=rho_sym, rng=rng)
        αrw = copy(Arew); match_IS!(αrw, IS0)
        push!(y_rew, resilience(build_J_from(αrw,u)))
    end
    return (x_full=x_full, y_ush=y_ush, y_rp=y_rp, y_rew=y_rew)
end

arr = _resilience_arrays(; makeA, makeu, reps=96, rho_sym=0.5, mag_cv=0.60, seed=20251111)
println(("var_ush", var(arr.y_ush)), ("var_rew", var(arr.y_rew)))
println(("spearman_ush", spearman(arr.x_full, arr.y_ush)),
        ("spearman_rp",  spearman(arr.x_full, arr.y_rp)),
        ("spearman_rew", spearman(arr.x_full, arr.y_rew)))
