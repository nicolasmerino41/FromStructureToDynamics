################ THRESHOLDS: mod, gap_A, degree-CV (global + restricted) ################
# ---------- helpers: column handling (robust to Symbol/String names) ----------
# Return true if a column named `name` (Symbol or String) exists.
col_in(df::DataFrame, name) = begin
    nms = names(df)
    (name isa Symbol && (name in nms)) ||
    (name isa String && (name in String.(nms))) ||
    (name isa Symbol && (String(name) in String.(nms))) ||
    (name isa String && (Symbol(name) in nms))
end

# Return the canonical column key to index df[!, key] if present, else nothing.
function col_key(df::DataFrame, name)
    nms = names(df)
    if name isa Symbol
        if name in nms; return name end
        s = String(name); for k in nms; if String(k) == s; return k end; end
    else
        s = String(name)
        for k in nms; if String(k) == s; return k end; end
    end
    return nothing
end

# ---------- tiny helper: safe numeric extraction -> Float64 or nothing ----------
@inline function _safe_real(x)
    x isa Real || return nothing
    xf = float(x)
    isfinite(xf) ? xf : nothing
end

# ---------- discover available steps by pattern *_S{step} (any metric) ----------
function detect_steps(df::DataFrame; metrics = [:resilience, :reactivity, :rt_pulse, :after_press])
    found = Int[]
    txt = String.(names(df))
    for s in 1:8
        pat = "_S$(s)"
        okm = any( (string(m) * pat) in txt for m in metrics )
        okm && push!(found, s)
    end
    # fallback: accept any *_S{step} occurrence if specific metrics missing
    if isempty(found)
        for s in 1:8
            pat = "_S$(s)"
            any(endswith(n, pat) for n in txt) && push!(found, s)
        end
        found = unique(found)
    end
    # prefer 2, then 1, then ascending remainder
    pref = [2, 1]
    rest = setdiff(sort(unique(found)), pref)
    return vcat(intersect(pref, found), rest)
end

# ---------- clipped relative-error composite ----------
function add_err_clip_mean!(df::DataFrame;
    step::Union{Int,Symbol} = :auto,
    metrics = [:resilience, :reactivity, :rt_pulse, :after_press],
    clip_frac::Float64 = 0.10
)
    # choose step
    s_used::Union{Int,Nothing} = nothing
    if step === :auto
        steps = detect_steps(df; metrics=metrics)
        isempty(steps) && error("No *_S{step} columns found in DataFrame.")
        s_used = first(steps)
        @info "Using step=$(s_used) (auto-detected)"
    else
        s_used = Int(step)
        # if none of the requested metrics have this step, fall back
        if !any(col_in(df, Symbol("$(m)_S", s_used)) for m in metrics)
            @warn "Requested step=$(s_used) not present for given metrics; falling back to auto."
            steps = detect_steps(df; metrics=metrics)
            isempty(steps) && error("No *_S{step} columns found in DataFrame.")
            s_used = first(steps)
            @info "Using step=$(s_used) (auto fallback)"
        end
    end

    # keep only metrics that actually exist (both *_full and *_S{s_used})
    avail = Symbol[]
    for m in metrics
        fullk = col_key(df, Symbol("$(m)_full"))
        stepk = col_key(df, Symbol("$(m)_S", s_used))
        if fullk !== nothing && stepk !== nothing
            push!(avail, m)
        end
    end
    isempty(avail) && error("None of the requested metrics exist for step=$(s_used).")

    # build clipped rel error per metric
    for m in avail
        fullk = col_key(df, Symbol("$(m)_full"))
        stepk = col_key(df, Symbol("$(m)_S", s_used))

        # robust median scale for clipping
        tmp = Float64[]
        for v in df[!, fullk]
            sv = _safe_real(v); isnothing(sv) && continue
            push!(tmp, abs(sv))
        end
        τ = isempty(tmp) ? 1e-6 : quantile(tmp, 0.50) * clip_frac

        rel = Vector{Float64}(undef, nrow(df))
        for i in 1:nrow(df)
            f = _safe_real(df[i, fullk])
            s = _safe_real(df[i, stepk])
            rel[i] = (isnothing(f) || isnothing(s)) ? NaN : abs(s - f) / max(abs(f), τ + 1e-12)
        end
        df[!, Symbol("err_clip_", m)] = rel
    end

    # row-wise mean over the available err_clip_* columns
    cols = Symbol.("err_clip_" .* String.(avail))
    err_mean = Vector{Float64}(undef, nrow(df))
    for i in 1:nrow(df)
        vals = Float64[]
        for c in cols
            ck = col_key(df, c)
            ck === nothing && continue
            v = _safe_real(df[i, ck]); isnothing(v) && continue
            push!(vals, v)
        end
        err_mean[i] = isempty(vals) ? NaN : mean(vals)
    end
    df.err_clip_mean = err_mean
    return s_used
end

# ---------- 1D change-point over binned, trimmed means ----------
function changepoint_1d(xv, yv; n_bins::Int = 12, trim_frac::Float64 = 0.10)
    # collect finite Float64 pairs
    x = Float64[]; y = Float64[]
    n = min(length(xv), length(yv))
    for i in 1:n
        sx = _safe_real(xv[i]); sy = _safe_real(yv[i])
        (isnothing(sx) || isnothing(sy)) && continue
        push!(x, sx); push!(y, sy)
    end
    isempty(x) && return (NaN, Float64[], Float64[], NaN)

    ord = sortperm(x); x = x[ord]; y = y[ord]
    N = length(x); chunk = max(1, fld(N, n_bins))
    bx = Float64[]; by = Float64[]

    for b in 1:n_bins
        lo = (b - 1) * chunk + 1
        hi = b == n_bins ? N : min(b * chunk, N)
        if lo <= hi
            push!(bx, mean(x[lo:hi]))
            seg = copy(y[lo:hi]); sort!(seg)
            m = length(seg)
            if m == 0
                push!(by, NaN)
            else
                t = clamp(floor(Int, trim_frac * m), 0, max(0, (m - 1) ÷ 2))
                push!(by, mean(seg[(t + 1):(m - t)]))
            end
        end
    end

    # drop NaN bins
    good = [isfinite(v) for v in by]
    nb_good = count(identity, good)
    nb_good < 4 && return (NaN, bx, by, NaN)
    bx = [bx[i] for i in eachindex(bx) if good[i]]
    by = [by[i] for i in eachindex(by) if good[i]]
    nb = length(bx)

    best_k, best_sse = 2, Inf
    for k in 2:(nb - 1)
        μ1 = mean(by[1:k]); μ2 = mean(by[(k + 1):nb])
        sse = sum((by[1:k] .- μ1).^2) + sum((by[(k + 1):nb] .- μ2).^2)
        if sse < best_sse
            best_sse = sse; best_k = k
        end
    end
    x_thr = bx[best_k]
    return (x_thr, bx, by, best_k)
end

# ---------- fallback: compute consumer out-degree CV if missing ----------
function ensure_degcv_cons_out!(df::DataFrame)
    if col_in(df, :deg_cv_cons_out_realized)
        return df
    end
    degcv = Vector{Float64}(undef, nrow(df))
    for (i,row) in enumerate(eachrow(df))
        K, A = row.p_final
        n = size(A, 1)
        # infer R: maximize negatives in resource->consumer block
        bestR, bestScore = 1, -1
        for Rt in 1:(n-1)
            blk = A[1:Rt, (Rt+1):n]
            score = count(<(0.0), blk)
            if score > bestScore
                bestScore = score; bestR = Rt
            end
        end
        R = bestR; C = n - R
        d = Vector{Int}(undef, C)
        for ic in 1:C
            d[ic] = count(!iszero, A[R + ic, 1:R])
        end
        μ = mean(d)
        degcv[i] = (isempty(d) || μ == 0) ? NaN : std(d) / μ
    end
    df[!, :deg_cv_cons_out_realized] = degcv
    return df
end

# ---------- log one threshold ----------
function log_threshold(df::DataFrame, axis::Symbol; axis_label::String = String(axis),
                       n_bins::Int = 12, trim_frac::Float64 = 0.10, prefix::String = "")
    axk = col_key(df, axis)
    if axk === nothing
        @warn "Missing column $(axis); skipping."
        return NaN
    end
    thr, bx, by, k = changepoint_1d(df[!, axk], df.err_clip_mean; n_bins=n_bins, trim_frac=trim_frac)
    if isfinite(thr)
        @info "$(prefix)Estimated threshold $(axis_label) ≈ $(round(thr, digits=3))"
    else
        @warn "$(prefix)Could not estimate threshold for $(axis_label)"
    end
    return thr
end

# ---------- main driver ----------
function threshold_suite!(df::DataFrame;
    step::Union{Int,Symbol} = :auto,
    metrics = [:resilience, :reactivity, :rt_pulse, :after_press],
    n_bins::Int = 12, trim_frac::Float64 = 0.10,
    conn_low::Float64 = 0.08, conn_high::Float64 = 0.12,
    S_target::Int = 150
)
    ensure_degcv_cons_out!(df)
    if !col_in(df, :gap_A)
        @warn "add_community_diagnostics! not found (gap_A missing). Run it first for gap-based thresholds."
    end

    s_used = add_err_clip_mean!(df; step=step, metrics=metrics, clip_frac=0.10)
    @info "Thresholds computed for step=$(s_used)"

    # GLOBAL thresholds
    log_threshold(df, :realized_mod; axis_label="realized_mod",
                  n_bins=n_bins, trim_frac=trim_frac, prefix="[Global] ")
    log_threshold(df, :gap_A; axis_label="gap_A (1−ρ(A))",
                  n_bins=n_bins, trim_frac=trim_frac, prefix="[Global] ")
    log_threshold(df, :deg_cv_cons_out_realized; axis_label="deg_cv_cons_out_realized",
                  n_bins=n_bins, trim_frac=trim_frac, prefix="[Global] ")

    # RESTRICTED SLICE
    if !col_in(df, :conn)
        @warn "Missing :conn for slicing; skipping restricted slice."
        return nothing
    end
    connk = col_key(df, :conn); Sk = col_key(df, :S)
    mask = (df[!, connk] .>= conn_low) .& (df[!, connk] .<= conn_high) .& (df[!, Sk] .== S_target)
    df_slice = df[mask, :]
    if nrow(df_slice) < 20
        @warn "Restricted slice too small (n=$(nrow(df_slice))). Adjust filters."
        return nothing
    end
    @info "[Slice] Using $(nrow(df_slice)) rows with $(conn_low) ≤ conn ≤ $(conn_high), S=$(S_target)"

    log_threshold(df_slice, :realized_mod; axis_label="realized_mod",
                  n_bins=n_bins, trim_frac=trim_frac, prefix="[Slice] ")
    log_threshold(df_slice, :gap_A; axis_label="gap_A (1−ρ(A))",
                  n_bins=n_bins, trim_frac=trim_frac, prefix="[Slice] ")
    log_threshold(df_slice, :deg_cv_cons_out_realized; axis_label="deg_cv_cons_out_realized",
                  n_bins=n_bins, trim_frac=trim_frac, prefix="[Slice] ")

    return nothing
end

# ---------------------- run it ----------------------
# df should be your DataFrame (e.g., `sim_results`) with diagnostics already added.
# Example:
# df = sim_results
# add_community_diagnostics!(df; k_consumer_cutoff=0.01)

threshold_suite!(
    df;
    step = :4,  # prefers 2, then 1
    metrics = [:resilience, :reactivity, :rt_pulse, :after_press],
    n_bins = 12,
    trim_frac = 0.10,
    conn_low = 0.08,
    conn_high = 0.12,
    S_target = 150
)
