############################ DISORDERED PREDATION: BASELINE + STEPS (+ Step 3) ############################
using Random, Statistics, LinearAlgebra, DataFrames
using DifferentialEquations
using CairoMakie

# =============================== core gLV ===================================
const EXTINCTION_THRESHOLD = 1e-6

function gLV_rhs!(du, u, p, t)
    K, A = p
    Au = A * u
    @inbounds for i in eachindex(u)
        du[i] = u[i] * (K[i] - u[i] + Au[i])
    end
end

jacobian_at(u::AbstractVector, A::AbstractMatrix) = Diagonal(u) * (A - I)

# shrink A until max Re(eig(J)) ≤ -margin; returns K=(I-A)u0 so u0 is equilibrium for baseline only
function stabilize_shrink!(A::Matrix{Float64}, u0::Vector{Float64};
                           margin::Float64=0.05, shrink::Float64=0.85, max_iter::Int=60)
    α = 1.0
    for _ in 1:max_iter
        λmax = maximum(real, eigvals(jacobian_at(u0, A)))
        if λmax <= -margin
            return (I - A) * u0, α, λmax
        end
        A .*= shrink
        α *= shrink
    end
    return (I - A) * u0, α, maximum(real, eigvals(jacobian_at(u0, A)))
end

# robust equilibrium solver with fixed K (NO K recompute)
function equilibrium_from(K::Vector{Float64}, A::Matrix{Float64};
                          u0::Vector{Float64}, tmax::Float64=1200.0)
    # try linear solve first
    u = try
        (I - A) \ K
    catch
        fill(NaN, length(K))
    end
    if all(isfinite, u) && all(>(0.0), u)
        return u
    end
    # fallback: integrate ODE to steady
    prob = ODEProblem(gLV_rhs!, max.(u0, EXTINCTION_THRESHOLD), (0.0, tmax), (K, A))
    sol  = solve(prob, Tsit5(); abstol=1e-7, reltol=1e-7, save_everystep=false)
    uT   = sol.u[end]
    any(!isfinite, uT) && return fill(NaN, length(K))
    return max.(uT, 0.0)
end

# ========================== random predation (no groups) ====================
# Build antisymmetric predation: for each active unordered pair, set (Aij, Aji) = (+m, -m) or (-m, +m)
function build_random_predation(S::Int; conn::Float64=0.10, mag_cv::Float64=0.6,
                                mean_abs::Float64=0.1, rng=Random.default_rng())
    A = zeros(Float64, S, S)
    pairs = [(i,j) for i in 1:S for j in (i+1):S]
    E = clamp(round(Int, conn * S*(S-1)/2), 0, length(pairs))
    sel = rand(rng, 1:length(pairs), E)

    σ = sqrt(log(1 + mag_cv^2))
    μ = log(mean_abs) - σ^2/2

    for idx in sel
        i, j = pairs[idx]
        m = rand(rng, LogNormal(μ, σ))
        if rand(rng) < 0.5
            A[i,j] = +m; A[j,i] = -m
        else
            A[i,j] = -m; A[j,i] = +m
        end
    end
    return A
end

# Empirical coarse stats of an antisymmetric predation matrix
function coarse_stats(A::AbstractMatrix)
    S = size(A,1)
    act = Float64[]
    for i in 1:S, j in (i+1):S
        if A[i,j] != 0.0 || A[j,i] != 0.0
            push!(act, abs(A[i,j]) > 0 ? abs(A[i,j]) : abs(A[j,i]))
        end
    end
    total_pairs = S*(S-1)/2
    E = length(act)
    conn = E / total_pairs
    if isempty(act)
        return (conn=0.0, mean_abs=0.0, mag_cv=0.0)
    else
        μ = mean(act)
        σ = std(act)
        cv = μ == 0 ? 0.0 : σ/μ
        return (conn=conn, mean_abs=μ, mag_cv=cv)
    end
end

# random positive K and u0
function random_K_u0(S::Int; K_mean::Float64=1.0, K_cv::Float64=0.3,
                     u_mean::Float64=1.0, u_cv::Float64=0.5, rng=Random.default_rng())
    σK = sqrt(log(1 + K_cv^2)); μK = log(K_mean) - σK^2/2
    σu = sqrt(log(1 + u_cv^2)); μu = log(u_mean) - σu^2/2
    return rand(rng, LogNormal(μK, σK), S), rand(rng, LogNormal(μu, σu), S)
end

# ============================== metrics =====================================
resilience(A::AbstractMatrix, u::AbstractVector) = maximum(real, eigvals(jacobian_at(u, A)))

# For antisymmetric pairwise predation, (A + A')/2 = 0 on active pairs ⇒ reactivity = -min(u*)
reactivity_from_u(u::AbstractVector) = -minimum(u)

# ========================= step transforms (preserve pair antisymmetry) =====
# list of active unordered pairs
function active_pairs(A::AbstractMatrix)
    S = size(A,1)
    out = Tuple{Int,Int}[]
    for i in 1:S, j in (i+1):S
        if A[i,j] != 0.0 || A[j,i] != 0.0
            push!(out, (i,j))
        end
    end
    return out
end

# Step 1: row-mean magnitudes, pair-consistent (keep original pair signs)
function step1_rowmean(A::AbstractMatrix)
    S = size(A,1)
    rmean = zeros(Float64, S)
    for i in 1:S
        mags = Float64[]
        for j in 1:S
            if i != j && A[i,j] != 0.0
                push!(mags, abs(A[i,j]))
            end
        end
        rmean[i] = isempty(mags) ? 0.0 : mean(mags)
    end
    A1 = zeros(Float64, S, S)
    for (i,j) in active_pairs(A)
        sgn = A[i,j] != 0.0 ? sign(A[i,j]) : -sign(A[j,i])
        m = 0.5*(rmean[i] + rmean[j])
        A1[i,j] = sgn*m
        A1[j,i] = -A1[i,j]
    end
    return A1
end

# Step 2: global mean magnitude (keep original pair signs)
function step2_globalmean(A::AbstractMatrix)
    mags = Float64[]
    S = size(A,1)
    for i in 1:S, j in (i+1):S
        if A[i,j] != 0.0 || A[j,i] != 0.0
            push!(mags, abs(A[i,j])>0 ? abs(A[i,j]) : abs(A[j,i]))
        end
    end
    m̄ = isempty(mags) ? 0.0 : mean(mags)
    A2 = zeros(Float64, S, S)
    for (i,j) in active_pairs(A)
        sgn = A[i,j] != 0.0 ? sign(A[i,j]) : -sign(A[j,i])
        A2[i,j] = sgn*m̄
        A2[j,i] = -A2[i,j]
    end
    return A2
end

# Step 3: redraw a fresh A' from the same coarse-grained stats as baseline A
function step3_resampled(A::AbstractMatrix; rng=Random.default_rng())
    S = size(A,1)
    cs = coarse_stats(A)  # (conn, mean_abs, mag_cv)
    return build_random_predation(S; conn=cs.conn, mag_cv=cs.mag_cv, mean_abs=cs.mean_abs, rng=rng)
end

# ============================ evaluation of one instance =====================
relerr(s, f; clip=0.10) = begin
    if !(isfinite(f) && isfinite(s)); return NaN; end
    τ = max(abs(f)*clip, 1e-6)
    abs(s - f) / max(abs(f), τ)
end

"""
evaluate_instance_disordered(...)

Baseline: build A, draw K,u0, shrink A (only) so u0 is stable, and set K=(I-A)u0.
Steps 1–3: modify A → A_S, keep that same K, compute u*_S from (K, A_S), then metrics at u*_S.
"""
function evaluate_instance_disordered(S::Int;
    conn=0.10, mag_cv=0.6, mean_abs=0.1,
    K_mean=1.0, K_cv=0.3, u_mean=1.0, u_cv=0.5,
    margin=0.05, rng=Random.default_rng())

    # baseline draw
    A = build_random_predation(S; conn=conn, mag_cv=mag_cv, mean_abs=mean_abs, rng=rng)
    K0, u0 = random_K_u0(S; K_mean=K_mean, K_cv=K_cv, u_mean=u_mean, u_cv=u_cv, rng=rng)

    # stabilize baseline by shrinking A (not K), then define K := (I - A)u0  (baseline only)
    K, α, λmax = stabilize_shrink!(A, u0; margin=margin)

    # equilibrium & metrics at baseline
    u_full = u0  # by construction
    res_full = resilience(A, u_full)
    rea_full = reactivity_from_u(u_full)

    # Step 1
    A1 = step1_rowmean(A)
    u_S1 = equilibrium_from(K, A1; u0=u_full)
    res_S1 = resilience(A1, u_S1)
    rea_S1 = reactivity_from_u(u_S1)

    # Step 2
    A2 = step2_globalmean(A)
    u_S2 = equilibrium_from(K, A2; u0=u_full)
    res_S2 = resilience(A2, u_S2)
    rea_S2 = reactivity_from_u(u_S2)

    # Step 3 (fresh redraw with same coarse stats)
    A3 = step3_resampled(A; rng=rng)
    u_S3 = equilibrium_from(K, A3; u0=u_full)
    res_S3 = resilience(A3, u_S3)
    rea_S3 = reactivity_from_u(u_S3)

    return (;
        S=S, conn=conn, mag_cv=mag_cv, mean_abs=mean_abs,
        shrink_alpha=α, lambda_max=λmax,
        res_full=res_full, rea_full=rea_full,
        res_S1=res_S1,  rea_S1=rea_S1,
        res_S2=res_S2,  rea_S2=rea_S2,
        res_S3=res_S3,  rea_S3=rea_S3,
        err_res_S1 = relerr(res_S1, res_full),
        err_rea_S1 = relerr(rea_S1, rea_full),
        err_res_S2 = relerr(res_S2, res_full),
        err_rea_S2 = relerr(rea_S2, rea_full),
        err_res_S3 = relerr(res_S3, res_full),
        err_rea_S3 = relerr(rea_S3, rea_full),
    )
end

# ============================== threaded scan =================================
@inline function _splitmix64(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
    z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
    return z ⊻ (z >>> 31)
end

function scan_disordered(; S=150, conn=0.10, mag_cv=0.6, mean_abs=0.1,
                          reps=300, margin=0.05, seed=42, threaded=true)
    base = _splitmix64(UInt64(seed))
    out_per_thread = [Vector{NamedTuple}() for _ in 1:Threads.nthreads()]

    if threaded && Threads.nthreads() > 1
        Threads.@threads for j in 1:reps
            tid = Threads.threadid()
            rng = Random.Xoshiro(_splitmix64(base ⊻ UInt64(j)))
            row = evaluate_instance_disordered(S; conn=conn, mag_cv=mag_cv,
                                               mean_abs=mean_abs, margin=margin, rng=rng)
            push!(out_per_thread[tid], row)
        end
    else
        rng = Random.Xoshiro(base)
        for j in 1:reps
            row = evaluate_instance_disordered(S; conn=conn, mag_cv=mag_cv,
                                               mean_abs=mean_abs, margin=margin, rng=rng)
            push!(out_per_thread[1], row)
        end
    end
    return DataFrame(vcat(out_per_thread...))
end

# ========================== predictability correlations =======================
function plot_correlations(df::DataFrame; steps=[1,2,3], metrics=[:res, :rea],
                           resolution=(1200, 520))
    labels = Dict(:res=>"Resilience", :rea=>"Reactivity")
    colors = [:steelblue, :darkorange, :seagreen]

    fig = Figure(size=resolution)
    for (mi, m) in enumerate(metrics)
        xname = Symbol(m, :_full)
        xs = df[!, xname]
        for (si, s) in enumerate(steps)
            yname = Symbol(m, :_S, s)
            ys = df[!, yname]

            # finite pairs
            x = Float64[]; y = Float64[]
            n = min(length(xs), length(ys))
            for i in 1:n
                xi = xs[i]; yi = ys[i]
                if xi isa Real && yi isa Real && isfinite(xi) && isfinite(yi)
                    push!(x, float(xi)); push!(y, float(yi))
                end
            end
            ax = Axis(fig[mi, si];
                title="$(labels[m]) — Step $s",
                xlabel=string(xname), ylabel=string(yname),
                xgridvisible=false, ygridvisible=false)

            if !isempty(x)
                mn = min(minimum(x), minimum(y)); mx = max(maximum(x), maximum(y))
                if !(isfinite(mn) && isfinite(mx)) || mn == mx
                    c = isfinite(mn) ? mn : 0.0; pad = max(abs(c)*0.1, 1.0); mn, mx = c-pad, c+pad
                end
                lines!(ax, [mn, mx], [mn, mx]; color=:black, linestyle=:dash)
                scatter!(ax, x, y; color=colors[mi], markersize=4, transparency=true, alpha=0.35)
                μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
                r2 = sst == 0 ? NaN : 1 - ssr/sst
                isfinite(r2) && text!(ax, "R²=$(round(r2, digits=3))"; position=(mx, mn), align=(:right,:bottom))
            end
        end
    end
    display(fig)
end

################################################################################
# Example run
df = scan_disordered(
    ; S=150, conn=0.10, mag_cv=0.6, mean_abs=0.1,
    reps=300, margin=0.05, seed=123, threaded=true
)

plot_correlations(df; steps=[1,2,3], metrics=[:res, :rea], resolution=(1200, 520))
