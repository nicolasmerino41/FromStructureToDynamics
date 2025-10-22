############################ FULLY BIPARTITE: BASELINE + STEPS ############################
using Random, Statistics, LinearAlgebra, DataFrames
using DifferentialEquations
using CairoMakie
using Base.Threads

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

# shrink A until max Re(eig(J)) ≤ -margin; keep u* by recomputing K=(I-A)u
function stabilize_shrink!(A::Matrix{Float64}, u::Vector{Float64};
                           margin::Float64=0.05, shrink::Float64=0.85, max_iter::Int=60)
    α = 1.0
    for _ in 1:max_iter
        λmax = maximum(real, eigvals(jacobian_at(u, A)))
        if λmax <= -margin
            return (I - A) * u, α, λmax
        end
        A .*= shrink
        α *= shrink
    end
    return (I - A) * u, α, maximum(real, eigvals(jacobian_at(u, A)))
end

# ========================= fully bipartite predation ========================
"""
build_random_bipartite(S, R; conn, mag_cv, mean_abs, rng)

- Two layers: resources 1..R, consumers R+1..S.
- Activate E ≈ conn * R * C consumer→resource links uniformly at random.
- For each active pair (i=consumer, j=resource) draw magnitude m>0 (LogNormal)
  and set A[i,j]=+m, A[j,i]=-m. No within-layer edges. No self-links.
"""
function build_random_bipartite(S::Int, R::Int; conn::Float64=0.10, mag_cv::Float64=0.6,
                                mean_abs::Float64=0.1, rng=Random.default_rng())
    @assert 1 ≤ R < S
    C = S - R
    A = zeros(Float64, S, S)

    # edge budget
    E = clamp(round(Int, conn * R * C), 0, R*C)

    # uniform random selection of consumer–resource pairs
    # encode pairs as linear indices (ic, jr)
    pairs = [(ic, jr) for ic in 1:C, jr in 1:R] |> vec
    sel = rand(rng, 1:length(pairs), E)

    σ = sqrt(log(1 + mag_cv^2))
    μ = log(mean_abs) - σ^2/2

    @inbounds for idx in sel
        ic, jr = pairs[idx]
        i = R + ic       # consumer row
        j = jr           # resource col
        m = rand(rng, LogNormal(μ, σ))
        A[i, j] =  m     # consumer benefits
        A[j, i] = -m     # resource suffers
    end
    return A
end

# random positive K and u0 (no extra structure here)
function random_K_u0(S::Int; K_mean::Float64=1.0, K_cv::Float64=0.3,
                     u_mean::Float64=1.0, u_cv::Float64=0.5, rng=Random.default_rng())
    σK = sqrt(log(1 + K_cv^2)); μK = log(K_mean) - σK^2/2
    σu = sqrt(log(1 + u_cv^2)); μu = log(u_mean) - σu^2/2
    return rand(rng, LogNormal(μK, σK), S), rand(rng, LogNormal(μu, σu), S)
end

# ============================== metrics =====================================
resilience(A::AbstractMatrix, u::AbstractVector) = maximum(real, eigvals(jacobian_at(u, A)))
reactivity(A::AbstractMatrix, u::AbstractVector) = maximum(real, eigvals((jacobian_at(u, A) + jacobian_at(u, A)')/2))

# pulse return time to final (robust)
function mean_return_time_pulse(u0, K, A; δ=0.1, tol=0.10, tcap=5_000.0)
    # rough time-scale from linearization
    λ = maximum(real, eigvals(Diagonal(u0) * (A - I)))
    τ = (isfinite(λ) && λ < 0) ? 50.0/abs(λ) : 800.0            # generous factor
    tmax = clamp(τ, 400.0, tcap)

    pulsed = u0 .* (1 .- δ)
    prob = ODEProblem(gLV_rhs!, pulsed, (0.0, tmax), (K, A))
    sol = solve(prob, Tsit5(); abstol=1e-6, reltol=1e-6, save_everystep=false)

    uT = sol.u[end]
    # first time each coord is within tol of its final value
    hits = Float64[]
    @inbounds for i in eachindex(uT)
        target = uT[i]
        hit = NaN
        for (t, u) in zip(sol.t, sol.u)
            if abs(u[i] - target) / (abs(target) + 1e-12) < tol
                hit = t; break
            end
        end
        push!(hits, hit)
    end
    vals = filter(isfinite, hits)
    return isempty(vals) ? NaN : mean(vals)
end

# return time based on state-norm distance to the final state
function return_time_norm(u0, K, A; δ=0.15, tol=0.10, tcap=5_000.0)
    pulsed = u0 .* (1 .- δ)

    # adaptive horizon from linear time scale (but with generous ceiling)
    J = Diagonal(u0) * (A - I)
    λ = maximum(real, eigvals(J))
    τ = (isfinite(λ) && λ < 0) ? 60.0/abs(λ) : 800.0
    tmax = clamp(τ, 600.0, tcap)

    prob = ODEProblem(gLV_rhs!, pulsed, (0.0, tmax), (K, A))
    sol  = solve(prob, Tsit5(); abstol=1e-6, reltol=1e-6, save_everystep=false)

    u★ = sol.u[end]
    denom = norm(u★) + 1e-12
    # first time the whole state is within tol of the final state
    for (t, u) in zip(sol.t, sol.u)
        if norm(u .- u★) / denom ≤ tol
            return t
        end
    end
    return NaN   # didn’t settle within tol before tmax
end

# ========================= step transforms (preserve CR sign) ===============
# Active consumer–resource pairs from a bipartite A
function active_CR_pairs(A::AbstractMatrix, R::Int)
    S = size(A,1); C = S - R
    pairs = Tuple{Int,Int}[]
    @inbounds for ic in 1:C, jr in 1:R
        i = R + ic; j = jr
        if A[i,j] != 0.0 || A[j,i] != 0.0
            push!(pairs, (i,j))
        end
    end
    return pairs
end

# Step 1: row-mean |A| magnitudes, pair-consistent averaging of i, j rows
function step1_rowmean_bip(A::AbstractMatrix, R::Int)
    S = size(A,1)
    rmean = zeros(Float64, S)
    @inbounds for i in 1:S
        mags = Float64[]
        for j in 1:S
            if i != j && A[i,j] != 0.0
                push!(mags, abs(A[i,j]))
            end
        end
        rmean[i] = isempty(mags) ? 0.0 : mean(mags)
    end
    A1 = zeros(Float64, S, S)
    for (i,j) in active_CR_pairs(A, R)
        # sign must be consumer→resource positive
        m = 0.5 * (rmean[i] + rmean[j])
        A1[i,j] =  m
        A1[j,i] = -m
    end
    return A1
end

# Step 2: single global-mean magnitude across all active pairs
function step2_globalmean_bip(A::AbstractMatrix, R::Int)
    S = size(A,1)
    mags = Float64[]
    @inbounds for (i,j) in active_CR_pairs(A, R)
        push!(mags, abs(A[i,j])>0 ? abs(A[i,j]) : abs(A[j,i]))
    end
    m̄ = isempty(mags) ? 0.0 : mean(mags)
    A2 = zeros(Float64, S, S)
    for (i,j) in active_CR_pairs(A, R)
        A2[i,j] =  m̄
        A2[j,i] = -m̄
    end
    return A2
end

# ============================ evaluation of one instance =====================
relerr(s, f; clip=0.10) = begin
    if !(isfinite(f) && isfinite(s)); return NaN; end
    τ = max(abs(f)*clip, 1e-6)
    abs(s - f) / max(abs(f), τ)
end

"""
evaluate_instance_bipartite(S, R; ...)

Build fully bipartite A, stabilize (shrink) under u0, compute metrics; then apply Step1/2
and set K_s=(I-A_s)u0 so u0 stays equilibrium. Returns a NamedTuple row.
"""
function evaluate_instance_bipartite(S::Int, R::Int;
    conn=0.10, mag_cv=0.6, mean_abs=0.1,
    K_mean=1.0, K_cv=0.3, u_mean=1.0, u_cv=0.5,
    margin=0.05, rng=Random.default_rng())

    A = build_random_bipartite(S, R; conn=conn, mag_cv=mag_cv, mean_abs=mean_abs, rng=rng)
    K0, u0 = random_K_u0(S; K_mean=K_mean, K_cv=K_cv, u_mean=u_mean, u_cv=u_cv, rng=rng)

    # stabilize baseline
    K, α, λmax = stabilize_shrink!(A, u0; margin=margin)

    # baseline
    res_full = resilience(A, u0)
    rea_full = reactivity(A, u0)
    # rt_full  = mean_return_time_pulse(u0, K, A; δ=0.2)
    rt_full = return_time_norm(u0, K, A; δ=0.15, tol=0.10)
    # Step 1
    A1 = step1_rowmean_bip(A, R)
    K1 = (I - A1) * u0
    res_S1 = resilience(A1, u0)
    rea_S1 = reactivity(A1, u0)
    # rt_S1  = mean_return_time_pulse(u0, K1, A1; δ=0.2)
    rt_S1   = return_time_norm(u0, K1, A1; δ=0.15, tol=0.10)
    
    # Step 2
    A2 = step2_globalmean_bip(A, R)
    K2 = (I - A2) * u0
    res_S2 = resilience(A2, u0)
    rea_S2 = reactivity(A2, u0)
    # rt_S2  = mean_return_time_pulse(u0, K2, A2; δ=0.2)
    rt_S2   = return_time_norm(u0, K2, A2; δ=0.15, tol=0.10)

    return (;
        S=S, R=R, C=S-R, conn=conn, mag_cv=mag_cv, mean_abs=mean_abs,
        shrink_alpha=α, lambda_max=λmax,
        res_full=res_full, rea_full=rea_full, rt_full=rt_full,
        res_S1=res_S1,  rea_S1=rea_S1,  rt_S1=rt_S1,
        res_S2=res_S2,  rea_S2=rea_S2,  rt_S2=rt_S2,
        err_res_S1 = relerr(res_S1, res_full),
        err_rea_S1 = relerr(rea_S1, rea_full),
        err_rt_S1  = relerr(rt_S1,  rt_full),
        err_res_S2 = relerr(res_S2, res_full),
        err_rea_S2 = relerr(rea_S2, rea_full),
        err_rt_S2  = relerr(rt_S2,  rt_full),
    )
end

# ============================== threaded scan ===============================
@inline function _splitmix64(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
    z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
    return z ⊻ (z >>> 31)
end

function scan_bipartite(; S=150, R=75, conn=0.10, mag_cv=0.6, mean_abs=0.1,
                         reps=300, margin=0.05, seed=123, threaded=true)
    base = _splitmix64(UInt64(seed))
    buckets = [Vector{NamedTuple}() for _ in 1:nthreads()]

    if threaded && nthreads() > 1
        Threads.@threads for j in 1:reps
            tid = threadid()
            rng = Random.Xoshiro(_splitmix64(base ⊻ UInt64(j)))
            row = evaluate_instance_bipartite(S, R; conn=conn, mag_cv=mag_cv,
                                              mean_abs=mean_abs, margin=margin, rng=rng)
            push!(buckets[tid], row)
        end
    else
        rng = Random.Xoshiro(base)
        for j in 1:reps
            row = evaluate_instance_bipartite(S, R; conn=conn, mag_cv=mag_cv,
                                              mean_abs=mean_abs, margin=margin, rng=rng)
            push!(buckets[1], row)
        end
    end
    return DataFrame(vcat(buckets...))
end

# ========================== predictability correlations ======================
function plot_correlations(df::DataFrame; steps=[1,2], metrics=[:res, :rea, :rt],
                           resolution=(1100, 700))
    labels = Dict(:res=>"Resilience", :rea=>"Reactivity", :rt=>"Return time")
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
            
            varx = var(x); vary = var(y)
            if (varx + vary) < 1e-10
                c = mean(vcat(x, y))
                pad = max(abs(c)*0.1, 1.0)
                mn, mx = c - pad, c + pad
                # tiny jitter to avoid exact overlap with y=x
                ϵ = max(1e-6, 1e-3*pad)
                y = y .+ (rand(length(y)) .- 0.5) .* ϵ
            else
                mn = min(minimum(x), minimum(y))
                mx = max(maximum(x), maximum(y))
            end
            
            ax = Axis(fig[mi, si];
                title="$(labels[m]) — Step $s",
                xlabel=string(xname), ylabel=string(yname),
                xgridvisible=false, ygridvisible=false)
            if isempty(x)
                continue
            end
            # if the data are (near) constant, expand limits and jitter a hair so points are visible

            scatter!(ax, x, y; markersize=5, alpha=0.5)   # draw points first
            lines!(ax, [mn, mx], [mn, mx]; color=:black, linestyle=:dash)  # line behind


            μy = mean(y)
            sst = sum((y .- μy).^2)
            ssr = sum((y .- x).^2)
            r2 = sst == 0 ? NaN : 1 - ssr/sst
            isfinite(r2) && text!(ax, "R²=$(round(r2, digits=3))"; position=(mx, mn),
                                  align=(:right,:bottom))
        end
    end
    display(fig)
end

######################################## RUN ########################################
df_bip = scan_bipartite(; S=50, R=30, conn=0.10, mag_cv=0.6, mean_abs=0.1,
                        reps=300, margin=0.05, seed=2025, threaded=true)

plot_correlations(df_bip; steps=[1,2], metrics=[:res, :rea, :rt], resolution=(1100, 700))
