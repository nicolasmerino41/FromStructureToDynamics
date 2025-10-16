######################## DISORDERED PREDATION: NO K RECALC ########################
using Random, Statistics, LinearAlgebra, DataFrames
using DifferentialEquations
using CairoMakie

const EXTINCTION_THRESHOLD = 1e-6

# ------------------------------ gLV core -----------------------------------
function gLV_rhs!(du, u, p, t)
    K, A = p
    Au = A * u
    @inbounds for i in eachindex(u)
        du[i] = u[i] * (K[i] - u[i] + Au[i])
    end
end

jacobian_at(u::AbstractVector, A::AbstractMatrix) = Diagonal(u) * (A - I)

resilience(A::AbstractMatrix, u::AbstractVector) =
    maximum(real, eigvals(jacobian_at(u, A)))

reactivity(A::AbstractMatrix, u::AbstractVector) =
    maximum(real, eigvals((jacobian_at(u, A) + jacobian_at(u, A)')/2))

# --------------------------- random builders --------------------------------
function build_random_predation(S::Int; conn=0.10, mag_cv=0.6,
                                mean_abs=0.1, rng=Random.default_rng())
    A = zeros(Float64, S, S)
    pairs = [(i,j) for i in 1:S for j in (i+1):S]
    E = clamp(round(Int, conn * S*(S-1) / 2), 0, length(pairs))
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

function random_K_u0(S::Int; K_mean=1.0, K_cv=0.3,
                     u_mean=1.0, u_cv=0.5, rng=Random.default_rng())
    σK = sqrt(log(1 + K_cv^2)); μK = log(K_mean) - σK^2/2
    σu = sqrt(log(1 + u_cv^2)); μu = log(u_mean) - σu^2/2
    return rand(rng, LogNormal(μK, σK), S), rand(rng, LogNormal(μu, σu), S)
end

# ------------------------- find equilibrium with fixed K --------------------
"""
steady_state_fixedK(K, A; u0, tmax) -> u*
Try algebraic solve u = (I - A) \\ K; if any component <= 0 or ill-conditioned,
integrate ODE from positive u0 to steady state. Always returns a nonnegative u*.
"""
function steady_state_fixedK(K::Vector{Float64}, A::Matrix{Float64};
                             u0::Vector{Float64}=abs.(K), tmax::Float64=4000.0)
    S = length(K)
    # algebraic attempt
    try
        u = (I - A) \ K
        if all(isfinite, u) && all(>(0.0), u)
            return u
        end
    catch
        # fall through to ODE
    end
    # ODE fallback
    u0 = max.(u0, 1e-6)                 # strictly positive start
    prob = ODEProblem(gLV_rhs!, u0, (0.0, tmax), (K, A))
    sol = solve(prob, Tsit5(); abstol=1e-8, reltol=1e-8,
                save_everystep=false, save_start=false)
    uT = sol.u[end]
    return max.(uT, 0.0)
end

"""
stabilize_shrink_fixedK!(A, K; margin, shrink, max_iter) -> (α, u*, λmax)
Shrink A (A .*= α cumulatively) while keeping K fixed until max Re eig(J(u*,A)) ≤ -margin,
where u* is the steady state for (K, A) at that iteration.
"""
function stabilize_shrink_fixedK!(A::Matrix{Float64}, K::Vector{Float64};
                                  margin::Float64=0.05, shrink::Float64=0.85,
                                  max_iter::Int=60, u0::Vector{Float64}=abs.(K))
    α = 1.0
    u = copy(u0)
    for _ in 1:max_iter
        u = steady_state_fixedK(K, A; u0=u)
        λ = resilience(A, u)
        if isfinite(λ) && λ <= -margin
            return (α, u, λ)
        end
        A .*= shrink
        α *= shrink
    end
    # return whatever we have
    u = steady_state_fixedK(K, A; u0=u)
    return (α, u, resilience(A, u))
end

# --------------------------- step transforms --------------------------------
function active_pairs(A::AbstractMatrix)
    S = size(A,1)
    ps = Tuple{Int,Int}[]
    for i in 1:S, j in (i+1):S
        if A[i,j] != 0.0 || A[j,i] != 0.0
            push!(ps, (i,j))
        end
    end
    return ps
end

function step1_rowmean(A::AbstractMatrix)
    S = size(A,1)
    rmean = zeros(Float64, S)
    value = rand(1:S-1)
    for i in value:(value+1)
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
        sgn = sign(A[i,j]) != 0 ? sign(A[i,j]) : -sign(A[j,i])
        m = 0.5 * (rmean[i] + rmean[j])
        A1[i,j] = sgn * m
        A1[j,i] = -A1[i,j]
    end
    return A1
end

function step2_globalmean(A::AbstractMatrix)
    S = size(A,1)
    mags = Float64[]
    for i in 1:S, j in 1:S
        if i != j && A[i,j] != 0.0
            push!(mags, abs(A[i,j]))
        end
    end
    m̄ = isempty(mags) ? 0.0 : mean(mags)
    A2 = zeros(Float64, S, S)
    for (i,j) in active_pairs(A)
        sgn = sign(A[i,j]) != 0 ? sign(A[i,j]) : -sign(A[j,i])
        A2[i,j] = sgn * m̄
        A2[j,i] = -A2[i,j]
    end
    return A2
end

# ---------------------------- evaluation (NO K CHANGE) ----------------------
relerr(s, f; clip=0.10) = begin
    if !(isfinite(f) && isfinite(s)); return NaN; end
    τ = max(abs(f)*clip, 1e-6)
    abs(s - f) / max(abs(f), τ)
end

function evaluate_instance_disordered(S::Int;
    conn=0.10, mag_cv=0.6, mean_abs=0.1,
    K_mean=1.0, K_cv=0.3, u_mean=1.0, u_cv=0.5,
    margin=0.05, rng=Random.default_rng())

    # draw K,u0 once; K is FROZEN for everything that follows
    K, u0 = random_K_u0(S; K_mean=K_mean, K_cv=K_cv, u_mean=u_mean, u_cv=u_cv, rng=rng)

    # baseline A and stabilization by shrinking A only (K fixed)
    A = build_random_predation(S; conn=conn, mag_cv=mag_cv, mean_abs=mean_abs, rng=rng)
    α, u_full, λmax = stabilize_shrink_fixedK!(A, K; margin=margin, u0=u0)

    # baseline metrics at u_full
    res_full = resilience(A, u_full)
    rea_full = reactivity(A, u_full)

    # Step 1: modify A -> get new u1 from SAME K -> metrics
    A1 = step1_rowmean(A)
    u1 = steady_state_fixedK(K, A1; u0=u_full)
    res_S1 = resilience(A1, u1)
    rea_S1 = reactivity(A1, u1)

    # Step 2: modify A -> get new u2 from SAME K -> metrics
    A2 = step2_globalmean(A)
    u2 = steady_state_fixedK(K, A2; u0=u1)
    res_S2 = resilience(A2, u2)
    rea_S2 = reactivity(A2, u2)

    return (;
        S=S, conn=conn, mag_cv=mag_cv, mean_abs=mean_abs,
        shrink_alpha=α, lambda_max=λmax,
        res_full=res_full, rea_full=rea_full,
        res_S1=res_S1, rea_S1=rea_S1,
        res_S2=res_S2, rea_S2=rea_S2,
        err_res_S1=relerr(res_S1, res_full),
        err_rea_S1=relerr(rea_S1, rea_full),
        err_res_S2=relerr(res_S2, res_full),
        err_rea_S2=relerr(rea_S2, rea_full),
    )
end

# ------------------------------ threaded scan -------------------------------
@inline function _splitmix64(x::UInt64)
    x += 0x9E3779B97F4A7C15
    z = x
    z ⊻= z >>> 30;  z *= 0xBF58476D1CE4E5B9
    z ⊻= z >>> 27;  z *= 0x94D049BB133111EB
    return z ⊻ (z >>> 31)
end

function scan_disordered(; S=150, conn=0.10, mag_cv=0.6, mean_abs=0.1,
                          reps=300, margin=0.05, seed=123, threaded=true)
    base = _splitmix64(UInt64(seed))
    out = [Vector{NamedTuple}() for _ in 1:Threads.nthreads()]

    if threaded && Threads.nthreads() > 1
        Threads.@threads for j in 1:reps
            tid = Threads.threadid()
            rng = Random.Xoshiro(_splitmix64(base ⊻ UInt64(j)))
            push!(out[tid], evaluate_instance_disordered(S;
                conn=conn, mag_cv=mag_cv, mean_abs=mean_abs, margin=margin, rng=rng))
        end
    else
        rng = Random.Xoshiro(base)
        for j in 1:reps
            push!(out[1], evaluate_instance_disordered(S;
                conn=conn, mag_cv=mag_cv, mean_abs=mean_abs, margin=margin, rng=rng))
        end
    end
    return DataFrame(vcat(out...))
end

# --------------------------- correlation plots ------------------------------
function plot_correlations(df::DataFrame; steps=[1,2], metrics=[:res, :rea],
                           resolution=(1100, 700))
    labels = Dict(:res=>"Resilience", :rea=>"Reactivity")
    colors = [:steelblue, :darkorange]

    fig = Figure(size=resolution)
    for (mi, m) in enumerate(metrics)
        xname = Symbol(m, :_full)
        xs = df[!, xname]
        for (si, s) in enumerate(steps)
            yname = Symbol(m, :_S, s)
            ys = df[!, yname]

            x = Float64[]; y = Float64[]
            n = min(length(xs), length(ys))
            for i in 1:n
                xi = xs[i]; yi = ys[i]
                if xi isa Real && yi isa Real && isfinite(xi) && isfinite(yi)
                    push!(x, float(xi)); push!(y, float(yi))
                end
            end
            mn = min(minimum(x), minimum(y))
            mx = max(maximum(x), maximum(y))
            if !(isfinite(mn) && isfinite(mx)) || mn == mx
                c = isfinite(mn) ? mn : 0.0
                pad = max(abs(c)*0.1, 1.0)
                mn, mx = c - pad, c + pad
            end

            ax = Axis(fig[mi, si];
                title="$(labels[m]) — Step $s",
                xlabel=string(xname),
                ylabel=string(yname),
                limits=((mn, mx), (mn, mx)),
                xgridvisible=false, ygridvisible=false)

            scatter!(ax, x, y; color=colors[mi], markersize=4, transparency=true, alpha=0.35)
            lines!(ax, [mn, mx], [mn, mx]; color=:black, linestyle=:dash)

            μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
            r2 = sst == 0 ? NaN : 1 - ssr/sst
            isfinite(r2) && text!(ax, "R²=$(round(r2,digits=3))"; position=(mx, mn), align=(:right,:bottom))
        end
    end
    display(fig)
end

# --------------------------------- run --------------------------------------
df = scan_disordered(; S=150, conn=0.10, mag_cv=0.6, mean_abs=0.1,
                     reps=300, margin=0.05, seed=123, threaded=true)

plot_correlations(df; steps=[1,2], metrics=[:res, :rea], resolution=(1100, 700))
