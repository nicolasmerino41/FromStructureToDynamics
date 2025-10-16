################## ProvingNonTrophicDisorderedSystemsArePredictable.jl ##################
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

# shrink A until max Re(eig(J)) ≤ -margin; return K=(I-A)u0 (baseline only)
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

# robust equilibrium with fixed K (NO K recompute in steps)
function equilibrium_from(K::Vector{Float64}, A::Matrix{Float64};
                          u0::Vector{Float64}, tmax::Float64=1200.0)
    u = try
        (I - A) \ K
    catch
        fill(NaN, length(K))
    end
    if all(isfinite, u) && all(>(0.0), u)
        return u
    end
    prob = ODEProblem(gLV_rhs!, max.(u0, EXTINCTION_THRESHOLD), (0.0, tmax), (K, A))
    sol  = solve(prob, Tsit5(); abstol=1e-7, reltol=1e-7, save_everystep=false)
    uT   = sol.u[end]
    any(!isfinite, uT) && return fill(NaN, length(K))
    return max.(uT, 0.0)
end

# ====================== non-trophic random interaction matrix ======================
"""
build_regular_nontrophic(S; conn, m_abs, pos_frac, rng)

- Exactly k = round(conn*(S-1)) nonzeros in **each row** (row-regular).
- Each nonzero has magnitude m_abs and sign + with prob pos_frac (else −).
- Diagonal = 0.
"""
function build_regular_nontrophic(S::Int; conn::Float64=0.10, m_abs::Float64=0.1,
                                  pos_frac::Float64=0.5, rng=Random.default_rng())
    A = zeros(Float64, S, S)
    k = clamp(round(Int, conn * (S-1)), 0, S-1)
    @inbounds for i in 1:S
        # choose exactly k distinct targets j ≠ i
        pool = [j for j in 1:S if j != i]
        k == 0 && continue
        js = rand(rng, pool, k)
        for j in js
            s = (rand(rng) < pos_frac) ? 1.0 : -1.0
            A[i,j] = s * rand(Normal(m_abs, m_abs))
        end
    end
    return A
end
"""
build_random_nontrophic(S; conn, mag_cv, mean_abs, pos_frac, rng)

Off-diagonal entries:
- with prob `conn`, set A[i,j] = s * m   (s ∈ {+1, -1}, P(s=+1)=pos_frac; m>0 LogNormal(mean_abs, mag_cv))
- with prob (1-conn), A[i,j] = 0
Diagonal is zero. No pair constraint (A[i,j] and A[j,i] independent).
"""
function build_random_nontrophic(S::Int; conn::Float64=0.10, mag_cv::Float64=0.6,
                                 mean_abs::Float64=0.1, pos_frac::Float64=0.5,
                                 rng=Random.default_rng())
    A = zeros(Float64, S, S)
    σ = sqrt(log(1 + mag_cv^2))
    μ = log(mean_abs) - σ^2/2
    @inbounds for i in 1:S, j in 1:S
        i == j && continue
        if rand(rng) < conn
            m = rand(rng, LogNormal(μ, σ))
            s = (rand(rng) < pos_frac) ? 1.0 : -1.0
            A[i,j] = s * m
        end
    end
    return A
end

# coarse stats from an arbitrary (non-trophic) A
function coarse_stats_nontrophic(A::AbstractMatrix)
    S = size(A,1)
    mags = Float64[]; npos = 0; nact = 0
    @inbounds for i in 1:S, j in 1:S
        i == j && continue
        a = A[i,j]
        if a != 0.0
            push!(mags, abs(a))
            npos += (a > 0.0)
            nact += 1
        end
    end
    total = S*(S-1)
    conn = total == 0 ? 0.0 : nact / total
    if isempty(mags)
        return (conn=0.0, mean_abs=0.0, mag_cv=0.0, pos_frac=0.5)
    else
        μ = mean(mags); σ = std(mags); cv = μ == 0 ? 0.0 : σ/μ
        pf = nact == 0 ? 0.5 : npos / nact
        return (conn=conn, mean_abs=μ, mag_cv=cv, pos_frac=pf)
    end
end

# random K and u0
function random_K_u0(S::Int; K_mean::Float64=1.0, K_cv::Float64=0.3,
                     u_mean::Float64=1.0, u_cv::Float64=0.5, rng=Random.default_rng())
    σK = sqrt(log(1 + K_cv^2)); μK = log(K_mean) - σK^2/2
    σu = sqrt(log(1 + u_cv^2)); μu = log(u_mean) - σu^2/2
    return rand(rng, LogNormal(μK, σK), S), rand(rng, LogNormal(μu, σu), S)
end
# perfectly homogeneous K,u0
function homogeneous_K_u0(S::Int; K_val::Float64=1.0, u_val::Float64=1.0)
    return fill(K_val, S), fill(u_val, S)
end

# ============================== metrics =====================================
resilience(A::AbstractMatrix, u::AbstractVector) = maximum(real, eigvals(jacobian_at(u, A)))
reactivity(A::AbstractMatrix, u::AbstractVector) = begin
    J = jacobian_at(u, A)
    maximum(real, eigvals((J + J')/2))
end

# ========================= step transforms (entrywise) ======================
# Step 1: row-mean |A| magnitudes; keep each entry’s original sign
function step1_rowmean(A::AbstractMatrix)
    S = size(A,1)
    rmean = zeros(Float64, S)
    # value = rand(1:S-1)
    # @inbounds for i in value:(value+1)
    @inbounds for i in 1:S
        mags = Float64[]
        for j in 1:S
            i == j && continue
            a = A[i,j]
            a == 0.0 && continue
            push!(mags, abs(a))
        end
        rmean[i] = isempty(mags) ? 0.0 : mean(mags)
    end
    A1 = zeros(Float64, S, S)
    @inbounds for i in 1:S, j in 1:S
        i == j && continue
        a = A[i,j]
        if a != 0.0
            A1[i,j] = sign(a) * rmean[i]
        end
    end
    return A1
end
#FAKE
# function step1_rowmean(A::AbstractMatrix)
#     S = size(A,1)
#     rmean = zeros(Float64, S)
#     value = rand(1:S-1)
#     @inbounds for i in 1:S
#         mags = Float64[]
#         for j in 1:S
#             i == j && continue
#             a = A[i,j]
#             a == 0.0 && continue
#             push!(mags, abs(a))
#         end
#         rmean[i] = isempty(mags) ? 0.0 : mean(mags)
#     end
#     A1 = zeros(Float64, S, S)
#     @inbounds for i in 1:S, j in 1:S
#         i == j && continue
#         a = A[i,j]
#         if a != 0.0
#             A1[i,j] = sign(a) * abs(a*rmean[i]*0.01)
#         end
#     end
#     return A1
# end

# Step 2: global-mean |A| across all active off-diagonals; keep original signs
function step2_globalmean(A::AbstractMatrix)
    S = size(A,1)
    mags = Float64[]
    @inbounds for i in 1:S, j in 1:S
        i == j && continue
        a = A[i,j]
        a == 0.0 && continue
        push!(mags, abs(a))
    end
    m̄ = isempty(mags) ? 0.0 : mean(mags)
    A2 = zeros(Float64, S, S)
    @inbounds for i in 1:S, j in 1:S
        i == j && continue
        a = A[i,j]
        if a != 0.0
            A2[i,j] = sign(a) * m̄
        end
    end
    return A2
end

# Step 3: redraw a fresh A′ with the SAME coarse stats (conn, mean|A|, CV|A|, pos_frac)
function step3_resampled(A::AbstractMatrix; rng=Random.default_rng())
    S = size(A,1)
    cs = coarse_stats_nontrophic(A)
    # return build_random_nontrophic(S; conn=cs.conn, mag_cv=cs.mag_cv,
                                #    mean_abs=cs.mean_abs, pos_frac=cs.pos_frac, rng=rng)
    return build_regular_nontrophic(S; conn=cs.conn, m_abs=cs.mean_abs, pos_frac=cs.pos_frac, rng=rng)
end

# ============================ evaluation of one instance =====================
relerr(s, f; clip=0.10) = begin
    if !(isfinite(f) && isfinite(s)); return NaN; end
    τ = max(abs(f)*clip, 1e-6)
    abs(s - f) / max(abs(f), τ)
end

"""
evaluate_instance_nontrophic(S; ...)

Baseline: build A, draw K,u0, shrink A (only) so u0 stable, then set K=(I-A)u0.
Steps: modify A → A_S, keep K fixed; compute u*_S from (K, A_S); metrics at u*_S.
"""
function evaluate_instance_nontrophic(S::Int;
    conn=0.10, mag_cv=0.6, mean_abs=0.1, pos_frac=0.5,
    K_mean=1.0, K_cv=0.3, u_mean=1.0, u_cv=0.5,
    margin=0.05, rng=Random.default_rng())

    # baseline
    # A = build_random_nontrophic(S; conn=conn, mag_cv=mag_cv, mean_abs=mean_abs, pos_frac=pos_frac, rng=rng)
    A = build_regular_nontrophic(S; conn=conn, m_abs=mean_abs, pos_frac=pos_frac, rng=rng)

    # optional tiny heterogeneity
    A .+= 1e-3 .* mean_abs .* (randn(rng, size(A)) .* (A .!= 0.0))

    K0, u0 = homogeneous_K_u0(S; K_val=K_mean, u_val=u_mean)
    K, α, λmax = stabilize_shrink!(A, u0; margin=margin)

    u_full = u0  # by construction
    res_full = resilience(A, u_full)
    rea_full = reactivity(A, u_full)

    # Step 1
    A1 = step1_rowmean(A)
    u_S1 = equilibrium_from(K, A1; u0=u_full)
    res_S1 = resilience(A1, u_S1)
    rea_S1 = reactivity(A1, u_S1)

    # Step 2
    A2 = step2_globalmean(A)
    u_S2 = equilibrium_from(K, A2; u0=u_full)
    res_S2 = resilience(A2, u_S2)
    rea_S2 = reactivity(A2, u_S2)

    # Step 3
    A3 = step3_resampled(A; rng=rng)
    u_S3 = equilibrium_from(K, A3; u0=u_full)
    res_S3 = resilience(A3, u_S3)
    rea_S3 = reactivity(A3, u_S3)

    return (;
        S=S, conn=conn, mag_cv=mag_cv, mean_abs=mean_abs, pos_frac=pos_frac,
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

function scan_nontrophic(; S=150, conn=0.10, mag_cv=0.6, mean_abs=0.1, pos_frac=0.5,
                          reps=300, margin=0.05, seed=777, threaded=true)
    base = _splitmix64(UInt64(seed))
    out_per_thread = [Vector{NamedTuple}() for _ in 1:nthreads()]

    if threaded && nthreads() > 1
        Threads.@threads for j in 1:reps
            tid = threadid()
            rng = Random.Xoshiro(_splitmix64(base ⊻ UInt64(j)))
            row = evaluate_instance_nontrophic(S; conn=conn, mag_cv=mag_cv,
                                               mean_abs=mean_abs, pos_frac=pos_frac,
                                               margin=margin, rng=rng)
            push!(out_per_thread[tid], row)
        end
    else
        rng = Random.Xoshiro(base)
        for j in 1:reps
            row = evaluate_instance_nontrophic(S; conn=conn, mag_cv=mag_cv,
                                               mean_abs=mean_abs, pos_frac=pos_frac,
                                               margin=margin, rng=rng)
            push!(out_per_thread[1], row)
        end
    end
    return DataFrame(vcat(out_per_thread...))
end

# ========================== predictability correlations =======================
function plot_correlations(df::DataFrame; steps=[1,2,3], metrics=[:res, :rea],
                           resolution=(1200, 520))
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

            ax = Axis(fig[mi, si]; title="$(labels[m]) — Step $s",
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
                isfinite(r2) && text!(ax, "R²=$(round(r2, digits=3))";
                                      position=(mx, mn), align=(:right,:bottom))
            end
        end
    end
    display(fig)
end

# =================================== RUN ====================================
df = scan_nontrophic(; S=50, conn=0.10, mag_cv=0.6, mean_abs=0.1, pos_frac=0.5,
                     reps=300, margin=0.05, seed=2026, threaded=true)

plot_correlations(df; steps=[1,2,3], metrics=[:res, :rea], resolution=(1200, 520))
