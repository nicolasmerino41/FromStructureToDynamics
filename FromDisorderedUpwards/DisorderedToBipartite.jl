########################### STRUCTURE AXIS: disorder → bipartite ###########################
using Random, Statistics, LinearAlgebra, DataFrames
using DifferentialEquations
using CairoMakie

# --- reuse these from your disordered script (or keep here if standalone) ---
const EXTINCTION_THRESHOLD = 1e-8
function gLV_rhs!(du, u, p, t)
    K, A = p
    Au = A * u
    @inbounds for i in eachindex(u)
        du[i] = u[i] * (K[i] - u[i] + Au[i])
    end
end
jacobian_at(u,A) = Diagonal(u) * (A - I)
resilience(A,u)  = maximum(real, eigvals(jacobian_at(u,A)))
reactivity(A,u)  = maximum(real, eigvals((jacobian_at(u,A) + jacobian_at(u,A)')/2))
relerr(s,f; clip=0.10) = (isfinite(f)&&isfinite(s)) ? abs(s-f)/max(abs(f),max(1e-6,clip*abs(f))) : NaN

function mean_return_time_pulse(u0,K,A; δ=0.2, tspan=(0.0,400.0))
    pulsed = u0 .* (1 .- δ)
    sol = solve(ODEProblem(gLV_rhs!, pulsed, (0.0, tspan[2]), (K, A)),
                Tsit5(); abstol=1e-6, reltol=1e-6, save_everystep=false, save_start=false)
    uT = sol.u[end]
    hits = Float64[]
    for i in eachindex(uT)
        target = uT[i]; t_hit = NaN
        for (t,u) in zip(sol.t, sol.u)
            if abs(u[i]-target)/ (abs(target)+1e-12) < 0.10; t_hit=t; break; end
        end
        push!(hits, t_hit)
    end
    vals = filter(isfinite, hits)
    return isempty(vals) ? NaN : mean(vals)
end

function stabilize_shrink!(A::Matrix{Float64}, u::Vector{Float64};
                           margin::Float64=0.05, shrink::Float64=0.85, max_iter::Int=60)
    α = 1.0
    for _ in 1:max_iter
        λ = maximum(real, eigvals(jacobian_at(u,A)))
        λ <= -margin && return ( (I-A)*u, α, λ )
        A .*= shrink; α *= shrink
    end
    return ( (I-A)*u, α, maximum(real, eigvals(jacobian_at(u,A))) )
end

# -------------------------- structure-controlled generator ---------------------------
"""
build_bipartivity_axis(S; conn, s in [0,1], mag_cv, mean_abs, rng)

- Partition species into two equal blocks (A,B).
- Total edges E ≈ conn * S*(S-1)/2 (unordered pairs).
- Fraction across blocks f_across = 0.5 + 0.5*s  (0.5=random; 1.0=purely bipartite).
- Sample round(E*f_across) pairs from across-block set and the rest within-block.
- For each chosen pair {i,j}, draw |Aij| from LogNormal(mean_abs, cv) and assign antisymmetric signs at random.
"""
function build_bipartivity_axis(S::Int; conn=0.10, s::Float64=0.0,
                                mag_cv=0.6, mean_abs=0.1, rng=Random.default_rng())
    A = zeros(Float64, S, S)
    g = fill(1, S); g[div(S,2)+1:end] .= 2  # two layers
    within = Tuple{Int,Int}[]; across = Tuple{Int,Int}[]
    for i in 1:S, j in (i+1):S
        if g[i]==g[j]; push!(within,(i,j)) else push!(across,(i,j)) end
    end
    E = clamp(round(Int, conn * S*(S-1)/2), 0, length(within)+length(across))
    frac_across = 0.5 + 0.5*s
    Ea = clamp(round(Int, frac_across * E), 0, length(across))
    Ew = clamp(E - Ea, 0, length(within))

    sel_a = rand(rng, 1:length(across), Ea)
    sel_w = rand(rng, 1:length(within), Ew)

    σ = sqrt(log(1 + mag_cv^2))
    μ = log(mean_abs) - σ^2/2

    for idx in sel_a
        i,j = across[idx]
        m = rand(rng, LogNormal(μ,σ))
        if rand(rng) < 0.5; A[i,j]=+m; A[j,i]=-m else; A[i,j]=-m; A[j,i]=+m end
    end
    for idx in sel_w
        i,j = within[idx]
        m = rand(rng, LogNormal(μ,σ))
        if rand(rng) < 0.5; A[i,j]=+m; A[j,i]=-m else; A[i,j]=-m; A[j,i]=+m end
    end

    # realized structure summaries (for plotting later)
    E_real = count(!iszero, A) ÷ 2
    Ea_real = 0
    for i in 1:S, j in (i+1):S
        if A[i,j]!=0.0 || A[j,i]!=0.0
            Ea_real += (g[i]!=g[j]) ? 1 : 0
        end
    end
    bipartivity_real = E_real==0 ? NaN : Ea_real / E_real

    # degree CV (undirected neighbors)
    deg = zeros(Int, S)
    for i in 1:S, j in (i+1):S
        if A[i,j]!=0.0 || A[j,i]!=0.0; deg[i]+=1; deg[j]+=1 end
    end
    μd = mean(deg); degcv = μd==0 ? NaN : std(deg)/μd

    return A, bipartivity_real, degcv
end

# ------------------------ steps (keep antisymmetry, as before) -------------------------
function _active_pairs(A)
    S=size(A,1); ps=Tuple{Int,Int}[]
    for i in 1:S, j in (i+1):S
        (A[i,j]!=0.0 || A[j,i]!=0.0) && push!(ps,(i,j))
    end
    ps
end
function step1_rowmean(A)
    S=size(A,1)
    rmean=zeros(Float64,S)
    for i in 1:S
        mags=Float64[]
        for j in 1:S
            if i!=j && A[i,j]!=0.0; push!(mags,abs(A[i,j])) end
        end
        rmean[i]=isempty(mags) ? 0.0 : mean(mags)
    end
    A1=zeros(Float64,S,S)
    for (i,j) in _active_pairs(A)
        sgn = (A[i,j]!=0.0) ? sign(A[i,j]) : -sign(A[j,i])
        m = 0.5*(rmean[i]+rmean[j])
        A1[i,j]=sgn*m; A1[j,i]=-A1[i,j]
    end
    A1
end
function step2_globalmean(A)
    S=size(A,1); mags=Float64[]
    for i in 1:S, j in 1:S
        if i!=j && A[i,j]!=0.0; push!(mags,abs(A[i,j])) end
    end
    m̄ = isempty(mags) ? 0.0 : mean(mags)
    A2=zeros(Float64,S,S)
    for (i,j) in _active_pairs(A)
        sgn = (A[i,j]!=0.0) ? sign(A[i,j]) : -sign(A[j,i])
        A2[i,j]=sgn*m̄; A2[j,i]=-A2[i,j]
    end
    A2
end

# ------------------------ one evaluation at given s --------------------------
function evaluate_structured_instance(S::Int; conn=0.10, s=0.0,
    mag_cv=0.6, mean_abs=0.1, K_mean=1.0, K_cv=0.3, u_mean=1.0, u_cv=0.5,
    margin=0.05, rng=Random.default_rng())

    A, b_real, degcv = build_bipartivity_axis(S; conn=conn, s=s, mag_cv=mag_cv, mean_abs=mean_abs, rng=rng)

    # random K,u; stabilize baseline by shrinking A only
    σK = sqrt(log(1 + K_cv^2)); μK = log(K_mean) - σK^2/2
    σu = sqrt(log(1 + u_cv^2)); μu = log(u_mean) - σu^2/2
    K0 = rand(rng, LogNormal(μK,σK), S); u0 = rand(rng, LogNormal(μu,σu), S)
    K, α, λ = stabilize_shrink!(A, u0; margin=margin)

    resF = resilience(A,u0); reaF = reactivity(A,u0); rtF = mean_return_time_pulse(u0,K,A)

    A1 = step1_rowmean(A); K1 = (I-A1)*u0
    res1 = resilience(A1,u0); rea1 = reactivity(A1,u0); rt1 = mean_return_time_pulse(u0,K1,A1)

    A2 = step2_globalmean(A); K2 = (I-A2)*u0
    res2 = resilience(A2,u0); rea2 = reactivity(A2,u0); rt2 = mean_return_time_pulse(u0,K2,A2)

    return (s=s, bipartivity=b_real, degcv=degcv,
            res_full=resF, rea_full=reaF, rt_full=rtF,
            res_S1=res1, rea_S1=rea1, rt_S1=rt1,
            res_S2=res2, rea_S2=rea2, rt_S2=rt2)
end

# ------------------------ threaded scan along s ------------------------------
@inline _mix(x::UInt64) = begin
    y=x+0x9E3779B97F4A7C15; y ⊻= y>>>30; y*=0xBF58476D1CE4E5B9; y ⊻= y>>>27; y*=0x94D049BB133111EB; y ⊻= y>>>31
end

function scan_structure_axis(; S=150, conn=0.10, s_grid=0.0:0.05:1.0,
                              reps_per_s=100, margin=0.05, seed=7, threaded=true)
    base=_mix(UInt64(seed))
    per_thread=[Vector{NamedTuple}() for _ in 1:Threads.nthreads()]
    if threaded && Threads.nthreads()>1
        Threads.@threads for idx in eachindex(s_grid)
            s = s_grid[idx]
            tid=Threads.threadid()
            rng = Random.Xoshiro(_mix(base ⊻ UInt64(idx)))
            # use independent rngs per replicate
            for r in 1:reps_per_s
                row = evaluate_structured_instance(S; conn=conn, s=s, margin=margin,
                                                   rng = Random.Xoshiro(rand(rng, UInt64)))
                push!(per_thread[tid], row)
            end
        end
    else
        rng = Random.Xoshiro(base)
        for (idx,s) in enumerate(s_grid)
            for r in 1:reps_per_s
                row = evaluate_structured_instance(S; conn=conn, s=s, margin=margin,
                                                   rng = Random.Xoshiro(rand(rng, UInt64)))
                push!(per_thread[1], row)
            end
        end
    end
    DataFrame(vcat(per_thread...))
end

# ------------------------ summarize by s & plot ------------------------------
# R² to the 1:1 line for a metric at a given s
_r2_1to1(x,y) = begin
    xf=Float64[]; yf=Float64[]
    n=min(length(x),length(y))
    for i in 1:n
        xi=x[i]; yi=y[i]
        if xi isa Real && yi isa Real && isfinite(xi) && isfinite(yi)
            push!(xf,float(xi)); push!(yf,float(yi))
        end
    end
    isempty(xf) && return NaN
    μy=mean(yf); sst=sum((yf .- μy).^2); ssr=sum((yf .- xf).^2)
    sst==0 ? NaN : 1 - ssr/sst
end

function summarize_by_s(df::DataFrame)
    g = groupby(df, :s)
    rows = NamedTuple[]
    for sub in g
        push!(rows, (; s = first(sub.s),
                       R2_res_S1 = _r2_1to1(sub.res_full, sub.res_S1),
                       R2_res_S2 = _r2_1to1(sub.res_full, sub.res_S2),
                       R2_rea_S1 = _r2_1to1(sub.rea_full, sub.rea_S1),
                       R2_rea_S2 = _r2_1to1(sub.rea_full, sub.rea_S2),
                       bipartivity = mean(skipmissing(sub.bipartivity)),
                       degcv = mean(skipmissing(sub.degcv)) ))
    end
    sort!(DataFrame(rows), :s)
end

function plot_structure_vs_predictability(summary::DataFrame; resolution=(1100,700))
    fig = Figure(size=resolution)

    # top: R² vs s
    ax1 = Axis(
        fig[1,1]; title="Predictability (R² to y=x) vs structure s",
        xlabel="structure s (0=random → 1=bipartite)", ylabel="R²",
        # ylimits=(0,1),
        xgridvisible=false, ygridvisible=false
    )
    lines!(ax1, summary.s, summary.R2_res_S1; label="Resilience S1")
    lines!(ax1, summary.s, summary.R2_res_S2; label="Resilience S2")
    lines!(ax1, summary.s, summary.R2_rea_S1; label="Reactivity S1")
    lines!(ax1, summary.s, summary.R2_rea_S2; label="Reactivity S2")
    axislegend(ax1; position=:lb, framevisible=false)

    # bottom: realized structure for context
    ax2 = Axis(fig[2,1]; title="Realized structure along s",
               xlabel="structure s", ylabel="value",
               xgridvisible=false, ygridvisible=false)
    lines!(ax2, summary.s, summary.bipartivity; label="across-block fraction (bipartivity)")
    lines!(ax2, summary.s, summary.degcv;      label="degree CV (undirected)")
    axislegend(ax2; position=:rt, framevisible=false)

    display(fig)
end

function plot_predictability_points(summary::DataFrame; resolution=(1100,700))
    fig = Figure(size = resolution)

    # Top panel: predictability vs structure (points only)
    ax1 = Axis(fig[1, 1];
        title = "Predictability (R² to y=x) vs structure s",
        xlabel = "structure s (0=random → 1=bipartite)",
        ylabel = "R²",
        # ylimits = (0, 1),
        xgridvisible = false,
        ygridvisible = false
    )

    scatter!(ax1, summary.s, summary.R2_res_S1; label = "Resilience S1",
             color = :dodgerblue, markersize = 8, alpha = 0.7)
    scatter!(ax1, summary.s, summary.R2_res_S2; label = "Resilience S2",
             color = :seagreen, markersize = 8, alpha = 0.7)
    scatter!(ax1, summary.s, summary.R2_rea_S1; label = "Reactivity S1",
             color = :darkorange, markersize = 8, alpha = 0.7)
    scatter!(ax1, summary.s, summary.R2_rea_S2; label = "Reactivity S2",
             color = :firebrick, markersize = 8, alpha = 0.7)

    axislegend(ax1; position = :lb, framevisible = false)

    # Bottom panel: realized structural metrics (points only)
    ax2 = Axis(fig[2, 1];
        title = "Realized structure along s",
        xlabel = "structure s",
        ylabel = "value",
        xgridvisible = false,
        ygridvisible = false
    )

    scatter!(ax2, summary.s, summary.bipartivity; label = "across-block fraction (bipartivity)",
             color = :royalblue, markersize = 8, alpha = 0.7)
    scatter!(ax2, summary.s, summary.degcv; label = "degree CV (undirected)",
             color = :limegreen, markersize = 8, alpha = 0.7)

    axislegend(ax2; position = :rt, framevisible = false)

    display(fig)
end

################################################################################
# ------------------------------- example run ---------------------------------
df_s = scan_structure_axis(; S=150, conn=0.05, s_grid=0.0:0.05:1.0, reps_per_s=80,
                            margin=0.05, seed=123, threaded=true)
summ = summarize_by_s(df_s)
plot_structure_vs_predictability(summ; resolution=(1100,700))
fig = plot_predictability_points(summ)
################################################################################