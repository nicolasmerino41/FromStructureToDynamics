using LinearAlgebra, Random, Statistics
using CairoMakie

# ============================================================
# 0) R̃med(t) — overflow-safe via real-Schur + spectral shift
# ============================================================
function compute_rmed_series_stable(J::AbstractMatrix{<:Real},
                                   u::AbstractVector{<:Real},
                                   t_vals::AbstractVector{<:Real};
                                   perturb::Symbol=:biomass,
                                   margin::Float64=1e-6)
    F = schur(Matrix{Float64}(J))     # J = Z*T*Z'
    Z, T, vals = F.Z, F.T, F.values

    w = perturb === :biomass ? (u .^ 2) :
        perturb === :uniform ? fill(1.0, length(u)) :
        error("Unknown perturbation: $perturb")
    sqrtw = sqrt.(w)
    M = transpose(Z) * Diagonal(sqrtw)

    μ = maximum(real.(vals)) + margin
    I_T = Matrix{Float64}(I, size(T,1), size(T,2))

    out = Vector{Float64}(undef, length(t_vals))
    @inbounds for (k, t) in pairs(t_vals)
        Y = exp(t .* (T .- μ .* I_T)) * M
        s = sum(abs2, Y)
        num = (log(s) + 2.0*μ*t)
        den = log(sum(w))
        out[k] = - (num - den) / (2.0*t)
        if !isfinite(out[k]); out[k] = NaN; end
    end
    return out
end

# ============================================================
# 1) Scramblers
# ============================================================
"""
    scramble_A_space_keep_diag(J; rng, lock_pairs=0, tol_diag=1e-10)

MAIN PATH. Work in the Schur space of the interaction part A = J - Diag(diag J).
- Rotate only the BULK Schur subspace (optionally freeze first `lock_pairs` blocks).
- Enforce ONLY the diagonal pin (||diag(J')-diag(J)||∞ ≤ tol_diag). If violated, return `nothing`.

Returns (J′, info::NamedTuple) or `nothing`.
"""
function scramble_A_space_keep_diag(J::AbstractMatrix{<:Real};
                                    rng::AbstractRNG=Xoshiro(42),
                                    lock_pairs::Int=0)
    # J = D + A
    S = size(J,1)
    D = Diagonal(diag(J))
    A = Matrix{Float64}(J) .- Matrix(D)

    # Real Schur of A
    F = schur(A)                # A = ZA * TA * ZA'
    ZA, TA = F.Z, F.T
    n = size(TA,1)

    # Parse 1x1/2x2 blocks
    blocks = UnitRange{Int}[]
    i = 1
    while i ≤ n
        if i < n && abs(TA[i+1,i]) > 0.0
            push!(blocks, i:(i+1)); i += 2
        else
            push!(blocks, i:i);     i += 1
        end
    end
    k = clamp(sum(length.(blocks[1:clamp(lock_pairs,0,length(blocks))])), 0, n)
    m = n - k
    if m == 0
        # No rotation requested
        A2 = ZA*TA*ZA'
        A2 .-= Diagonal(diag(A2))           # enforce zero diag on interaction
        Jp = D + A2
        diagerr   = maximum(abs.(diag(Jp) .- diag(J)))   # will be ~0
        edgedrift = abs(maximum(real(eigvals(Jp))) - maximum(real(eigvals(J))))
        return (Jp, (mode=:Aspace, locked=k, rotated=0, diagerr=diagerr, edgedrift=edgedrift))
    end

    # Rotate bulk of the Schur space
    Q = qr!(randn(rng, m, m)).Q
    U = Matrix{Float64}(I, n, n)
    @views U[k+1:end, k+1:end] .= Q

    TA2 = transpose(U) * TA * U
    A2  = ZA * TA2 * ZA'

    # --- CRUCIAL: remove induced diagonal so time-scales stay fixed
    A2 .-= Diagonal(diag(A2))               # kills diag drift exactly
    Jp  = D + A2

    diagerr   = maximum(abs.(diag(Jp) .- diag(J)))       # ≈ 0 by construction
    edgedrift = abs(maximum(real(eigvals(Jp))) - maximum(real(eigvals(J))))

    return (Jp, (mode=:Aspace, locked=k, rotated=m, diagerr=diagerr, edgedrift=edgedrift))
end

"""
    scramble_J_space_lock_edge(J; rng, lock_pairs=0, tol_edge=1e-10)

CONTROL. Work in the Schur space of the full J.
- Rotate only the BULK; optionally freeze first `lock_pairs` blocks.
- Enforce ONLY the edge/spectrum pin (|λmax(J′)-λmax(J)| ≤ tol_edge). If violated, return `nothing`.

Returns (J′, info::NamedTuple) or `nothing`.
"""
function scramble_J_space_lock_edge(J::AbstractMatrix{<:Real};
                                    rng::AbstractRNG=Xoshiro(42),
                                    lock_pairs::Int=0,
                                    tol_edge::Float64=1e-10)
    F = schur(Matrix{Float64}(J))   # J = ZJ * TJ * ZJ'
    ZJ, TJ, vals = F.Z, F.T, F.values
    n = size(TJ,1)

    # blocks
    blocks = UnitRange{Int}[]
    i = 1
    while i ≤ n
        if i < n && abs(TJ[i+1,i]) > 0.0
            push!(blocks, i:(i+1)); i += 2
        else
            push!(blocks, i:i);     i += 1
        end
    end
    k = clamp(sum(length.(blocks[1:clamp(lock_pairs,0,length(blocks))])), 0, n)
    m = n - k
    if m == 0
        return (ZJ*TJ*ZJ', (mode=:Jspace, locked=k, rotated=0, diagerr=0.0, edgedrift=0.0))
    end

    Q = qr!(randn(rng, m, m)).Q
    U = Matrix{Float64}(I, n, n)
    @views U[k+1:end, k+1:end] .= Q

    TJ2 = transpose(U) * TJ * U
    Jp  = ZJ * TJ2 * ZJ'

    diagerr   = maximum(abs.(diag(Jp) .- diag(J)))
    edgedrift = abs(maximum(real(eigvals(Jp))) - maximum(real(eigvals(J))))

    # Enforce ONLY the edge pin
    if edgedrift > tol_edge
        return nothing
    end
    return (Jp, (mode=:Jspace, locked=k, rotated=m, diagerr=diagerr, edgedrift=edgedrift))
end

# ============================================================
# 2) Builders (thin wrappers to rescale by realized IS)
# ============================================================
build_trophic_ER = function (S; conn, mean_abs, mag_cv, deg_cv, rho_mag, rho_sign, rng)
    A = build_ER_degcv(S, conn, mean_abs, mag_cv, rho_mag, rho_sign, deg_cv; rng=rng)
    isA = realized_IS(A); isA == 0 ? A : A .* (mean_abs / isA)
end

build_ER_baseline = function (S; conn, mean_abs, mag_cv, rng)
    A = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                degree_family=:uniform, deg_param=0.0,
                                rho_sym=0.0, rng=rng)
    isA = realized_IS(A); isA == 0 ? A : A .* (mean_abs / isA)
end

build_niche_rescaled = function (S; conn, mean_abs, mag_cv, deg_cv, rho_mag, rng)
    A = build_niche_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                            degree_family=:lognormal, deg_param=deg_cv,
                            rho_sym=rho_mag, rng=rng)
    isA = realized_IS(A); isA == 0 ? A : A .* (mean_abs / isA)
end

# ============================================================
# 3) One level run (A-space main, J-space control)
# ============================================================
"""
    run_level(; kind, deg_cv, rho_mag, rho_sign, ...)

Returns NamedTuple:
  t, Aspace=(delta, fbar, gbar, n, diagerrs, edgedrifts),
     Jspace=(delta, fbar, gbar, n, diagerrs, edgedrifts),
  table::Vector{NamedTuple}   # compact diagnostics rows (means)
"""
function run_level(; S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.50,
        mag_cv::Float64=0.60, u_mean::Float64=1.0, u_cv::Float64=0.6,
        t_vals = 10 .^ range(-2, 2; length=40), reps::Int=40, seed::Int=0xBEEF,
        perturb::Symbol=:biomass, lock_pairs::Int=0,
        kind::Symbol=:ER, deg_cv::Float64=0.0, rho_mag::Float64=0.0, rho_sign::Float64=0.0)

    rng_master = Xoshiro(seed)
    nt = length(t_vals)

    # accumulators
    ΔA = zeros(nt); fA = zeros(nt); gA = zeros(nt); nA = 0
    ΔJ = zeros(nt); fJ = zeros(nt); gJ = zeros(nt); nJ = 0
    diagA = Float64[]; edgeA = Float64[]
    diagJ = Float64[]; edgeJ = Float64[]

    for r in 1:reps
        rng = Xoshiro(rand(rng_master, UInt64))
        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

        # draw A according to level
        A =
            kind === :NICHE ? build_niche_rescaled(S; conn, mean_abs, mag_cv,
                                                   deg_cv=deg_cv, rho_mag=rho_mag, rng=rng) :
            kind === :BASE  ? build_ER_baseline(S; conn, mean_abs, mag_cv, rng) :
                              build_trophic_ER(S; conn, mean_abs, mag_cv,
                                               deg_cv=deg_cv, rho_mag=rho_mag,
                                               rho_sign=rho_sign, rng=rng)
        J = jacobian(A, u)

        # A-space MAIN
        ascr = scramble_A_space_keep_diag(J; rng=rng, lock_pairs=lock_pairs)
        if ascr !== nothing
            Jp, info = ascr
            f = compute_rmed_series_stable(J,  u, t_vals; perturb=perturb)
            g = compute_rmed_series_stable(Jp, u, t_vals; perturb=perturb)
            if !(any(!isfinite, f) || any(!isfinite, g))
                ΔA .+= abs.(g .- f); fA .+= f; gA .+= g; nA += 1
                push!(diagA, info.diagerr); push!(edgeA, info.edgedrift)
            end
        end

        # J-space CONTROL
        jscr = scramble_J_space_lock_edge(J; rng=rng, lock_pairs=lock_pairs)
        if jscr !== nothing
            Jp, info = jscr
            f = compute_rmed_series_stable(J,  u, t_vals; perturb=perturb)
            g = compute_rmed_series_stable(Jp, u, t_vals; perturb=perturb)
            if !(any(!isfinite, f) || any(!isfinite, g))
                ΔJ .+= abs.(g .- f); fJ .+= f; gJ .+= g; nJ += 1
                push!(diagJ, info.diagerr); push!(edgeJ, info.edgedrift)
            end
        end
    end

    # finalize means; leave NaNs if no accepted reps
    Aspace = (t=t_vals,
              delta = nA>0 ? ΔA ./ nA : fill(NaN, nt),
              fbar  = nA>0 ? fA ./ nA  : fill(NaN, nt),
              gbar  = nA>0 ? gA ./ nA  : fill(NaN, nt),
              n = nA, diagerrs = diagA, edgedrifts = edgeA)

    Jspace = (t=t_vals,
              delta = nJ>0 ? ΔJ ./ nJ : fill(NaN, nt),
              fbar  = nJ>0 ? fJ ./ nJ  : fill(NaN, nt),
              gbar  = nJ>0 ? gJ ./ nJ  : fill(NaN, nt),
              n = nJ, diagerrs = diagJ, edgedrifts = edgeJ)

    # small compact table (means of drifts)
    row = (; n_A=nA, n_J=nJ,
           mean_diagerr_A = isempty(diagA) ? NaN : mean(diagA),
           mean_edgedrift_A = isempty(edgeA) ? NaN : mean(edgeA),
           mean_diagerr_J = isempty(diagJ) ? NaN : mean(diagJ),
           mean_edgedrift_J = isempty(edgeJ) ? NaN : mean(edgeJ))

    return (; t=t_vals, Aspace, Jspace, table=[row])
end

# ============================================================
# 4) Multi-level driver + plotting
# ============================================================
function run_path(; S=120, conn=0.10, mean_abs=0.50, mag_cv=0.60,
                   u_mean=1.0, u_cv=0.6, reps=40, seed=0xBEEF,
                   t_vals = 10 .^ range(-2, 2; length=40),
                   lock_pairs=0, perturb=:biomass)

    levels = [
        (name="baselineStruct", kind=:BASE,  deg_cv=0.0, rho_mag=0.0,  rho_sign=0.0),
        (name="ER",             kind=:ER,    deg_cv=0.0, rho_mag=0.0,  rho_sign=0.0),
        (name="degCV",          kind=:ER,    deg_cv=0.8, rho_mag=0.0,  rho_sign=0.0),
        (name="deg+mag",        kind=:ER,    deg_cv=0.8, rho_mag=0.99, rho_sign=0.0),
        (name="trophic",        kind=:ER,    deg_cv=0.8, rho_mag=0.99, rho_sign=1.0),
        (name="trophic+",       kind=:ER,    deg_cv=1.2, rho_mag=0.99, rho_sign=1.0),
        (name="niche",          kind=:NICHE, deg_cv=1.0, rho_mag=0.99, rho_sign=1.0),
    ]

    results = Dict{String,Any}()
    println("level | nA  nJ  |  ⟨diagerr⟩_A  ⟨edge drift⟩_A  |  ⟨diagerr⟩_J  ⟨edge drift⟩_J")
    println("------+--------+-------------------------------+------------------------------")

    for L in levels
        out = run_level(; S, conn, mean_abs, mag_cv, u_mean, u_cv,
                         t_vals, reps, seed, perturb, lock_pairs,
                         kind=L.kind, deg_cv=L.deg_cv, rho_mag=L.rho_mag, rho_sign=L.rho_sign)
        results[L.name] = (; label=L.name, data=Dict(:Aspace=>out.Aspace, :Jspace=>out.Jspace))
        row = out.table[1]
        @printf("%-10s %3d %3d   %10.3e   %10.3e      %10.3e   %10.3e\n",
                L.name, row.n_A, row.n_J,
                row.mean_diagerr_A, row.mean_edgedrift_A,
                row.mean_diagerr_J, row.mean_edgedrift_J)
    end
    return results
end

# ---- plotting like your previous helpers ----
function plot_condition_pair(res_for_level; title="Where to shuffle")
    A = res_for_level.data[:Aspace]
    J = res_for_level.data[:Jspace]
    fig = Figure(size=(1100,700))

    # Δ curves
    ax1 = Axis(fig[1,1], xscale=log10, xlabel="t", ylabel="|ΔR̃med|", title=title)
    lines!(ax1, A.t, A.delta, color=:seagreen,  linewidth=2, label="A-space shuffle")
    lines!(ax1, J.t, J.delta, color=:orangered, linewidth=2, label="J-space shuffle")
    axislegend(ax1, position=:rt, framevisible=false)

    # mean R̃med curves
    ax2 = Axis(fig[2,1], xscale=log10, xlabel="t", ylabel="R̃med (means)")
    lines!(ax2, A.t, A.fbar, color=:steelblue,  label="orig (A)")
    lines!(ax2, A.t, A.gbar, color=:gray,      label="scr (A)")
    lines!(ax2, J.t, J.fbar, color=:purple,    label="orig (J)")
    lines!(ax2, J.t, J.gbar, color=:black,     label="scr (J)")
    axislegend(ax2, position=:lb, framevisible=false)

    display(fig)
    return fig
end

# ============================================================
# 5) Run and plot
# ============================================================
S, conn, mean_abs, mag_cv = 120, 0.10, 0.50, 0.60
t_vals = 10 .^ range(-2, 2; length=40)

results = run_path(; S, conn, mean_abs, mag_cv, u_mean=1.0, u_cv=0.6,
                    reps=40, seed=Int(0xBEEF), t_vals, lock_pairs=0, perturb=:biomass)

# show two representative levels
for name in ["ER","trophic"]
    haskey(results, name) || continue
    plot_condition_pair(results[name]; title="Where to shuffle — $(name)")
end
