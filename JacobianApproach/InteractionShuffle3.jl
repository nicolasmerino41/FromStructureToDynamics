using LinearAlgebra, Random, Statistics
using CairoMakie

# =============== helpers you already have (referenced) =================
# - compute_rmed_series_stable(J,u,t_vals; perturb=:biomass)
# - scramble_interaction_keep_diag(J; rng=Xoshiro(42), lock_pairs=0)
# - build_ER_degcv, build_random_nontrophic, build_niche_trophic
# - realized_IS, random_u, jacobian
# - build_trophic_ER, build_ER_baseline, build_niche_rescaled (from your last working file)

# ---------------- diagnostics (lightweight but useful) -----------------
# non-normality (Henrici departure): ||J'J - JJ'||_F
_nn(J) = norm(J'J - J*J')

# commutator with D=diag(J), measures TS/interaction mismatch scale
_comm(J) = begin D = Diagonal(diag(J)); A = J - D; norm(D*A - A*D) end

# off-diagonal magnitude proxy for the interaction strength distribution
_off(J) = begin D = Diagonal(diag(J)); A = J - D; s = sum(abs, A) - sum(abs, diag(A)); s / max(1, length(J)^2 - length(J)) end

# eigenvector condition number (guarded). If eig fails or ill-conditioned, return NaN.
function _kappaV(J)
    try
        F = eigen(J)              # J = V Λ V^{-1}
        V = F.vectors
        κ = opnorm(V) * opnorm(inv(V))
        isfinite(κ) ? κ : NaN
    catch
        NaN
    end
end

# scalar “edge drift”: difference in max real eigenvalue (resilience)
_edge(J1,J2) = abs(maximum(real(eigvals(J1))) - maximum(real(eigvals(J2))))

# ---- per-rep diagnostic bundle
struct DiagonalSummary
    Δnn::Float64
    comm::Float64
    Δkappa::Float64
    off::Float64
    diagerr::Float64
    edgedrift::Float64
end

function diag_pair(J, Jscr)
    Δnn   = _nn(Jscr) - _nn(J)
    comm  = _comm(Jscr)                     # absolute level after scramble
    κ1    = _kappaV(J); κ2 = _kappaV(Jscr)
    Δκ    = (isfinite(κ1) && isfinite(κ2)) ? (κ2 - κ1) : NaN
    off   = _off(Jscr)
    diagerr = maximum(abs.(diag(Jscr) .- diag(J)))
    edged  = _edge(J, Jscr)
    DiagonalSummary(Δnn, comm, Δκ, off, diagerr, edged)
end

# -------------------- per-level runner (A-space vs J-space) --------------------
# Scramble in A-space (interaction-only, keeps D exactly), or in J-space (full).
function scramble_Aspace(J; rng, lock_pairs::Int=0)
    # J = D + A, scramble A in its real-Schur basis, keep D identical
    Jp, info = scramble_interaction_keep_diag(J; rng=rng, lock_pairs=lock_pairs)
    return Jp, info
end

function scramble_Jspace(J; rng, lock_pairs::Int=0)
    # shuffle in Schur(J); keeps edge if lock_pairs>0, but will move diag(J)
    F = schur(Matrix{Float64}(J))
    Z, T = F.Z, F.T
    # parse blocks
    blocks = UnitRange{Int}[]
    i = 1; n = size(T,1)
    while i ≤ n
        if i < n && abs(T[i+1,i]) > 0.0; push!(blocks, i:(i+1)); i += 2
        else;                            push!(blocks, i:i);     i += 1
        end
    end
    k = clamp(sum(length.(blocks[1:clamp(lock_pairs,0,length(blocks))])), 0, n)
    m = n - k
    if m == 0
        return Z*T*Z', (locked=k, rotated=0)
    end
    Q = qr!(randn(rng, m, m)).Q
    U = Matrix{Float64}(I, n, n)
    @views U[k+1:end, k+1:end] .= Q
    T2 = transpose(U) * T * U
    Jp = Z*T2*Z'
    return Jp, (locked=k, rotated=m)
end

# -------------------- main suite: runs A-space and J-space ---------------------
function run_falsification_suite(; S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.50,
    mag_cv::Float64=0.60, u_mean::Float64=1.0, u_cv::Float64=0.6,
    t_vals = 10 .^ range(-2, 2; length=40), reps::Int=40, seed::Int=Int(0xC0FFEE),
    lock_pairs::Int=0, perturb::Symbol=:biomass, P_list=(0.2, 0.1, 0.05))

    rng_master = Xoshiro(seed)
    nt = length(t_vals)

    levels = [
        (name="baselineStruct", kind=:BASE,  deg_cv=0.0, rho_mag=0.0,  rho_sign=0.0),
        (name="ER",             kind=:ER,    deg_cv=0.0, rho_mag=0.0,  rho_sign=0.0),
        (name="degCV",          kind=:ER,    deg_cv=0.8, rho_mag=0.0,  rho_sign=0.0),
        (name="deg+mag",        kind=:ER,    deg_cv=0.8, rho_mag=0.99, rho_sign=0.0),
        (name="trophic",        kind=:ER,    deg_cv=0.8, rho_mag=0.99, rho_sign=1.0),
        (name="trophic+",       kind=:ER,    deg_cv=1.2, rho_mag=0.99, rho_sign=1.0),
        (name="niche",          kind=:NICHE, deg_cv=1.0, rho_mag=0.99, rho_sign=1.0),
    ]

    function draw_A(L, rng)
        A =
            L.kind === :NICHE ? build_niche_rescaled(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                                     deg_cv=L.deg_cv, rho_mag=L.rho_mag, rng=rng) :
            L.kind === :BASE  ? build_ER_baseline(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng) :
                                build_trophic_ER(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                                 deg_cv=L.deg_cv, rho_mag=L.rho_mag,
                                                 rho_sign=L.rho_sign, rng=rng)
        return A
    end

    # t_P proxy from the averaged series: smallest t where 2 t Rmed(t) ≥ -log P
    function t_recovery_from_series(rmed::Vector{Float64}, t::Vector{Float64}, P::Float64)
        thresh = -log(P)
        for (i,ti) in pairs(t)
            s = 2ti * rmed[i]
            if isfinite(s) && s ≥ thresh; return ti; end
        end
        return NaN
    end

    # container per level
    results = Dict{String, Any}()

    for L in levels
        acc = Dict{Symbol,Any}()

        for mode in (:Aspace, :Jspace)
            Δ   = zeros(nt)
            f̄   = zeros(nt); ḡ = zeros(nt)
            n   = 0
            diags = DiagonalSummary[]

            for r in 1:reps
                rng = Xoshiro(rand(rng_master, UInt64))
                u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

                A  = draw_A(L, rng); J = jacobian(A, u)

                Jscr, _ = mode === :Aspace ?
                    scramble_Aspace(J; rng=rng, lock_pairs=lock_pairs) :
                    scramble_Jspace(J; rng=rng, lock_pairs=lock_pairs)

                f = compute_rmed_series_stable(J,    u, t_vals; perturb=perturb)
                g = compute_rmed_series_stable(Jscr, u, t_vals; perturb=perturb)

                if any(!isfinite, f) || any(!isfinite, g)
                    continue
                end

                Δ .+= abs.(g .- f);  f̄ .+= f; ḡ .+= g; n += 1
                push!(diags, diag_pair(J, Jscr))
            end

            Δ ./= max(n,1); f̄ ./= max(n,1); ḡ ./= max(n,1)

            # recovery-time proxies per P, robust to NaN
            tP = Dict{Float64,Float64}()
            for P in P_list
                tP[P] = t_recovery_from_series(f̄, t_vals, P)
            end

            acc[mode] = (t=t_vals, n=n, delta=Δ, fbar=f̄, gbar=ḡ,
                         diags=diags, t_recovery=tP)
        end

        results[L.name] = (label=L.name, data=acc)
    end

    return results
end

# -------------------------- correlations & summaries --------------------------
# Mid-t window default (as before)
function mid_indices(t; frac=(0.35,0.65))
    lt = log10.(t); lo, hi = lt[1], lt[end]
    a = lo + frac[1]*(hi-lo); b = lo + frac[2]*(hi-lo)
    findall(i -> lt[i] ≥ a && lt[i] ≤ b, eachindex(t))
end

# compact “per-level” correlation struct
function correlate_diagnostics(level_record)
    A  = level_record.data[:Aspace]
    J  = level_record.data[:Jspace]
    t  = A.t
    mid = mid_indices(t)

    mid_A = mean(skipmissing(@view A.delta[mid]))
    mid_J = mean(skipmissing(@view J.delta[mid]))

    # means of diagnostics
    meanA(f) = mean(getfield.(A.diags, f))
    meanJ(f) = mean(getfield.(J.diags, f))

    return (
        mid_A = mid_A, mid_J = mid_J,
        A_nnc = meanA(:Δnn),    A_comm = meanA(:comm),
        A_kc  = meanA(:Δkappa), A_off  = meanA(:off),
        J_nnc = meanJ(:Δnn),    J_comm = meanJ(:comm),
        J_kc  = meanJ(:Δkappa), J_off  = meanJ(:off),
        A_diagerr = meanA(:diagerr), A_edged = meanA(:edgedrift),
        J_diagerr = meanJ(:diagerr), J_edged = meanJ(:edgedrift)
    )
end

# ------------------------------- plotting -------------------------------------
# Pair plot (A vs J)
function plot_condition_pair(level_record; title="Where to shuffle")
    A = level_record.data[:Aspace]
    J = level_record.data[:Jspace]
    t = A.t

    fig = Figure(size=(1100,700))

    # ΔRmed
    ax1 = Axis(fig[1,1], xscale=log10, xlabel="t", ylabel="|ΔR̃med|", title=title * " — $(level_record.label)")
    if any(isfinite, A.delta); lines!(ax1, t, A.delta, color=:seagreen, linewidth=3, label="A-space shuffle"); end
    if any(isfinite, J.delta); lines!(ax1, t, J.delta, color=:orangered, linewidth=3, label="J-space shuffle"); end
    axislegend(ax1, position=:rt, framevisible=false)

    # mean Rmed curves
    ax2 = Axis(fig[2,1], xscale=log10, xlabel="t", ylabel="R̃med (means)")
    if any(isfinite, A.fbar); lines!(ax2, t, A.fbar, color=:steelblue, label="orig (A)"); end
    if any(isfinite, A.gbar); lines!(ax2, t, A.gbar, color=:gray40,   label="scr (A)");  end
    if any(isfinite, J.fbar); lines!(ax2, t, J.fbar, color=:purple,   label="orig (J)"); end
    if any(isfinite, J.gbar); lines!(ax2, t, J.gbar, color=:black,    label="scr (J)");  end
    axislegend(ax2, position=:rt, framevisible=false)

    display(fig)
    return fig
end

# Path summary (mid-t excess only; A and J side-by-side)
function plot_path_summary(results::Dict{String,Any}, order::Vector{String})
    xs = 1:length(order)
    midA = Float64[]; midJ = Float64[]; names = String[]
    for name in order
        haskey(results, name) || continue
        push!(names, name)
        r = results[name]; A = r.data[:Aspace]; J = r.data[:Jspace]
        mids = mid_indices(A.t)
        push!(midA, mean(@view A.delta[mids]))
        push!(midJ, mean(@view J.delta[mids]))
    end
    fig = Figure(size=(900,420))
    ax = Axis(fig[1,1], xlabel="structure level", ylabel="mid-t |ΔR̃med|")
    lines!(ax, xs, midA, linewidth=3, label="A-space")
    lines!(ax, xs, midJ, linewidth=3, label="J-space (control)")
    ax.xticks = (xs, names); axislegend(ax, framevisible=false, position=:lt)
    display(fig)
    return fig
end

# Recovery-time proxies (robust: silently skips NaN)
function plot_recovery_times(results::Dict{String,Any}, order::Vector{String}, P_list=(0.2,0.1,0.05))
    xs = 1:length(order)
    fig = Figure(size=(1000,420))
    ax = Axis(fig[1,1], xlabel="structure level", ylabel="t_P proxy (from 2tR̃med)")
    for P in P_list
        vals = Float64[]
        names = String[]
        for name in order
            if !haskey(results,name); continue; end
            push!(names, name)
            r = results[name].data[:Aspace]  # show A-space by default
            tp = get(r.t_recovery, P, NaN)
            push!(vals, tp)
        end
        lines!(ax, 1:length(vals), vals, linewidth=2, label="P=$(P)")
        ax.xticks = (1:length(vals), names)
    end
    axislegend(ax, framevisible=false, position=:rt)
    display(fig)
    return fig
end

# ================================ RUN =========================================
S, conn, mean_abs, mag_cv = 120, 0.10, 0.50, 0.60
t_vals = 10 .^ range(-2, 2; length=40)

results = run_falsification_suite(; S, conn, mean_abs, mag_cv,
    u_mean=1.0, u_cv=0.6, t_vals, reps=40, seed=Int(0xC0FFEE),
    lock_pairs=0, perturb=:biomass, P_list=(0.2, 0.1, 0.05))

level_order = ["baselineStruct","ER","degCV","deg+mag","trophic","trophic+","niche"]

for name in level_order
    haskey(results, name) || continue
    plot_condition_pair(results[name]; title="Where to shuffle")
end

plot_path_summary(results, level_order)
plot_recovery_times(results, level_order, (0.2, 0.1, 0.05))

println("\n=== Mid-t excess vs diagnostics (means per level) ===")
for name in level_order
    haskey(results, name) || continue
    s = correlate_diagnostics(results[name])
    println(rpad(name,12), " | mid_A=$(round(s.mid_A,digits=4))  mid_J=$(round(s.mid_J,digits=4))",
            "  | A: ΔNN=$(round(s.A_nnc,digits=3)) [D,A]≈$(round(s.A_comm,digits=3)) Δκ=$(round(s.A_kc,digits=3)) off≈$(round(s.A_off,digits=3))",
            "  | J: ΔNN=$(round(s.J_nnc,digits=3)) [D,A]≈$(round(s.J_comm,digits=3)) Δκ=$(round(s.J_kc,digits=3)) off≈$(round(s.J_off,digits=3))")
end
