using LinearAlgebra, Random, Statistics

# ------------------------------------------------------------
# 0) R̃med(t) computation (overflow-safe, exact shift handling)
# ------------------------------------------------------------
function compute_rmed_series_stable(J::AbstractMatrix{<:Real},
                                   u::AbstractVector{<:Real},
                                   t_vals::AbstractVector{<:Real};
                                   perturb::Symbol=:biomass,
                                   margin::Float64=1e-6)
    F = schur(Matrix{Float64}(J))     # real Schur: J = Z*T*Z'
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

# ------------------------------------------------------------
# 1) Interaction-only scramble, diagonal preserved exactly
#    J = D + A, with D = diag(J). We scramble A in its real-Schur basis.
#    IMPORTANT: we NEVER touch D, and we NEVER post-multiply Z by U.
# ------------------------------------------------------------
function scramble_interaction_keep_diag(J::AbstractMatrix{<:Real};
                                       rng::AbstractRNG=Xoshiro(42),
                                       lock_pairs::Int=0)
    S = size(J,1)
    D = Diagonal(diag(J))
    A = Matrix{Float64}(J) .- Matrix(D)

    # Real Schur of A
    F = schur(A)              # A = Z*T*Z'
    Z, T = F.Z, F.T
    n = size(T,1)

    # Parse 1x1/2x2 blocks
    blocks = UnitRange{Int}[]
    i = 1
    while i ≤ n
        if i < n && abs(T[i+1,i]) > 0.0
            push!(blocks, i:(i+1)); i += 2
        else
            push!(blocks, i:i);     i += 1
        end
    end
    # "lock_pairs" leading blocks are frozen
    k = clamp(sum(length.(blocks[1:clamp(lock_pairs,0,length(blocks))])), 0, n)
    m = n - k
    if m == 0
        return (D + Z*T*Z', (locked=k, rotated=0, warn=false))
    end

    Q = qr!(randn(rng, m, m)).Q
    U = Matrix{Float64}(I, n, n)
    @views U[k+1:end, k+1:end] .= Q

    T2 = transpose(U) * T * U
    Jp = D + Z * T2 * transpose(Z)

    # --- sanity checks (warn, do not stop)
    warn = false
    if maximum(abs.(diag(Jp) .- diag(J))) > 1e-12
        @warn "interaction-shuffle: diagonal changed"
        warn = true
    end
    # eigenvalues of A and A' should match up to ~1e-10–1e-12
    evA  = sort(real.(eigvals(A)))
    evAp = sort(real.(eigvals(Matrix(Jp) .- Matrix(D))))
    if norm(evA - evAp) > 1e-8*max(1.0, norm(evA))
        error("interaction-shuffle: A-spectrum drift = $(norm(evA-evAp))")
        warn = true
    end
    return (Jp, (locked=k, rotated=m, warn=warn))
end

# ------------------------------------------------------------
# 2) Builders you already have (we call them exactly as before)
# ------------------------------------------------------------
# We assume build_ER_degcv, build_random_nontrophic, build_niche_trophic,
# realized_IS, random_u, jacobian are defined elsewhere.

# Convenience wrappers that rescale to hit mean_abs via realized_IS
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

"""
    lock_edge_to_reference(J_prop, J_ref)

Aligns the *slowest real-Schur block* of `J_prop` to that of `J_ref`,
so that both matrices share exactly the same dominant eigenpair.
Preserves:

  • the diagonal of J_prop (time scales)
  • all other Schur blocks of J_prop (bulk)
  • reference slowest block (copied exactly)

Returns:
    (J_locked, info)

`info` contains:
    mode::String       # "block" or "scalar-shift"
    k::Int             # block size used (1 or 2)
    dlam::Float64      # |λmax(J_locked) − λmax(J_ref)|
    ddiag::Float64     # ∞-norm of diagonal drift
    message::String    # optional diagnostic
"""
function lock_edge_to_reference(J_prop::AbstractMatrix, J_ref::AbstractMatrix)

    # --- Real Schur decompositions ---
    Fref = schur(Matrix{Float64}(J_ref))   # J_ref = Zr * Tr * Zr'
    Zr, Tr, vals_r = Fref.Z, Fref.T, Fref.values

    Fp   = schur(Matrix{Float64}(J_prop))  # J_prop = Zp * Tp * Zp'
    Zp, Tp, vals_p = Fp.Z, Fp.T, Fp.values

    # --- Find slowest eigenindex in each ---
    ir = argmax(real.(vals_r))
    ip = argmax(real.(vals_p))

    # --- Helper: detect Schur blocks ---
    function schur_blocks(T)
        n = size(T,1)
        blocks = UnitRange{Int}[]
        i = 1
        while i ≤ n
            if i < n && abs(T[i+1,i]) > 0   # 2×2 block
                push!(blocks, i:(i+1))
                i += 2
            else
                push!(blocks, i:i)
                i += 1
            end
        end
        return blocks
    end

    br = schur_blocks(Tr)
    bp = schur_blocks(Tp)

    Br = br[findfirst(b -> ir in b, br)]   # block of slowest mode in ref
    Bp = bp[findfirst(b -> ip in b, bp)]   # block of slowest mode in prop

    kr = length(Br)   # 1 or 2
    kp = length(Bp)   # 1 or 2

    # --- Helper: permute block B to the front of Schur decomposition ---
    function move_block_first(T, Z, B::UnitRange{Int})
        S = size(T,1)
        rest = collect(1:S)
        deleteat!(rest, B)
        perm = vcat(collect(B), rest)
        P = Matrix{Float64}(I, S, S)[:, perm]
        return transpose(P) * T * P, Z * P
    end

    # --- CASE 1: block sizes match (1↔1 or 2↔2) ---
    if kr == kp
        Tref_p, Zref_p = move_block_first(Tr, Zr, Br)
        Tprop_p, Zprop_p = move_block_first(Tp, Zp, Bp)

        # Overwrite the block
        @views Tprop_p[1:kr, 1:kr] .= Tref_p[1:kr, 1:kr]

        # Reconstruct locked J
        J_locked = Zprop_p * Tprop_p * transpose(Zprop_p)

        info = (
            mode = "block",
            k    = kr,
            dlam = abs(maximum(real(eigvals(J_locked))) -
                       maximum(real(eigvals(J_ref)))),
            ddiag= maximum(abs.(diag(J_locked) .- diag(J_prop))),
            message = "block overwrite"
        )
        return J_locked, info
    end

    # ------------------------------------------------------------------
    # --- CASE 2: block sizes differ (1↔2 or 2↔1)
    # ------------------------------------------------------------------
    # Fallback: match the dominant eigenvalue by scalar shift
    lam_ref = maximum(real.(vals_r))
    lam_pro = maximum(real.(vals_p))
    δ = lam_ref - lam_pro

    S = size(J_prop, 1)
    J_locked = J_prop .+ δ .* Diagonal(ones(S))

    info = (
        mode  = "scalar-shift",
        k     = min(kr,kp),
        dlam  = abs(maximum(real(eigvals(J_locked))) - lam_ref),
        ddiag = abs(δ),
        message = "fallback: mismatched Schur-block sizes (ref=$kr, prop=$kp)"
    )
    return J_locked, info
end

# ------------------------------------------------------------
# 3) One experiment: for a given (level), compare
#    structured J vs interaction-scrambled(J)  and
#    baseline ER J0 vs interaction-scrambled(J0)
# ------------------------------------------------------------
function run_interaction_only_path(; S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.50,
    mag_cv::Float64=0.60, u_mean::Float64=1.0, u_cv::Float64=0.6,
    t_vals = 10 .^ range(-2, 2; length=40), reps::Int=40, seed::Int=45000,
    perturb::Symbol=:biomass, lock_pairs::Int=0)

    rng_master = Xoshiro(seed)
    nt = length(t_vals)

    # Discrete "structure" path including NICHE (last)
    # NEW FIRST LEVEL: same configuration as baseline
    levels = [
        (name="baselineStruct", kind=:BASE,  deg_cv=0.0, rho_mag=0.0,  rho_sign=0.0),
        (name="ER",             kind=:ER,    deg_cv=0.0, rho_mag=0.0,  rho_sign=0.0),
        (name="degCV",          kind=:ER,    deg_cv=0.8, rho_mag=0.0,  rho_sign=0.0),
        (name="deg+mag",        kind=:ER,    deg_cv=0.8, rho_mag=0.99, rho_sign=0.0),
        (name="trophic",        kind=:ER,    deg_cv=0.8, rho_mag=0.0, rho_sign=1.0),
        (name="trophic+",       kind=:ER,    deg_cv=1.2, rho_mag=0.99, rho_sign=1.0),
        (name="niche",          kind=:NICHE, deg_cv=1.0, rho_mag=0.99, rho_sign=1.0),
    ]

    results = Dict{String,Any}()

    for L in levels
        Δ_struct = zeros(nt);  B = zeros(nt)
        sum_f = zeros(nt); sum_g = zeros(nt)
        sum_b0 = zeros(nt); sum_b1 = zeros(nt)
        nS = 0; nB = 0

        for r in 1:reps
            rng = Xoshiro(rand(rng_master, UInt64))
            u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

            # Structured draw
            A =
                L.kind === :NICHE ? build_niche_rescaled(S; conn, mean_abs, mag_cv,
                                                         deg_cv=L.deg_cv, rho_mag=L.rho_mag, rng=rng) :
                L.kind === :BASE  ? build_ER_baseline(S; conn, mean_abs, mag_cv, rng) :
                                    build_trophic_ER(S; conn, mean_abs, mag_cv,
                                                     deg_cv=L.deg_cv, rho_mag=L.rho_mag,
                                                     rho_sign=L.rho_sign, rng=rng)

            J = jacobian(A, u)
            J_prop, infoS = scramble_interaction_keep_diag(J; rng=rng, lock_pairs=lock_pairs)
            Js = J_prop
            # Js, infoLock = lock_edge_to_reference(J_prop, J)

            # diagnostics
            Δdiag  = maximum(abs.(diag(Js) .- diag(J)))
            Δλedge = abs(maximum(real(eigvals(Js))) - maximum(real(eigvals(J))))
            comm   = norm((A*Diagonal(diag(J)) - Diagonal(diag(J))*A))  # crude [A,D] proxy if D=diag(J)-Adiag

            @info "checks" Δdiag=Δdiag Δλedge=Δλedge comm=comm

            f = compute_rmed_series_stable(J,  u, t_vals; perturb=perturb)
            g = compute_rmed_series_stable(Js, u, t_vals; perturb=perturb)

            if any(!isfinite, f) || any(!isfinite, g)
                @info "skip structured non-finite" L.name infoS
            else
                Δ_struct .+= abs.(g .- f);  nS += 1
                sum_f .+= f; sum_g .+= g
            end

            # Baseline ER draw (independent directions & signs)
            A0 = build_ER_baseline(S; conn, mean_abs, mag_cv, rng)
            J0 = jacobian(A0, u)
            J1, infoB = scramble_interaction_keep_diag(J0; rng=rng, lock_pairs=lock_pairs)
            # J1, infoLockBase = lock_edge_to_reference(J1, J0)

            b0 = compute_rmed_series_stable(J0, u, t_vals; perturb=perturb)
            b1 = compute_rmed_series_stable(J1, u, t_vals; perturb=perturb)

            if any(!isfinite, b0) || any(!isfinite, b1)
                @info "skip baseline non-finite" L.name infoB
            else
                B .+= abs.(b1 .- b0);  nB += 1
                sum_b0 .+= b0; sum_b1 .+= b1
            end
        end

        Δ_struct ./= max(nS,1);  B ./= max(nB,1)

        results[L.name] = (
            t = t_vals,
            delta_struct = Δ_struct,
            baseline = B,
            excess = Δ_struct .- B,
            rmed_struct = nS>0 ? sum_f./nS : fill(NaN, nt),
            rmed_scram  = nS>0 ? sum_g./nS : fill(NaN, nt),
            rmed_base0  = nB>0 ? sum_b0./nB : fill(NaN, nt),
            rmed_base1  = nB>0 ? sum_b1./nB : fill(NaN, nt),
            reps_struct = nS, reps_base = nB
        )
    end

    return results
end

using Statistics
using CairoMakie  # or GLMakie

# -------- per-level figure --------
function plot_condition(res; title::AbstractString="Interaction-shuffle vs baseline")
    fig = Figure(size=(1100,700))

    ax1 = Axis(
        fig[1,1], xscale=log10, xlabel="t", ylabel="|ΔR̃med|", title=title,
        limits = ((nothing, nothing), (0.0, 0.55)),
    )
    lines!(ax1, res.t, res.delta_struct, color=:orangered, linewidth=2,
        label="structured → interaction-shuffled")
    lines!(ax1, res.t, res.baseline, color=:steelblue, linewidth=2,
        label="baseline ER→ interaction-shuffled")
    axislegend(ax1, position=:rt, framevisible=false)

    ax2 = Axis(fig[2,1], xscale=log10, xlabel="t", ylabel="excess |Δ|",
        title="Excess structural effect (structured − ER baseline)")
    lines!(ax2, res.t, res.excess, color=:purple, linewidth=2)

    display(fig)
    return fig
end

# --- summarize dict-of-levels into scalars for the path plot ---
function summarize_results(results::Dict{String,Any};
                           level_order::Vector{String} = collect(keys(results)),
                           mid_frac::Tuple{Float64,Float64} = (0.35, 0.65),
                           large_k::Int = 5)

    out = NamedTuple[]
    for name in level_order
        r = results[name]
        t  = r.t
        ex = r.excess
        ds = r.delta_struct
        bl = r.baseline

        # indices for mid window on log10 scale
        lt = log10.(t)
        lo = lt[1]; hi = lt[end]
        a = lo + mid_frac[1]*(hi-lo)
        b = lo + mid_frac[2]*(hi-lo)
        mid_idx = findall(i -> (lt[i] ≥ a && lt[i] ≤ b), eachindex(t))
        isempty(mid_idx) && (mid_idx = round.(Int, range(length(t)÷3, 2length(t)÷3, length=5)))
        println("mid_idx: ", mid_idx, "which equals: ", t[mid_idx])
        # large-t indices (last K points)
        k = min(large_k, length(t))
        big = (length(t)-k+1):length(t)

        mid_excess   = mean(skipmissing(@view ex[mid_idx]))
        large_struct = mean(skipmissing(@view ds[big]))
        large_base   = mean(skipmissing(@view bl[big]))

        push!(out, (; label=name, mid_excess, large_struct, large_base))
    end
    return out
end

# -------- path summary figure --------
function plot_path_summary(summaries::Vector{<:NamedTuple})
    names = [r.label for r in summaries]
    x = 1:length(summaries)
    mid = abs.([r.mid_excess for r in summaries])
    Ls  = abs.([r.large_struct for r in summaries])
    Lb  = abs.([r.large_base   for r in summaries])

    fig = Figure(size=(900,420))
    ax = Axis(fig[1,1], xlabel="STRUCTURAL LEVEL", ylabel="|ΔStructured - ΔBaseline|")
    lines!(ax, x, mid, linewidth=3, label="mid-t excess (non-trivial)")
    lines!(ax, x, Ls,  linewidth=2, label="large-t (structured)")
    lines!(ax, x, Lb,  linewidth=2, label="large-t (ER baseline)")
    axislegend(ax, position=:lc, framevisible=false)
    ax.xticks = (x, names)
    display(fig)
end

# ---------------- usage ----------------
resdict_zerolock = run_interaction_only_path(; S=120, conn=0.10, mean_abs=0.50, mag_cv=0.60,
                                u_mean=1.0, u_cv=0.6, reps=40, lock_pairs=0)
resdict_onelock = run_interaction_only_path(; S=120, conn=0.10, mean_abs=0.50, mag_cv=0.60,
                                u_mean=1.0, u_cv=0.6, reps=40, lock_pairs=1)
A = [resdict_zerolock]

# keep intended order, with new first level
level_order = ["baselineStruct","ER","degCV","deg+mag","trophic","trophic+","niche"]
for name in level_order
    for lock in 1
        resdict = A[lock]
        haskey(resdict, name) || continue
        plot_condition(resdict[name]; title="Interaction-shuffle vs baseline — $(name) lock-pairs=$(lock)")
    end
end

summ = summarize_results(resdict_onelock; level_order=level_order, mid_frac=(0.35,0.65), large_k=5)
plot_path_summary(summ)

