# --- Assumes you already have:
# SchurPack, schur_pack, _rmed_series_schur, compute_rmed_series,
# rewire_pairs_preserving_values, build_ER_degcv, build_random_nontrophic,
# random_u, jacobian, median_return_rate, r2_to_identity, realized_IS
using LinearAlgebra, Random, Statistics

"""
    compute_rmed_series_stable(J, u, t_vals; perturb=:biomass, margin=1e-6)

Exact, overflow-safe evaluation of R̃med(t) via real-Schur with a spectral shift.
Takes J’s Schur J = Z*T*Z'. Let μ = max Re(λ(J)) + margin. Define Es(t) = exp(t*(J-μI)).
Then exp(tJ) = exp(μ t) * Es(t), and

    tr( exp(tJ) C exp(tJ)' ) = exp(2 μ t) * tr( Es(t) C Es(t)' )

so inside log we add 2 μ t back. This is algebraically exact and prevents overflow.
"""
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

    # We compute Y(t) = exp(t*(T - μI)) * (Z' * diag(sqrt(w)))  and then ||Y||_F^2.
    M = transpose(Z) * Diagonal(sqrtw)

    μ = maximum(real.(vals)) + margin
    I_T = Matrix{Float64}(I, size(T,1), size(T,2))   # explicit I for clarity

    out = Vector{Float64}(undef, length(t_vals))
    @inbounds for (k, t) in pairs(t_vals)
        # Stable exponential of the shifted real-Schur factor
        Y = exp(t .* (T .- μ .* I_T)) * M
        s = sum(abs2, Y)                 # Frobenius norm squared
        # exact shift correction inside the log:
        num = (log(s) + 2.0*μ*t)
        den = log(sum(w))
        out[k] = - (num - den) / (2.0*t)
        if !isfinite(out[k])
            out[k] = NaN                 # caller will skip this rep if any NaN
        end
    end
    return out
end

"""
    bulk_scramble_lock_edge(J; freeze_k=1, rng=Xoshiro(42))

Freeze the slowest real-Schur block (1×1 or 2×2); orthogonally scramble the
complementary Schur subspace. Preserves the entire spectrum and the frozen block.
No `ordschur` is used; we permute explicitly for robustness.
"""
function bulk_scramble_lock_edge(J::AbstractMatrix{<:Real};
                                 freeze_k::Int=1,
                                 rng::AbstractRNG=Xoshiro(42))

    # --- 1) Real Schur: J = Z*T*Z'
    F = schur(Matrix{Float64}(J))
    Z = F.Z
    T = F.T
    vals = F.values
    S   = length(vals)

    # --- 2) Parse Schur blocks (1×1 or 2×2) and locate the slowest eigenvalue’s block
    blocks = UnitRange{Int}[]
    i = 1
    while i ≤ S
        if i < S && abs(T[i+1,i]) > 0.0
            push!(blocks, i:(i+1))
            i += 2
        else
            push!(blocks, i:i)
            i += 1
        end
    end

    slow_idx      = argmax(real.(vals))
    slow_block_id = findfirst(br -> slow_idx ∈ br, blocks)
    @assert slow_block_id !== nothing
    edge = blocks[slow_block_id]                 # either length 1 or 2
    k    = length(edge)
    m    = S - k
    m ≤ 0 && return Z*T*Z'                       # nothing to scramble

    # --- 3) Build an explicit permutation that brings EDGE first
    bulk_idx = collect(1:S)
    deleteat!(bulk_idx, edge)                    # remove the edge indices
    perm = vcat(collect(edge), bulk_idx)         # EDGE, then BULK

    # Permutation matrix P (Float64, not Bool)
    P = Matrix{Float64}(I, S, S)[:, perm]

    # Permuted Schur form and vectors
    Tp = transpose(P) * T * P
    Zp = Z * P
    
    # in bulk_scramble_lock_edge(..) just before Q = qr!(randn(...)).Q
    if m == 0
        return Z*T*Z'    # nothing to scramble
    end

    # --- after you compute Tp and Zp (Schur in permuted basis) ---
    Q = qr!(randn(rng, m, m)).Q
    U = Matrix{Float64}(I, S, S)
    @views U[k+1:end, k+1:end] .= Q

    # DO: only change T by similarity in the Schur space
    Tpp = transpose(U) * Tp * U

    # DON'T: post-multiply Z by U (that would undo the similarity)
    # Zpp = Zp * U      # <-- remove this line entirely

    # Reconstruct with Zp unchanged
    Jscr = Zp * Tpp * Zp'

    # --- 5) Undo the permutation
    # Pinv = transpose(P)                # permutation is orthogonal
    # Jscr = Zpp * Tpp * transpose(Zpp)  # back to the original basis

    return Jscr
end

# ---------- 2) Build communities ----------
build_trophic_ER = function (S; conn, mean_abs, mag_cv, deg_cv, rng)
    # trophic-ER with rho_mag=1.0, rho_sign=1.0 (pure trophic signs and full mag-corr)
    A = build_ER_degcv(S, conn, mean_abs, mag_cv, 0.9999, 1.0, deg_cv; rng=rng)
    isA = realized_IS(A); isA == 0 && return A
    A .* (mean_abs / isA)
end

build_ER_baseline = function (S; conn, mean_abs, mag_cv, rng)
    # plain ER-like non-trophic baseline (no sign pattern, rho_mag=1)
    A = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                degree_family=:uniform, deg_param=0.0, rho_sym=0.0, rng=rng)
    isA = realized_IS(A); isA == 0 && return A
    A .* (mean_abs / isA)
end

# ---------- 3) One experiment run ----------
"""
    run_bulk_vs_baseline(; S, conn, mean_abs, mag_cv, deg_cv, u_mean, u_cv,
                           t_vals, reps, seed, perturb=:biomass, freeze_k=1)

Returns a NamedTuple with:
  t, B(t)=baseline |ΔR̃med| (ER→ER), Δ_struct(t)=|R̃(J)-R̃(J_scramble)|,
  E(t)=Δ_struct - B(t), resilience(J), resilience(J_scramble),
  small_t_pred, edge_pred, nonnormality(J), nonnormality(J_scramble).
"""
function run_bulk_vs_baseline(; S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.50,
        mag_cv::Float64=0.60, deg_cv::Float64=0.8, u_mean::Float64=1.0, u_cv::Float64=0.6,
        t_vals=10 .^ range(-2, 2; length=40), reps::Int=50, seed::Int=20251114,
        perturb::Symbol=:biomass, freeze_k::Int=1)

    rng_master = Xoshiro(seed)
    nt = length(t_vals)

    # accumulators
    Δ_struct = zeros(length(t_vals))
    B        = zeros(length(t_vals))
    nS = 0; nB = 0

    # NEW: accumulate mean series
    sum_struct   = zeros(nt)  # R̃(J)
    sum_scram    = zeros(nt)  # R̃(J_scr)
    sum_base0    = zeros(nt)  # R̃(J0)    (optional)
    sum_base1    = zeros(nt)  # R̃(J1)

    # diagnostics (averaged)
    res_edge  = Float64[]; res_edge2 = Float64[]
    smallpred = Float64[]; edgepred  = Float64[]
    nn_J = Float64[]; nn_Jp = Float64[]

    # put these before the reps loop
    n_attempts_struct = 0; n_skipped_struct = 0
    n_attempts_base   = 0; n_skipped_base   = 0
    
    for r in 1:reps
        rng = Xoshiro(rand(rng_master, UInt64))

        # --- draw u once and reuse (same-u comparisons)
        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

        # --- structured case: trophic-ER
        A  = build_trophic_ER(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, deg_cv=deg_cv, rng=rng)
        J  = jacobian(A, u)
        Jp = bulk_scramble_lock_edge(J; freeze_k=freeze_k, rng=rng)   # edge locked, bulk scrambled
        
        n_attempts_struct += 1
        λ0 = maximum(real(eigvals(J)))
        λb = maximum(real(eigvals(Jp)))
        if λ0 >= -1e-10 || λb >= -1e-10
            n_skipped_struct += 1
            @info "Skip rep (struct) due to near-marginal stability: λ0=$(λ0), λb=$(λb)"
            continue
        end

        # ----- structured series -----
        f = compute_rmed_series_stable(J,  u, t_vals; perturb=perturb)
        g = compute_rmed_series_stable(Jp, u, t_vals; perturb=perturb)

        # NEW: store means for plotting
        sum_struct .+= f
        sum_scram  .+= g

        # after computing f,g:
        if any(!isfinite, f) || any(!isfinite, g)
            @info "Skip rep (struct) due to non-finite series"
            continue
        end

        Δ_struct .+= abs.(g .- f);  nS += 1


        # diag + edge predictors + non-normality
        push!(smallpred, -mean(diag(J)))
        push!(edgepred,  -maximum(real(eigvals(J))))
        push!(res_edge,  -maximum(real(eigvals(J))))
        push!(res_edge2, -maximum(real(eigvals(Jp))))
        push!(nn_J,  norm(J'J - J*J'))
        push!(nn_Jp, norm(Jp'Jp - Jp*Jp'))

        # --- baseline: ER→ER by pair rewiring (same-u)
        J0 = nothing
        J1 = nothing
        stable = false

        for attempt in 1:4
            A0 = build_ER_baseline(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
            A1 = rewire_pairs_preserving_values(A0; rng=rng, random_targets=true)

            J0_try = jacobian(A0, u)
            J1_try = jacobian(A1, u)

            λ0 = maximum(real(eigvals(J0_try)))
            λ1 = maximum(real(eigvals(J1_try)))

            if λ0 <= -1e-10 && λ1 <= -1e-10
                # success — keep these
                J0 = J0_try
                J1 = J1_try
                n_attempts_base += 1
                stable = true
                break
            elseif attempt == 4
                # give up after 4 attempts
                n_skipped_base += 1
                @info "Skip rep (baseline) after 4 failed attempts: λ0=$(λ0), λ1=$(λ1)"
            else
                # retry
                n_skipped_base += 1
                @info "Retry baseline (attempt $(attempt)) due to instability: λ0=$(λ0), λ1=$(λ1)"
            end
        end

        # if we never obtained a stable pair → skip the baseline part for this rep
        if !stable
            continue
        end
        # ----- baseline series -----
        b0 = compute_rmed_series_stable(J0, u, t_vals; perturb=perturb)
        b1 = compute_rmed_series_stable(J1, u, t_vals; perturb=perturb)

        # after computing b0,b1:
        if any(!isfinite, b0) || any(!isfinite, b1)
            @info "Skip rep (baseline) due to non-finite series"
            continue
        end

        B .+= abs.(b1 .- b0); nB += 1
    end

    println("STRUCT: skipped $n_skipped_struct / $n_attempts_struct  (", 
            round(100n_skipped_struct/max(n_attempts_struct,1), digits=1), "%)")
    println("BASE:   skipped $n_skipped_base / $n_attempts_base      (", 
            round(100n_skipped_base/max(n_attempts_base,1), digits=1), "%)")

    Δ_struct ./= max(nS,1);  B ./= max(nB,1)

    # NEW: finalize mean series (NaN if none accepted)
    mean_struct = (nS > 0 ? sum_struct ./ nS : fill(NaN, nt))
    mean_scram  = (nS > 0 ? sum_scram  ./ nS : fill(NaN, nt))
    mean_b0     = (nB > 0 ? sum_base0 ./ nB : fill(NaN, nt))
    mean_b1     = (nB > 0 ? sum_base1 ./ nB : fill(NaN, nt))

    return (
        ; t=t_vals,
        baseline=B,
        delta_struct=Δ_struct,
        excess=Δ_struct .- B,
        rmed_struct=mean_struct,      # NEW
        rmed_scram=mean_scram,        # NEW
        rmed_base0=mean_b0,           # NEW
        rmed_base1=mean_b1,           # NEW
        res_struct=mean(res_edge), res_struct_scr=mean(res_edge2),
        small_t_pred=mean(smallpred), edge_pred=mean(edgepred),
        nonnormality_struct=mean(nn_J), nonnormality_scr=mean(nn_Jp)
    )
end

# ---------- 4) Annex: u_cv sweep (kept compact) ----------
function annex_u_cv_sweep(; u_cvs=collect(range(0.2, 2.0; length=7)), kwargs...)
    out = Dict{Float64,Any}()
    for ucv in u_cvs
        out[ucv] = run_bulk_vs_baseline(; u_cv=ucv, kwargs...)
    end
    out
end

S, conn, mean_abs, mag_cv = 120, 0.10, 0.25, 0.60
t_vals = 10 .^ range(-2, 2; length=40)

res = run_bulk_vs_baseline(; S, conn, mean_abs, mag_cv,
                           deg_cv=0.9, u_mean=1.0, u_cv=0.6,
                           t_vals, reps=40, seed=Int64(0xBEEF), perturb=:biomass, freeze_k=1)

# res.excess is the non-trivial mid-t signal (structured minus baseline ER→ER)
# You can also inspect res.baseline (gray band), res.delta_struct, and diagnostics.

ann = annex_u_cv_sweep(; u_cvs=0.2:0.3:2.0, S=120, conn=0.10, mean_abs=0.50, mag_cv=0.60,
                       deg_cv=0.9, u_mean=1.0, t_vals=t_vals, reps=30, seed=Int64(0xBEEF),
                       perturb=:biomass, freeze_k=1)

# For each u_cv, plot ann[u_cv].excess and track how long-t sensitivity (|Δresilience|)
# and mid-t excess change with u_cv.

using CairoMakie

function plot_bulk_vs_baseline(res; title="Bulk vs baseline")
    fig = Figure(size=(1100, 700))
    # 1) Δ_struct and baseline
    ax1 = Axis(fig[1,1], xscale=log10, xlabel="t", ylabel="|ΔR̃med|", title=title)
    lines!(ax1, res.t, res.delta_struct, linewidth=2, color=:orangered, label="Δ_struct (tER → bulk-scrambled)")
    lines!(ax1, res.t, res.baseline,     linewidth=2, color=:steelblue, label="baseline (ER → ER)")
    axislegend(ax1, position=:rt, framevisible=false)

    # 2) Excess = Δ_struct − baseline
    ax2 = Axis(fig[2,1], xscale=log10, xlabel="t", ylabel="excess |Δ|",
               title="Excess structural effect (Δ_struct − baseline)")
    lines!(ax2, res.t, res.excess, linewidth=2, color=:purple)

    display(fig)
end

plot_bulk_vs_baseline(res)

# Suppose you also computed f̄(t) = mean R̃med(J) and ḡ(t) = mean R̃med(Jscr)
# then plot R̃med and t·R̃med on twin axes:

function plot_rmed_and_tres(t, rmed; title="R̃med and t·R̃med")
    fig = Figure(size=(900, 400))
    axL = Axis(fig[1,1], xscale=log10, xlabel="t", ylabel="R̃med", title=title)
    lines!(axL, t, rmed, linewidth=2)

    axR = Axis(fig[1,1], yaxisposition=:right, xscale=log10, ylabel="t·R̃med")
    hidespines!(axR, :l, :t); linkxaxes!(axL, axR)
    lines!(axR, t, t .* rmed, linewidth=2, linestyle=:dash)

    display(fig)
end

plot_rmed_and_tres(res.t, res.rmed_struct; title="Structured community")
plot_rmed_and_tres(res.t, res.rmed_scram;  title="Bulk-scrambled (edge frozen)")

plot_rmed_and_tres(res.t, res.rmed_base0; title="Baseline ER (source)")
plot_rmed_and_tres(res.t, res.rmed_base1; title="Baseline ER (rewired)")
