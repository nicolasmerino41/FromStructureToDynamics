using LinearAlgebra, Random, Statistics
# ------------------------------------------------------------
# 1) Scramble only the INTERACTION part K = J - diag(J)
#    - Preserves small-t (keeps diag(J) exactly)
#    - Changes eigenvectors of K; spectrum of full J will generally change.
# ------------------------------------------------------------
"""
    scramble_interaction_keep_diag(J; rng=Xoshiro(42))

Let J = D + K with D = Diagonal(diag(J)), K = J - D.
Compute a real-Schur of K, apply a random orthogonal similarity on the Schur basis,
and reconstruct Kscr. Finally, **force diag(Kscr) = diag(K)** so that
diag(D + Kscr) == diag(J). Return J' = D + Kscr and diagnostics.
"""
function scramble_interaction_keep_diag(J::AbstractMatrix{<:Real};
                                        rng::AbstractRNG = Xoshiro(42))
    S = size(J,1); @assert size(J,2) == S
    J = Matrix{Float64}(J)

    # Split J
    d  = diag(J)
    D  = Diagonal(d)
    K  = J .- D                     # by construction, diag(K) == 0

    # Real Schur of K
    F  = schur(K)                   # K = Z*T*Z'
    Z, T = F.Z, F.T

    # Random orthogonal similarity in Schur space: T -> Q' T Q
    Q  = qr!(randn(rng, S, S)).Q
    Tq = transpose(Q) * T * Q

    # Back-transform
    Kscr = Z * Tq * Z'

    # Repair diagonal so small-t is pinned (keep diag(K)=0 exactly)
    # This step breaks exact isospectrality of Kscr, but that's the point: we fix TS’s.
    @inbounds for i in 1:S
        Kscr[i,i] = K[i,i]          # K's diagonal is zero; enforce it exactly
    end

    Jprime = D + Kscr

    # Diagnostics
    diag_err = norm(diag(Jprime) .- d, Inf)                  # should be ~0
    lam0 = maximum(real(eigvals(J)))
    lamp = maximum(real(eigvals(Jprime)))
    lam_drift = lamp - lam0                                  # expect nonzero

    return Jprime, (; diag_err, lam_drift)
end

# ------------------------------------------------------------
# 2) Builders you already use
# ------------------------------------------------------------
build_trophic_ER = function (S; conn, mean_abs, mag_cv, deg_cv, rng)
    A = build_ER_degcv(S, conn, mean_abs, mag_cv, 0.0, 0.0, 0.0, deg_cv; rng=rng)
    isA = realized_IS(A); isA == 0 && return A
    A .* (mean_abs / isA)
end

build_ER_baseline = function (S; conn, mean_abs, mag_cv, rng)
    A = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                degree_family=:uniform, deg_param=0.0, rho_sym=0.0, rng=rng)
    isA = realized_IS(A); isA == 0 && return A
    A .* (mean_abs / isA)
end

# ------------------------------------------------------------
# 3) One analysis run: tER → (interaction-scrambled) vs baseline ER→ER
# ------------------------------------------------------------
function run_interaction_shuffle_vs_baseline(; S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.50,
        mag_cv::Float64=0.60, deg_cv::Float64=0.9, u_mean::Float64=1.0, u_cv::Float64=0.6,
        t_vals=10 .^ range(-2, 2; length=40), reps::Int=40, seed::Int=0xBEEF,
        perturb::Symbol=:biomass)

    rng_master = Xoshiro(seed)
    nt = length(t_vals)

    Δ_struct = zeros(nt)   # |Rmed(J) - Rmed(J')|
    B        = zeros(nt)   # baseline |Rmed(J0) - Rmed(J1)|
    nS = 0; nB = 0

    # optional: mean series for inspection
    sum_struct = zeros(nt); sum_scram = zeros(nt)
    sum_b0 = zeros(nt);     sum_b1   = zeros(nt)

    # counters
    n_attempts_struct = 0; n_skip_struct = 0
    n_attempts_base   = 0; n_skip_base   = 0

    for r in 1:reps
        rng = Xoshiro(rand(rng_master, UInt64))

        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

        # ----- Structured community -----
        A = build_trophic_ER(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, deg_cv=deg_cv, rng=rng)
        J = jacobian(A, u)
        n_attempts_struct += 1

        # Interaction-only eigenvector scramble
        Jp, info = scramble_interaction_keep_diag(J; rng=rng)

        # Small/large-t sanity: diagonal preserved ⇒ small-t diffs should be tiny.
        # We still require actual series to be finite.
        f = compute_rmed_series_stable(J,  u, t_vals; perturb=perturb)
        g = compute_rmed_series_stable(Jp, u, t_vals; perturb=perturb)
        if any(!isfinite, f) || any(!isfinite, g)
            n_skip_struct += 1
            @info "Skip struct rep (non-finite series) diag_err=$(info.diag_err) Δλ=$(info.lam_drift)"
            continue
        end

        Δ_struct .+= abs.(g .- f);  nS += 1
        sum_struct .+= f; sum_scram .+= g

        # ----- Baseline (ER → ER by pair rewiring), same-u -----
        # A few attempts to avoid marginal/unstable pairs
        stable = false
        for attempt in 1:4
            A0 = build_ER_baseline(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
            # A1 = rewire_pairs_preserving_values(A0; rng=rng, random_targets=true)

            J0 = jacobian(A0, u)
            # J1 = jacobian(A1, u)
            J1, info = scramble_interaction_keep_diag(J0; rng=rng)

            b0 = compute_rmed_series_stable(J0, u, t_vals; perturb=perturb)
            b1 = compute_rmed_series_stable(J1, u, t_vals; perturb=perturb)
            if any(!isfinite, b0) || any(!isfinite, b1)
                n_skip_base += 1
                @info "Retry baseline (attempt $attempt): non-finite series"
                continue
            end
            B .+= abs.(b1 .- b0);  nB += 1
            sum_b0 .+= b0; sum_b1 .+= b1
            stable = true
            break
        end
        if !stable
            @info "Skip baseline rep after retries"
        end
    end

    println("STRUCT: used $nS / $n_attempts_struct")
    println("BASE:   used $nB")

    Δ_struct ./= max(nS,1);  B ./= max(nB,1)
    mean_struct = (nS>0 ? sum_struct ./ nS : fill(NaN, nt))
    mean_scram  = (nS>0 ? sum_scram  ./ nS : fill(NaN, nt))
    mean_b0     = (nB>0 ? sum_b0     ./ nB : fill(NaN, nt))
    mean_b1     = (nB>0 ? sum_b1     ./ nB : fill(NaN, nt))

    return (; t=t_vals,
            delta_struct=Δ_struct,
            baseline=B,
            excess=Δ_struct .- B,
            rmed_struct=mean_struct,
            rmed_scram=mean_scram,
            rmed_base0=mean_b0,
            rmed_base1=mean_b1)
end

# ------------------------------------------------------------
# 4) Plots
# ------------------------------------------------------------
using CairoMakie

function plot_interaction_shuffle_results(res; title_top="Interaction-shuffle vs baseline when rho_sign=0, rho_mag=0.0 and deg_cv=0.0")
    fig = Figure(size=(1100, 720))

    ax1 = Axis(fig[1,1], xscale=log10, xlabel="t", ylabel="|ΔR̃med|", title=title_top)
    lines!(ax1, res.t, res.delta_struct, linewidth=2, color=:orangered, label="tER → interaction-scrambled")
    lines!(ax1, res.t, res.baseline,     linewidth=2, color=:steelblue, label="baseline ER→ER")
    axislegend(ax1, position=:rt, framevisible=false)

    ax2 = Axis(fig[2,1], xscale=log10, xlabel="t", ylabel="excess |Δ|",
               title="Excess structural effect (interaction-shuffle − baseline)")
    lines!(ax2, res.t, res.excess, linewidth=2, color=:purple)

    display(fig)
end

function plot_rmed_dual(t, rmed; title="R̃med and t·R̃med")
    fig = Figure(size=(900, 400))
    axL = Axis(fig[1,1], xscale=log10, xlabel="t", ylabel="R̃med", title=title)
    lines!(axL, t, rmed, linewidth=2)
    axR = Axis(fig[1,1], xscale=log10, yaxisposition=:right, ylabel="t·R̃med")
    hidespines!(axR, :l, :t); linkxaxes!(axL, axR)
    lines!(axR, t, t .* rmed, linewidth=2, linestyle=:dash)
    display(fig)
end

# ------------------------------------------------------------
# 5) Run once
# ------------------------------------------------------------
S, conn, mean_abs, mag_cv = 120, 0.10, 0.50, 0.60
t_vals = 10 .^ range(-2, 2; length=40)

res_base = run_interaction_shuffle_vs_baseline(; S, conn, mean_abs, mag_cv,
    deg_cv=0.0, u_mean=1.0, u_cv=0.6, t_vals, reps=40, seed=10000, perturb=:biomass)

plot_interaction_shuffle_results(res_base)
plot_rmed_dual(res_base.t, res_base.rmed_struct; title="Structured tER")
plot_rmed_dual(res_base.t, res_base.rmed_scram;  title="Interaction-scrambled (diag pinned)")
plot_rmed_dual(res_base.t, res_base.rmed_base0;  title="Baseline ER (source)")
plot_rmed_dual(res_base.t, res_base.rmed_base1;  title="Baseline ER (rewired)")
