########################  Collectivity vs Shape — Experiment Script  ########################
using LinearAlgebra, Random, Statistics
# Optional: CairoMakie or GLMakie for plotting
using CairoMakie

# ===== 0) Time-resolved recoverability (overflow-safe Schur-shift) =====
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

# ===== 1) Helpers: A-space shuffles, rho-matching, diagnostics =====

"""
    shuffle_A_eigvecs(A; rng=Xoshiro(42), lock_pairs=0)

Spectrum-preserving **A-space** shuffle: A = Z*T*Z', rotate only the bulk Schur
subspace by an orthogonal matrix. `lock_pairs` counts leading 1×1/2×2 Schur
blocks to freeze (0 means rotate all).
"""
function shuffle_A_eigvecs(A::AbstractMatrix{<:Real};
                           rng::AbstractRNG=Xoshiro(42), lock_pairs::Int=0)
    F = schur(Matrix{Float64}(A))
    Z, T = F.Z, F.T
    n = size(T,1)

    # parse real-Schur blocks
    blocks = UnitRange{Int}[]
    i = 1
    while i ≤ n
        if i < n && abs(T[i+1,i]) > 0.0
            push!(blocks, i:(i+1)); i += 2
        else
            push!(blocks, i:i);     i += 1
        end
    end
    k = clamp(sum(length.(blocks[1:clamp(lock_pairs,0,length(blocks))])), 0, n)
    m = n - k
    if m == 0
        return Z*T*Z'
    end

    Q = qr!(randn(rng, m, m)).Q
    U = Matrix{Float64}(I, n, n)
    @views U[k+1:end, k+1:end] .= Q

    T2 = transpose(U) * T * U
    return Z * T2 * transpose(Z)
end

"""
    rho_match(A, ρ_target) -> Ã

Rescale A so that `maximum(abs,eigvals(Ã)) == ρ_target` (no change in shape).
"""
function rho_match(A::AbstractMatrix{<:Real}, ρ_target::Real)
    ρA = maximum(abs.(eigvals(Matrix{Float64}(A))))
    if ρA == 0.0
        return A   # nothing to scale (degenerate)
    else
        return (ρ_target / ρA) .* A
    end
end

# ---- diagnostics on a single (D,A) pair ----
struct Diag
    rho::Float64
    fro::Float64
    cplx::Float64       # May-like complexity ≈ ||A||_F / sqrt(S)
    rho_over_fro::Float64
    comm_DA::Float64    # ||[D,A]||_F
    J_nonnorm::Float64  # ||J'J - JJ'||_F
    kappaZ::Float64     # cond(Z) from real-Schur of J
end

function diagnostics(D::Diagonal{<:Real}, A::AbstractMatrix{<:Real})
    S = size(A,1)
    J = Matrix(D) + Matrix(A)
    ρ = maximum(abs.(eigvals(Matrix{Float64}(A))))
    fro = norm(A)
    cplx = fro / sqrt(S)
    Jn = norm(J'J - J*J')
    C = D*A - A*D
    FJ = schur(Matrix{Float64}(J))
    κZ = cond(FJ.Z)
    return Diag(ρ, fro, cplx, ρ/(fro+eps()), norm(C), Jn, κZ)
end

# ===== 2) Builders (thin wrappers with realized_IS rescaling) =====
build_trophic_ER = function (S; conn, mean_abs, mag_cv, deg_cv, rho_pair, rho_mag, rho_sign, rng)
    A = build_ER_degcv(S, conn, mean_abs, mag_cv, rho_pair, rho_mag, rho_sign, deg_cv; rng=rng)
    isA = realized_IS(A); isA == 0 ? A : A .* (mean_abs / isA)
end

build_ER_baseline = function (S; conn, mean_abs, mag_cv, rng)
    A = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                                degree_family=:uniform, deg_param=0.0, rho_sym=0.0, rng=rng)
    isA = realized_IS(A); isA == 0 ? A : A .* (mean_abs / isA)
end

build_niche_rescaled = function (S; conn, mean_abs, mag_cv, deg_cv, rho_mag, rng)
    A = build_niche_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                            degree_family=:lognormal, deg_param=deg_cv,
                            rho_sym=rho_mag, rng=rng)
    isA = realized_IS(A); isA == 0 ? A : A .* (mean_abs / isA)
end

# ===== 3) Per-replicate runner (single level) =====
"""
    run_rep(A; u, t_vals, rng, perturb=:biomass, lock_pairs=0)

Returns a NamedTuple with series for:
- raw rewire (A → A_raw via rewire_pairs_preserving_values)
- rho-matched rewire (A_raw scaled to match ρ(A))
- A-space spectrum-preserving shuffle (A_shuf)

All with the *same* D (extracted from the baseline J0).
"""
function run_rep(A::AbstractMatrix{<:Real};
                 u::AbstractVector{<:Real},
                 t_vals::AbstractVector{<:Real},
                 rng::AbstractRNG,
                 perturb::Symbol=:biomass,
                 lock_pairs::Int=0)

    # Baseline J0 and D (pin small-t)
    J0 = jacobian(A, u)
    D  = Diagonal(diag(J0))              # keep this fixed across variants

    # --- RAW REWIRE: degree-preserving pair rewiring (changes ρ generically)
    A_raw = rewire_pairs_preserving_values(A; rng=rng, random_targets=true, preserving_pairs=false)
    J_raw = Matrix(D) + Matrix(A_raw)

    # --- RHO-MATCHED SHAPE: rescale A_raw to match collectivity of A
    ρA = maximum(abs.(eigvals(Matrix{Float64}(A))))
    A_rmatch = rho_match(A_raw, ρA)

    # froA = norm(A)
    # norm_raw = norm(A_raw)
    # A_rmatch = norm_raw == 0 ? A_raw : (froA / norm_raw) .* A_raw

    J_rmatch = Matrix(D) + Matrix(A_rmatch)

    # --- A-SPACE SHUFFLE (spectrum preserved): control for “pure shape”
    A_shuf = shuffle_A_eigvecs(A; rng=rng, lock_pairs=lock_pairs)
    J_shuf = Matrix(D) + Matrix(A_shuf)

    # time series
    f0 = compute_rmed_series_stable(J0,     u, t_vals; perturb=perturb)
    fR = compute_rmed_series_stable(J_raw,  u, t_vals; perturb=perturb)
    fS = compute_rmed_series_stable(J_rmatch,u, t_vals; perturb=perturb)
    fA = compute_rmed_series_stable(J_shuf, u, t_vals; perturb=perturb)

    # deltas
    Δ_raw    = abs.(fR .- f0)
    Δ_shape  = abs.(fS .- f0)
    Δ_control= abs.(fA .- f0)
    Δ_sigma  = Δ_raw .- Δ_shape

    # diagnostics (baseline vs variants)
    d0 = diagnostics(D, A)
    dR = diagnostics(D, A_raw)
    dS = diagnostics(D, A_rmatch)
    dA = diagnostics(D, A_shuf)

    # @info "The nonorm J'J - JJ' is $(d0.J_nonnorm), $(dR.J_nonnorm), $(dS.J_nonnorm) for rho-matched"
    # @info "The difference in J'J - JJ' is $(dR.J_nonnorm-d0.J_nonnorm), $(dS.J_nonnorm-d0.J_nonnorm)"

    return (t = t_vals,
            f0=f0, f_raw=fR, f_rmatch=fS, f_A=fA,
            Δ_raw=Δ_raw, Δ_shape=Δ_shape, Δ_sigma=Δ_sigma, Δ_control=Δ_control,
            diag = (; base=d0, raw=dR, rmatch=dS, A=dA))
end

# ===== 4) Run the suite across a structure path =====
"""
    run_collectivity_suite(; S, conn, mean_abs, mag_cv, u_mean, u_cv,
                             t_vals, reps, seed, perturb, lock_pairs)

Levels (discrete gradient of structure): baselineStruct, ER, degCV, deg+mag, trophic, trophic+, niche.
Returns a Dict(name => result) where each result bundles mean curves and diagnostics.
"""
function run_collectivity_suite(; S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.50,
    mag_cv::Float64=0.60, u_mean::Float64=1.0, u_cv::Float64=0.6,
    t_vals=10 .^ range(-2, 2; length=40), reps::Int=40, seed::Int=0xBEEF,
    perturb::Symbol=:biomass, lock_pairs::Int=0)

    rng_master = Xoshiro(seed)
    nt = length(t_vals)

    levels = [
        (name="baselineStruct", kind=:BASE,  deg_cv=0.0, rho_pair=0.0, rho_mag=0.0,  rho_sign=0.0),
        (name="baselineStructFromER", kind=:ER,  deg_cv=0.0, rho_pair=0.0, rho_mag=0.0,  rho_sign=0.0),
        (name="ER",             kind=:ER,    deg_cv=0.0, rho_pair=1.0, rho_mag=0.0,  rho_sign=0.0),
        (name="degCV_notPaired",          kind=:ER,    deg_cv=5.0, rho_pair=0.0, rho_mag=0.0,  rho_sign=0.0),
        (name="degCV_paired",          kind=:ER,    deg_cv=5.0, rho_pair=1.0, rho_mag=0.0,  rho_sign=0.0),
        # (name="deg+mag",        kind=:ER,    deg_cv=0.8, rho_pair=1.0, rho_mag=0.99, rho_sign=0.0),
        (name="trophic_nonskew",        kind=:ER,    deg_cv=0.8, rho_pair=1.0, rho_mag=0.0, rho_sign=1.0),
        (name="trophic_pureskew",        kind=:ER,    deg_cv=0.8, rho_pair=1.0, rho_mag=1.0, rho_sign=1.0),
        # (name="trophic+",       kind=:ER,    deg_cv=1.2, rho_pair=0.0, rho_mag=1.0, rho_sign=1.0),
        # (name="niche",          kind=:NICHE, deg_cv=1.0, rho_pair=0.0, rho_mag=1.0, rho_sign=1.0),
    ]

    results = Dict{String,Any}()

    println("level | reps |  ⟨small-t Δ (shape)⟩   ⟨large-t Δ (shape)⟩   ⟨ρ drift (raw)⟩")
    println("------+------|-----------------------|-----------------------|---------------")

    for L in levels
        @info "Running level $(L.name)"
        sum_f0  = zeros(nt)
        sum_raw = zeros(nt); sum_rmatch = zeros(nt); sum_A = zeros(nt)
        sum_Δraw = zeros(nt); sum_Δshape = zeros(nt); sum_Δsigma = zeros(nt); sum_Δctrl = zeros(nt)

        # diagnostics accumulators
        d_keys = (:rho,:fro,:cplx,:rho_over_fro,:comm_DA,:J_nonnorm,:kappaZ)
        acc = Dict(k => Float64[] for k in d_keys)  # store (base, raw, rmatch, A) as tuples later

        n = 0
        for r in 1:reps
            rng = Xoshiro(rand(rng_master, UInt64))
            u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

            A =
                L.kind === :NICHE ? build_niche_rescaled(S; conn, mean_abs, mag_cv,
                                                         deg_cv=L.deg_cv, rho_mag=L.rho_mag, rng=rng) :
                L.kind === :BASE  ? build_ER_baseline(S; conn, mean_abs, mag_cv, rng) :
                                    build_trophic_ER(S; conn, mean_abs, mag_cv,
                                                     deg_cv=L.deg_cv, rho_pair=L.rho_pair, rho_mag=L.rho_mag,
                                                     rho_sign=L.rho_sign, rng=rng)

            rep = run_rep(A; u=u, t_vals=t_vals, rng=rng, perturb=perturb, lock_pairs=lock_pairs)

            if any(!isfinite, rep.f0) || any(!isfinite, rep.f_rmatch) || any(!isfinite, rep.f_raw)
                @info "skip non-finite series at level $(L.name)"
                continue
            end

            sum_f0    .+= rep.f0
            sum_raw   .+= rep.f_raw
            sum_rmatch .+= rep.f_rmatch
            sum_A     .+= rep.f_A

            sum_Δraw  .+= rep.Δ_raw
            sum_Δshape .+= rep.Δ_shape
            sum_Δsigma .+= rep.Δ_sigma
            sum_Δctrl .+= rep.Δ_control

            # diag deltas (store raw numbers; you can compare later)
            for (tag, d) in pairs(rep.diag)
                # accumulate only base/raw/rmatch for collectivity story
                if tag === :base || tag === :raw || tag === :rmatch
                    push!(acc[:rho], d.rho)
                    push!(acc[:fro], d.fro)
                    push!(acc[:cplx], d.cplx)
                    push!(acc[:rho_over_fro], d.rho_over_fro)
                    push!(acc[:comm_DA], d.comm_DA)
                    push!(acc[:J_nonnorm], d.J_nonnorm)
                    push!(acc[:kappaZ], d.kappaZ)
                end
            end

            n += 1
        end

        n == 0 && (results[L.name] = (; label=L.name, n=0); continue)

        f0_bar  = sum_f0 ./ n
        raw_bar = sum_raw ./ n
        rmatch_bar = sum_rmatch ./ n
        A_bar   = sum_A ./ n

        Δraw_bar   = sum_Δraw ./ n
        Δshape_bar = sum_Δshape ./ n
        Δsigma_bar = sum_Δsigma ./ n
        Δctrl_bar  = sum_Δctrl ./ n

        # quick console summary
        small_ix = 1:clamp(max(2, nt ÷ 6), 1, nt)      # earliest sixth
        large_ix = (nt - (nt ÷ 6) + 1):nt               # latest sixth
        smallΔ = mean(Δshape_bar[small_ix])
        largeΔ = mean(Δshape_bar[large_ix])

        # ρ drift (raw vs base)
        ρdrift = mean(@view acc[:rho][2:3:end]) - mean(@view acc[:rho][1:3:end])  # crude: uses appended order base,raw,rmatch
        @printf("%-10s %4d | %7.3f                | %7.3f                | %7.3f\n",
                L.name, n, smallΔ, largeΔ, ρdrift)

        results[L.name] = (;
            label = L.name, n=n, t=t_vals,
            f0=f0_bar, f_raw=raw_bar, f_rmatch=rmatch_bar, f_A=A_bar,
            Δ_raw=Δraw_bar, Δ_shape=Δshape_bar, Δ_sigma=Δsigma_bar, Δ_control=Δctrl_bar,
            diag_means = (; rho = mean(acc[:rho]), fro = mean(acc[:fro]),
                           cplx = mean(acc[:cplx]), rho_over_fro = mean(acc[:rho_over_fro]),
                           comm_DA = mean(acc[:comm_DA]), J_nonnorm = mean(acc[:J_nonnorm]),
                           kappaZ = mean(acc[:kappaZ]))
        )
    end

    return results
end

# ===== 5) Plotting =====
function plot_deltas(res; title="Collectivity vs shape")
    fig = Figure(size=(1100,700))
    ax1 = Axis(fig[1,1], xscale=log10, xlabel="t", ylabel="|ΔR̃med|", title=title)
    lines!(ax1, res.t, res.Δ_raw,    linewidth=2, color=:crimson,   label="raw rewire")
    lines!(ax1, res.t, res.Δ_shape,  linewidth=2, color=:seagreen,  label="Φ-matched")
    lines!(ax1, res.t, res.Δ_sigma,  linewidth=2, color=:orange,    label="collectivity component (raw - shape)")
    lines!(ax1, res.t, res.Δ_control,linewidth=2, color=:slateblue, label="A-space shuffle (spectrum-preserving)")
    axislegend(ax1, position=:rt, framevisible=false)

    ax2 = Axis(fig[2,1], xscale=log10, xlabel="t", ylabel="R̃med (means)")
    lines!(ax2, res.t, res.f0,      color=:black,    linewidth=2, label="baseline")
    lines!(ax2, res.t, res.f_raw,   color=:crimson,  linewidth=2, linestyle=:dash, label="raw")
    lines!(ax2, res.t, res.f_rmatch,color=:seagreen, linewidth=2, linestyle=:dash, label="ρ-matched")
    lines!(ax2, res.t, res.f_A,     color=:slateblue,linewidth=2, linestyle=:dash, label="A-shuffle")
    axislegend(ax2, position=:rb, framevisible=false)

    display(fig)
end

function plot_path_summary(results::Dict{String,Any}, level_order::Vector{String})
    x = 1:length(level_order)
    mid_idx = (i->i)(round.(Int, range(length(results[level_order[1]].t)÷3,
                                       2length(results[level_order[1]].t)÷3, length=7)))

    mids = [mean(@view results[name].Δ_shape[mid_idx]) for name in level_order]
    Lend = [mean(@view results[name].Δ_shape[end-4:end]) for name in level_order]
    Rdr  = [results[name].diag_means.rho for name in level_order]

    fig = Figure(size=(950,420))
    ax = Axis(fig[1,1], xlabel="structure level", ylabel="magnitude")
    lines!(ax, x, mids, linewidth=3, label="mid-t (ρ-matched / shape)")
    lines!(ax, x, Lend, linewidth=2, label="large-t (ρ-matched)")
    axislegend(ax, position=:lt, framevisible=false)
    ax.xticks = (x, level_order)
    display(fig)

    fig2 = Figure(size=(950,420))
    ax2 = Axis(fig2[1,1], xlabel="structure level", ylabel="⟨ρ(A)⟩")
    lines!(ax2, x, Rdr, linewidth=3, label="collectivity ⟨ρ⟩ (baseline)")
    axislegend(ax2, position=:lt, framevisible=false)
    ax2.xticks = (x, level_order)
    display(fig2)

end

# ===== 6) Example usage =====
S, conn, mean_abs, mag_cv = 120, 0.10, 0.50, 0.60
t_vals = 10 .^ range(-2, 2; length=40)

results = run_collectivity_suite(; S, conn, mean_abs, mag_cv,
    u_mean=1.0, u_cv=0.6, t_vals, reps=40, seed=Int(0xBEEF),
    perturb=:biomass, lock_pairs=0)

# Per-level visualisation
level_order = ["baselineStruct","ER","degCV","deg+mag","trophic","trophic+","niche"]
level_order = ["baselineStruct", "BaselineStructFromER", "ER", "degCV_notPaired", "degCV_paired", "trophic_nonskew", "trophic_pureskew"]
for name in level_order
    haskey(results, name) || continue
    plot_deltas(results[name]; title="Collectivity vs shape — $(name) matched by frobenius norm")
end

# Path summary
plot_path_summary(results, level_order)
############################################################################################
