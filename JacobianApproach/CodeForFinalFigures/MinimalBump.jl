################################################################################
# MINIMAL “BUMP” PIPELINE (biomass-weighted rmed only)
#
# What this does (no resolvents, no τ, no sensitivity spectra):
#   1) Build a stable base Jacobian J = -diag(u) + B
#   2) Destroy structure by rewiring the OFF-DIAGONAL entries of B (including zeros)
#      -> Jrew = -diag(u) + reshuffle(B)
#   3) Compute biomass-weighted rmed(t) for both
#   4) Define Δ(t) = |rmed_base(t) - rmed_rew(t)|
#   5) Summarize “bump-ness” by:
#        bump_strength = max_t Δ(t) / Δ(t_max)
#      ( > 1 means intermediate error exceeds the late-time level )
#
# One knob only:
#   η in [0,1] controls how “directed / feedforward” B is.
#   η=0: mostly symmetric pairs (more normal-ish)
#   η=1: mostly upper-triangular directed edges (more non-normal-ish)
#
# Output plots (only 3):
#   (1) mean Δ(t) vs t for each η
#   (2) bump_strength summary vs η (median ± IQR)
#   (3) bump_strength scatter (jittered) vs η
################################################################################

using Random, LinearAlgebra, Statistics, Distributions
using CairoMakie
using Base.Threads

# Avoid BLAS oversubscription when threading
try
    BLAS.set_num_threads(1)
catch
end

# ------------------------------------------------------------------------------
# 1) Biomass-weighted rmed(t)
# ------------------------------------------------------------------------------
"""
Biomass-weighted median return rate rmed(t):
  E = exp(J t)
  C = diag(u^2)
  rmed(t) = - ( log(tr(E C E')) - log(tr(C)) ) / (2t)
"""
function rmed_biomass(J::AbstractMatrix, u::AbstractVector; t::Real)
    t = float(t)
    t <= 0 && return NaN
    E = exp(t * J)
    any(!isfinite, E) && return NaN

    w = u .^ 2
    Ttr = tr(E * Diagonal(w) * transpose(E))
    (!isfinite(Ttr) || Ttr <= 0) && return NaN

    r = -(log(Ttr) - log(sum(w))) / (2t)
    return isfinite(r) ? r : NaN
end

function rmed_curve(J::AbstractMatrix, u::AbstractVector, tvals::Vector{Float64})
    r = Vector{Float64}(undef, length(tvals))
    for (i,t) in enumerate(tvals)
        r[i] = rmed_biomass(J, u; t=t)
    end
    return r
end

function delta_curve(r1::AbstractVector, r2::AbstractVector)
    n = length(r1)
    @assert length(r2) == n
    Δ = Vector{Float64}(undef, n)
    for i in 1:n
        a, b = r1[i], r2[i]
        Δ[i] = (isfinite(a) && isfinite(b)) ? abs(a - b) : NaN
    end
    return Δ
end

# ------------------------------------------------------------------------------
# 2) Timescales u
# ------------------------------------------------------------------------------
function random_u(S::Int; mean::Real=1.0, cv::Real=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + float(cv)^2))
    mu = log(float(mean)) - sigma^2/2
    return rand(rng, LogNormal(mu, sigma), S)
end

# ------------------------------------------------------------------------------
# 3) Simple structure knob η: symmetric pairs ↔ upper-triangular edges
# ------------------------------------------------------------------------------
"""
Build an off-diagonal interaction matrix B (diag=0) with exactly L directed edges.

Construction:
  - Choose npairs = round((1-η)*L/2) unordered pairs (i<j),
    set B[i,j]=w and B[j,i]=w  (symmetric pair, more normal-ish)
  - Remaining Lrem = L - 2*npairs edges go ONLY in the upper triangle (i<j),
    set B[i,j]=w                (directed feedforward-ish)

Then rescale B so ||B||_F = strength * ||u||_2 (constant relative strength).
"""
function build_B_eta(S::Int, η::Real, L::Int, σ::Real, u::Vector{Float64};
                     strength::Real=0.25, rng=Random.default_rng())
    η = float(η)
    @assert 0.0 <= η <= 1.0
    @assert L >= 1

    B = zeros(Float64, S, S)

    # all unordered pairs i<j
    pairs = Tuple{Int,Int}[]
    sizehint!(pairs, (S*(S-1)) ÷ 2)
    for i in 1:S-1, j in i+1:S
        push!(pairs, (i,j))
    end

    # number of symmetric pairs
    npairs = round(Int, (1.0 - η) * L / 2)
    npairs = clamp(npairs, 0, length(pairs))

    # sample symmetric pairs
    if npairs > 0
        perm = randperm(rng, length(pairs))
        for k in 1:npairs
            (i,j) = pairs[perm[k]]
            w = randn(rng) * float(σ)
            B[i,j] = w
            B[j,i] = w
        end
    end

    # remaining directed edges, only upper triangle, excluding those already used
    Lrem = L - 2*npairs
    if Lrem > 0
        uppers = Tuple{Int,Int}[]
        for i in 1:S-1, j in i+1:S
            # skip if already filled by symmetric pair
            if B[i,j] == 0.0 && B[j,i] == 0.0
                push!(uppers, (i,j))
            end
        end
        Lrem = min(Lrem, length(uppers))
        permu = randperm(rng, length(uppers))
        for k in 1:Lrem
            (i,j) = uppers[permu[k]]
            B[i,j] = randn(rng) * float(σ)
        end
    end

    # rescale strength: ||B||_F = strength * ||u||_2
    nB = norm(B)
    nB == 0 && return B
    target = float(strength) * norm(u)
    B .*= (target / nB)

    return B
end

function build_B_eta_trophic(
    S::Int,
    η::Real,
    L::Int,
    σ::Real,
    u::Vector{Float64};
    strength::Real = 0.25,
    rng = Random.default_rng()
)
    η = float(η)
    @assert 0.0 ≤ η ≤ 1.0
    @assert L ≥ 1

    B = zeros(Float64, S, S)

    # all unordered pairs i < j
    pairs = Tuple{Int,Int}[]
    for i in 1:S-1, j in i+1:S
        push!(pairs, (i,j))
    end

    # number of reciprocal trophic pairs (+/-)
    nrecip = round(Int, (1.0 - η) * L / 2)
    nrecip = clamp(nrecip, 0, length(pairs))

    perm = randperm(rng, length(pairs))

    # --- reciprocal trophic (+/-) pairs ---
    for k in 1:nrecip
        i, j = pairs[perm[k]]
        w = abs(randn(rng) * σ)

        if rand(rng) < 0.5
            B[i,j] =  w   # i eats j
            B[j,i] = -w
        else
            B[i,j] = -w   # j eats i
            B[j,i] =  w
        end
    end

    # --- unilateral trophic (+/0 or -/0) pairs ---
    Lrem = L - 2*nrecip
    if Lrem > 0
        unused = Tuple{Int,Int}[]
        for (i,j) in pairs
            if B[i,j] == 0.0 && B[j,i] == 0.0
                push!(unused, (i,j))
            end
        end

        Lrem = min(Lrem, length(unused))
        permu = randperm(rng, length(unused))

        for k in 1:Lrem
            i, j = unused[permu[k]]
            w = abs(randn(rng) * σ)

            if rand(rng) < 0.5
                # i affects j
                B[i,j] =  w * (rand(rng) < 0.5 ? 1.0 : -1.0)
            else
                # j affects i
                B[j,i] =  w * (rand(rng) < 0.5 ? 1.0 : -1.0)
            end
        end
    end

    # --- rescale overall strength ---
    nB = norm(B)
    nB == 0 && return B
    B .*= (float(strength) * norm(u) / nB)

    return B
end

# ------------------------------------------------------------------------------
# 4) Rewire structure by permuting off-diagonal entries (including zeros)
# ------------------------------------------------------------------------------
function reshuffle_offdiagonal(M::AbstractMatrix; rng=Random.default_rng())
    S = size(M, 1)
    M2 = copy(Matrix(M))

    vals = Float64[]
    idxs = Tuple{Int,Int}[]
    for i in 1:S, j in 1:S
        if i != j
            push!(vals, float(M2[i,j]))
            push!(idxs, (i,j))
        end
    end

    perm = randperm(rng, length(vals))
    for k in 1:length(vals)
        (i,j) = idxs[k]
        M2[i,j] = vals[perm[k]]
    end
    return M2
end

function reshuffle_offdiagonal_pairs(M::AbstractMatrix; rng=Random.default_rng())
    S = size(M, 1)
    M2 = copy(Matrix(M))

    # collect unordered pairs i < j
    pairs = Tuple{Float64,Float64}[]
    idxs  = Tuple{Int,Int}[]

    for i in 1:S-1, j in i+1:S
        push!(pairs, (float(M2[i,j]), float(M2[j,i])))
        push!(idxs,  (i, j))
    end

    # shuffle the PAIRS
    perm = randperm(rng, length(pairs))

    # clear off-diagonal
    for i in 1:S, j in 1:S
        i != j && (M2[i,j] = 0.0)
    end

    # reassign shuffled pairs
    for k in eachindex(pairs)
        (i,j) = idxs[k]
        (a,b) = pairs[perm[k]]
        M2[i,j] = a
        M2[j,i] = b
    end

    return M2
end

# ------------------------------------------------------------------------------
# 5) Stability: keep it simple
# ------------------------------------------------------------------------------
spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigen(J).values))

"""
Build J = -diag(u) + B and (rarely) shrink B until stable enough.
This keeps everything simple and avoids low acceptance.

Returns (J, B_used, shrinks, alpha).
"""
function build_stable_J(u::Vector{Float64}, B::Matrix{Float64};
                        margin::Real=1e-3, shrink_factor::Real=0.85, max_shrinks::Int=20)
    S = length(u)
    D = -Diagonal(u)
    Bout = copy(B)
    J = Matrix(D + Bout)

    α = spectral_abscissa(J)
    k = 0
    while !(isfinite(α) && α < -float(margin)) && k < max_shrinks
        Bout .*= float(shrink_factor)
        J = Matrix(D + Bout)
        α = spectral_abscissa(J)
        k += 1
    end

    return (J=J, B=Bout, shrinks=k, alpha=α)
end

# ------------------------------------------------------------------------------
# 6) Bump summary (super minimal)
# ------------------------------------------------------------------------------
"""
Given Δ(t), define:
  Δ_end = Δ(t_max)
  bump_strength = max(Δ)/Δ_end
  bump_exists = 1 if argmax(Δ) < last_index and max(Δ) > (1+δ)*Δ_end
"""
function bump_summary(Δ::Vector{Float64}; δ::Real=0.05)
    # Use only finite values for max location/value
    idx = findall(isfinite, Δ)
    isempty(idx) && return (
        Δ_end=NaN, Δ_max=NaN, t_idx_max=0,
        bump_strength=NaN, bump_excess=NaN, bump_exists=0
    )

    Δf = Δ[idx]
    imax_local = argmax(Δf)
    i_max = idx[imax_local]
    Δ_max = Δ[i_max]
    Δ_end = Δ[end]

    # bump_excess works even if Δ_end ≈ 0 (no division)
    bump_excess = (isfinite(Δ_max) && isfinite(Δ_end)) ? (Δ_max - Δ_end) : NaN

    # bump_strength only defined if Δ_end > 0
    bump_strength = (isfinite(Δ_end) && Δ_end > 0 && isfinite(Δ_max) && Δ_max > 0) ? (Δ_max / Δ_end) : NaN

    # "bump exists" if interior maximum and clearly above end level
    bump_exists = (
        isfinite(Δ_end) && isfinite(Δ_max) &&
        i_max < length(Δ) &&
        Δ_end >= 0 &&
        Δ_max > (1.0 + float(δ)) * max(Δ_end, 0.0)
    ) ? 1 : 0

    return (
        Δ_end=Δ_end, Δ_max=Δ_max, t_idx_max=i_max,
        bump_strength=bump_strength, bump_excess=bump_excess,
        bump_exists=bump_exists
    )
end

# ------------------------------------------------------------------------------
# 7) MAIN PIPELINE (threaded-friendly)
# ------------------------------------------------------------------------------
"""
Run the simplified experiment:
  For each η:
    repeat n_reps times:
      - draw u
      - build B(η) with fixed L, σ and strength
      - build stable J
      - build Jrew by reshuffling B
      - compute rmed curves and Δ(t)
      - compute bump_strength
Returns a NamedTuple with everything.
"""
function run_simple_bump_pipeline(;
    S::Int=80,
    η_grid = collect(range(0.0, 1.0; length=7)),
    n_reps::Int=80,
    seed::Int=1234,
    # timescales
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    # interactions
    connectance::Real=0.05,     # sets L ≈ p*S*(S-1)
    σ::Real=1.0,
    strength::Real=0.25,        # ||B||_F = strength*||u||
    # stability
    margin::Real=1e-3,
    # time grid
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=50),
    # bump
    bump_delta::Real=0.05
)
    tvals = collect(float.(tvals))
    η_grid = collect(float.(η_grid))
    L = max(1, round(Int, float(connectance) * S * (S-1)))  # directed edges count target

    nη = length(η_grid)

    # store Δ curves + metrics per η
    Δcurves = [Vector{Vector{Float64}}() for _ in 1:nη]

    bump_strengths = [Float64[] for _ in 1:nη]
    bump_excesses  = [Float64[] for _ in 1:nη]
    max_deltas     = [Float64[] for _ in 1:nη]
    end_deltas     = [Float64[] for _ in 1:nη]
    bump_exists    = [Int[] for _ in 1:nη]
    shrinks_used   = [Int[] for _ in 1:nη]

    # reproducible per-job seeds
    seeds = [seed + 100_000*iη + irep for iη in 1:nη for irep in 1:n_reps]
    N = nη * n_reps

    ok = falses(N)
    out_iη = fill(0, N)
    out_Δ = Vector{Vector{Float64}}(undef, N)

    out_bs = fill(NaN, N)
    out_be = fill(NaN, N)
    out_mx = fill(NaN, N)
    out_ed = fill(NaN, N)
    out_ex = fill(NaN, N)
    out_sh = fill(0, N)

    Threads.@threads for job in 1:N
        iη = (job - 1) ÷ n_reps + 1
        η = η_grid[iη]
        rng = MersenneTwister(seeds[job])

        u = collect(random_u(S; mean=u_mean, cv=u_cv, rng=rng))
        B = build_B_eta_trophic(S, η, L, σ, u; strength=strength, rng=rng)

        built = build_stable_J(u, B; margin=margin)
        if !(isfinite(built.alpha) && built.alpha < -float(margin))
            ok[job] = false
            continue
        end

        Brew = reshuffle_offdiagonal(built.B; rng=rng)
        J = built.J
        Jrew = Matrix(-Diagonal(u) + Brew)

        r1 = rmed_curve(J, u, tvals)
        r2 = rmed_curve(Jrew, u, tvals)
        Δ = delta_curve(r1, r2)

        bs = bump_summary(Δ; δ=bump_delta)

        ok[job] = true
        out_iη[job] = iη
        out_Δ[job] = Δ

        out_bs[job] = bs.bump_strength
        out_ex[job] = bs.bump_excess
        out_mx[job] = bs.Δ_max
        out_ed[job] = bs.Δ_end
        out_be[job] = bs.bump_exists
        out_sh[job] = built.shrinks
    end

    # assemble single-threaded
    for job in 1:N
        ok[job] || continue
        iη = out_iη[job]
        push!(Δcurves[iη], out_Δ[job])

        push!(bump_strengths[iη], out_bs[job])
        push!(bump_excesses[iη],  out_ex[job])
        push!(max_deltas[iη],     out_mx[job])
        push!(end_deltas[iη],     out_ed[job])

        push!(bump_exists[iη], out_be[job])
        push!(shrinks_used[iη], out_sh[job])
    end

    accepted = sum(length.(bump_strengths))
    @info "Accepted reps total = $accepted / $(N)  (rate=$(accepted / max(1, N)))"

    return (
        params=(S=S, η_grid=η_grid, n_reps=n_reps, seed=seed,
                u_mean=u_mean, u_cv=u_cv, connectance=connectance, L=L,
                σ=σ, strength=strength, margin=margin,
                bump_delta=bump_delta),
        tvals=tvals,
        Δcurves=Δcurves,
        bump_strengths=bump_strengths,
        bump_excesses=bump_excesses,
        max_deltas=max_deltas,
        end_deltas=end_deltas,
        bump_exists=bump_exists,
        shrinks_used=shrinks_used
    )
end

# ------------------------------------------------------------------------------
# 8) Plotting (3 plots only)
# ------------------------------------------------------------------------------
mean_curve(curves::Vector{Vector{Float64}}) =
    isempty(curves) ? Float64[] : [mean(c[i] for c in curves if isfinite(c[i])) for i in 1:length(curves[1])]

function median_iqr(v::Vector{Float64})
    vv = filter(isfinite, v)
    isempty(vv) && return (med=NaN, q25=NaN, q75=NaN)
    return (med=median(vv), q25=quantile(vv, 0.25), q75=quantile(vv, 0.75))
end

function plot_simple_bump_results(res; figsize=(1900, 1100))
    η_grid = res.params.η_grid
    tvals = res.tvals

    Δcurves = res.Δcurves
    bump_strengths = res.bump_strengths
    bump_excesses  = res.bump_excesses
    max_deltas     = res.max_deltas
    end_deltas     = res.end_deltas

    fig = Figure(size=figsize)

    # (1) mean Δ(t) vs t for each η
    ax1 = Axis(fig[1, 1],
        xscale=log10,
        xlabel="t",
        ylabel="mean Δ(t) = mean |rmed_base(t) - rmed_rewired(t)|",
        title="Mean structure effect profile Δ(t) (biomass rmed)"
    )
    for (k, η) in enumerate(η_grid)
        mc = mean_curve(Δcurves[k])
        isempty(mc) && continue
        lines!(ax1, tvals, mc, linewidth=3, label="η=$(round(η, digits=2))")
    end
    axislegend(ax1; position=:lt)

    # Helpers for summaries
    function summary_line!(ax, xs, vecs; ylab, title, yscale=nothing, hline=nothing)
        meds = Float64[]; q25s = Float64[]; q75s = Float64[]
        for v in vecs
            st = median_iqr(v)
            push!(meds, st.med); push!(q25s, st.q25); push!(q75s, st.q75)
        end
        lines!(ax, xs, meds, linewidth=3)
        scatter!(ax, xs, meds, markersize=10)
        for i in eachindex(xs)
            if isfinite(q25s[i]) && isfinite(q75s[i])
                lines!(ax, [xs[i], xs[i]], [q25s[i], q75s[i]], linewidth=4)
            end
        end
        if hline !== nothing
            hlines!(ax, [hline], linestyle=:dash)
        end
        ax.ylabel = ylab
        ax.title = title
        if yscale !== nothing
            ax.yscale = yscale
        end
    end

    # (2) bump_strength summary vs η
    ax2 = Axis(fig[1, 2], xlabel="η")
    summary_line!(ax2, η_grid, bump_strengths;
        ylab="bump_strength = max Δ(t) / Δ(t_max)",
        title="Does intermediate dominate late?",
        hline=1.0
    )

    # (3) bump_excess summary vs η (absolute, no division)
    ax3 = Axis(fig[1, 3], xlabel="η")
    summary_line!(ax3, η_grid, bump_excesses;
        ylab="bump_excess = max Δ(t) - Δ(t_max)",
        title="Absolute intermediate excess over end"
    )
    hlines!(ax3, [0.0], linestyle=:dash)

    # (4) median maxΔ vs η
    ax4 = Axis(fig[2, 1], xlabel="η")
    summary_line!(ax4, η_grid, max_deltas;
        ylab="maxΔ = max_t Δ(t)",
        title="Median maxΔ vs η"
    )

    # (5) median Δ_end vs η
    ax5 = Axis(fig[2, 2], xlabel="η")
    summary_line!(ax5, η_grid, end_deltas;
        ylab="Δ_end = Δ(t_max)",
        title="Median endΔ vs η"
    )

    # (6) distribution of bump_strength (each dot)
    ax6 = Axis(fig[2, 3],
        xlabel="η",
        ylabel="bump_strength",
        title="Bump strength distribution (each dot = one base/rewire pair)"
    )
    for (k, η) in enumerate(η_grid)
        v = filter(isfinite, bump_strengths[k])
        isempty(v) && continue
        x = η .+ 0.02 .* (rand(length(v)) .- 0.5)
        scatter!(ax6, x, v, markersize=4)
    end
    hlines!(ax6, [1.0], linestyle=:dash)

    display(fig)
end

# ------------------------------------------------------------------------------
# MAIN (edit only these knobs)
# ------------------------------------------------------------------------------
tvals = collect(10 .^ range(log10(0.01), log10(100.0); length=50))

res = run_simple_bump_pipeline(
    S=120,
    η_grid=collect(range(0.0, 1.0; length=12)),
    n_reps=80,
    seed=1234,
    u_mean=1.0,
    u_cv=0.5,
    connectance=0.05,
    σ=1.0,
    strength=0.25,      # if too many unstable: lower to 0.20
    margin=1e-3,
    tvals=tvals,
    bump_delta=0.05
)

plot_simple_bump_results(res)