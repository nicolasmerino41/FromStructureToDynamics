###############################################################
#   NEW SCRIPT 2 — TROPHIC-GEOMETRY REWIRED PIPELINE
#   (Baseline vs Rewired-within-q vs Min-q-rewired vs Max-q-rewired)
###############################################################

using LinearAlgebra
using Statistics
using Random
using Distributions
using CairoMakie
using Base.Threads

###############################################################
# 0. Utilities
###############################################################

"""
random_u(S; mean, cv)
Generate metabolic time scales u_i > 0 with (mean, cv).
"""
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    σ = mean * cv
    return rand(rng, LogNormal(log(mean^2 / sqrt(σ^2 + mean^2)),
                               sqrt(log(1 + (σ^2 / mean^2)))), S)
end

###############################################################
# 1. TROPHIC COHERENCE (as in Script 1)
###############################################################

"""
    trophic_coherence(Aplus)

Aplus = consumer → resource adjacency with positive magnitudes.
Returns (q, ℓ).
"""
function trophic_coherence(Ap)
    S = size(Ap,1)
    kin = vec(sum(Ap .> 0, dims=2))

    if sum(kin) == 0
        return 0.0, ones(S)
    end

    D = Diagonal(kin .+ 1e-6)
    M = D - (Ap .> 0)     # structure only

    ℓ = try
        M \ ones(S)
    catch
        ones(S)
    end

    diffs = Float64[]
    for i in 1:S, j in 1:S
        if Ap[i,j] > 0
            push!(diffs, ℓ[j] - ℓ[i] - 1)
        end
    end

    return std(diffs), ℓ
end

###############################################################
# 2. BUILD TROPHIC NETWORK WITH TARGET q*  (Script 1 style)
###############################################################

"""
    build_trophic_network_qstar(S; conn, qstar, mean_abs, mag_cv, rng)

Construct trophic geometry:
- Start with random ℓ's
- For each edge, enforce Δℓ = Normal(1, qstar)
- Aplus: consumer→resource flows
- A: signed predator/prey LV-like interactions

Returns (Aplus, A, q, ℓ)
"""
function build_trophic_network_qstar(
    S; conn=0.1, qstar=0.5, mean_abs=0.5, mag_cv=0.5,
    rng=Random.default_rng()
)
    # initial trophic levels
    ℓ = sort(rand(rng, S)) .* 4

    Aplus = zeros(Float64,S,S)

    pairs = [(i,j) for i in 1:S for j in 1:S if i!=j]
    K = round(Int, conn*length(pairs))
    chosen = sample(rng, pairs, K; replace=false)

    for (i,j) in chosen
        Δ = rand(rng, Normal(1, qstar))
        ℓj = ℓ[i] + Δ
        ℓ[j] = ℓj

        if Δ > 0
            # consumer i eats resource j
            Aplus[i,j] = rand(rng, LogNormal(log(mean_abs), mag_cv))
        end
    end

    q, ℓnew = trophic_coherence(Aplus)

    # LV-like signed matrix
    A = zeros(Float64,S,S)
    for i in 1:S, j in 1:S
        if Aplus[i,j] > 0
            m = Aplus[i,j]
            A[i,j] = +m
            A[j,i] = -m/2    # asymmetric predator-prey strength
        end
    end

    return Aplus, A, q, ℓnew
end

###############################################################
# 3. WITHIN-Q REWIRING — preserve Δℓ distribution (Script 1 style)
###############################################################

"""
    within_q_rewire(Aplus, ℓ; rng)

Produces Aplus2 preserving:
- Δℓ values across edges
- direction (consumer→resource)
- magnitudes
- connectivity count

Equivalent to Script 1's within-q variant.
"""
function within_q_rewire(Aplus, ℓ; rng=Random.default_rng())
    S = size(Aplus,1)
    edges = [(i,j, Aplus[i,j]) for i in 1:S for j in 1:S if Aplus[i,j] > 0]
    L = length(edges)

    # All possible directed pairs except self
    allpairs = [(i,j) for i in 1:S for j in 1:S if i!=j]
    perm = shuffle(rng, allpairs)

    Aplus2 = zeros(Float64,S,S)

    for k in 1:L
        (i_old, j_old, m) = edges[k]
        (u,v) = perm[k]

        # enforce trophic direction based on ℓ
        if ℓ[v] > ℓ[u]
            Aplus2[u,v] = m
        else
            Aplus2[v,u] = m
        end
    end

    return Aplus2
end

###############################################################
# 4. Convert Aplus → signed A  (consistent prey/predator asymmetry)
###############################################################

function Aplus_to_A(Aplus)
    S = size(Aplus,1)
    A = zeros(Float64,S,S)
    for i in 1:S, j in 1:S
        if Aplus[i,j] > 0
            m = Aplus[i,j]
            A[i,j] = +m
            A[j,i] = -m/2
        end
    end
    return A
end

###############################################################
# 5. Jacobian & median return rate
###############################################################

function jacobian(A, u)
    D = Diagonal(1 ./ u)
    return D * (A - I)
end

function median_return_rate(J::AbstractMatrix, u::AbstractVector; t=0.01)
    E = exp(t*J)
    w = u.^2
    C = Diagonal(w)
    num = log(tr(E * C * transpose(E)))
    den = log(sum(w))
    return -(num-den)/(2t)
end

###############################################################
# 6. FULL REWIRED PIPELINE WITH TROPHIC GEOMETRY
###############################################################

function run_rewire_pipeline(
    qvals;
    reps=10,
    S=120,
    conn=0.1,
    tvec = 10 .^ range(-2,2,length=50),
    mean_abs=0.5,
    mag_cv=0.5
)

    qmin, qmax = minimum(qvals), maximum(qvals)
    results = Dict{Float64,Dict}()

    U = random_u(S; mean=1.0, cv=0.5)

    for qstar in qvals
        println("Running q* = $qstar")

        groups = Dict(
            :baseline      => Vector{Vector{Float64}}(undef, reps),
            :rewired_q     => Vector{Vector{Float64}}(undef, reps),
            :rewired_minq  => Vector{Vector{Float64}}(undef, reps),
            :rewired_maxq  => Vector{Vector{Float64}}(undef, reps)
        )

        @threads for rep in 1:reps
            rng = Random.TaskLocalRNG()

            # BASELINE (trophic geometry)
            Aplus, A_base, qA, ℓ = build_trophic_network_qstar(
                S; conn, qstar, mean_abs, mag_cv, rng
            )

            mags_base = [Aplus[i,j] for i in 1:S for j in 1:S if Aplus[i,j] > 0]

            # WITHIN-Q REWIRED (preserve Δℓ geometry)
            Aplus_q = within_q_rewire(Aplus, ℓ; rng)
            A_q     = Aplus_to_A(Aplus_q)

            # MIN-q REWIRED (low-coherence geometry)
            Aplus_min, _, qmin_tmp, ℓmin = build_trophic_network_qstar(
                S; conn, qstar=qmin, mean_abs, mag_cv, rng
            )
            Aplus_min = replace_magnitudes(Aplus_min, mags_base, ℓmin)
            A_min = Aplus_to_A(Aplus_min)

            # MAX-q REWIRED (high-coherence geometry)
            Aplus_max, _, qmax_tmp, ℓmax = build_trophic_network_qstar(
                S; conn, qstar=qmax, mean_abs, mag_cv, rng
            )
            Aplus_max = replace_magnitudes(Aplus_max, mags_base, ℓmax)
            A_max = Aplus_to_A(Aplus_max)

            # Return rate curve
            function rr(A)
                J = jacobian(A, U)
                [median_return_rate(J,U;t=t) for t in tvec]
            end

            groups[:baseline][rep]      = rr(A_base)
            groups[:rewired_q][rep]     = rr(A_q)
            groups[:rewired_minq][rep]  = rr(A_min)
            groups[:rewired_maxq][rep]  = rr(A_max)
        end

        results[qstar] = groups
    end

    return results, tvec
end

###############################################################
# Helper: Replace magnitudes in a new Aplus with old magnitudes
# preserving Δℓ signs
###############################################################

function replace_magnitudes(Aplus_new, mags_base, ℓ)
    S = size(Aplus_new,1)
    indices = [(i,j) for i in 1:S for j in 1:S if Aplus_new[i,j] > 0]
    shuffle!(mags_base)
    k = 1
    L = length(mags_base)

    Aplus2 = zeros(Float64,S,S)
    for (i,j) in indices
        m = mags_base[k]
        k = (k == L ? 1 : k+1)

        # enforce feeding direction ℓ[j] > ℓ[i]
        if ℓ[j] > ℓ[i]
            Aplus2[i,j] = m
        else
            Aplus2[j,i] = m
        end
    end

    return Aplus2
end

###############################################################
# 7. PLOTTING FUNCTIONS (3×3 grids)
###############################################################

function plot_rmed_grid(results, tvec, qvals)
    fig = Figure(size=(1500,1500))
    rows, cols = 3,3

    for (k, q) in enumerate(qvals)
        i = ceil(Int,k/cols)
        j = k - (i-1)*cols
        ax = fig[i,j] = Axis(fig, title="q* = $q", xscale=log10)

        R = results[q]
        base = mean(reduce(hcat,R[:baseline]),dims=2)[:]
        rq   = mean(reduce(hcat,R[:rewired_q]),dims=2)[:]
        rmin = mean(reduce(hcat,R[:rewired_minq]),dims=2)[:]
        rmax = mean(reduce(hcat,R[:rewired_maxq]),dims=2)[:]

        lines!(ax,tvec, base, label="baseline")
        lines!(ax,tvec, rq,   label="within-q")
        lines!(ax,tvec, rmin, label="min-q")
        lines!(ax,tvec, rmax, label="max-q")
        axislegend(ax)
    end
    display(fig)
end

function plot_delta_rmed(results, tvec, qvals)
    fig = Figure(size=(1500,1500))
    rows, cols = 3,3

    for (k, q) in enumerate(qvals)
        i = ceil(Int,k/cols)
        j = k - (i-1)*cols
        ax = fig[i,j] = Axis(fig, title="Δrmed(t), q* = $q", xscale=log10)

        R = results[q]
        base = mean(reduce(hcat,R[:baseline]),dims=2)[:]
        rq   = mean(reduce(hcat,R[:rewired_q]),dims=2)[:]
        rmin = mean(reduce(hcat,R[:rewired_minq]),dims=2)[:]
        rmax = mean(reduce(hcat,R[:rewired_maxq]),dims=2)[:]

        lines!(ax,tvec, rq .- base, label="within-q")
        lines!(ax,tvec, rmin .- base, label="min-q")
        lines!(ax,tvec, rmax .- base, label="max-q")
        axislegend(ax)
    end
    display(fig)
end

function plot_scatter_deltaq(results, tvec, qvals; tmid=(0.5,5))
    idx = findall(t -> tmid[1] ≤ t ≤ tmid[2], tvec)
    X, Y = Float64[], Float64[]

    qmin, qmax = minimum(qvals), maximum(qvals)

    for q in qvals
        R = results[q]
        base = mean(reduce(hcat,R[:baseline]),dims=2)[:]
        minq = mean(reduce(hcat,R[:rewired_minq]),dims=2)[:]
        maxq = mean(reduce(hcat,R[:rewired_maxq]),dims=2)[:]

        push!(X, abs(q-qmin)); push!(Y, mean(minq[idx]) - mean(base[idx]))
        push!(X, abs(q-qmax)); push!(Y, mean(maxq[idx]) - mean(base[idx]))
    end

    fig = Figure(size=(800,600))
    ax = fig[1,1] = Axis(fig, title="|Δq| vs mid-time Δrmed")
    scatter!(ax, X, Y)
    display(fig)
end

###############################################################
# 8. RUN EXAMPLE
###############################################################
qvals = collect(range(0.0, 1.3, length=9))

results, tvec = run_rewire_pipeline(
    qvals;
    reps=8,
    S=120,
    conn=0.08,
    mean_abs=0.5,
    mag_cv=0.6
)

plot_rmed_grid(results,tvec,qvals)
plot_delta_rmed(results,tvec,qvals)
plot_scatter_deltaq(results,tvec,qvals)
