using Random
using LinearAlgebra
using Statistics
using CairoMakie

# ------------------------------------------------------------------------------
# 1. Johnson trophic coherence
# ------------------------------------------------------------------------------

"""
    trophic_coherence(Aplus)

Compute (q, trophic_levels ℓ) using Johnson et al. (2014).

Aplus must be a positive adjacency matrix representing consumer→resource feeding.
"""
function trophic_coherence(Ap)
    S = size(Ap,1)

    kin = vec(sum(Ap .> 0, dims=2))

    # If no edges → coherence undefined but return trivial
    if sum(kin) == 0
        return 0.0, ones(S)
    end

    D = Diagonal(kin .+ 1e-6)
    M = D - Ap

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

# ------------------------------------------------------------------------------
# 2. Strict trophic generator with target q*
# ------------------------------------------------------------------------------

"""
    build_trophic_network_qstar(S; conn, qstar, mean_abs, mag_cv, rng)

Build a strict trophic network with target coherence q*.

- Each edge has a consumer→resource direction.
- No reciprocal feeding (strict Option A).
- q* controls the std of trophic distances.

Returns:
    Aplus (energy flow adjacency)
    A     (Jacobian)
    q     (achieved trophic coherence)
"""
function build_trophic_network_qstar(
    S; conn=0.1, qstar=0.5, mean_abs=0.5, mag_cv=0.6, rng=Xoshiro(42)
)
    # Start with random trophic ordering
    ℓ = sort(rand(rng, S)) .* 4             # spread trophic levels

    Aplus = zeros(Float64,S,S)

    pairs = [(i,j) for i in 1:S for j in 1:S if i!=j]
    K = round(Int, conn*length(pairs))
    chosen = sample(rng, pairs, K; replace=false)

    for (i,j) in chosen
        Δ = rand(rng, Normal(1, qstar))       # draw trophic distance
        ℓj = ℓ[i] + Δ                          # enforce desired Δℓ
        ℓ[j] = ℓj

        if Δ > 0                                # consumer i feeds on j
            Aplus[i,j] = rand(rng, LogNormal(log(mean_abs), mag_cv))
        end
    end

    # Compute coherence
    q, ℓnew = trophic_coherence(Aplus)

    # Construct Jacobian form: predator benefit positive, prey negative
    A = zeros(Float64,S,S)
    for i in 1:S, j in 1:S
        if Aplus[i,j] > 0
            m = Aplus[i,j]
            A[i,j] = +m            # predator benefits
            A[j,i] = -m/2          # prey harmed
        end
    end

    return Aplus, A, q, ℓnew
end

# ------------------------------------------------------------------------------
# 3. Within-q variants: preserve Δℓ distribution exactly
# ------------------------------------------------------------------------------
"""
    within_q_variant(Aplus; rng)

Shuffle links but preserve:
- who is above/below whom (sign of Δℓ)
- the distribution of Δℓ magnitudes
"""
function within_q_variant(Aplus; rng=Xoshiro(42))
    S = size(Aplus,1)

    # Extract all edges with their Δℓ
    q, ℓ = trophic_coherence(Aplus)

    edges = []
    for i in 1:S, j in 1:S
        if Aplus[i,j] > 0
            push!(edges, (i,j, ℓ[j]-ℓ[i]))
        end
    end

    # Shuffle only positions, not Δℓ
    Anewplus = zeros(Float64,S,S)

    perm = shuffle(rng, [(i,j) for i in 1:S for j in 1:S if i!=j])
    for (k,(i,j,_)) in enumerate(edges)
        (u,v) = perm[k]
        # enforce feeding direction only if Δℓ>0
        if ℓ[v] > ℓ[u]
            Anewplus[u,v] = Aplus[i,j]
        else
            Anewplus[v,u] = Aplus[i,j]
        end
    end

    # Build Jacobian again
    Anew = zeros(Float64,S,S)
    for i in 1:S, j in 1:S
        if Anewplus[i,j] > 0
            m = Anewplus[i,j]
            Anew[i,j] = +m
            Anew[j,i] = -m/2
        end
    end

    return Anewplus, Anew
end

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

function compute_curve(A, u, t)
    compute_rmed_series_stable(jacobian(A,u), u, t)
end

# ------------------------------------------------------------------------------
# 6. Full experiment: multiple q* groups
# ------------------------------------------------------------------------------
function run_experiment(; 
    qstars = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
    S=120, conn=0.08, reps=8,
    t_vals = 10 .^ range(-2,2,length=50)
)
    results = Dict()

    for qstar in qstars
        println("Generating q* = $qstar")

        baseline = []
        within   = []
        qs       = []

        for r in 1:reps
            rng = Xoshiro(rand(UInt64))

            # baseline
            Aplus, A, qA, ℓ = build_trophic_network_qstar(
                S; conn, qstar, rng
            )
            push!(qs, qA)

            u = random_u(S; mean=1.0, cv=0.6, rng)
            f0 = compute_curve(A, u, t_vals)

            # within-q variant
            Aplus2, A2 = within_q_variant(Aplus; rng)
            f1 = compute_curve(A2, u, t_vals)

            push!(baseline, f0)
            push!(within, f1)
        end

        results[qstar] = (
            baseline=baseline,
            within=within,
            qs=qs
        )
    end

    return results, t_vals
end

function plot_rmed(results, t)
    fig = Figure(size=(1400,900))
    i = 1
    for qstar in sort(collect(keys(results)))
        base = results[qstar].baseline
        var  = results[qstar].within

        f0 = mean(reduce(hcat, base), dims=2)
        f1 = mean(reduce(hcat, var),  dims=2)

        ax = Axis(fig[(i-1)÷3+1, mod(i-1,3)+1],
                  title="q* = $qstar", xscale=log10)

        lines!(ax, t, vec(f0), color=:black, label="baseline")
        lines!(ax, t, vec(f1), color=:blue,  label="within-q")

        axislegend(ax)
        i+=1
    end
    display(fig)
end

function plot_delta(results, t)
    fig = Figure(size=(1400,900))
    i = 1
    for qstar in sort(collect(keys(results)))
        base = results[qstar].baseline
        var  = results[qstar].within

        d = mean(reduce(hcat, [v .- b for (v,b) in zip(var,base)]), dims=2)

        ax = Axis(fig[(i-1)÷3+1, mod(i-1,3)+1],
                  title="ΔRmed, q* = $qstar",
                  xscale=log10)

        hlines!(ax, [0.0], color=:gray)
        lines!(ax, t, vec(d), color=:blue)

        i+=1
    end
    display(fig)
end

function plot_q_distance(results, t_vals)
    qlist = Float64[]
    dlist = Float64[]

    for q1 in keys(results)
        for q2 in keys(results)
            if q1 >= q2; continue; end

            f0 = mean(reduce(hcat, results[q1].baseline), dims=2)
            f1 = mean(reduce(hcat, results[q2].baseline), dims=2)

            mid = findall(t_vals .>= 0.5 .&& t_vals .<= 5)
            d = mean(abs.(f0[mid] .- f1[mid]))

            push!(qlist, abs(q1-q2))
            push!(dlist, d)
        end
    end

    fig = Figure(size=(800,600))
    ax = Axis(fig[1,1], xlabel="|Δq|", ylabel="mid-time distance")

    scatter!(ax, qlist, dlist, color=:blue)

    display(fig)
end

results, t = run_experiment()
plot_rmed(results, t)
plot_delta(results, t)
plot_q_distance(results, t)