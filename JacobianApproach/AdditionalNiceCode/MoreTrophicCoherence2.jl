###############################################################
#   REWIRED-TOPOLOGY TROPHIC COHERENCE PIPELINE
#   Baseline vs Rewired-within-q vs Min-q-rewired vs Max-q-rewired
###############################################################

using LinearAlgebra
using Statistics
using Random
using Distributions
using Graphs
using CairoMakie

###############################################################
# 1. TROPHIC COHERENCE
###############################################################

function trophic_levels(G::SimpleDiGraph)
    S = nv(G)
    A = adjacency_matrix(G)
    s = ones(Float64, S)
    max_iter = 100
    tol = 1e-9

    for _ in 1:max_iter
        sold = copy(s)
        for i in 1:S
            ki = sum(A[:, i])
            if ki > 0
                s[i] = 1 + sum(A[:, i] .* sold) / ki
            else
                s[i] = 1
            end
        end
        if maximum(abs.(s .- sold)) < tol
            break
        end
    end
    return s
end

function trophic_coherence(G::SimpleDiGraph)
    s = trophic_levels(G)
    A = adjacency_matrix(G)
    deltas = Float64[]
    for i in 1:nv(G)
        for j in 1:nv(G)
            if A[i, j] == 1
                push!(deltas, s[j] - s[i])
            end
        end
    end
    return std(deltas)
end

###############################################################
# 2. GENERATE GRAPH AT TARGET q
###############################################################

function generate_graph_target_q(S, L, q_target; max_iter=40)

    function gen()
        G = SimpleDiGraph(S)
        while ne(G) < L
            i = rand(1:S)
            j = rand(1:S)
            if i != j && !has_edge(G, i, j)
                add_edge!(G, i, j)
            end
        end
        return G
    end

    Gbest = nothing
    besterr = Inf

    for _ in 1:max_iter
        G = gen()
        q = trophic_coherence(G)
        err = abs(q - q_target)
        if err < besterr
            Gbest = G
            besterr = err
        end
    end

    return Gbest
end

###############################################################
# 3. INTERACTION MATRIX BUILDER
###############################################################

function interaction_matrix(G::SimpleDiGraph;
                            mag_abs=0.5,
                            mag_cv=0.5,
                            correlated=true)

    S = nv(G)
    A = zeros(Float64, S, S)
    dist = Normal(mag_abs, mag_abs*mag_cv)

    for i in 1:S
        for j in outneighbors(G, i)
            if correlated
                m = abs(rand(dist))
                A[i,j] = +m
                A[j,i] = -m
            else
                m1 = abs(rand(dist))
                m2 = abs(rand(dist))
                A[i,j] = +m1
                A[j,i] = -m2
            end
        end
    end

    return A
end

###############################################################
# 4. REWIRING WITH MAGNITUDE TRANSFER
###############################################################

# Extract list of magnitudes from baseline A
function extract_pair_magnitudes(A)
    mags = Float64[]
    S = size(A,1)
    for i in 1:S
        for j in 1:S
            if A[i,j] > 0   # predator receives +m
                push!(mags, A[i,j])
            end
        end
    end
    return mags
end

# Build new A from new adjacency and old magnitudes
function rebuild_A_from_adjacency(Gnew::SimpleDiGraph,
                                  mags::Vector{Float64})
    S = nv(Gnew)
    Anew = zeros(Float64, S, S)

    shuffle!(mags)
    k = 1
    n = length(mags)

    for i in 1:S
        for j in outneighbors(Gnew, i)
            # wrap around if needed
            m = mags[k]
            k += 1
            if k > n
                k = 1
            end

            Anew[i,j] = +m
            Anew[j,i] = -m
        end
    end

    return Anew
end


###############################################################
# 5. JACOBIAN
###############################################################

# Correct formula: J = D(A - I)
function jacobian(A, u)
    D = Diagonal(1 ./ u)
    return D * (A - I)
end

###############################################################
# 6. RETURN RATE
###############################################################

function median_return_rate(
    J::AbstractMatrix, u::AbstractVector;
    t::Real=0.01, perturbation::Symbol=:biomass
)
    S = size(J,1)
    if S == 0 || any(!isfinite, J)
        return NaN
    end
    E = exp(t*J)
    if perturbation === :uniform
        num = log(tr(E * transpose(E)))
        den = log(S)
    elseif perturbation === :biomass
        w = u .^ 2
        C = Diagonal(w)
        num = log(tr(E * C * transpose(E)))
        den = log(sum(w))
    else
        error("Unknown perturbation")
    end
    return -(num - den) / (2t)
end

###############################################################
# 7. PIPELINE WITH REWIRE COMPARISONS
###############################################################

using Base.Threads
using Random

function run_rewire_pipeline(qvals;
                             reps=10,
                             S=120,
                             connectance=0.1,
                             tvec = 10 .^ range(-2,2,length=50),
                             mag_abs=0.5,
                             mag_cv=0.5)

    L = round(Int, connectance*S*S)
    qmin, qmax = minimum(qvals), maximum(qvals)

    # fixed u vector
    U = random_u(S; mean=1.0, cv=0.5)

    results = Dict{Float64,Dict}()

    # Loop over q-values (can thread this too safely)
    for q in qvals
        println("Running q = $q ...")

        # Preallocate container for this q
        groups = Dict(
            :baseline      => Vector{Vector{Float64}}(undef, reps),
            :rewired_q     => Vector{Vector{Float64}}(undef, reps),
            :rewired_minq  => Vector{Vector{Float64}}(undef, reps),
            :rewired_maxq  => Vector{Vector{Float64}}(undef, reps)
        )

        # Threaded loop over reps
        @threads for rep in 1:reps
            rng = Random.TaskLocalRNG()  # thread-local RNG

            # BASELINE GRAPH
            Gbase = generate_graph_target_q(S,L,q)
            A_base = interaction_matrix(Gbase; mag_abs, mag_cv)
            mags_base = extract_pair_magnitudes(A_base)

            # WITHIN-Q REWIRED
            Gq = generate_graph_target_q(S,L,q)
            A_q = rebuild_A_from_adjacency(Gq, mags_base)

            # MIN-Q REWIRED
            Gmin = generate_graph_target_q(S,L,qmin)
            A_min = rebuild_A_from_adjacency(Gmin, mags_base)

            # MAX-Q REWIRED
            Gmax = generate_graph_target_q(S,L,qmax)
            A_max = rebuild_A_from_adjacency(Gmax, mags_base)

            # Return rate helper
            function rr(A)
                J = jacobian(A, U)
                [median_return_rate(J, U; t=t) for t in tvec]
            end

            # Store results (thread-safe: each rep has its own slot)
            groups[:baseline][rep]     = rr(A_base)
            groups[:rewired_q][rep]    = rr(A_q)
            groups[:rewired_minq][rep] = rr(A_min)
            groups[:rewired_maxq][rep] = rr(A_max)
        end

        # Store result for this q
        results[q] = groups
    end

    return results, collect(tvec)
end


###############################################################
# 8. PLOTTING (same as before but 3×3 grids)
###############################################################

function plot_rmed_grid(results, tvec, qvals)
    fig = Figure(; size=(1500,1500))
    rows, cols = 3, 3

    for (k,q) in enumerate(qvals)
        i = ceil(Int, k/cols)
        j = k - (i-1)*cols
        ax = fig[i,j] = Axis(fig, title="q = $q", xscale=log10)

        R = results[q]
        base  = mean(reduce(hcat, R[:baseline]),      dims=2)[:]
        rq    = mean(reduce(hcat, R[:rewired_q]),     dims=2)[:]
        rmin  = mean(reduce(hcat, R[:rewired_minq]),  dims=2)[:]
        rmax  = mean(reduce(hcat, R[:rewired_maxq]),  dims=2)[:]

        lines!(ax, tvec, base, label="baseline")
        lines!(ax, tvec, rq,   label="rewired-q")
        lines!(ax, tvec, rmin, label="rewired-minq")
        lines!(ax, tvec, rmax, label="rewired-maxq")
        axislegend(ax)
    end
    display(fig)
end

function plot_delta_rmed(results, tvec, qvals)
    fig = Figure(; size=(1500,1500))
    rows, cols = 3, 3

    for (k,q) in enumerate(qvals)
        i = ceil(Int, k/cols)
        j = k - (i-1)*cols
        ax = fig[i,j] = Axis(fig, title="Δrmed(t), q = $q", xscale=log10)

        R = results[q]
        base  = mean(reduce(hcat, R[:baseline]),      dims=2)[:]
        rq    = mean(reduce(hcat, R[:rewired_q]),     dims=2)[:]
        rmin  = mean(reduce(hcat, R[:rewired_minq]),  dims=2)[:]
        rmax  = mean(reduce(hcat, R[:rewired_maxq]),  dims=2)[:]

        lines!(ax, tvec, rq   .- base, label="rewired-q")
        lines!(ax, tvec, rmin .- base, label="rewired-minq")
        lines!(ax, tvec, rmax .- base, label="rewired-maxq")
        axislegend(ax)
    end
    display(fig)
end

function plot_scatter_deltaq(results, tvec, qvals;
                             tmid=(0.02,0.03))

    idx = findall(t->tmid[1]≤t≤tmid[2], tvec)
    X = Float64[]
    Y = Float64[]

    qmin, qmax = minimum(qvals), maximum(qvals)

    for q in qvals
        R = results[q]
        base = mean(reduce(hcat, R[:baseline]), dims=2)[:]
        minq = mean(reduce(hcat, R[:rewired_minq]), dims=2)[:]
        maxq = mean(reduce(hcat, R[:rewired_maxq]), dims=2)[:]

        push!(X, abs(q - qmin));  push!(Y, mean(minq[idx]) - mean(base[idx]))
        push!(X, abs(q - qmax));  push!(Y, mean(maxq[idx]) - mean(base[idx]))
    end

    fig = Figure(; size=(800,600))
    ax = fig[1,1] = Axis(fig, title="|Δq| vs mid-t Δrmed")
    scatter!(ax, X, Y)
    display(fig)
end

qvals = collect(range(0.1, 0.9, length=9))

results, tvec = run_rewire_pipeline(
    qvals;
    reps=10,
    S=100,
    connectance=0.1,
    mag_abs=0.5,
    mag_cv=0.5
)

fig1 = plot_rmed_grid(results, tvec, qvals)

fig2 = plot_delta_rmed(results, tvec, qvals)

fig3 = plot_scatter_deltaq(results, tvec, qvals)
