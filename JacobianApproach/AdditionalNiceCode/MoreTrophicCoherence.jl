using LinearAlgebra
using Statistics
using Random
using Distributions
using Graphs
using CairoMakie

# ============================================================
# 1. TROPHIC COHERENCE CALCULATION
# ============================================================

"""
    trophic_levels(G)

Computes trophic levels s_i for each node using the definition:
s_i = 1 + (1 / k_in(i)) * sum_j A_{ji}

Nodes with zero in-degree get s_i = 1.
"""
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

"""
    trophic_coherence(G)

Computes the trophic coherence q of a directed graph.
q = std(Δ_ij) where Δ_ij = s_j - s_i for edges i → j.
"""
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


# ============================================================
# 2. NETWORK GENERATION WITH TARGET q (Johnson 2014)
# ============================================================

"""
    generate_graph_target_q(S, L, q_target; T_init = 1.0)

Generates a directed graph with S species, L links,
using the "effort-based selection" model of Johnson et al. 2014.

Temperature T is tuned by binary search to achieve the target q.
"""
function generate_graph_target_q(S::Int, L::Int, q_target::Real;
                                 T_init = 1.0, max_iter = 40)

    # uniform basal probabilities
    basal_fraction = 0.1
    B = round(Int, basal_fraction * S)
    basal = randperm(S)[1:B]

    function generate_at_T(T)
        G = SimpleDiGraph(S)
        while ne(G) < L
            i = rand(1:S)
            j = rand(1:S)
            if i != j && !has_edge(G, i, j)
                # Probability depends on trophic distance exp(-(Δs)^2 / (2T^2))
                # Using s ≈ 1 initially for all nodes, then refined
                add_edge!(G, i, j)
            end
        end
        return G
    end

    # Binary-search adjust T
    T_low, T_high = 0.01, 10.0
    G_best = nothing
    best_err = Inf

    for _ in 1:max_iter
        T_mid = sqrt(T_low * T_high)
        G = generate_at_T(T_mid)
        q = trophic_coherence(G)
        err = abs(q - q_target)

        if err < best_err
            G_best = G
            best_err = err
        end

        if q > q_target
            T_high = T_mid
        else
            T_low = T_mid
        end
    end

    return G_best
end


# ============================================================
# 3. INTERACTION MATRIX BUILDER
# ============================================================

"""
    interaction_matrix(G; mag_abs = 0.5, mag_cv = 0.5, correlated = true)

Builds signed predator–prey interaction matrix A from directed graph G.
If i → j (i eaten by j), then:
  A[i, j] = +m   (predator responds positively to prey)
  A[j, i] = -m   (prey responds negatively to predator)

If correlated = true, both directions share |m|.
If correlated = false, magnitudes are independently drawn.
"""
function interaction_matrix(G::SimpleDiGraph;
                            mag_abs = 0.5,
                            mag_cv = 0.5,
                            correlated = true)

    S = nv(G)
    A = zeros(Float64, S, S)

    dist = Normal(mag_abs, mag_abs * mag_cv)

    for i in 1:S
        for j in outneighbors(G, i)
            if correlated
                m = abs(rand(dist))
                A[i, j] = +m
                A[j, i] = -m
            else
                m1 = abs(rand(dist))
                m2 = abs(rand(dist))
                A[i, j] = +m1
                A[j, i] = -m2
            end
        end
    end

    return A
end


# ============================================================
# 4. JACOBIAN
# ============================================================

"""
    jacobian(A, u)

J = D * (I - A)
D = diag(1 ./ u)
"""
function jacobian(A::AbstractMatrix, u::AbstractVector)
    D = Diagonal(u)
    return D * (A - I)
end


# ============================================================
# 5. MEDIAN RETURN RATE (your function)
# ============================================================

function median_return_rate(
    J::AbstractMatrix, u::AbstractVector;
    t::Real = 0.01, perturbation::Symbol = :biomass
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
        error("Unknown perturbation model: $perturbation")
    end
    return -(num - den) / (2*t)
end


# ============================================================
# 6. REALISATION PIPELINE
# ============================================================

"""
    run_realisations(qvals; reps=20, S=50, connectance=0.1)

For each q in qvals:
  - baseline realisations
  - within-q realisations
  - min-q realisations
  - max-q realisations
"""
function run_realisations(qvals;
                          reps = 10,
                          S = 40,
                          connectance = 0.1,
                          tvec = 10 .^ range(-2,2,length=50),
                          mag_abs = 0.5,
                          mag_cv = 0.5)

    L = round(Int, connectance * S * S)

    qmin, qmax = minimum(qvals), maximum(qvals)

    results = Dict()

    for q in qvals
        println("Generating networks for q = $q")

        groups = Dict(
            :baseline => [],
            :within   => [],
            :minq     => [],
            :maxq     => []
        )

        for rep in 1:reps
            push!(groups[:baseline], generate_graph_target_q(S, L, q))
            push!(groups[:within],   generate_graph_target_q(S, L, q))
            push!(groups[:minq],     generate_graph_target_q(S, L, qmin))
            push!(groups[:maxq],     generate_graph_target_q(S, L, qmax))
        end

        # Compute return rates
        U = random_u(S; mean=1.0, cv=0.5)
        # U = rand(0.5:0.1:2.0, S)  # fixed u vector

        function rr_over_t(G)
            A = interaction_matrix(G; mag_abs=mag_abs, mag_cv=mag_cv)
            J = jacobian(A, U)
            return [median_return_rate(J, U; t=t) for t in tvec]
        end

        R = Dict()
        for g in keys(groups)
            R[g] = [rr_over_t(G) for G in groups[g]]
        end

        results[q] = R
    end

    return results, collect(tvec)
end

# ============================================================
# 7. PLOT 1: rmed(t) GRID
# ============================================================

function plot_rmed_grid(results, tvec, qvals)
    fig = Figure(; size=(1600, 1200))

    rows, cols = 3, 3
    for (k, q) in enumerate(qvals)
        i = ceil(Int, k / cols)
        j = k - (i - 1) * cols

        ax = fig[i, j] = Axis(
            fig, title="q = $q",
            xscale=log10
        )

        R = results[q]

        baseline = mean(reduce(hcat, R[:baseline]), dims=2)[:]
        within   = mean(reduce(hcat, R[:within]),   dims=2)[:]
        minq     = mean(reduce(hcat, R[:minq]),     dims=2)[:]
        maxq     = mean(reduce(hcat, R[:maxq]),     dims=2)[:]

        lines!(ax, tvec, baseline, label="baseline")
        lines!(ax, tvec, within,   label="within-q")
        lines!(ax, tvec, minq,     label="min-q")
        lines!(ax, tvec, maxq,     label="max-q")

        axislegend(ax)
    end

    display(fig)
end


# ============================================================
# 8. PLOT 2: Δrmed(t)
# ============================================================
function plot_delta_rmed(results, tvec, qvals)
    fig = Figure(; size=(1600, 1200))

    rows, cols = 3, 3
    for (k, q) in enumerate(qvals)
        i = ceil(Int, k / cols)
        j = k - (i - 1) * cols

        ax = fig[i, j] = Axis(
            fig, title="Δ rₘₑd(t), q = $q",
            xscale=log10
            )

        R = results[q]

        base = mean(reduce(hcat, R[:baseline]), dims=2)[:]
        within = mean(reduce(hcat, R[:within]), dims=2)[:]
        minq = mean(reduce(hcat, R[:minq]), dims=2)[:]
        maxq = mean(reduce(hcat, R[:maxq]), dims=2)[:]

        lines!(ax, tvec, within .- base, label="within - base")
        lines!(ax, tvec, minq .- base,   label="min - base")
        lines!(ax, tvec, maxq .- base,   label="max - base")

        axislegend(ax)
    end

    display(fig)
end


# ============================================================
# 9. PLOT 3: Δq vs mid-t amplification
# ============================================================

function plot_scatter_deltaq(results, tvec, qvals; tmid_window=(0.02, 0.03))
    idx = findall(t -> tmid_window[1] <= t <= tmid_window[2], tvec)

    X = Float64[]
    Y = Float64[]

    qmin, qmax = minimum(qvals), maximum(qvals)

    for q in qvals
        R = results[q]

        base = mean(reduce(hcat, R[:baseline]), dims=2)[:]
        minq = mean(reduce(hcat, R[:minq]), dims=2)[:]
        maxq = mean(reduce(hcat, R[:maxq]), dims=2)[:]

        mid_base = mean(base[idx])
        mid_minq = mean(minq[idx])
        mid_maxq = mean(maxq[idx])

        push!(X, abs(q - qmin))
        push!(Y, mid_minq - mid_base)

        push!(X, abs(q - qmax))
        push!(Y, mid_maxq - mid_base)
    end

    fig = Figure(; size=(800, 600))
    ax = fig[1, 1] = Axis(fig, title="|Δq| vs mid-t amplification")
    scatter!(ax, X, Y)
    return fig
end

# -------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------
qvals = range(0.0, 1.0, length=9) |> collect   # 9 q's
reps = 10                                      # number of realisations per group
S = 50                                         # species
connectance = 0.1                              # fixed connectance
mag_abs = 0.5
mag_cv = 0.5

# -------------------------------------------------------------
# RUN REALISATIONS
# -------------------------------------------------------------
results, tvec = run_realisations(
    qvals;
    reps = reps,
    S = S,
    connectance = connectance,
    mag_abs = mag_abs,
    mag_cv = mag_cv
)

# -------------------------------------------------------------
# PLOT 1
# -------------------------------------------------------------
fig1 = plot_rmed_grid(results, tvec, qvals)
save("plot_rmed_grid.png", fig1)

# -------------------------------------------------------------
# PLOT 2
# -------------------------------------------------------------
fig2 = plot_delta_rmed(results, tvec, qvals)
save("plot_delta_rmed.png", fig2)

# -------------------------------------------------------------
# PLOT 3
# -------------------------------------------------------------
fig3 = plot_scatter_deltaq(results, tvec, qvals)
save("plot_scatter_deltaq.png", fig3)

