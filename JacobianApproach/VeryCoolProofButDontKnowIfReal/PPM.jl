using LinearAlgebra, Measures
using Distributions
using Graphs
using GraphPlot
using CategoricalArrays
using NetworkLayout, GraphMakie
"""
    trophic_levels(A)

Compute trophic levels sᵢ from adjacency matrix A.
"""
function trophic_levels(A::Matrix{Int})
    S = size(A,1)
    kin = sum(A, dims=2)
    v = [max(kin[i],1) for i in 1:S]
    Λ = Diagonal(v) - A
    s = Λ \ v
    return s
end

"""
    trophic_incoherence(A, s)

Compute incoherence q = sqrt(E[x²] - 1) where x = sᵢ - sⱼ for each link i<-j.
"""
function trophic_incoherence(A, s)
    xs = Float64[]
    S = length(s)
    for i in 1:S, j in 1:S
        if A[i,j] == 1
            push!(xs, s[i] - s[j])
        end
    end
    return sqrt(mean(xs.^2) - 1)
end

# """
#     interaction_matrix(A, η=0.2)

# Compute W = ηA - Aᵀ.
# """
# function interaction_matrix(A; η=0.2)
#     return η .* A .- transpose(A)
# end


"""
    ppm(S, B, L, T)

Generate adjacency matrix A using the Preferential Preying Model.
"""
function ppm(S, B, L, T)
    A = zeros(Int, S, S)

    β = (S^2 - B^2) / (2L - 1)
    beta_dist = Beta(β, β)

    prey_count = zeros(Int, S)

    current = B

    while current < S
        current += 1
        i = current

        existing = 1:(i-1)
        n_i = length(existing)

        # 1) First prey
        j = rand(existing)
        A[i,j] = 1
        prey_count[i] += 1

        # provisional TLs:
        s_hat = 1 .+ prey_count[1:i]

        # 2) Expected prey count
        k_exp = rand(beta_dist) * n_i
        k_total = max(1, round(Int, k_exp))
        k_extra = k_total - 1

        if k_extra > 0
            # probabilities based on provisional TLs
            probs = [exp(-abs(s_hat[j] - s_hat[ℓ]) / T) for ℓ in existing]
            probs ./= sum(probs)

            chosen = rand(Distributions.Categorical(probs), k_extra)
            for idx in unique(chosen)
                prey = existing[idx]
                A[i, prey] = 1
                prey_count[i] += 1
            end
        end
    end

    # final, correct TL computation
    s = trophic_levels(A)

    return A, s
end

"""
    visualize(A)

Plot the network using GraphPlot.
"""
function visualize(A)
    g = DiGraph(A)
    gplot(g, nodelabel=1:nv(g))
end


# --- Builder Object --------------------------------------------------------
mutable struct PPMBuilder
    S::Int
    B::Int
    L::Int
    T::Float64
    η::Float64
    PPMBuilder() = new(0,0,0,0.0,0.2)
end

function set!(b::PPMBuilder; S=nothing, B=nothing, L=nothing, T=nothing, η=nothing)
    if S !== nothing; b.S = S; end
    if B !== nothing; b.B = B; end
    if L !== nothing; b.L = L; end
    if T !== nothing; b.T = T; end
    if η !== nothing; b.η = η; end
end

"""
    build(builder)

Construct the adjacency matrix, interaction matrix, trophic levels, and q.
"""
function build(b::PPMBuilder)
    @assert b.S > 0 && b.B > 0 && b.L > 0 && b.T > 0 "Incomplete builder fields"

    A, s = ppm(b.S, b.B, b.L, b.T;)
    s = trophic_levels(A)
    q = trophic_incoherence(A, s)
    W = build_interaction_matrix(
        A;
        mag_abs=1.0,
        mag_cv=0.5,
        corr_aij_aji=0.99
    )

    return (A=A, s=s, q=q, W=W)
end

for q in q_targets
    avg_q_vec = Float64[]
    for i in 1:10
        b = PPMBuilder()
        set!(b; S=120, B=24, L=2142, T=q)
        net = build(b)
        q = net.q
        push!(avg_q_vec, q)
    end

    avg_q = mean(avg_q_vec)
    println("q target= ", q)
    println("Average q = ", avg_q)
end

result = build(b)

A = result.A
s = result.s
q = result.q
W = result.W

println("trophic incoherence = ", q)

# visualization
# visualize(A)

"""
    trophic_layout(s; xspread=1.0, yspread=1.0)

Return (x, y) coordinates for nodes arranged by trophic level.
"""
function trophic_layout(s; xspread=1.0, yspread=1.0)
    S = length(s)
    levels = sort(unique(s))
    x = zeros(Float64, S)
    y = zeros(Float64, S)

    for lvl in levels
        inds = findall(s .== lvl)
        n = length(inds)

        if n == 1
            xs = [0.0]
        else
            xs = range(-0.5*xspread*(n-1), 0.5*xspread*(n-1), length=n)
        end

        for (k,i) in enumerate(inds)
            x[i] = xs[k]
            y[i] = -lvl * yspread
        end
    end
    return x, y
end

"""
    visualize_escalator(
        A, s; B,
        xnoise = 0.25,
        figres = (1600,1200)
    )

Plots the PPM network with:
- x wrapping every B species
- exact TL on y-axis
- jitter on x-axis for non-basal nodes
- arrow edges
- color bins by TL (basal green, TL∈[1,2)=blue, [2,3)=orange, [3,4)=red)
"""
function visualize_escalator(A, s; B, xnoise=0.75, figres=(1100,620))

    g = DiGraph(A)
    S = length(s)

    # --- Base x wrapping ---
    x = Float64.([(i-1) % B for i in 1:S])
    y = s

    # --- Add jitter ONLY to non-basals ---
    for i in 1:S
        if s[i] > 1.0
            x[i] += (rand() - 0.5) * xnoise   # small shift
        end
    end

    # --- Color nodes by TL bins ---
    colors = Vector{Symbol}(undef, S)
    for i in 1:S
        if s[i] == 1
            colors[i] = :green
        elseif 1 < s[i] <= 2
            colors[i] = :blue
        elseif 2 <= s[i] <= 3
            colors[i] = :orange
        elseif 3 < s[i]
            colors[i] = :red
        end
    end

    # --- Create figure ---
    fig = Figure(; size=figres)
    ax = Axis(fig[1,1];
        title = "Escalator Trophic Layout",
        xlabel = "X (wrapped every B species)",
        ylabel = "Trophic Level",
    )

    for prey in 1:S
        for pred in inneighbors(g, prey)

            # line from prey to predator
            lines!(ax,
                [x[prey], x[pred]],
                [y[prey], y[pred]];
                color=:black, linewidth=0.5
            )

            # arrowhead at predator
            dx = x[pred] - x[prey]
            dy = y[pred] - y[prey]
            θ = atan(dy, dx)

            arrow_x = x[pred] - 0.08*cos(θ)
            arrow_y = y[pred] - 0.08*sin(θ)

            poly!(ax,
                Point2f[
                    (arrow_x, arrow_y),
                    (arrow_x - 0.05*cos(θ + 0.3),
                    arrow_y - 0.05*sin(θ + 0.3)),
                    (arrow_x - 0.05*cos(θ - 0.3),
                    arrow_y - 0.05*sin(θ - 0.3))
                ],
                color=:black)
        end
    end


    # --- Draw nodes ---
    scatter!(ax, x, y; color=colors, markersize=10)

    # --- Axis limits ---
    xlims!(ax, -1, B)
    ylims!(ax, minimum(y)-0.2, maximum(y)+0.2)

    display(fig)
end

visualize_escalator(A, s; B=b.B)

# General trophic levels
using LinearAlgebra

function general_trophic_levels(A::AbstractMatrix{<:Real})
    S = size(A, 1)
    @assert size(A, 2) == S "Adjacency matrix must be square"

    # Number of prey per species (row sums)
    k = sum(A, dims=2)

    # Identify basal species (no prey)
    basal = vec(k .== 0)

    # Diet fraction matrix P
    P = zeros(Float64, S, S)
    for i in 1:S
        if k[i] > 0
            P[i, :] .= A[i, :] ./ k[i]
        end
    end

    # Identity matrix
    I = Diagonal(ones(Float64, S))
    I = Matrix(I)

    # Right-hand side: 1 for all species
    b = ones(Float64, S)

    # Solve linear system: (I - P) * TL = 1
    TL = (I - P) \ b

    return TL
end

Z = build_niche_trophic(
    S; conn=0.15, mean_abs=0.5, mag_cv=0.60,
    degree_family=:uniform, deg_param=1.0,
    rho_sym=0.0, rng=Random.default_rng()
)
Z[Z .< 0.0] .= 0.0
Z_bool = Z .!= 0.0
s = general_trophic_levels(Int.(Z_bool))

visualize_escalator(Z, s; B=24)