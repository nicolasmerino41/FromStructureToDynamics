##############################################################
# SCRIPT 1 — Conceptual Collectivity–Shape Demonstration
##############################################################
using CairoMakie
# ----------------------------
# Helpers for shape metrics
# ----------------------------
# Rotational (skew-symmetric) content
skew_measure(A) = norm(A - A')

# Feed-forward (upper-triangular) content
triangular_measure(A) = norm(triu(A, 1))

# Non-normality metric
nonnormal_measure(A) = norm(A'*A - A*A')

# Collectivity (spectral radius)
rho(A) = maximum(abs.(eigvals(A)))

# ----------------------------
# Build canonical matrices
# ----------------------------

function build_symmetric_MF(S; a = 0.2)
    A = fill(a, S, S)
    for i in 1:S
        A[i,i] = 0
    end
    return (A + A')/2
end

function build_skew_MF(S; a = 0.2)
    A = fill(0.0, S, S)
    for i in 1:S, j in i+1:S
        A[i,j] = a
        A[j,i] = -a
    end
    return A
end

function build_triangular(S; a = 0.2)
    A = zeros(S,S)
    for i in 1:S, j in i+1:S
        A[i,j] = a
    end
    return A
end

function build_random(S; a = 0.2, rng=Xoshiro(42))
    A = a * randn(rng, S, S)
    for i in 1:S
        A[i,i] = 0
    end
    return A
end

# Synthetic “real trophic” network
function build_trophic_mixed(S; rng=Xoshiro(1234), a=0.2)
    A = zeros(S,S)

    # 1. feed-forward backbone
    for i in 1:S, j in i+1:S
        A[i,j] = a * rand(rng)
    end

    # 2. random predator–prey pairs (skew)
    for _ in 1:round(Int, S/3)
        i, j = rand(rng, 1:S, 2)
        A[i,j] =  a
        A[j,i] = -a
    end

    # 3. random noise
    A .+= 0.2a * randn(rng, S, S)

    # zero diagonal
    for i in 1:S
        A[i,i] = 0
    end

    return A
end

# ----------------------------
# Evaluate the set
# ----------------------------
S = 50

A_MF       = build_symmetric_MF(S)
A_skew     = build_skew_MF(S)
A_tri      = build_triangular(S)
A_rand     = build_random(S)
A_trophic  = build_trophic_mixed(S)

nms = ["Symmetric MF","Skew MF","Triangular","Random","Trophic mix"]
As    = [A_MF, A_skew, A_tri, A_rand, A_trophic]

collectivities = rho.(As)
skews          = skew_measure.(As)
triangles      = triangular_measure.(As)

# ----------------------------
# Plot
# ----------------------------
begin
    fig = Figure(; size=(900,700))
    ax = Axis3(fig[1,1], xlabel="Collectivity ρ(A)",
                        ylabel="Rotational shape (skewness)",
                        zlabel="Feed-forward shape (triangularity)")

    for i in 1:length(As)
        scatter!(ax, [collectivities[i]], [skews[i]], [triangles[i]],
                markersize=16, label=nms[i])
    end
    axislegend(ax, position=:lt)
    display(fig)
end
