using LinearAlgebra, Random

# Mean-field trophic (fully connected predator-prey matrix)
function meanfield_trophic(S, a=1.0)
    A = zeros(S, S)
    for i in 1:S-1, j in i+1:S
        A[i,j] =  a     # predator i consumes j
        A[j,i] = -a     # prey j is harmed by i
    end
    return A
end

# Hierarchical trophic "star" – ONE apex predator aggregates many prey
function trophic_star(S, a=1.0)
    A = zeros(S, S)
    for i in 1:S-1
        A[S, i] =  a          # apex predator benefits from all prey
        A[i, S] = -a          # prey lose to apex predator
    end
    return A
end

# ER random trophic network with same mean magnitude
function random_trophic_ER(S; conn=0.3, a=1.0)
    A = zeros(S, S)
    for i in 1:S-1, j in i+1:S
        if rand() < conn
            A[i,j] =  a*(rand()<0.5 ? +1 : -1)
            A[j,i] = -A[i,j]
        end
    end
    return A
end

function collectivity(A)
    maximum(abs, eigvals(A))   # φ = spectral radius
end

# ========= RUN TEST =========
S = 50
A1 = meanfield_trophic(S)
A2 = trophic_star(S)
A3 = random_trophic_ER(S; conn=0.3)

println("\n=== COLLECTIVITY (ϕ = ρ(A)) ===")
println("Mean-field (C=1 antisymmetric):   ", collectivity(A1))
println("Trophic star (hierarchical):       ", collectivity(A2))
println("Random ER trophic (same <|A|>):    ", collectivity(A3))
