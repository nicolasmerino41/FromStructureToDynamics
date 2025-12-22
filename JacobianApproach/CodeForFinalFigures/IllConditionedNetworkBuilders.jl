using Random, LinearAlgebra, Distributions

"""
Feed-forward (upper-triangular) interaction matrix A.

Each species i influences only higher-index species j>i.
All eigenvalues of (A - I) are -1, but J = diag(u)*(A - I) becomes strongly non-normal.

Parameters:
- S  : number of species
- p  : connection probability in the upper triangle
- σ  : weight scale (standard deviation)
- signed : if true, random ± weights; if false, all positive
"""
function A_feedforward(S::Int; p::Real=0.1, σ::Real=1.0, signed::Bool=true, rng=Random.default_rng())
    A = Matrix{Float64}(I, S, S)
    dist = signed ? Normal(0, σ) : LogNormal(log(σ), 0.5)
    for i in 1:S-1, j in i+1:S
        if rand(rng) < p
            A[i,j] = rand(rng, dist)
        end
    end
    A[diagind(A)] .= 0
    return A
end

"""
Jordan-like chain A: nearly defective interaction structure.
"""
function A_jordan(S::Int; β::Real=1.0)
    A = Matrix{Float64}(I, S, S)
    for i in 1:S-1
        A[i, i+1] = β
    end
    A[diagind(A)] .= 0
    return A
end

"""
Construct A = V * Diagonal(λ) * inv(V), with ill-conditioned V.

All eigenvalues λ near 1 (so A - I has small negative real parts),
but eigenvectors are highly non-orthogonal → strong non-normality.
"""
function A_illconditioned(S::Int; cond_target::Real=1e4, spread::Real=0.1, rng=Random.default_rng())
    # random orthogonal U, W
    U = qr!(randn(rng, S, S)).Q
    W = qr!(randn(rng, S, S)).Q

    # singular values spanning [1, cond_target]
    s = exp.(range(0, log(cond_target); length=S))
    V = U * Diagonal(s) * W'

    # eigenvalues near 1 (so A - I has small negative reals)
    λ = 1 .- spread .* rand(rng, S)

    A = V * Diagonal(λ) * inv(V)
    A = Matrix(A)
    A[diagind(A)] .= 0
    return A
end

