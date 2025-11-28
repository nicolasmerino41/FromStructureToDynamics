using Random, Distributions, LinearAlgebra

function correlated_magnitudes(mag_abs, mag_cv, ρ; rng=Random.default_rng())
    σ = mag_abs * mag_cv   # normal std

    Σ = [σ^2  ρ*σ^2;
         ρ*σ^2  σ^2]

    d = MvNormal([mag_abs, mag_abs], Σ)

    mag = rand(rng, d)
    return abs.(mag)   # ensure positive magnitudes
end

"""
    build_interaction_matrix(A;
        mag_abs=1.0,
        mag_cv=0.5,
        corr_aij_aji=1.0,
        rng=Random.default_rng())

Construct signed interaction matrix W from adjacency A (prey→predator).
"""
function build_interaction_matrix(A;
        mag_abs=1.0,
        mag_cv=0.5,
        corr_aij_aji=1.0,
        rng=Random.default_rng())

    S = size(A,1)
    W = zeros(Float64, S, S)

    for prey in 1:S
        for pred in findall(A[prey,:] .== 1)

            # draw magnitudes with correlation ρ
            m_preypred, m_predprey = correlated_magnitudes(mag_abs, mag_cv, corr_aij_aji; rng=rng)

            # sign structure: prey → predator
            W[prey, pred] = +m_preypred   # effect on predator
            W[pred, prey] = -m_predprey   # effect on prey
        end
    end

    return W
end

W = build_interaction_matrix(A;
    mag_abs = 0.5,
    mag_cv = 0.3,
    corr_aij_aji = 0.8,
)

u = random_u(S)
J = jacobian(W, u)
