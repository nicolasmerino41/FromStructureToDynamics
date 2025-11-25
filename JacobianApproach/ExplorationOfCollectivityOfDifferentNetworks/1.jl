using LinearAlgebra, Random, Statistics
using CairoMakie

###########################################################
# 1. Build a matrix with controlled directionality θ ∈ [0,1]
#
# θ = 0   → perfectly reciprocal (Aij = Aji)
# θ = 1   → perfectly directional (Aij ≠ Aji, random orientation)
###########################################################
function build_directional_matrix(S; θ, rng=Xoshiro(1234), mean_abs=0.5)
    A = zeros(Float64, S, S)

    for i in 1:S-1, j in i+1:S
        # magnitude
        m = rand(rng, Exponential(mean_abs))

        if rand(rng) < θ
            # directional (feed-forward)
            if rand(rng) < 0.5
                A[i,j] =  m
                A[j,i] =  0
            else
                A[i,j] =  0
                A[j,i] =  m
            end
        else
            # reciprocal (symmetric)
            A[i,j] =  m
            A[j,i] =  m
        end
    end

    return A
end

###########################################################
# 2. Non-normality measure: N(A) = ||AᵀA − AAᵀ||_F
#
# For a normal matrix N(A) = 0.
# Directional matrices violate AᵀA = AAᵀ → N(A) grows.
###########################################################
function nonnormality(A)
    return norm(A' * A - A * A')
end

###########################################################
# 3. Experiment: sweep θ and compute non-normality
###########################################################
function run_experiment(; S=80, θs=range(0,1,length=20))
    rng = Xoshiro(2025)
    nn = Float64[]

    for θ in θs
        A = build_directional_matrix(S; θ=θ, rng=rng)
        push!(nn, nonnormality(A))
    end

    return θs, nn
end

###########################################################
# 4. Run
###########################################################
θs, nn = run_experiment()

###########################################################
# 5. Plot
###########################################################
begin
    fig = Figure(size=(900,500))
    ax = Axis(fig[1,1],
        xlabel="Directionality θ",
        ylabel="Non-normality  ‖AᵀA − AAᵀ‖_F",
        title="Increasing directionality → increasing non-normality"
    )

    lines!(ax, θs, nn, linewidth=4, color=:dodgerblue)
    scatter!(ax, θs, nn, color=:dodgerblue)

    fig

end