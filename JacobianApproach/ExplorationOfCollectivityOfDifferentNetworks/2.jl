using LinearAlgebra, Random, CairoMakie

############################################################
#  Build TROPHIC networks from antisymmetric → triangular
#
#  θ = 0 : purely antisymmetric (normal)
#  θ = 1 : purely triangular (max non-normal)
#
#  Always predator–prey (±).
#  Magnitudes vary and break symmetry unless θ = 0.
############################################################

function build_trophic_antisym_to_triangular(S; θ, conn=0.15, rng=Xoshiro(42))

    A = zeros(Float64, S, S)

    # pick interacting pairs
    pairs = [(i,j) for i in 1:S-1 for j in i+1:S]
    K = round(Int, conn*length(pairs))
    chosen = sample(rng, pairs, K; replace=false)

    for (i,j) in chosen
        # random magnitudes for each direction (breaks anti-symmetry)
        m1 = rand(rng)
        m2 = rand(rng)

        # randomly decide who is predator
        if rand(rng) < 0.5
            # i is predator (i gains, j loses)
            base_ij = +m1
            base_ji = -m2
        else
            # j is predator
            base_ij = -m2
            base_ji = +m1
        end

        if θ == 0
            # pure antisymmetry: force m1=m2
            A[i,j] = +abs(base_ij)
            A[j,i] = -abs(base_ij)
        elseif θ == 1
            # pure triangular: keep only trophic-forward direction
            if i < j
                A[i,j] = base_ij
                A[j,i] = 0
            else
                A[j,i] = base_ji
                A[i,j] = 0
            end
        else
            # interpolate:
            #   with prob θ → triangular
            #   with prob (1-θ) → antisymmetric
            if rand(rng) < θ
                # triangular contribution
                if i < j
                    A[i,j] = base_ij
                    A[j,i] = 0
                else
                    A[j,i] = base_ji
                    A[i,j] = 0
                end
            else
                # antisymmetric contribution (magnitudes forced equal)
                m = abs(base_ij)  # or abs(base_ji)
                if base_ij > 0
                    A[i,j] = +m
                    A[j,i] = -m
                else
                    A[i,j] = -m
                    A[j,i] = +m
                end
            end
        end
    end

    return A
end


############################################################
# non-normality
############################################################
nonnormality(A) = norm(A' * A - A * A')

############################################################
# Sweep θ and plot non-normality
############################################################
function sweep_nonnormality(; S=60, θs=range(0,1,length=20))
    rng = Xoshiro(2025)
    nn = Float64[]

    for θ in θs
        A = build_trophic_antisym_to_triangular(S; θ=θ, rng=rng)
        push!(nn, nonnormality(A))
    end

    return θs, nn
end

θs, nn = sweep_nonnormality()

begin
    fig = Figure(size=(900,500))
    ax = Axis(fig[1,1],
        xlabel="θ (0 = antisymmetric, 1 = triangular)",
        ylabel="Non-normality  ‖AᵀA − AAᵀ‖_F",
        title="Directionality increases non-normality (within trophic networks)"
    )

    scatter!(ax, θs, nn, color=:orange)
    lines!(ax, θs, nn, linewidth=4, color=:orange)
    display(fig)
end