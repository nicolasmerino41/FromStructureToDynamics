using LinearAlgebra, Random
using CairoMakie

# ------------------------------
# Lognormal helper (mean, cv)
# ------------------------------
function lognormal_params(mean::Float64, cv::Float64)
    σ2 = log(1 + cv^2)
    σ  = sqrt(σ2)
    μ  = log(mean) - σ2/2
    return μ, σ
end

# ------------------------------
# Trophic mean-field Jacobian
# ------------------------------
"""
    trophic_meanfield_J(S; conn=1.0, mean_abs=0.2, mag_cv=0.6,
                        u_mean=1.0, u_cv=0.4, rng)

Build a Jacobian J for a trophic-like mean-field system:
  - u ~ LogNormal(u_mean, u_cv)  (diagonal ~ -u_i, "quite homogeneous")
  - off-diagonals: predator–prey-ish antisymmetric structure,
    but scaled by u_i, u_j so antisymmetry is broken -> 2D spectrum.
"""
function trophic_meanfield_J(S::Int;
                             conn::Float64 = 1.0,
                             mean_abs::Float64 = 0.2,
                             mag_cv::Float64 = 0.6,
                             u_mean::Float64 = 1.0,
                             u_cv::Float64   = 0.4,
                             rng::AbstractRNG = MersenneTwister(1234))
    # abundances u (quite homogeneous: cv ≈ 0.4)
    μu, σu = lognormal_params(u_mean, u_cv)
    u = exp.(μu .+ σu .* randn(rng, S))

    # self-regulation on the diagonal
    J = -Matrix(Diagonal(u))

    # magnitudes for interactions
    μm, σm = lognormal_params(mean_abs, mag_cv)

    # mean-field-ish trophic structure: every unordered pair can be predator–prey
    for i in 1:S-1, j in i+1:S
        if rand(rng) < conn
            # base magnitude (lognormal)
            m = exp(μm + σm * randn(rng))
            # random trophic sign (who benefits)
            s = rand(rng) < 0.5 ? 1.0 : -1.0

            # break perfect antisymmetry by scaling with u_i, u_j
            # w_i = u[i] / (u[i] + u[j])
            # w_j = u[j] / (u[i] + u[j])

            # i benefits, j loses (or vice versa)
            J[i,j] +=  s * m #* w_i
            J[j,i] += -s * m # w_j
        end
    end

    return J
end

# ------------------------------
# Collect eigenvalues over many realizations
# ------------------------------
S      = 80          # number of species
nreal  = 50          # number of Jacobian realizations to fill the cloud
rng    = MersenneTwister(0xBEEF)

reals  = Float64[]
imags  = Float64[]

for r in 1:nreal
    J = trophic_meanfield_J(S; conn=0.1, mean_abs=0.2, mag_cv=0.6,
                            u_mean=1.0, u_cv=0.4, rng=rng)
    λ = eigvals(J)
    append!(reals, real.(λ))
    append!(imags, imag.(λ))
end

println("Max real part over all eigenvalues: ", maximum(reals))

# ------------------------------
# Plot spectrum in complex plane
# ------------------------------
begin
    fig = Figure(size = (700, 700))
    ax  = Axis(fig[1, 1],
            xlabel = "Re(λ)",
            ylabel = "Im(λ)",
            title  = "Spectrum of trophic mean-field Jacobians\n(S=$S, u_cv=0.4, conn=1.0)")

    scatter!(ax, reals, imags; markersize=6, strokewidth=0.0)
    hlines!(ax, [0.0], color=:gray, linestyle=:dash)
    vlines!(ax, [0.0], color=:gray, linestyle=:dash)

    # axislegend(ax; framevisible=false)
    display(fig)
end