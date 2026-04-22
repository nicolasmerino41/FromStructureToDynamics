# ============================================================
# claim3_pointwise_envelope_6node.jl
#
# 6-node modular example for a richer illustrative figure.
#
# Pointwise structural-sensitivity landscape over admissible T:
#   for each frequency ω, across admissible timescale configurations:
#       - worst-case line  = max_T  ‖S(ω) P S(ω)‖₂
#       - best-case line   = min_T  ‖S(ω) P S(ω)‖₂
#       - mean-case line   = mean_T ‖S(ω) P S(ω)‖₂
#       - homogeneous reference line
#
# Important:
#   worst/best are pointwise envelopes in ω, not generally the
#   profile of a single fixed T.
#
# Only the profile plot is produced.
# ============================================================

using LinearAlgebra
using Statistics
using Random
using CairoMakie
using Printf

# ------------------------------------------------------------
# Helpers: dynamics
# ------------------------------------------------------------
resolvent(A, T, ω) = inv(im * ω * T - A)

structured_sensitivity(A, T, P, ω) =
    opnorm(resolvent(A, T, ω) * P * resolvent(A, T, ω), 2)

function structured_profile(A, T, P, ωs)
    [structured_sensitivity(A, T, P, ω) for ω in ωs]
end

# ------------------------------------------------------------
# Helpers: centered log-timescale space
# ------------------------------------------------------------
function centered_basis(n::Int)
    M = Matrix{Float64}(I, n, n - 1)
    for j in 1:n-1
        M[n, j] = -1.0
    end
    F = qr(M).Q
    return Matrix(F[:, 1:n-1])
end

function coords_to_T(c::AbstractVector, B::AbstractMatrix; tau0=1.0)
    x = B * c
    τ = tau0 .* exp.(x)
    return Diagonal(τ)
end

function sample_ball(rng::AbstractRNG, dim::Int, rho::Float64)
    v = randn(rng, dim)
    v ./= norm(v)
    r = rho * rand(rng)^(1 / dim)
    return r * v
end

function sample_shell_biased(rng::AbstractRNG, dim::Int, rho::Float64; p=0.25)
    v = randn(rng, dim)
    v ./= norm(v)
    r = rho * rand(rng)^p
    return r * v
end

function sample_ball_points(rng::AbstractRNG, N::Int, dim::Int, rho::Float64; shell_bias=false, p=0.25)
    pts = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        pts[i] = shell_bias ? sample_shell_biased(rng, dim, rho; p=p) :
                              sample_ball(rng, dim, rho)
    end
    return pts
end

# ------------------------------------------------------------
# Helpers: modular 6-node network
# ------------------------------------------------------------
rotblock(α, β) = [-α -β; β -α]

"""
Build a 6×6 network from three 2×2 damped-oscillatory modules.

Module 1 = slow
Module 2 = intermediate
Module 3 = fast

Couplings are hand-tuned for illustration: strong enough to create
competition, but not so strong that everything collapses into one mode.
"""
function three_module_network()
    A1 = rotblock(0.14, 0.75)   # slow, lightly damped
    A2 = rotblock(0.22, 1.45)   # intermediate
    A3 = rotblock(0.30, 1.95)   # fast

    A = zeros(6, 6)
    A[1:2, 1:2] .= A1
    A[3:4, 3:4] .= A2
    A[5:6, 5:6] .= A3

    # Inter-module couplings (asymmetric on purpose for richer pattern)
    A[1, 3] = 0.20
    A[2, 4] = 0.16
    A[3, 1] = 0.11
    A[4, 2] = 0.09

    A[3, 5] = 0.24
    A[4, 6] = 0.18
    A[5, 3] = 0.13
    A[6, 4] = 0.11

    A[1, 5] = 0.07
    A[2, 6] = 0.05
    A[5, 1] = 0.04
    A[6, 2] = 0.06

    return A
end

# ------------------------------------------------------------
# Landscape evaluation
# ------------------------------------------------------------
function evaluate_profiles_matrix(A, P, ωs, B, coords_list; tau0=1.0)
    N = length(coords_list)
    M = length(ωs)
    profiles_mat = Matrix{Float64}(undef, N, M)

    for i in 1:N
        T = coords_to_T(coords_list[i], B; tau0=tau0)
        profiles_mat[i, :] .= structured_profile(A, T, P, ωs)
    end

    return profiles_mat
end

function pointwise_stats(profiles_mat, sample_rows)
    worst_line = vec(maximum(profiles_mat[sample_rows, :], dims=1))
    best_line  = vec(minimum(profiles_mat[sample_rows, :], dims=1))
    mean_line  = vec(mean(profiles_mat[sample_rows, :], dims=1))
    return best_line, mean_line, worst_line
end

# ------------------------------------------------------------
# Model setup
# ------------------------------------------------------------
A = three_module_network()

println("Eigenvalues of A:")
println(sort(eigvals(A), by = x -> imag(x)))

# Mixed perturbation class: within-module + cross-module structure
Pstruct = [
    0.0  0.4  0.8  0.0  0.4  0.0;
    0.4  0.0  0.0  0.7  0.0  0.3;
    0.8  0.0  0.0  0.5  0.9  0.0;
    0.0  0.7  0.5  0.0  0.0  0.8;
    0.4  0.0  0.9  0.0  0.0  0.6;
    0.0  0.3  0.0  0.8  0.6  0.0
]

# ------------------------------------------------------------
# Timescale-space setup
# ------------------------------------------------------------
n = 6
B = centered_basis(n)
dim = n - 1

tau0 = 1.0
rho_max = 2.4
rng = MersenneTwister(42)

# Wider window to reveal low/mid/high frequency regimes
ωs = 10 .^ range(-1.5, 1.1, length=1400)

# More samples + shell bias for richer envelopes
Nsamp = 5000
coords_samples = sample_ball_points(rng, Nsamp, dim, rho_max; shell_bias=true, p=0.22)

c_ref = zeros(dim)
coords_all = vcat([c_ref], coords_samples)

profiles_mat = evaluate_profiles_matrix(A, Pstruct, ωs, B, coords_all; tau0=tau0)

idx_ref = 1
sample_rows = 2:size(profiles_mat, 1)

profile_ref = vec(profiles_mat[idx_ref, :])
best_line, mean_line, worst_line = pointwise_stats(profiles_mat, sample_rows)

println("\nPointwise envelope statistics:")
@printf("reference profile min/max = %.6e / %.6e\n", minimum(profile_ref), maximum(profile_ref))
@printf("best-case line  min/max   = %.6e / %.6e\n", minimum(best_line), maximum(best_line))
@printf("mean-case line  min/max   = %.6e / %.6e\n", minimum(mean_line), maximum(mean_line))
@printf("worst-case line min/max   = %.6e / %.6e\n", minimum(worst_line), maximum(worst_line))

# Optional diagnostic: how many distinct sampled T attain the envelopes?
submat = profiles_mat[sample_rows, :]
idx_worst_by_ω = [argmax(view(submat, :, j)) + 1 for j in axes(submat, 2)]
idx_best_by_ω  = [argmin(view(submat, :, j)) + 1 for j in axes(submat, 2)]

println()
@printf("Distinct sampled T attaining pointwise worst-case: %d\n", length(unique(idx_worst_by_ω)))
@printf("Distinct sampled T attaining pointwise best-case : %d\n", length(unique(idx_best_by_ω)))

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
begin
    fig = Figure(size = (980, 680))

    ax = Axis(
        fig[1, 1],
        xlabel = "ω",
        ylabel = "‖S(ω) P S(ω)‖₂",
        title = "6-node modular network: pointwise structural-sensitivity envelopes",
        xscale = log10,
        yscale = log10
    )

    # Band makes the figure visually richer without changing the data
    band!(ax, ωs, best_line, worst_line, alpha = 0.22)

    lines!(ax, ωs, profile_ref, linewidth = 3, label = "homogeneous reference", color = :blue)
    # lines!(ax, ωs, mean_line,   linewidth = 3, label = "mean case")
    lines!(ax, ωs, worst_line,  linewidth = 3, label = "worst case", color = :red)
    lines!(ax, ωs, best_line,   linewidth = 3, label = "best case", color = :green)

    axislegend(ax, position = :lb)

    display(fig)
end