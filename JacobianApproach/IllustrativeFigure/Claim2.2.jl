# ============================================================
# four_cases_same_resilience_makie.jl
#
# Four systems with the same asymptotic resilience:
#   1) symmetric normal
#   2) non-symmetric normal
#   3) triangular
#   4) modular
#
# Top row: matrix heatmaps
# Middle row: intrinsic sensitivity
# Bottom row: structured sensitivity
#
# "Same resilience" is enforced as same spectral abscissa:
#   α(A) = maximum(real.(eigvals(A)))
# and we set α(A) = target_alpha for all four matrices.
# ============================================================

using LinearAlgebra
using Printf
using Random
using CairoMakie

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
resolvent(A, T, ω) = inv(im * ω * T - A)

intrinsic_sensitivity(A, T, ω) = opnorm(resolvent(A, T, ω), 2)

structured_sensitivity(A, T, P, ω) = opnorm(resolvent(A, T, ω) * P * resolvent(A, T, ω), 2)

profile_intrinsic(A, T, ωs) = [intrinsic_sensitivity(A, T, ω) for ω in ωs]
profile_structured(A, T, P, ωs) = [structured_sensitivity(A, T, P, ω) for ω in ωs]

function trapz(x, y)
    s = 0.0
    for i in 1:length(x)-1
        s += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return s
end

spectral_abscissa(A) = maximum(real.(eigvals(A)))
resilience(A) = -spectral_abscissa(A)

function enforce_alpha(A, target_alpha)
    α = spectral_abscissa(A)
    return A + (target_alpha - α) * I
end

function orthogonal_matrix(n; seed=1)
    rng = MersenneTwister(seed)
    Q, _ = qr(randn(rng, n, n))
    return Matrix(Q)
end

# ------------------------------------------------------------
# Matrix builders
# ------------------------------------------------------------

"""
Symmetric normal matrix:
Q * Diagonal(λ) * Q'
"""
function build_symmetric_normal(λ; seed=10)
    n = length(λ)
    Q = orthogonal_matrix(n, seed=seed)
    A = Q * Diagonal(λ) * Q'
    return 0.5 * (A + A')
end

"""
Non-symmetric normal real matrix:
orthogonal similarity of 2x2 damped-rotation blocks.
Each block [ -a  -b; b  -a ] is normal and non-symmetric if b != 0.
"""
function build_nonsymmetric_normal(alphas, betas; seed=20)
    @assert length(alphas) == length(betas)
    k = length(alphas)
    n = 2k
    B = zeros(n, n)
    for i in 1:k
        a = alphas[i]
        b = betas[i]
        idx = (2i-1):(2i)
        B[idx, idx] .= [-a -b; b -a]
    end
    Q = orthogonal_matrix(n, seed=seed)
    A = Q * B * Q'
    return A
end

"""
Upper triangular non-normal matrix.
Eigenvalues are exactly the diagonal entries.
"""
function build_triangular(λ)
    n = length(λ)
    A = zeros(n, n)
    for i in 1:n
        A[i, i] = λ[i]
    end

    # strong upper-triangular feedforward couplings
    for i in 1:n-1
        A[i, i+1] = 1.6 - 0.12*(i-1)
    end
    for i in 1:n-2
        A[i, i+2] = 0.65 + 0.05*(i-1)
    end
    for i in 1:n-3
        A[i, i+3] = 0.22
    end

    return A
end
function build_triangular_oscillatory()
    # 4 oscillatory blocks => 8x8 total
    B1 = [-0.35 -0.9;  0.9 -0.35]
    B2 = [-0.55 -1.7;  1.7 -0.55]
    B3 = [-0.85 -2.8;  2.8 -0.85]
    B4 = [-1.20 -4.2;  4.2 -1.20]

    A = zeros(8, 8)

    # diagonal 2x2 oscillatory blocks
    A[1:2, 1:2] = B1
    A[3:4, 3:4] = B2
    A[5:6, 5:6] = B3
    A[7:8, 7:8] = B4

    # feedforward couplings: block upper triangular
    A[1:2, 3:4] = [1.2  0.3;
                   -0.4 0.9]

    A[3:4, 5:6] = [1.0 -0.2;
                    0.5 0.8]

    A[5:6, 7:8] = [0.9  0.4;
                   -0.3 0.7]

    # longer-range feedforward couplings
    A[1:2, 5:6] = [0.45 0.10;
                  -0.15 0.35]

    A[3:4, 7:8] = [0.35 -0.08;
                   0.12  0.28]

    A[1:2, 7:8] = [0.18  0.00;
                   0.00  0.14]

    return A
end

"""
Two-community modular matrix with directed but weak inter-community coupling.
Community sizes chosen as 4 + 4 for a clear block structure.
"""
function build_modular()
    n1, n2 = 4, 4
    n = n1 + n2
    A = zeros(n, n)

    # community 1: denser internal interactions
    C1 = [
        0.0  0.90 0.65 0.55;
        0.35 0.0  0.80 0.45;
        0.25 0.50 0.0  0.85;
        0.40 0.30 0.55 0.0
    ]

    # community 2: different internal geometry
    C2 = [
        0.0  0.75 0.50 0.35;
        0.55 0.0  0.70 0.25;
        0.40 0.60 0.0  0.78;
        0.20 0.35 0.48 0.0
    ]

    # weak inter-community couplings, directed and asymmetric
    B12 = [
        0.00 0.08 0.00 0.04;
        0.03 0.00 0.07 0.00;
        0.00 0.05 0.00 0.06;
        0.02 0.00 0.03 0.00
    ]

    B21 = [
        0.00 0.03 0.00 0.02;
        0.01 0.00 0.02 0.00;
        0.00 0.02 0.00 0.03;
        0.01 0.00 0.02 0.00
    ]

    A[1:n1, 1:n1] .= C1
    A[n1+1:end, n1+1:end] .= C2
    A[1:n1, n1+1:end] .= B12
    A[n1+1:end, 1:n1] .= B21

    # negative self-regulation, not all identical to avoid triviality
    A .-= Diagonal([2.1, 2.0, 2.2, 2.05, 1.95, 2.15, 2.05, 2.0])

    return A
end
function build_modular_oscillatory()
    # Two communities, each 4x4, each made of two oscillatory 2x2 blocks
    A = zeros(8, 8)

    # Community 1
    C1a = [-0.35 -0.8;  0.8 -0.35]
    C1b = [-0.55 -1.4;  1.4 -0.55]

    C1 = zeros(4, 4)
    C1[1:2, 1:2] = C1a
    C1[3:4, 3:4] = C1b
    C1[1:2, 3:4] = [0.45  0.12;
                   -0.08  0.35]
    C1[3:4, 1:2] = [0.20  0.00;
                    0.00  0.18]

    # Community 2
    C2a = [-0.40 -2.2;  2.2 -0.40]
    C2b = [-0.65 -3.6;  3.6 -0.65]

    C2 = zeros(4, 4)
    C2[1:2, 1:2] = C2a
    C2[3:4, 3:4] = C2b
    C2[1:2, 3:4] = [0.38 -0.06;
                    0.10  0.30]
    C2[3:4, 1:2] = [0.16  0.00;
                    0.00  0.14]

    A[1:4, 1:4] = C1
    A[5:8, 5:8] = C2

    # weak inter-community coupling
    A[1:4, 5:8] = [
        0.00 0.05 0.00 0.02;
        0.03 0.00 0.04 0.00;
        0.00 0.04 0.00 0.03;
        0.02 0.00 0.03 0.00
    ]

    A[5:8, 1:4] = [
        0.00 0.02 0.00 0.01;
        0.01 0.00 0.02 0.00;
        0.00 0.02 0.00 0.01;
        0.01 0.00 0.01 0.00
    ]

    return A
end
# ------------------------------------------------------------
# Shared frequency grid and perturbation class
# ------------------------------------------------------------
ωs = 10 .^ range(-1.5, 1.1, length=900)
T_hom = Diagonal(ones(8))

# structured perturbation emphasizing community-aware perturbations
function community_perturbation(n1, n2; cross_weight=0.35)
    n = n1 + n2
    P = zeros(n, n)
    for i in 1:n, j in 1:n
        if i != j
            same_comm = (i <= n1 && j <= n1) || (i > n1 && j > n1)
            P[i, j] = same_comm ? 1.0 : cross_weight
        end
    end
    return P
end

P = community_perturbation(4, 4; cross_weight=0.35)

# ------------------------------------------------------------
# Build four cases with the same resilience
# ------------------------------------------------------------
target_alpha = -0.35   # same spectral abscissa for all
target_resilience = -target_alpha

# 1) symmetric normal
λ_sym = [-0.35, -0.55, -0.80, -1.05, -1.30, -1.55, -1.80, -2.10]
A_sym = build_symmetric_normal(λ_sym; seed=11)
A_sym = enforce_alpha(A_sym, target_alpha)

# 2) non-symmetric normal
alphas_ns = [0.35, 0.70, 1.20, 1.75]
betas_ns  = [0.80, 1.45, 2.60, 4.10]
A_nsn = build_nonsymmetric_normal(alphas_ns, betas_ns; seed=22)
A_nsn = enforce_alpha(A_nsn, target_alpha)

# 3) triangular
λ_tri = [-0.35, -0.52, -0.74, -0.98, -1.24, -1.52, -1.83, -2.15]
A_tri = build_triangular_oscillatory()
A_tri = enforce_alpha(A_tri, target_alpha)

A_mod = build_modular_oscillatory()
A_mod = enforce_alpha(A_mod, target_alpha)

interaction_matrix(A) = A - Diagonal(diag(A))

function match_offdiag_frobenius(A, target_frob)
    D = Diagonal(diag(A))
    G = A - D
    ng = norm(G)
    if ng == 0
        return A
    end
    return D + (target_frob / ng) * G
end
target_frob = 3.5   # choose whatever scale you want

A_sym = match_offdiag_frobenius(A_sym, target_frob)
A_nsn = match_offdiag_frobenius(A_nsn, target_frob)
A_tri = match_offdiag_frobenius(A_tri, target_frob)
A_mod = match_offdiag_frobenius(A_mod, target_frob)

A_sym = enforce_alpha(A_sym, target_alpha)
A_nsn = enforce_alpha(A_nsn, target_alpha)
A_tri = enforce_alpha(A_tri, target_alpha)
A_mod = enforce_alpha(A_mod, target_alpha)
# ------------------------------------------------------------
# Sensitivity profiles
# ------------------------------------------------------------
intr_profiles = Dict{String, Vector{Float64}}()
struct_profiles = Dict{String, Vector{Float64}}()

cases = [
    ("Symmetric normal", A_sym),
    ("Non-symmetric normal", A_nsn),
    ("Triangular", A_tri),
    ("Modular", A_mod),
]
for (name, A) in cases
    intr_profiles[name] = profile_intrinsic(A, T_hom, ωs)
    struct_profiles[name] = profile_structured(A, T_hom, P, ωs)
end
# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------
println("============================================================")
println("Four cases with the same resilience")
println("============================================================")
@printf("Target spectral abscissa α* = %.4f\n", target_alpha)
@printf("Target resilience        r* = %.4f\n\n", target_resilience)

for (name, A) in cases
    α = spectral_abscissa(A)
    r = resilience(A)
    normal_defect = norm(A'A - A*A', 2)
    @printf("%-24s  α(A) = % .6f   resilience = %.6f   normal defect = %.3e\n",
        name, α, r, normal_defect)
end

println("\nMatrices:")
for (name, A) in cases
    println("\n--- $name ---")
    show(stdout, "text/plain", round.(A, digits=3))
    println()
end

println("\n============================================================")
println("Peak sensitivities")
println("============================================================")
for (name, A) in cases
    yi = intr_profiles[name]
    ys = struct_profiles[name]
    ii = argmax(yi)
    jj = argmax(ys)
    @printf("%-24s intrinsic max = %10.4f at ω = %.4g\n", name, yi[ii], ωs[ii])
    @printf("%-24s structured max = %9.4f at ω = %.4g\n\n", "", ys[jj], ωs[jj])
end

println("Integrated sensitivities")
for (name, _) in cases
    ai = trapz(ωs, intr_profiles[name])
    as = trapz(ωs, struct_profiles[name])
    @printf("%-24s ∫‖S(ω)‖ dω = %10.4f   ∫‖SPS‖ dω = %10.4f\n", name, ai, as)
end
interaction_matrix(A) = A - Diagonal(diag(A))

function interaction_stats(A)
    G = interaction_matrix(A)
    total_abs = sum(abs.(G))
    frob = norm(G)
    op2 = opnorm(G, 2)
    maxrow = maximum(sum(abs.(G), dims=2))
    return total_abs, frob, op2, maxrow
end

println("\n============================================================")
println("Interaction-strength diagnostics")
println("============================================================")
for (name, A) in cases
    total_abs, frob, op2, maxrow = interaction_stats(A)
    @printf("%-24s total|offdiag| = %8.4f   ‖offdiag‖F = %8.4f   ‖offdiag‖2 = %8.4f   max row sum = %8.4f\n",
        name, total_abs, frob, op2, maxrow)
end
# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
allvals = vcat([vec(A) for (_, A) in cases]...)
mx = maximum(abs.(allvals))

begin
    fig = Figure(size = (1600, 1100))

    allvals = vcat([vec(A) for (_, A) in cases]...)
    mx = maximum(abs.(allvals))

    for (j, (name, A)) in enumerate(cases)
    ax = Axis(
        fig[1, j],
        title = name,
        xlabel = "Species j",
        ylabel = "Species i",
        aspect = DataAspect()
    )

    heatmap!(
        ax, 1:size(A,2), 1:size(A,1), A;
        colorrange = (-mx, mx),
        colormap = :balance
    )

    ylims!(ax, size(A,1) + 0.5, 0.5)
    xlims!(ax, 0.5, size(A,2) + 0.5)

    ax.xticksvisible = false
    ax.yticksvisible = false
    ax.xticklabelsvisible = false
    ax.yticklabelsvisible = false
    ax.xminorticksvisible = false
    ax.yminorticksvisible = false
    ax.xgridvisible = false
    ax.ygridvisible = false
end

    Colorbar(fig[1, 5], limits = (-mx, mx), colormap = :balance, label = "Interaction strength")
    # --- middle row: intrinsic sensitivity
    ax_intr = Axis(
        fig[2, 1:4],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Intrinsic sensitivity",
        xscale = log10,
        # yscale = log10
    )

    for (name, _) in cases
        lines!(ax_intr, ωs, intr_profiles[name], linewidth=3, label=name)
    end
    axislegend(ax_intr, position=:rt)

    # # --- bottom row: structured sensitivity
    # ax_struct = Axis(
    #     fig[3, 1:4],
    #     xlabel = "ω",
    #     ylabel = "‖S(ω) P S(ω)‖₂",
    #     title = "Structured sensitivity",
    #     xscale = log10,
    #     # yscale = log10
    # )

    # for (name, _) in cases
    #     lines!(ax_struct, ωs, struct_profiles[name], linewidth=3, label=name)
    # end
    # axislegend(ax_struct, position=:rt)

    Label(
        fig[0, 1:5],
        "Four architectures with identical asymptotic resilience",
        fontsize = 22,
        font = :bold
    )

    display(fig)
end