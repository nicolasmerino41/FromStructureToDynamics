# ============================================================
# four_cases_same_resilience_makie_20species.jl
#
# Four 20-species systems with the same asymptotic resilience:
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

function interaction_stats(A)
    G = interaction_matrix(A)
    total_abs = sum(abs.(G))
    frob = norm(G)
    op2 = opnorm(G, 2)
    maxrow = maximum(sum(abs.(G), dims=2))
    return total_abs, frob, op2, maxrow
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
20x20 block-upper-triangular oscillatory non-normal matrix.
Built from ten 2x2 oscillatory blocks with feedforward coupling.
Eigenvalues are exactly those of the diagonal blocks.
"""
function build_triangular_oscillatory_20()
    nblocks = 10
    n = 2nblocks
    A = zeros(n, n)

    # diagonal oscillatory 2x2 blocks
    for k in 1:nblocks
        a = 0.35 + 0.10*(k-1)
        b = 0.9 + 0.45*(k-1)
        idx = (2k-1):(2k)
        A[idx, idx] .= [-a -b; b -a]
    end

    # nearest-neighbor feedforward couplings
    for k in 1:nblocks-1
        i = (2k-1):(2k)
        j = (2(k+1)-1):(2(k+1))
        A[i, j] .= [
            1.15 - 0.05*(k-1)    0.22 + 0.02*(k-1);
           -0.18 + 0.01*(k-1)    0.88 - 0.04*(k-1)
        ]
    end

    # next-nearest-neighbor feedforward couplings
    for k in 1:nblocks-2
        i = (2k-1):(2k)
        j = (2(k+2)-1):(2(k+2))
        A[i, j] .= [
            0.42 - 0.015*(k-1)    0.09;
           -0.10                  0.30 - 0.01*(k-1)
        ]
    end

    # weaker long-range feedforward couplings
    for k in 1:nblocks-3
        i = (2k-1):(2k)
        j = (2(k+3)-1):(2(k+3))
        A[i, j] .= [
            0.15   0.00;
            0.00   0.11
        ]
    end

    return A
end

"""
20x20 modular oscillatory matrix:
two communities of size 10, each made of five oscillatory 2x2 blocks,
with stronger within-community coupling and weaker asymmetric between-community coupling.
"""
function build_modular_oscillatory_20()
    n1, n2 = 10, 10
    A = zeros(20, 20)

    # ---------- Community 1 ----------
    C1 = zeros(n1, n1)
    for k in 1:5
        a = 0.35 + 0.08*(k-1)
        b = 0.8 + 0.35*(k-1)
        idx = (2k-1):(2k)
        C1[idx, idx] .= [-a -b; b -a]
    end

    # within-community feedforward structure
    for k in 1:4
        i = (2k-1):(2k)
        j = (2(k+1)-1):(2(k+1))
        C1[i, j] .= [
            0.42 - 0.03*(k-1)   0.10;
           -0.08                0.34 - 0.02*(k-1)
        ]
    end
    for k in 1:3
        i = (2k-1):(2k)
        j = (2(k+2)-1):(2(k+2))
        C1[i, j] .= [
            0.18   0.04;
           -0.03   0.14
        ]
    end
    for k in 2:5
        i = (2k-1):(2k)
        j = (2(k-1)-1):(2(k-1))
        C1[i, j] .= [
            0.14   0.00;
            0.00   0.12
        ]
    end

    # ---------- Community 2 ----------
    C2 = zeros(n2, n2)
    for k in 1:5
        a = 0.40 + 0.09*(k-1)
        b = 2.0 + 0.55*(k-1)
        idx = (2k-1):(2k)
        C2[idx, idx] .= [-a -b; b -a]
    end

    for k in 1:4
        i = (2k-1):(2k)
        j = (2(k+1)-1):(2(k+1))
        C2[i, j] .= [
            0.36 - 0.02*(k-1)   -0.05;
            0.09                 0.28 - 0.015*(k-1)
        ]
    end
    for k in 1:3
        i = (2k-1):(2k)
        j = (2(k+2)-1):(2(k+2))
        C2[i, j] .= [
            0.16  -0.03;
            0.04   0.12
        ]
    end
    for k in 2:5
        i = (2k-1):(2k)
        j = (2(k-1)-1):(2(k-1))
        C2[i, j] .= [
            0.12   0.00;
            0.00   0.10
        ]
    end

    A[1:10, 1:10] = C1
    A[11:20, 11:20] = C2

    # weak asymmetric inter-community coupling
    for i in 1:10, j in 11:20
        if isodd(i + j)
            A[i, j] = 0.05 + 0.01 * ((i + j) % 3)
        elseif abs(i - (j - 10)) <= 2
            A[i, j] = 0.025
        end
    end

    for i in 11:20, j in 1:10
        if isodd(i + j) && abs((i - 10) - j) <= 3
            A[i, j] = 0.018
        elseif ((i + j) % 5 == 0)
            A[i, j] = 0.012
        end
    end

    return A
end

# ------------------------------------------------------------
# Shared frequency grid and perturbation class
# ------------------------------------------------------------
n = 20
ωs = 10 .^ range(-1.5, 1.2, length=900)
T_hom = Diagonal(ones(n))

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

P = community_perturbation(10, 10; cross_weight=0.35)

# ------------------------------------------------------------
# Build four cases with the same resilience
# ------------------------------------------------------------
target_alpha = -0.35
target_resilience = -target_alpha

# 1) symmetric normal (20 eigenvalues)
λ_sym = [
    -0.35, -0.45, -0.55, -0.65, -0.78,
    -0.90, -1.02, -1.15, -1.28, -1.40,
    -1.52, -1.65, -1.78, -1.92, -2.05,
    -2.18, -2.32, -2.46, -2.60, -2.75
]
A_sym = build_symmetric_normal(λ_sym; seed=11)
A_sym = enforce_alpha(A_sym, target_alpha)

# 2) non-symmetric normal (10 oscillatory blocks = 20 species)
alphas_ns = [0.35, 0.48, 0.62, 0.78, 0.95, 1.13, 1.32, 1.52, 1.73, 1.95]
betas_ns  = [0.80, 1.10, 1.45, 1.85, 2.30, 2.80, 3.35, 3.95, 4.60, 5.30]
A_nsn = build_nonsymmetric_normal(alphas_ns, betas_ns; seed=22)
A_nsn = enforce_alpha(A_nsn, target_alpha)

# 3) triangular oscillatory (20 species)
A_tri = build_triangular_oscillatory_20()
A_tri = enforce_alpha(A_tri, target_alpha)

# 4) modular oscillatory (20 species)
A_mod = build_modular_oscillatory_20()
A_mod = enforce_alpha(A_mod, target_alpha)

# optional normalization of total off-diagonal interaction scale
target_frob = 7.0

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
println("Four 20-species cases with the same resilience")
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
    fig = Figure(size = (1700, 1150))

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

    Colorbar(
        fig[1, 5],
        limits = (-mx, mx),
        colormap = :balance,
        label = "Interaction strength"
    )

    # --- middle row: intrinsic sensitivity
    ax_intr = Axis(
        fig[2, 1:4],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Intrinsic sensitivity",
        xscale = log10,
    )

    for (name, _) in cases
        lines!(ax_intr, ωs, intr_profiles[name], linewidth=3, label=name)
    end
    axislegend(ax_intr, position=:rt)

    # --- bottom row: structured sensitivity
    ax_struct = Axis(
        fig[3, 1:4],
        xlabel = "ω",
        ylabel = "‖S(ω) P S(ω)‖₂",
        title = "Structured sensitivity",
        xscale = log10,
    )

    for (name, _) in cases
        lines!(ax_struct, ωs, struct_profiles[name], linewidth=3, label=name)
    end
    axislegend(ax_struct, position=:rt)

    Label(
        fig[0, 1:5],
        "Four 20-species architectures with identical asymptotic resilience",
        fontsize = 22,
        font = :bold
    )

    display(fig)
end