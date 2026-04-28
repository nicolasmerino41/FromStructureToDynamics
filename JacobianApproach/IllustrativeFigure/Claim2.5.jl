# ============================================================
# four_cases_same_resilience_full_makie.jl
#
# Four systems with identical asymptotic resilience:
#   1) symmetric normal
#   2) non-symmetric normal
#   3) triangular/feedforward oscillatory
#   4) modular oscillatory
#
# Figure:
#   Row 1: network drawings
#   Row 2: matrix heatmaps, with colored architecture labels
#   Row 3: intrinsic sensitivity profiles, with matching colors
#
# Same resilience enforced by same spectral abscissa:
#   α(A) = maximum(real.(eigvals(A)))
# ============================================================

using LinearAlgebra
using Printf
using Random
using CairoMakie
using Colors

# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------
case_cols = Dict(
    "Symmetric normal"     => colorant"#2C7FB8",
    "Non-symmetric normal" => colorant"#D95F02",
    "Triangular"           => colorant"#1B9E77",
    "Modular"              => colorant"#7B3294",
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
resolvent(A, T, ω) = inv(im * ω * T - A)
intrinsic_sensitivity(A, T, ω) = opnorm(resolvent(A, T, ω), 2)
profile_intrinsic(A, T, ωs) = [intrinsic_sensitivity(A, T, ω) for ω in ωs]

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
    ng == 0 && return A
    return D + (target_frob / ng) * G
end

function trapz(x, y)
    s = 0.0
    for i in 1:length(x)-1
        s += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return s
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
function build_symmetric_normal(λ; seed=10)
    n = length(λ)
    Q = orthogonal_matrix(n, seed=seed)
    A = Q * Diagonal(λ) * Q'
    return 0.5 * (A + A')
end

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
    return Q * B * Q'
end

function build_triangular_oscillatory()
    B1 = [-0.35 -0.9;  0.9 -0.35]
    B2 = [-0.55 -1.7;  1.7 -0.55]
    B3 = [-0.85 -2.8;  2.8 -0.85]
    B4 = [-1.20 -4.2;  4.2 -1.20]

    A = zeros(8, 8)

    A[1:2, 1:2] = B1
    A[3:4, 3:4] = B2
    A[5:6, 5:6] = B3
    A[7:8, 7:8] = B4

    A[1:2, 3:4] = [1.2  0.3;
                   -0.4  0.9]

    A[3:4, 5:6] = [1.0 -0.2;
                    0.5  0.8]

    A[5:6, 7:8] = [0.9  0.4;
                   -0.3  0.7]

    A[1:2, 5:6] = [0.45  0.10;
                   -0.15  0.35]

    A[3:4, 7:8] = [0.35 -0.08;
                    0.12  0.28]

    A[1:2, 7:8] = [0.18  0.00;
                    0.00  0.14]

    return A
end

function build_modular_oscillatory()
    A = zeros(8, 8)

    C1a = [-0.35 -0.8;  0.8 -0.35]
    C1b = [-0.55 -1.4;  1.4 -0.55]

    C1 = zeros(4, 4)
    C1[1:2, 1:2] = C1a
    C1[3:4, 3:4] = C1b
    C1[1:2, 3:4] = [0.45  0.12;
                    -0.08  0.35]
    C1[3:4, 1:2] = [0.20  0.00;
                     0.00  0.18]

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
# Network layouts
# ------------------------------------------------------------
function circular_layout(n; radius=1.0, rotation=0.0)
    [
        Point2f(
            radius * cos(rotation + 2π * (k-1) / n),
            radius * sin(rotation + 2π * (k-1) / n)
        )
        for k in 1:n
    ]
end

function balanced_oval_layout(n)
    θs = range(0, 2π, length=n+1)[1:end-1]
    [Point2f(1.15 * cos(θ), 0.82 * sin(θ)) for θ in θs]
end

function triangular_layout()
    [
        Point2f(0.0,  0.55),
        Point2f(0.2, -0.15),

        Point2f(1.2,  0.95),
        Point2f(1.4,  0.20),

        Point2f(2.5,  0.35),
        Point2f(2.7, -0.45),

        Point2f(3.8,  0.75),
        Point2f(4.0, -0.05),
    ]
end

function modular4_layout()
    [
        Point2f(-2.2,   0.95),
        Point2f(-1.75,  0.55),

        Point2f(-2.0,  -0.45),
        Point2f(-1.55, -0.85),

        Point2f( 1.55,  0.85),
        Point2f( 2.0,   0.45),

        Point2f( 1.8,  -0.55),
        Point2f( 2.25, -0.95),
    ]
end

function network_layout(name, n)
    if name == "Symmetric normal"
        return circular_layout(n; radius=1.0, rotation=π/8)
    elseif name == "Non-symmetric normal"
        return balanced_oval_layout(n)
    elseif name == "Triangular"
        return triangular_layout()
    elseif name == "Modular"
        return modular4_layout()
    else
        return circular_layout(n)
    end
end

function draw_colored_network!(ax, A, name, mx; node_size=16, line_width=2.2)
    G = interaction_matrix(A)
    n = size(G, 1)
    pts = network_layout(name, n)

    tol = 1e-12

    for i in 1:n, j in 1:n
        if i != j && abs(G[i, j]) > tol
            lines!(
                ax,
                [pts[j][1], pts[i][1]],
                [pts[j][2], pts[i][2]];
                color = G[i, j],
                colorrange = (-mx, mx),
                colormap = :balance,
                linewidth = line_width
            )
        end
    end

    scatter!(
        ax,
        first.(pts),
        last.(pts);
        color = :black,
        markersize = node_size
    )

    hidedecorations!(ax)
    hidespines!(ax)
    ax.aspect = DataAspect()
end

# ------------------------------------------------------------
# Build four cases with same resilience
# ------------------------------------------------------------
ωs = 10 .^ range(-1.5, 1.1, length=900)
T_hom = Diagonal(ones(8))

target_alpha = -0.35
target_resilience = -target_alpha
target_frob = 3.5

λ_sym = [-0.35, -0.55, -0.80, -1.05, -1.30, -1.55, -1.80, -2.10]
A_sym = build_symmetric_normal(λ_sym; seed=11)

alphas_ns = [0.35, 0.70, 1.20, 1.75]
betas_ns  = [0.80, 1.45, 2.60, 4.10]
A_nsn = build_nonsymmetric_normal(alphas_ns, betas_ns; seed=22)

A_tri = build_triangular_oscillatory()
A_mod = build_modular_oscillatory()

A_sym = enforce_alpha(A_sym, target_alpha)
A_nsn = enforce_alpha(A_nsn, target_alpha)
A_tri = enforce_alpha(A_tri, target_alpha)
A_mod = enforce_alpha(A_mod, target_alpha)

A_sym = match_offdiag_frobenius(A_sym, target_frob)
A_nsn = match_offdiag_frobenius(A_nsn, target_frob)
A_tri = match_offdiag_frobenius(A_tri, target_frob)
A_mod = match_offdiag_frobenius(A_mod, target_frob)

A_sym = enforce_alpha(A_sym, target_alpha)
A_nsn = enforce_alpha(A_nsn, target_alpha)
A_tri = enforce_alpha(A_tri, target_alpha)
A_mod = enforce_alpha(A_mod, target_alpha)

cases = [
    ("Symmetric normal", A_sym),
    ("Non-symmetric normal", A_nsn),
    ("Triangular", A_tri),
    ("Modular", A_mod),
]

# ------------------------------------------------------------
# Sensitivity profiles
# ------------------------------------------------------------
intr_profiles = Dict{String, Vector{Float64}}()

for (name, A) in cases
    intr_profiles[name] = profile_intrinsic(A, T_hom, ωs)
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

    @printf(
        "%-24s  α(A) = % .6f   resilience = %.6f   normal defect = %.3e\n",
        name, α, r, normal_defect
    )
end

println("\n============================================================")
println("Peak intrinsic sensitivities")
println("============================================================")

for (name, _) in cases
    yi = intr_profiles[name]
    ii = argmax(yi)

    @printf(
        "%-24s intrinsic max = %10.4f at ω = %.4g\n",
        name, yi[ii], ωs[ii]
    )
end

println("\n============================================================")
println("Integrated intrinsic sensitivities")
println("============================================================")

for (name, _) in cases
    ai = trapz(ωs, intr_profiles[name])

    @printf(
        "%-24s ∫‖S(ω)‖₂ dω = %10.4f\n",
        name, ai
    )
end

println("\n============================================================")
println("Interaction-strength diagnostics")
println("============================================================")

for (name, A) in cases
    total_abs, frob, op2, maxrow = interaction_stats(A)

    @printf(
        "%-24s total|offdiag| = %8.4f   ‖offdiag‖F = %8.4f   ‖offdiag‖₂ = %8.4f   max row sum = %8.4f\n",
        name, total_abs, frob, op2, maxrow
    )
end

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
begin
    fig = Figure(
        size = (1600, 1000),
        fontsize = 16,
        figure_padding = (20, 20, 30, 32)
    )

    allvals = vcat([vec(A) for (_, A) in cases]...)
    mx = maximum(abs.(allvals))

    # -------------------------
    # Row 1: network drawings
    # -------------------------
    for (j, (name, A)) in enumerate(cases)
        ax_net = Axis(fig[1, j])
        draw_colored_network!(
            ax_net,
            A,
            name,
            mx;
            node_size = 14,
            line_width = 2.0
        )
    end

    # -------------------------
    # Row 2: colored matrix labels + heatmaps
    # -------------------------
    for (j, (name, A)) in enumerate(cases)

        Label(
            fig[2, j],
            name;
            color = case_cols[name],
            fontsize = 20,
            font = :bold,
            tellwidth = false
        )

        ax = Axis(
            fig[3, j],
            xlabel = "Species j",
            ylabel = "Species i",
            aspect = DataAspect()
        )

        heatmap!(
            ax,
            1:size(A, 2),
            1:size(A, 1),
            A;
            colorrange = (-mx, mx),
            colormap = :balance
        )

        ylims!(ax, size(A, 1) + 0.5, 0.5)
        xlims!(ax, 0.5, size(A, 2) + 0.5)

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
        fig[3, 5],
        limits = (-mx, mx),
        colormap = :balance,
        label = "Interaction strength"
    )

    # -------------------------
    # Row 3: intrinsic sensitivity
    # -------------------------
    ax_intr = Axis(
        fig[4, 1:4],
        xlabel = "ω",
        ylabel = "‖R(ω)‖₂",
        # title = "Intrinsic sensitivity",
        xscale = log10,
        xlabelsize = 17,
        ylabelsize = 17
        )

    ax_intr.xgridvisible = false
    ax_intr.ygridvisible = false
    ax_intr.xminorgridvisible = false
    ax_intr.yminorgridvisible = false
    for (name, _) in cases
        lines!(
            ax_intr,
            ωs,
            intr_profiles[name];
            linewidth = 3,
            color = case_cols[name],
            label = name
        )
    end

    axislegend(
        ax_intr;
        position = :rt,
        framevisible = false
    )

    # Label(
    #     fig[0, 1:5],
    #     "Four architectures with identical asymptotic resilience",
    #     fontsize = 22,
    #     font = :bold
    # )

    rowsize!(fig.layout, 1, Relative(0.27))
    rowsize!(fig.layout, 2, Fixed(34))
    rowsize!(fig.layout, 3, Relative(0.36))
    rowsize!(fig.layout, 4, Relative(0.37))

    rowgap!(fig.layout, 12)
    colgap!(fig.layout, 12)

    display(fig)
end