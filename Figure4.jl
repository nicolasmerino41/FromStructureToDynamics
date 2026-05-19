# ============================================================
# Figure4.jl
#
# This script builds four 8-species linear systems that have the same
# asymptotic resilience, measured by the spectral abscissa α(A),
# and the same Frobenius norm of their off-diagonal interactions.
# The figure compares how architectures with matched asymptotic
# resilience can still have very different intrinsic resolvent
# sensitivity profiles.
#
# ============================================================

using LinearAlgebra
using Printf
using Random
using CairoMakie
using Colors

const CASE_COLORS = Dict(
    "Symmetric normal"     => colorant"#2C7FB8",
    "Non-symmetric normal" => colorant"#D95F02",
    "Triangular"           => colorant"#1B9E77",
    "Modular"              => colorant"#7B3294",
)

resolvent(A, T, ω) = inv(im * ω * T - A)
intrinsic_sensitivity(A, T, ω) = opnorm(resolvent(A, T, ω), 2)
profile_intrinsic(A, T, ωs) = [intrinsic_sensitivity(A, T, ω) for ω in ωs]

spectral_abscissa(A) = maximum(real.(eigvals(A)))
resilience(A) = -spectral_abscissa(A)

function enforce_alpha(A, target_alpha)
    α = spectral_abscissa(A)
    A + (target_alpha - α) * I
end

function orthogonal_matrix(n; seed = 1)
    rng = MersenneTwister(seed)
    Q, _ = qr(randn(rng, n, n))
    Matrix(Q)
end

interaction_matrix(A) = A - Diagonal(diag(A))

function match_offdiag_frobenius(A, target_frob)
    D = Diagonal(diag(A))
    G = A - D
    current_frob = norm(G)

    current_frob == 0 && return A

    D + (target_frob / current_frob) * G
end

function trapz(x, y)
    area = 0.0

    for i in 1:(length(x) - 1)
        area += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    end

    area
end

function interaction_stats(A)
    G = interaction_matrix(A)

    total_abs = sum(abs.(G))
    frob = norm(G)
    op2 = opnorm(G, 2)
    maxrow = maximum(sum(abs.(G), dims = 2))

    total_abs, frob, op2, maxrow
end

function build_symmetric_normal(λ; seed = 10)
    Q = orthogonal_matrix(length(λ); seed = seed)
    A = Q * Diagonal(λ) * Q'

    0.5 * (A + A')
end

function build_nonsymmetric_normal(alphas, betas; seed = 20)
    @assert length(alphas) == length(betas)

    k = length(alphas)
    B = zeros(2k, 2k)

    for i in 1:k
        idx = (2i - 1):(2i)
        a = alphas[i]
        b = betas[i]

        B[idx, idx] .= [-a -b; b -a]
    end

    Q = orthogonal_matrix(2k; seed = seed)
    Q * B * Q'
end

function build_triangular_oscillatory()
    A = zeros(8, 8)

    A[1:2, 1:2] = [-0.35 -0.9;  0.9 -0.35]
    A[3:4, 3:4] = [-0.55 -1.7;  1.7 -0.55]
    A[5:6, 5:6] = [-0.85 -2.8;  2.8 -0.85]
    A[7:8, 7:8] = [-1.20 -4.2;  4.2 -1.20]

    A[1:2, 3:4] = [1.2 0.3; -0.4 0.9]
    A[3:4, 5:6] = [1.0 -0.2; 0.5 0.8]
    A[5:6, 7:8] = [0.9 0.4; -0.3 0.7]

    A[1:2, 5:6] = [0.45 0.10; -0.15 0.35]
    A[3:4, 7:8] = [0.35 -0.08; 0.12 0.28]
    A[1:2, 7:8] = [0.18 0.00; 0.00 0.14]

    A
end

function build_modular_oscillatory()
    A = zeros(8, 8)

    C1 = zeros(4, 4)
    C1[1:2, 1:2] = [-0.35 -0.8; 0.8 -0.35]
    C1[3:4, 3:4] = [-0.55 -1.4; 1.4 -0.55]
    C1[1:2, 3:4] = [0.45 0.12; -0.08 0.35]
    C1[3:4, 1:2] = [0.20 0.00; 0.00 0.18]

    C2 = zeros(4, 4)
    C2[1:2, 1:2] = [-0.40 -2.2; 2.2 -0.40]
    C2[3:4, 3:4] = [-0.65 -3.6; 3.6 -0.65]
    C2[1:2, 3:4] = [0.38 -0.06; 0.10 0.30]
    C2[3:4, 1:2] = [0.16 0.00; 0.00 0.14]

    A[1:4, 1:4] = C1
    A[5:8, 5:8] = C2

    A[1:4, 5:8] = [
        0.00 0.05 0.00 0.02
        0.03 0.00 0.04 0.00
        0.00 0.04 0.00 0.03
        0.02 0.00 0.03 0.00
    ]

    A[5:8, 1:4] = [
        0.00 0.02 0.00 0.01
        0.01 0.00 0.02 0.00
        0.00 0.02 0.00 0.01
        0.01 0.00 0.01 0.00
    ]

    A
end

function circular_layout(n; radius = 1.0, rotation = 0.0)
    [
        Point2f(
            radius * cos(rotation + 2π * (k - 1) / n),
            radius * sin(rotation + 2π * (k - 1) / n)
        )
        for k in 1:n
    ]
end

function balanced_oval_layout(n)
    θs = range(0, 2π, length = n + 1)[1:end - 1]
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

function modular_layout()
    [
        Point2f(-2.20,  0.95),
        Point2f(-1.75,  0.55),
        Point2f(-2.00, -0.45),
        Point2f(-1.55, -0.85),
        Point2f( 1.55,  0.85),
        Point2f( 2.00,  0.45),
        Point2f( 1.80, -0.55),
        Point2f( 2.25, -0.95),
    ]
end

function network_layout(name, n)
    if name == "Symmetric normal"
        circular_layout(n; radius = 1.0, rotation = π / 8)
    elseif name == "Non-symmetric normal"
        balanced_oval_layout(n)
    elseif name == "Triangular"
        triangular_layout()
    elseif name == "Modular"
        modular_layout()
    else
        circular_layout(n)
    end
end

function draw_colored_network!(ax, A, name, mx; node_size = 16, line_width = 2.2)
    G = interaction_matrix(A)
    pts = network_layout(name, size(G, 1))

    for i in axes(G, 1), j in axes(G, 2)
        if i != j && abs(G[i, j]) > 1e-12
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

function build_cases(; target_alpha = -0.35, target_frob = 3.5)
    A_sym = build_symmetric_normal(
        [-0.35, -0.55, -0.80, -1.05, -1.30, -1.55, -1.80, -2.10];
        seed = 11
    )

    A_nsn = build_nonsymmetric_normal(
        [0.35, 0.70, 1.20, 1.75],
        [0.80, 1.45, 2.60, 4.10];
        seed = 22
    )

    matrices = [
        "Symmetric normal"     => A_sym,
        "Non-symmetric normal" => A_nsn,
        "Triangular"           => build_triangular_oscillatory(),
        "Modular"              => build_modular_oscillatory(),
    ]

    matrices = [
        name => enforce_alpha(A, target_alpha)
        for (name, A) in matrices
    ]

    matrices = [
        name => match_offdiag_frobenius(A, target_frob)
        for (name, A) in matrices
    ]

    [
        name => enforce_alpha(A, target_alpha)
        for (name, A) in matrices
    ]
end

function print_diagnostics(cases, profiles, ωs, target_alpha, target_frob)
    println("============================================================")
    println("Four cases with the same resilience")
    println("============================================================")
    @printf("Target spectral abscissa α* = %.4f\n", target_alpha)
    @printf("Target resilience        r* = %.4f\n", -target_alpha)
    @printf("Target ‖A − diag(A)‖F     = %.4f\n\n", target_frob)

    for (name, A) in cases
        α = spectral_abscissa(A)
        normal_defect = norm(A' * A - A * A', 2)

        @printf(
            "%-24s  α(A) = % .6f   resilience = %.6f   normal defect = %.3e\n",
            name, α, resilience(A), normal_defect
        )
    end

    println("\n============================================================")
    println("Peak intrinsic sensitivities")
    println("============================================================")

    for (name, _) in cases
        profile = profiles[name]
        i = argmax(profile)

        @printf(
            "%-24s intrinsic max = %10.4f at ω = %.4g\n",
            name, profile[i], ωs[i]
        )
    end

    println("\n============================================================")
    println("Integrated intrinsic sensitivities")
    println("============================================================")

    for (name, _) in cases
        @printf(
            "%-24s ∫‖R(ω)‖₂ dω = %10.4f\n",
            name, trapz(ωs, profiles[name])
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
end

function plot_figure(cases, profiles, ωs)
    fig = Figure(
        size = (1600, 1000),
        fontsize = 16,
        figure_padding = (20, 20, 38, 40)
    )

    mx = maximum(abs.(vcat([vec(A) for (_, A) in cases]...)))

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

    for (j, (name, A)) in enumerate(cases)
        Label(
            fig[2, j],
            name;
            color = CASE_COLORS[name],
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

        xlims!(ax, 0.5, size(A, 2) + 0.5)
        ylims!(ax, size(A, 1) + 0.5, 0.5)

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

    Label(
        fig[4, 1:5],
        "Equal asymptotic resilience: α(A) = −0.35      Equal interaction magnitude: ‖A − diag(A)‖F = 3.5";
        fontsize = 18,
        tellwidth = false
    )

    ax_intr = Axis(
        fig[5, 1:4],
        xlabel = "ω",
        ylabel = "‖R(ω)‖₂",
        xscale = log10,
        xlabelsize = 20,
        ylabelsize = 20
    )

    ax_intr.xgridvisible = false
    ax_intr.ygridvisible = false
    ax_intr.xminorgridvisible = false
    ax_intr.yminorgridvisible = false

    hidespines!(ax_intr, :t, :r)

    for (name, _) in cases
        lines!(
            ax_intr,
            ωs,
            profiles[name];
            linewidth = 3,
            color = CASE_COLORS[name],
            label = name
        )
    end

    axislegend(
        ax_intr;
        position = (0.95, 0.95),
        framevisible = false,
        labelsize = 18,
        patchsize = (30, 18)
    )

    rowsize!(fig.layout, 1, Relative(0.27))
    rowsize!(fig.layout, 2, Fixed(34))
    rowsize!(fig.layout, 3, Relative(0.35))
    rowsize!(fig.layout, 4, Fixed(34))
    rowsize!(fig.layout, 5, Relative(0.37))

    rowgap!(fig.layout, 2, 18)
    colgap!(fig.layout, 12)

    fig
end

target_alpha = -0.35
target_frob = 3.5

ωs = 10 .^ range(-1.5, 1.1, length = 900)
T_hom = Diagonal(ones(8))

cases = build_cases(
    target_alpha = target_alpha,
    target_frob = target_frob
)

profiles = Dict(
    name => profile_intrinsic(A, T_hom, ωs)
    for (name, A) in cases
)

print_diagnostics(cases, profiles, ωs, target_alpha, target_frob)

fig = plot_figure(cases, profiles, ωs)

display(fig)
save("Figure4.png", fig; px_per_unit = 4)

fig
