using CairoMakie
using LinearAlgebra

interaction_matrix(A) = A - Diagonal(diag(A))

# ------------------------------------------------------------
# Layouts
# ------------------------------------------------------------
function circular_layout(n; radius=1.0, rotation=0.0)
    [Point2f(radius*cos(rotation + 2π*(k-1)/n),
             radius*sin(rotation + 2π*(k-1)/n)) for k in 1:n]
end

function balanced_oval_layout(n)
    θs = range(0, 2π, length=n+1)[1:end-1]
    [Point2f(1.15*cos(θ), 0.82*sin(θ)) for θ in θs]
end

"""
Triangular/feedforward layout:
not bipartite, but clearly hierarchical.
We place the four 2x2 oscillatory blocks in successive levels,
with slight vertical offsets to avoid the 'two columns' look.
"""
function triangular_layout()
    [
        Point2f(0.0,  0.55),   # block 1
        Point2f(0.2, -0.15),

        Point2f(1.2,  0.95),   # block 2
        Point2f(1.4,  0.20),

        Point2f(2.5,  0.35),   # block 3
        Point2f(2.7, -0.45),

        Point2f(3.8,  0.75),   # block 4
        Point2f(4.0, -0.05),
    ]
end
"""
Modular layout with 4 visible submodules:
(two higher-level communities, each split into two 2x2 blocks)
"""
function modular4_layout()
    [
        Point2f(-2.2,  0.95),  # submodule 1
        Point2f(-1.75, 0.55),

        Point2f(-2.0, -0.45),  # submodule 2
        Point2f(-1.55,-0.85),

        Point2f( 1.55, 0.85),  # submodule 3
        Point2f( 2.0,  0.45),

        Point2f( 1.8, -0.55),  # submodule 4
        Point2f( 2.25,-0.95),
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

# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------
function draw_colored_network!(ax, A, name, mx; node_size=16, line_width=2.2)
    G = interaction_matrix(A)
    n = size(G, 1)
    pts = network_layout(name, n)

    tol = 1e-12
    for i in 1:n, j in 1:n
        if i != j && abs(G[i,j]) > tol
            lines!(ax,
                [pts[j][1], pts[i][1]],
                [pts[j][2], pts[i][2]];
                color = G[i,j],
                colorrange = (-mx, mx),
                colormap = :balance,
                linewidth = line_width
            )
        end
    end

    scatter!(ax, first.(pts), last.(pts);
        color = :black,
        markersize = node_size
    )

    hidedecorations!(ax)
    hidespines!(ax)
    ax.aspect = DataAspect()
end
begin
    fig_net = Figure(size=(1000, 700))

    allG = vcat([vec(interaction_matrix(A)) for (_, A) in cases]...)
    mxG = maximum(abs.(allG))

    positions = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]

    for ((name, A), (r, c)) in zip(cases, positions)
        ax = Axis(fig_net[r, c])
        draw_colored_network!(ax, A, name, mxG; node_size=24, line_width=3.0)
    end

    # Colorbar(
    #     fig_net[1:2, 3],
    #     limits = (-mxG, mxG),
    #     colormap = :balance,
    #     label = "Interaction strength"
    # )

    # Label(
    #     fig_net[0, 1:3],
    #     "Network representations of the four interaction architectures",
    #     fontsize = 22,
    #     font = :bold
    # )

    display(fig_net)
end