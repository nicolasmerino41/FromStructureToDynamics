# ============================================================
# claim3_same_A_different_T_makie.jl
# ============================================================
using LinearAlgebra
using Printf
using CairoMakie

# ---------- Core definitions ----------
resolvent(A, T, ω) = inv(im * ω * T - A)

intrinsic_sensitivity(A, T, ω) = opnorm(resolvent(A, T, ω), 2)

structured_sensitivity(A, T, P, ω) = opnorm(resolvent(A, T, ω) * P * resolvent(A, T, ω), 2)

# ---------- Frequency grid ----------
ωs = 10 .^ range(-3, 2, length=700)

# ---------- Fixed interaction matrix ----------
A = [-1.0  1.35  0.0   0.0;
      0.0 -1.0   1.20  0.0;
      0.0  0.0  -1.0   1.05;
      0.0  0.0   0.0  -1.0]

println("Eigenvalues of A:")
println(sort(eigvals(A), by=x -> real(x)))

P = [0.0 1.0 0.0 0.0;
     0.0 0.0 1.0 0.0;
     0.0 0.0 0.0 1.0;
     0.0 0.0 0.0 0.0]

# ---------- Different timescale architectures ----------
T_hom = Diagonal([1.0, 1.0, 1.0, 1.0])
T_frontslow = Diagonal([2.5, 1.6, 0.9, 0.5])
T_backslow = Diagonal([0.5, 0.9, 1.6, 2.5])
T_alternating = Diagonal([2.2, 0.6, 2.2, 0.6])

Ts = Dict(
    "homogeneous" => T_hom,
    "front-slow" => T_frontslow,
    "back-slow" => T_backslow,
    "alternating" => T_alternating,
)

# ---------- Compute profiles ----------
intr = Dict{String, Vector{Float64}}()
strc = Dict{String, Vector{Float64}}()

for (name, T) in Ts
    intr[name] = [intrinsic_sensitivity(A, T, ω) for ω in ωs]
    strc[name] = [structured_sensitivity(A, T, P, ω) for ω in ωs]
end

# ---------- Peak summaries ----------
println("\nPeak intrinsic sensitivity")
for name in ["homogeneous", "front-slow", "back-slow", "alternating"]
    idx = argmax(intr[name])
    @printf("%-12s : max = %.4f at ω = %.4g\n", name, intr[name][idx], ωs[idx])
end

println("\nPeak structured sensitivity")
for name in ["homogeneous", "front-slow", "back-slow", "alternating"]
    idx = argmax(strc[name])
    @printf("%-12s : max = %.4f at ω = %.4g\n", name, strc[name][idx], ωs[idx])
end

# ---------- Plot ----------
begin
    fig = Figure(size = (1150, 450))

    ax1 = Axis(
        fig[1, 1],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Claim 3: intrinsic sensitivity for fixed A, different T",
        xscale = log10,
        yscale = log10
    )

    for name in ["homogeneous", "front-slow", "back-slow", "alternating"]
        lines!(ax1, ωs, intr[name], linewidth = 3, label = name)
    end
    axislegend(ax1, position = :lb)

    ax2 = Axis(
        fig[1, 2],
        xlabel = "ω",
        ylabel = "‖S(ω) P S(ω)‖₂",
        title = "Claim 3: structured sensitivity for fixed A, different T",
        xscale = log10,
        yscale = log10
    )

    for name in ["homogeneous", "front-slow", "back-slow", "alternating"]
        lines!(ax2, ωs, strc[name], linewidth = 3, label = name)
    end
    axislegend(ax2, position = :lb)

    display(fig)
end