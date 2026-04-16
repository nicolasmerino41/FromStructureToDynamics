# ============================================================
# claim2_same_eigenvalues_different_profiles_makie.jl
# ============================================================
using LinearAlgebra
using Printf
using CairoMakie

# ---------- Core definitions ----------
resolvent(A, T, ω) = inv(im * ω * T - A)

intrinsic_sensitivity(A, T, ω) = opnorm(resolvent(A, T, ω), 2)

structured_sensitivity(A, T, P, ω) = opnorm(resolvent(A, T, ω) * P * resolvent(A, T, ω), 2)

# ---------- Frequency grid ----------
ωs = 10 .^ range(-3, 2, length=600)

# ---------- Two matrices with identical eigenvalues ----------
λ = [-0.25, -0.8, -1.6]
Λ = Diagonal(λ)

A_normal = Matrix(Λ)

V = [1.0  4.0  0.0;
     0.0  1.0  3.0;
     0.0  0.0  1.0]

A_nonnormal = V * Λ * inv(V)

T = Diagonal(ones(3))

P = [0.0 1.0 0.0;
     0.0 0.0 1.0;
     0.0 0.0 0.0]

# ---------- Sanity checks ----------
println("Eigenvalues of A_normal:")
println(sort(eigvals(A_normal), by=x -> real(x)))

println("\nEigenvalues of A_nonnormal:")
println(sort(eigvals(A_nonnormal), by=x -> real(x)))

println("\nCondition number of V:")
@printf("%.3f\n", cond(V))

# ---------- Compute spectra ----------
intr_normal = [intrinsic_sensitivity(A_normal, T, ω) for ω in ωs]
intr_nonnormal = [intrinsic_sensitivity(A_nonnormal, T, ω) for ω in ωs]

struct_normal = [structured_sensitivity(A_normal, T, P, ω) for ω in ωs]
struct_nonnormal = [structured_sensitivity(A_nonnormal, T, P, ω) for ω in ωs]

# ---------- Peak summaries ----------
i1 = argmax(intr_normal)
i2 = argmax(intr_nonnormal)

println("\nPeak intrinsic sensitivity")
@printf("Normal      : max = %.4f at ω = %.4g\n", intr_normal[i1], ωs[i1])
@printf("Non-normal  : max = %.4f at ω = %.4g\n", intr_nonnormal[i2], ωs[i2])

j1 = argmax(struct_normal)
j2 = argmax(struct_nonnormal)

println("\nPeak structured sensitivity")
@printf("Normal      : max = %.4f at ω = %.4g\n", struct_normal[j1], ωs[j1])
@printf("Non-normal  : max = %.4f at ω = %.4g\n", struct_nonnormal[j2], ωs[j2])

# ---------- Plot ----------
begin
    fig = Figure(size = (1100, 450))

    ax1 = Axis(
        fig[1, 1],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Claim 2: intrinsic sensitivity",
        xscale = log10,
        yscale = log10
    )

    lines!(ax1, ωs, intr_normal, linewidth = 3, label = "Normal")
    lines!(ax1, ωs, intr_nonnormal, linewidth = 3, label = "Non-normal")
    axislegend(ax1, position = :lb)

    ax2 = Axis(
        fig[1, 2],
        xlabel = "ω",
        ylabel = "‖S(ω) P S(ω)‖₂",
        title = "Claim 2: structured sensitivity",
        xscale = log10,
        yscale = log10
    )

    lines!(ax2, ωs, struct_normal, linewidth = 3, label = "Normal")
    lines!(ax2, ωs, struct_nonnormal, linewidth = 3, label = "Non-normal")
    axislegend(ax2, position = :lb)

    display(fig)
end