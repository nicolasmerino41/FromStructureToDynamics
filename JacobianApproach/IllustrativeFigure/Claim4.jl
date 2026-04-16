# ============================================================
# claim4_best_worst_alignment_of_T_makie.jl
# ============================================================
using LinearAlgebra
using Printf
using CairoMakie
using StatsBase
# ---------- Core definitions ----------
resolvent(A, T, ω) = inv(im * ω * T - A)

intrinsic_sensitivity(A, T, ω) = opnorm(resolvent(A, T, ω), 2)

structured_sensitivity(A, T, P, ω) = opnorm(resolvent(A, T, ω) * P * resolvent(A, T, ω), 2)

function trapz(x, y)
    s = 0.0
    for i in 1:length(x)-1
        s += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return s
end

function all_permutations(v::Vector{T}) where T
    if length(v) == 1
        return [copy(v)]
    end
    perms = Vector{Vector{T}}()
    for i in eachindex(v)
        head = v[i]
        tail = [v[j] for j in eachindex(v) if j != i]
        for p in all_permutations(tail)
            push!(perms, [head; p])
        end
    end
    return perms
end

# ---------- Frequency grid ----------
ωs = 10 .^ range(-3, 1.5, length=700)
logω = log10.(ωs)

# ---------- Fixed interaction matrix ----------
A = [-1.0  1.45  0.0   0.0   0.0;
      0.0 -1.0   1.30  0.0   0.0;
      0.0  0.0  -1.0   1.15  0.0;
      0.0  0.0   0.0  -1.0   1.00;
      0.0  0.0   0.0   0.0  -1.0]

println("Eigenvalues of A:")
println(sort(eigvals(A), by=x -> real(x)))

P = [0.0 1.0 0.0 0.0 0.0;
     0.0 0.0 1.0 0.0 0.0;
     0.0 0.0 0.0 1.0 0.0;
     0.0 0.0 0.0 0.0 1.0;
     0.0 0.0 0.0 0.0 0.0]

# ---------- Timescale multiset ----------
times = [0.45, 0.75, 1.10, 1.70, 2.60]
perms = all_permutations(times)

println("\nNumber of permutations searched: $(length(perms))")

T_hom = Diagonal(fill(StatsBase.mean(times), length(times)))

# ---------- Objectives ----------
function objective_intrinsic(A, T, ωs, logω)
    vals = [intrinsic_sensitivity(A, T, ω) for ω in ωs]
    return trapz(logω, vals), vals
end

function objective_structured(A, T, P, ωs, logω)
    vals = [structured_sensitivity(A, T, P, ω) for ω in ωs]
    return trapz(logω, vals), vals
end

# ---------- Search best/worst alignments ----------
best_val = Inf
worst_val = -Inf

best_perm = nothing
worst_perm = nothing

best_profile = nothing
worst_profile = nothing

for perm in perms
    T = Diagonal(perm)
    val, prof = objective_intrinsic(A, T, ωs, logω)

    if val < best_val
        best_val = val
        best_perm = copy(perm)
        best_profile = copy(prof)
    end

    if val > worst_val
        worst_val = val
        worst_perm = copy(perm)
        worst_profile = copy(prof)
    end
end

hom_val, hom_profile = objective_intrinsic(A, T_hom, ωs, logω)

best_struct_val, best_struct_profile = objective_structured(A, Diagonal(best_perm), P, ωs, logω)
worst_struct_val, worst_struct_profile = objective_structured(A, Diagonal(worst_perm), P, ωs, logω)
hom_struct_val, hom_struct_profile = objective_structured(A, T_hom, P, ωs, logω)

# ---------- Report ----------
println("\nIntegrated intrinsic sensitivity over log-frequency")
@printf("Homogeneous baseline : %.6f\n", hom_val)
@printf("Best alignment       : %.6f\n", best_val)
@printf("Worst alignment      : %.6f\n", worst_val)

println("\nBest timescale assignment:")
println(best_perm)

println("\nWorst timescale assignment:")
println(worst_perm)

println("\nIntegrated structured sensitivity over log-frequency")
@printf("Homogeneous baseline : %.6f\n", hom_struct_val)
@printf("Best alignment       : %.6f\n", best_struct_val)
@printf("Worst alignment      : %.6f\n", worst_struct_val)

ih = argmax(hom_profile)
ib = argmax(best_profile)
iw = argmax(worst_profile)

println("\nPeak intrinsic sensitivity")
@printf("Homogeneous baseline : max = %.4f at ω = %.4g\n", hom_profile[ih], ωs[ih])
@printf("Best alignment       : max = %.4f at ω = %.4g\n", best_profile[ib], ωs[ib])
@printf("Worst alignment      : max = %.4f at ω = %.4g\n", worst_profile[iw], ωs[iw])

# ---------- Plot ----------
begin
    fig = Figure(size = (1150, 450))

    ax1 = Axis(
        fig[1, 1],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Claim 4: best/worst alignment of T with A",
        xscale = log10,
        yscale = log10
    )

    lines!(ax1, ωs, hom_profile, linewidth = 3, label = "homogeneous baseline")
    lines!(ax1, ωs, best_profile, linewidth = 3, label = "best alignment")
    lines!(ax1, ωs, worst_profile, linewidth = 3, label = "worst alignment")
    axislegend(ax1, position = :lb)

    ax2 = Axis(
        fig[1, 2],
        xlabel = "ω",
        ylabel = "‖S(ω) P S(ω)‖₂",
        title = "Claim 4: structured sensitivity under best/worst alignment",
        xscale = log10,
        yscale = log10
    )

    lines!(ax2, ωs, hom_struct_profile, linewidth = 3, label = "homogeneous baseline")
    lines!(ax2, ωs, best_struct_profile, linewidth = 3, label = "best alignment")
    lines!(ax2, ωs, worst_struct_profile, linewidth = 3, label = "worst alignment")
    axislegend(ax2, position = :lb)

    display(fig)
end