# ============================================================
# combined_claims_rich_dynamics_makie.jl
#
# One self-contained script for Claims 2, 3, and 4 using:
# - shared helpers
# - Makie only
# - richer 4x4 coupled oscillatory blocks
# - direct display only
#
# Claims:
#   2. Same eigenvalues, different sensitivity spectra
#   3. Same A, different T
#   4. Same A and same multiset of timescales, different alignments
# ============================================================
using LinearAlgebra
using Printf
using CairoMakie

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
resolvent(A, T, ω) = inv(im * ω * T - A)

intrinsic_sensitivity(A, T, ω) = opnorm(resolvent(A, T, ω), 2)

structured_sensitivity(A, T, P, ω) = opnorm(resolvent(A, T, ω) * P * resolvent(A, T, ω), 2)

function profile_intrinsic(A, T, ωs)
    [intrinsic_sensitivity(A, T, ω) for ω in ωs]
end

function profile_structured(A, T, P, ωs)
    [structured_sensitivity(A, T, P, ω) for ω in ωs]
end

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

# ------------------------------------------------------------
# Matrix builders
# ------------------------------------------------------------

"""
Two weakly coupled damped oscillatory blocks.
Each 2x2 block has eigenvalues -α ± iβ.
"""
function coupled_blocks(α1, β1, α2, β2, ε)
    [
        -α1  -β1   ε    0.0;
         β1  -α1   0.0  ε;
         ε    0.0 -α2  -β2;
         0.0  ε    β2  -α2
    ]
end

"""
A non-orthogonal similarity transform to preserve eigenvalues
while changing geometry / non-normality.
"""
function make_nonnormal(A; s12=1.6, s13=0.7, s24=1.2, s34=1.4)
    V = [
        1.0  s12  s13  0.0;
        0.0  1.0  0.0  s24;
        0.0  0.0  1.0  s34;
        0.0  0.0  0.0  1.0
    ]
    return V * A * inv(V), V
end

# ------------------------------------------------------------
# Perturbation classes
# ------------------------------------------------------------

# within slow block
Pslow = [
    0.0 1.0 0.0 0.0;
    1.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0
]

# within fast block
Pfast = [
    0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 1.0;
    0.0 0.0 1.0 0.0
]

# cross-block couplings
Pcross = [
    0.0 0.0 1.0 0.0;
    0.0 0.0 0.0 1.0;
    1.0 0.0 0.0 0.0;
    0.0 1.0 0.0 0.0
]

# aggregate structured class
Pall = Pslow + Pfast + 0.7 .* Pcross

# ------------------------------------------------------------
# Frequency grid
# ------------------------------------------------------------
ωs = 10 .^ range(-3, 2, length=900)
logω = log10.(ωs)

# ------------------------------------------------------------
# Shared base system
# ------------------------------------------------------------
# Choose sufficiently separated oscillatory frequencies and weak damping
A_base = coupled_blocks(
    0.18, 0.75,   # slow block
    0.12, 3.20,   # fast block
    0.22          # weak coupling between blocks
)

println("============================================================")
println("Shared base system")
println("============================================================")
println("Eigenvalues of A_base:")
println(sort(eigvals(A_base), by = x -> imag(x)))

# ============================================================
# CLAIM 2
# Same eigenvalues, different sensitivity spectra
# ============================================================
println("\n============================================================")
println("Claim 2: same eigenvalues, different sensitivity spectra")
println("============================================================")

A_normal = A_base
A_nonnormal, V = make_nonnormal(A_base; s12=2.2, s13=0.9, s24=1.7, s34=1.6)

println("Eigenvalues of A_normal:")
println(sort(eigvals(A_normal), by = x -> imag(x)))

println("\nEigenvalues of A_nonnormal:")
println(sort(eigvals(A_nonnormal), by = x -> imag(x)))

println("\nCondition number of similarity transform V:")
@printf("%.4f\n", cond(V))

T_hom4 = Diagonal(ones(4))

claim2_intr_normal = profile_intrinsic(A_normal, T_hom4, ωs)
claim2_intr_nonnormal = profile_intrinsic(A_nonnormal, T_hom4, ωs)

claim2_struct_normal = profile_structured(A_normal, T_hom4, Pall, ωs)
claim2_struct_nonnormal = profile_structured(A_nonnormal, T_hom4, Pall, ωs)

i_n = argmax(claim2_intr_normal)
i_nn = argmax(claim2_intr_nonnormal)

println("\nPeak intrinsic sensitivity")
@printf("Normal      : max = %.4f at ω = %.4g\n", claim2_intr_normal[i_n], ωs[i_n])
@printf("Non-normal  : max = %.4f at ω = %.4g\n", claim2_intr_nonnormal[i_nn], ωs[i_nn])

j_n = argmax(claim2_struct_normal)
j_nn = argmax(claim2_struct_nonnormal)

println("\nPeak structured sensitivity")
@printf("Normal      : max = %.4f at ω = %.4g\n", claim2_struct_normal[j_n], ωs[j_n])
@printf("Non-normal  : max = %.4f at ω = %.4g\n", claim2_struct_nonnormal[j_nn], ωs[j_nn])

begin
    fig = Figure(size = (1200, 460))

    ax1 = Axis(
        fig[1, 1],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Claim 2: intrinsic sensitivity",
        xscale = log10,
        yscale = log10
    )

    lines!(ax1, ωs, claim2_intr_normal, linewidth = 3, label = "Normal")
    lines!(ax1, ωs, claim2_intr_nonnormal, linewidth = 3, label = "Non-normal")
    axislegend(ax1, position = :rb)

    ax2 = Axis(
        fig[1, 2],
        xlabel = "ω",
        ylabel = "‖S(ω) P S(ω)‖₂",
        title = "Claim 2: structured sensitivity",
        xscale = log10,
        yscale = log10
    )

    lines!(ax2, ωs, claim2_struct_normal, linewidth = 3, label = "Normal")
    lines!(ax2, ωs, claim2_struct_nonnormal, linewidth = 3, label = "Non-normal")
    axislegend(ax2, position = :rb)

    display(fig)
end

# ============================================================
# CLAIM 3
# Same A, different T
# ============================================================
println("\n============================================================")
println("Claim 3: same A, different T")
println("============================================================")

A_claim3 = A_base

# Interpretation:
# species 1-2 belong to the slower oscillatory block
# species 3-4 belong to the faster oscillatory block

T_hom = Diagonal([1.0, 1.0, 1.0, 1.0])

# slow block made slower, fast block made faster
T_aligned = Diagonal([2.3, 2.0, 0.65, 0.55])

# slow block made faster, fast block made slower
T_misaligned = Diagonal([0.65, 0.55, 2.3, 2.0])

# alternating arrangement
T_alternating = Diagonal([2.3, 0.6, 2.0, 0.55])

Ts_claim3 = Dict(
    "homogeneous" => T_hom,
    "aligned" => T_aligned,
    "misaligned" => T_misaligned,
    "alternating" => T_alternating,
)

claim3_intr = Dict{String, Vector{Float64}}()
claim3_slow = Dict{String, Vector{Float64}}()
claim3_fast = Dict{String, Vector{Float64}}()
claim3_cross = Dict{String, Vector{Float64}}()

for (name, T) in Ts_claim3
    claim3_intr[name] = profile_intrinsic(A_claim3, T, ωs)
    claim3_slow[name] = profile_structured(A_claim3, T, Pslow, ωs)
    claim3_fast[name] = profile_structured(A_claim3, T, Pfast, ωs)
    claim3_cross[name] = profile_structured(A_claim3, T, Pcross, ωs)
end

println("Peak intrinsic sensitivity")
for name in ["homogeneous", "aligned", "misaligned", "alternating"]
    idx = argmax(claim3_intr[name])
    @printf("%-12s : max = %.4f at ω = %.4g\n", name, claim3_intr[name][idx], ωs[idx])
end

begin
    fig = Figure(size = (1450, 900))

    ax1 = Axis(
        fig[1, 1],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Claim 3: intrinsic sensitivity for fixed A, different T",
        xscale = log10,
        yscale = log10
    )

    for name in ["homogeneous", "aligned", "misaligned", "alternating"]
        lines!(ax1, ωs, claim3_intr[name], linewidth = 3, label = name)
    end
    axislegend(ax1, position = :lb)

    ax2 = Axis(
        fig[1, 2],
        xlabel = "ω",
        ylabel = "‖S(ω) Pslow S(ω)‖₂",
        title = "Claim 3: perturbations within slow block",
        xscale = log10,
        yscale = log10
    )

    for name in ["homogeneous", "aligned", "misaligned", "alternating"]
        lines!(ax2, ωs, claim3_slow[name], linewidth = 3, label = name)
    end
    axislegend(ax2, position = :lb)

    ax3 = Axis(
        fig[2, 1],
        xlabel = "ω",
        ylabel = "‖S(ω) Pfast S(ω)‖₂",
        title = "Claim 3: perturbations within fast block",
        xscale = log10,
        yscale = log10
    )

    for name in ["homogeneous", "aligned", "misaligned", "alternating"]
        lines!(ax3, ωs, claim3_fast[name], linewidth = 3, label = name)
    end
    axislegend(ax3, position = :lb)

    ax4 = Axis(
        fig[2, 2],
        xlabel = "ω",
        ylabel = "‖S(ω) Pcross S(ω)‖₂",
        title = "Claim 3: perturbations across blocks",
        xscale = log10,
        yscale = log10
    )

    for name in ["homogeneous", "aligned", "misaligned", "alternating"]
        lines!(ax4, ωs, claim3_cross[name], linewidth = 3, label = name)
    end
    axislegend(ax4, position = :lb)

    display(fig)
end

# ============================================================
# CLAIM 4
# Same A and same multiset of timescales, different alignments
# ============================================================
println("\n============================================================")
println("Claim 4: best/worst alignment of a fixed timescale multiset")
println("============================================================")

A_claim4 = A_base

times = [0.55, 0.80, 1.80, 2.30]
perms = all_permutations(times)

println("Number of permutations searched: $(length(perms))")

T_hom_baseline = Diagonal(fill(mean(times), length(times)))

# Objective:
# integrated intrinsic sensitivity over log-frequency
function objective_intrinsic(A, T, ωs, logω)
    vals = profile_intrinsic(A, T, ωs)
    return trapz(logω, vals), vals
end

best_val = Inf
worst_val = -Inf
best_perm = nothing
worst_perm = nothing
best_profile = nothing
worst_profile = nothing

for perm in perms
    T = Diagonal(perm)
    val, prof = objective_intrinsic(A_claim4, T, ωs, logω)

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

hom_val, hom_profile = objective_intrinsic(A_claim4, T_hom_baseline, ωs, logω)

best_T = Diagonal(best_perm)
worst_T = Diagonal(worst_perm)

claim4_slow_hom = profile_structured(A_claim4, T_hom_baseline, Pslow, ωs)
claim4_slow_best = profile_structured(A_claim4, best_T, Pslow, ωs)
claim4_slow_worst = profile_structured(A_claim4, worst_T, Pslow, ωs)

claim4_fast_hom = profile_structured(A_claim4, T_hom_baseline, Pfast, ωs)
claim4_fast_best = profile_structured(A_claim4, best_T, Pfast, ωs)
claim4_fast_worst = profile_structured(A_claim4, worst_T, Pfast, ωs)

claim4_cross_hom = profile_structured(A_claim4, T_hom_baseline, Pcross, ωs)
claim4_cross_best = profile_structured(A_claim4, best_T, Pcross, ωs)
claim4_cross_worst = profile_structured(A_claim4, worst_T, Pcross, ωs)

println("Integrated intrinsic sensitivity over log-frequency")
@printf("Homogeneous baseline : %.6f\n", hom_val)
@printf("Best alignment       : %.6f\n", best_val)
@printf("Worst alignment      : %.6f\n", worst_val)

println("\nBest timescale assignment:")
println(best_perm)

println("\nWorst timescale assignment:")
println(worst_perm)

ih = argmax(hom_profile)
ib = argmax(best_profile)
iw = argmax(worst_profile)

println("\nPeak intrinsic sensitivity")
@printf("Homogeneous baseline : max = %.4f at ω = %.4g\n", hom_profile[ih], ωs[ih])
@printf("Best alignment       : max = %.4f at ω = %.4g\n", best_profile[ib], ωs[ib])
@printf("Worst alignment      : max = %.4f at ω = %.4g\n", worst_profile[iw], ωs[iw])

begin
    fig = Figure(size = (1450, 900))

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
        ylabel = "‖S(ω) Pslow S(ω)‖₂",
        title = "Claim 4: within-slow perturbations",
        xscale = log10,
        yscale = log10
    )

    lines!(ax2, ωs, claim4_slow_hom, linewidth = 3, label = "homogeneous baseline")
    lines!(ax2, ωs, claim4_slow_best, linewidth = 3, label = "best alignment")
    lines!(ax2, ωs, claim4_slow_worst, linewidth = 3, label = "worst alignment")
    axislegend(ax2, position = :lb)

    ax3 = Axis(
        fig[2, 1],
        xlabel = "ω",
        ylabel = "‖S(ω) Pfast S(ω)‖₂",
        title = "Claim 4: within-fast perturbations",
        xscale = log10,
        yscale = log10
    )

    lines!(ax3, ωs, claim4_fast_hom, linewidth = 3, label = "homogeneous baseline")
    lines!(ax3, ωs, claim4_fast_best, linewidth = 3, label = "best alignment")
    lines!(ax3, ωs, claim4_fast_worst, linewidth = 3, label = "worst alignment")
    axislegend(ax3, position = :lb)

    ax4 = Axis(
        fig[2, 2],
        xlabel = "ω",
        ylabel = "‖S(ω) Pcross S(ω)‖₂",
        title = "Claim 4: cross-block perturbations",
        xscale = log10,
        yscale = log10
    )

    lines!(ax4, ωs, claim4_cross_hom, linewidth = 3, label = "homogeneous baseline")
    lines!(ax4, ωs, claim4_cross_best, linewidth = 3, label = "best alignment")
    lines!(ax4, ωs, claim4_cross_worst, linewidth = 3, label = "worst alignment")
    axislegend(ax4, position = :lb)

    display(fig)
end