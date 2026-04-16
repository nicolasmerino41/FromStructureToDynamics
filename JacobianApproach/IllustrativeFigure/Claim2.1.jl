# ============================================================
# claim2_peak_divergence_search_makie.jl
#
# Goal:
#   Construct a normal and a non-normal system with identical
#   eigenvalues, but with clearer divergence in dominant peak
#   location and profile shape.
#
# Strategy:
#   1. Use two closer oscillatory bands so interference matters.
#   2. Keep the normal baseline only weakly coupled.
#   3. Search over stronger non-orthogonal similarity transforms.
#   4. Score candidates primarily by dominant-peak shift.
#   5. Refine the dominant peak on a local fine grid.
#
# Makie only
# direct display only
# ============================================================
using LinearAlgebra
using CairoMakie
using Random
using Printf

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
resolvent(A, T, ω) = inv(im * ω * T - A)

intrinsic_sensitivity(A, T, ω) = opnorm(resolvent(A, T, ω), 2)

structured_sensitivity(A, T, P, ω) = opnorm(resolvent(A, T, ω) * P * resolvent(A, T, ω), 2)

function profile_intrinsic(A, T, ωs)
    y = Vector{Float64}(undef, length(ωs))
    @inbounds for k in eachindex(ωs)
        y[k] = intrinsic_sensitivity(A, T, ωs[k])
    end
    return y
end

function profile_structured(A, T, P, ωs)
    y = Vector{Float64}(undef, length(ωs))
    @inbounds for k in eachindex(ωs)
        y[k] = structured_sensitivity(A, T, P, ωs[k])
    end
    return y
end

function coupled_blocks(α1, β1, α2, β2, ε)
    return [
        -α1  -β1   ε    0.0;
         β1  -α1   0.0  ε;
         ε    0.0 -α2  -β2;
         0.0  ε    β2  -α2
    ]
end

function make_nonnormal(A, V)
    return V * A * inv(V)
end

# ------------------------------------------------------------
# Peak utilities
# ------------------------------------------------------------
function local_maxima(y::AbstractVector)
    idx = Int[]
    for i in 2:length(y)-1
        if y[i] > y[i-1] && y[i] > y[i+1]
            push!(idx, i)
        end
    end
    return idx
end

function prominent_peaks(y, ωs; relheight=0.15)
    idx = local_maxima(y)
    if isempty(idx)
        return Int[]
    end
    ymax = maximum(y)
    keep = [i for i in idx if y[i] >= relheight * ymax]
    sort!(keep, by = i -> y[i], rev = true)
    return keep
end

function dominant_peak_index(y)
    return argmax(y)
end

function dominant_peak(y, ωs)
    i = argmax(y)
    return i, ωs[i], y[i]
end

function refine_peak_intrinsic(A, T, ω0; factor=1.35, n=500)
    ωlo = max(ω0 / factor, 1e-8)
    ωhi = ω0 * factor
    ωfine = 10 .^ range(log10(ωlo), log10(ωhi), length=n)
    yfine = profile_intrinsic(A, T, ωfine)
    j = argmax(yfine)
    return j, ωfine[j], yfine[j], ωfine, yfine
end

function refine_peak_structured(A, T, P, ω0; factor=1.35, n=500)
    ωlo = max(ω0 / factor, 1e-8)
    ωhi = ω0 * factor
    ωfine = 10 .^ range(log10(ωlo), log10(ωhi), length=n)
    yfine = profile_structured(A, T, P, ωfine)
    j = argmax(yfine)
    return j, ωfine[j], yfine[j], ωfine, yfine
end

# ------------------------------------------------------------
# Similarity-transform family
# ------------------------------------------------------------
"""
Upper-triangular non-orthogonal transform with tunable strength.
The diagonal is 1, so invertibility is guaranteed.
"""
function random_upper_similarity(rng; strength=1.0)
    V = Matrix{Float64}(I, 4, 4)

    # Stronger weights farther from the diagonal to encourage mixing.
    V[1,2] = strength * rand(rng, 1.0:0.1:4.5)
    V[1,3] = strength * rand(rng, 2.0:0.1:8.5)
    V[1,4] = strength * rand(rng, 0.0:0.1:7.0)

    V[2,3] = strength * rand(rng, 1.0:0.1:4.5)
    V[2,4] = strength * rand(rng, 2.0:0.1:8.5)

    V[3,4] = strength * rand(rng, 1.0:0.1:5.0)

    return V
end

# ------------------------------------------------------------
# Candidate scoring
# ------------------------------------------------------------
"""
Primary goal: move the dominant intrinsic peak in log-frequency.

We also weakly reward:
- higher intrinsic amplification
- change in structured dominant peak location
- change in global shape

We penalize excessive conditioning to avoid pathological examples.
"""
function candidate_score(
    intr_normal, intr_nn,
    struct_normal, struct_nn,
    ωs, condV;
    cond_cap=2e4
)
    _, ωn_intr, yn_intr = dominant_peak(intr_normal, ωs)
    _, ωnn_intr, ynn_intr = dominant_peak(intr_nn, ωs)

    _, ωn_struct, yn_struct = dominant_peak(struct_normal, ωs)
    _, ωnn_struct, ynn_struct = dominant_peak(struct_nn, ωs)

    shift_intr = abs(log10(ωnn_intr) - log10(ωn_intr))
    shift_struct = abs(log10(ωnn_struct) - log10(ωn_struct))

    amp_intr = log(max(ynn_intr / yn_intr, 1e-12))
    amp_struct = log(max(ynn_struct / yn_struct, 1e-12))

    # Shape difference on normalized profiles
    nintr_normal = intr_normal ./ maximum(intr_normal)
    nintr_nn = intr_nn ./ maximum(intr_nn)
    nstruct_normal = struct_normal ./ maximum(struct_normal)
    nstruct_nn = struct_nn ./ maximum(struct_nn)

    shape_intr = norm(nintr_nn - nintr_normal) / sqrt(length(ωs))
    shape_struct = norm(nstruct_nn - nstruct_normal) / sqrt(length(ωs))

    # Hard penalty once condition number gets too large
    penalty = condV > cond_cap ? 2.5 * log(condV / cond_cap) : 0.0

    score =
        9.0 * shift_intr +
        3.0 * shift_struct +
        0.25 * amp_intr +
        0.15 * amp_struct +
        1.00 * shape_intr +
        0.60 * shape_struct -
        penalty

    return score
end

# ------------------------------------------------------------
# Frequency grid
# ------------------------------------------------------------
ωs = 10 .^ range(-3, 2, length=1800)

# ------------------------------------------------------------
# Base normal matrix
# ------------------------------------------------------------

# Closer oscillatory frequencies and weaker coupling:
# this makes interference and peak relocation easier.
A_normal = coupled_blocks(
    0.20, 1.00,
    0.22, 1.45,
    0.03
)

T = Matrix(Diagonal(ones(4)))

println("Eigenvalues of A_normal:")
println(sort(eigvals(A_normal), by = x -> imag(x)))

# ------------------------------------------------------------
# Structured perturbation operator
# ------------------------------------------------------------
Pslow = [
    0.0 1.0 0.0 0.0;
    1.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0
]

Pfast = [
    0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 1.0;
    0.0 0.0 1.0 0.0
]

Pcross = [
    0.0 0.0 1.0 0.0;
    0.0 0.0 0.0 1.0;
    1.0 0.0 0.0 0.0;
    0.0 1.0 0.0 0.0
]

Pall = 0.8 .* Pslow + 1.0 .* Pfast + 1.2 .* Pcross

# ------------------------------------------------------------
# Baseline profiles
# ------------------------------------------------------------
println("\nComputing baseline profiles...")
intr_normal = profile_intrinsic(A_normal, T, ωs)
struct_normal = profile_structured(A_normal, T, Pall, ωs)

i0_intr, ω0_intr, y0_intr = dominant_peak(intr_normal, ωs)
i0_struct, ω0_struct, y0_struct = dominant_peak(struct_normal, ωs)

println("\nBaseline dominant peaks:")
@printf("Intrinsic:  ω = %.6f, value = %.6f\n", ω0_intr, y0_intr)
@printf("Structured: ω = %.6f, value = %.6f\n", ω0_struct, y0_struct)

# ------------------------------------------------------------
# Search for a non-normal candidate with peak divergence
# ------------------------------------------------------------
rng = MersenneTwister(42)

best_score = -Inf
best_A = nothing
best_V = nothing
best_intr = nothing
best_struct = nothing
best_condV = NaN

ntrials = 700

println("\nSearching over non-normal similarity transforms...")
for trial in 1:ntrials
    # Mix moderate and strong transforms
    strength = rand(rng) < 0.65 ? 1.0 : 1.6
    V = random_upper_similarity(rng; strength=strength)
    condV = cond(V)

    # Skip extremely ill-conditioned transforms
    if condV > 8e4
        continue
    end

    A_nn = make_nonnormal(A_normal, V)

    intr_nn = profile_intrinsic(A_nn, T, ωs)
    struct_nn = profile_structured(A_nn, T, Pall, ωs)

    score = candidate_score(intr_normal, intr_nn, struct_normal, struct_nn, ωs, condV)

    if score > best_score
        best_score = score
        best_A = A_nn
        best_V = V
        best_intr = intr_nn
        best_struct = struct_nn
        best_condV = condV
    end
end

A_nonnormal = best_A

if isnothing(A_nonnormal)
    error("Search failed to find a valid non-normal candidate.")
end

println("\nChosen non-normal transform V:")
display(best_V)

println("\nCondition number of chosen V:")
@printf("%.6f\n", best_condV)

println("\nEigenvalues of A_nonnormal:")
println(sort(eigvals(A_nonnormal), by = x -> imag(x)))

# ------------------------------------------------------------
# Peak reporting: coarse + refined
# ------------------------------------------------------------
# Coarse dominant peaks
in_intr_n, ω_intr_n, y_intr_n = dominant_peak(intr_normal, ωs)
in_intr_nn, ω_intr_nn, y_intr_nn = dominant_peak(best_intr, ωs)

in_struct_n, ω_struct_n, y_struct_n = dominant_peak(struct_normal, ωs)
in_struct_nn, ω_struct_nn, y_struct_nn = dominant_peak(best_struct, ωs)

# Refined dominant peaks
_, ω_intr_n_ref, y_intr_n_ref, ωfine_intr_n, yfine_intr_n =
    refine_peak_intrinsic(A_normal, T, ω_intr_n)

_, ω_intr_nn_ref, y_intr_nn_ref, ωfine_intr_nn, yfine_intr_nn =
    refine_peak_intrinsic(A_nonnormal, T, ω_intr_nn)

_, ω_struct_n_ref, y_struct_n_ref, ωfine_struct_n, yfine_struct_n =
    refine_peak_structured(A_normal, T, Pall, ω_struct_n)

_, ω_struct_nn_ref, y_struct_nn_ref, ωfine_struct_nn, yfine_struct_nn =
    refine_peak_structured(A_nonnormal, T, Pall, ω_struct_nn)

println("\nDominant peak comparison (coarse grid):")
@printf("Intrinsic normal:     ω = %.6f, value = %.6f\n", ω_intr_n, y_intr_n)
@printf("Intrinsic non-normal: ω = %.6f, value = %.6f\n", ω_intr_nn, y_intr_nn)
@printf("Structured normal:     ω = %.6f, value = %.6f\n", ω_struct_n, y_struct_n)
@printf("Structured non-normal: ω = %.6f, value = %.6f\n", ω_struct_nn, y_struct_nn)

println("\nDominant peak comparison (refined):")
@printf("Intrinsic normal:     ω = %.6f, value = %.6f\n", ω_intr_n_ref, y_intr_n_ref)
@printf("Intrinsic non-normal: ω = %.6f, value = %.6f\n", ω_intr_nn_ref, y_intr_nn_ref)
@printf("Structured normal:     ω = %.6f, value = %.6f\n", ω_struct_n_ref, y_struct_n_ref)
@printf("Structured non-normal: ω = %.6f, value = %.6f\n", ω_struct_nn_ref, y_struct_nn_ref)

@printf("\nIntrinsic dominant-peak log10 shift = %.6f decades\n",
    abs(log10(ω_intr_nn_ref) - log10(ω_intr_n_ref)))

@printf("Structured dominant-peak log10 shift = %.6f decades\n",
    abs(log10(ω_struct_nn_ref) - log10(ω_struct_n_ref)))

# Also report prominent peaks
prom_intr_normal = prominent_peaks(intr_normal, ωs; relheight=0.18)
prom_intr_nn = prominent_peaks(best_intr, ωs; relheight=0.18)

prom_struct_normal = prominent_peaks(struct_normal, ωs; relheight=0.18)
prom_struct_nn = prominent_peaks(best_struct, ωs; relheight=0.18)

println("\nProminent intrinsic peaks (normal):")
for i in prom_intr_normal[1:min(end, 5)]
    @printf("ω = %.6f, value = %.6f\n", ωs[i], intr_normal[i])
end

println("\nProminent intrinsic peaks (non-normal):")
for i in prom_intr_nn[1:min(end, 5)]
    @printf("ω = %.6f, value = %.6f\n", ωs[i], best_intr[i])
end

println("\nProminent structured peaks (normal):")
for i in prom_struct_normal[1:min(end, 5)]
    @printf("ω = %.6f, value = %.6f\n", ωs[i], struct_normal[i])
end

println("\nProminent structured peaks (non-normal):")
for i in prom_struct_nn[1:min(end, 5)]
    @printf("ω = %.6f, value = %.6f\n", ωs[i], best_struct[i])
end

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

begin
    fig = Figure(size = (1450, 900))

    ax1 = Axis(
        fig[1, 1],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Claim 2: intrinsic sensitivity",
        xscale = log10,
        yscale = log10
    )

    lines!(ax1, ωs, intr_normal, linewidth = 3, label = "Normal")
    lines!(ax1, ωs, best_intr, linewidth = 3, label = "Non-normal")

    scatter!(ax1, [ω_intr_n], [y_intr_n], markersize = 13)
    scatter!(ax1, [ω_intr_nn], [y_intr_nn], markersize = 13)

    # refined peak markers
    scatter!(ax1, [ω_intr_n_ref], [y_intr_n_ref], markersize = 18)
    scatter!(ax1, [ω_intr_nn_ref], [y_intr_nn_ref], markersize = 18)

    axislegend(ax1, position = :rb)

    # show a small annotation with dominant peak shift
    text!(
        ax1,
        0.02, 0.08,
        text = @sprintf("Δ log10 ω* = %.3f", abs(log10(ω_intr_nn_ref) - log10(ω_intr_n_ref))),
        space = :relative,
        align = (:left, :bottom)
    )

    ax2 = Axis(
        fig[1, 2],
        xlabel = "ω",
        ylabel = "‖S(ω) P S(ω)‖₂",
        title = "Claim 2: structured sensitivity",
        xscale = log10,
        yscale = log10
    )

    lines!(ax2, ωs, struct_normal, linewidth = 3, label = "Normal")
    lines!(ax2, ωs, best_struct, linewidth = 3, label = "Non-normal")

    scatter!(ax2, [ω_struct_n], [y_struct_n], markersize = 13)
    scatter!(ax2, [ω_struct_nn], [y_struct_nn], markersize = 13)

    scatter!(ax2, [ω_struct_n_ref], [y_struct_n_ref], markersize = 18)
    scatter!(ax2, [ω_struct_nn_ref], [y_struct_nn_ref], markersize = 18)

    axislegend(ax2, position = :rb)

    text!(
        ax2,
        0.02, 0.08,
        text = @sprintf("Δ log10 ω* = %.3f", abs(log10(ω_struct_nn_ref) - log10(ω_struct_n_ref))),
        space = :relative,
        align = (:left, :bottom)
    )

    # Zoomed intrinsic panel around the dominant peaks
    ax3 = Axis(
        fig[2, 1],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Intrinsic sensitivity (zoom around dominant peaks)",
        xscale = log10,
        yscale = log10
    )

    ωmin_zoom_intr = min(ω_intr_n_ref, ω_intr_nn_ref) / 1.9
    ωmax_zoom_intr = max(ω_intr_n_ref, ω_intr_nn_ref) * 1.9

    mask_intr = (ωs .>= ωmin_zoom_intr) .& (ωs .<= ωmax_zoom_intr)

    lines!(ax3, ωs[mask_intr], intr_normal[mask_intr], linewidth = 3, label = "Normal")
    lines!(ax3, ωs[mask_intr], best_intr[mask_intr], linewidth = 3, label = "Non-normal")
    scatter!(ax3, [ω_intr_n_ref], [y_intr_n_ref], markersize = 18)
    scatter!(ax3, [ω_intr_nn_ref], [y_intr_nn_ref], markersize = 18)
    axislegend(ax3, position = :rb)

    # Zoomed structured panel around the dominant peaks
    ax4 = Axis(
        fig[2, 2],
        xlabel = "ω",
        ylabel = "‖S(ω) P S(ω)‖₂",
        title = "Structured sensitivity (zoom around dominant peaks)",
        xscale = log10,
        yscale = log10
    )

    ωmin_zoom_struct = min(ω_struct_n_ref, ω_struct_nn_ref) / 1.9
    ωmax_zoom_struct = max(ω_struct_n_ref, ω_struct_nn_ref) * 1.9

    mask_struct = (ωs .>= ωmin_zoom_struct) .& (ωs .<= ωmax_zoom_struct)

    lines!(ax4, ωs[mask_struct], struct_normal[mask_struct], linewidth = 3, label = "Normal")
    lines!(ax4, ωs[mask_struct], best_struct[mask_struct], linewidth = 3, label = "Non-normal")
    scatter!(ax4, [ω_struct_n_ref], [y_struct_n_ref], markersize = 18)
    scatter!(ax4, [ω_struct_nn_ref], [y_struct_nn_ref], markersize = 18)
    axislegend(ax4, position = :rb)

    display(fig)
end