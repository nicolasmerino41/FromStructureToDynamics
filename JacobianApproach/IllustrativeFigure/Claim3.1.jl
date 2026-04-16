# ============================================================
# claim3_profile_plus_heatmaps_makie.jl
#
# Claim 3:
#   same A, different T
#   top: intrinsic sensitivity profiles
#   bottom: raw structured sensitivity heatmaps by perturbation class
#
# Uses 4 species only
# Makie only
# Plotting code inside begin ... end, ending with display(fig)
# ============================================================

using LinearAlgebra
using CairoMakie
using Printf

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

function coupled_blocks(α1, β1, α2, β2, ε)
    [
        -α1  -β1   ε    0.0;
         β1  -α1   0.0  ε;
         ε    0.0 -α2  -β2;
         0.0  ε    β2  -α2
    ]
end

# local maxima indices
function local_maxima(y)
    idx = Int[]
    for i in 2:length(y)-1
        if y[i] > y[i-1] && y[i] > y[i+1]
            push!(idx, i)
        end
    end
    return idx
end

# choose up to n most prominent peaks, while avoiding near-duplicates in log-frequency
function separated_top_peaks(y, ωs; n=2, min_logsep=0.22)
    idx = local_maxima(y)
    isempty(idx) && return Int[]

    idx_sorted = sort(idx, by = i -> y[i], rev = true)
    chosen = Int[]

    for i in idx_sorted
        if all(abs(log10(ωs[i]) - log10(ωs[j])) > min_logsep for j in chosen)
            push!(chosen, i)
        end
        length(chosen) >= n && break
    end

    return sort(chosen)
end

# build 3 x nω matrix for heatmap
function structured_matrix(A, T, Pslow, Pfast, Pcross, ωs)
    sslow  = profile_structured(A, T, Pslow,  ωs)
    sfast  = profile_structured(A, T, Pfast,  ωs)
    scross = profile_structured(A, T, Pcross, ωs)
    M = vcat(sslow', sfast', scross')
    return M
end

# ------------------------------------------------------------
# Base matrix and perturbation classes
# ------------------------------------------------------------

A = coupled_blocks(
    0.24, 0.85,   # slow block
    0.20, 2.10,   # fast block
    0.14          # coupling
)

println("Eigenvalues of A:")
println(sort(eigvals(A), by = x -> imag(x)))

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

# ------------------------------------------------------------
# Frequency grid
# ------------------------------------------------------------

ωs = 10 .^ range(-3, 2, length=1000)
logωs = log10.(ωs)

# ------------------------------------------------------------
# Timescale assignments
# ------------------------------------------------------------

T_hom = Diagonal([1.0, 1.0, 1.0, 1.0])
T_aligned = Diagonal([2.2, 1.9, 0.65, 0.55])
T_misaligned = Diagonal([0.65, 0.55, 2.2, 1.9])

Ts = Dict(
    "homogeneous" => T_hom,
    "aligned" => T_aligned,
    "misaligned" => T_misaligned,
)

# ------------------------------------------------------------
# Compute profiles and heatmap matrices
# ------------------------------------------------------------

intr = Dict{String, Vector{Float64}}()
HM = Dict{String, Matrix{Float64}}()

for name in ["homogeneous", "aligned", "misaligned"]
    T = Ts[name]
    intr[name] = profile_intrinsic(A, T, ωs)
    HM[name] = structured_matrix(A, T, Pslow, Pfast, Pcross, ωs)
end

# Choose vertical guide lines from the homogeneous intrinsic profile
peak_idx = separated_top_peaks(intr["homogeneous"], ωs, n=2, min_logsep=0.20)
ω_guides = ωs[peak_idx]

println("\nGuide frequencies from homogeneous intrinsic profile:")
for ω in ω_guides
    @printf("ω = %.4g\n", ω)
end

println("\nPeak intrinsic sensitivity")
for name in ["homogeneous", "aligned", "misaligned"]
    idx = argmax(intr[name])
    @printf("%-12s : max = %.4f at ω = %.4g\n", name, intr[name][idx], ωs[idx])
end

# global color scale across all heatmaps, in log10 space
allvals = vcat(vec(HM["homogeneous"]), vec(HM["aligned"]), vec(HM["misaligned"]))
logall = log10.(allvals .+ 1e-12)
cmin, cmax = minimum(logall), maximum(logall)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

begin
    fig = Figure(size = (1500, 820))

    # ---------------- Top profile ----------------

    ax_top = Axis(
        fig[1, 1:3],
        xlabel = "ω",
        ylabel = "‖S(ω)‖₂",
        title = "Claim 3: timescale alignment reshapes intrinsic sensitivity",
        xscale = log10,
        yscale = log10
    )

    lines!(ax_top, ωs, intr["homogeneous"], linewidth = 3, label = "homogeneous")
    lines!(ax_top, ωs, intr["aligned"], linewidth = 3, label = "aligned")
    lines!(ax_top, ωs, intr["misaligned"], linewidth = 3, label = "misaligned")

    for ω in ω_guides
        vlines!(ax_top, [ω], linestyle = :dash, linewidth = 2)
    end

    axislegend(ax_top, position = :rb)

    # ---------------- Bottom heatmaps ----------------
    # x-axis is log10(ω) so spacing is visually honest on the log scale

    x = logωs
    y = 1:3

    function heatmap_panel!(ax, M, ttl, ω_guides, logωs, cmin, cmax)
        logM = log10.(M .+ 1e-12)

        hm = heatmap!(
            ax,
            x, y, logM;
            colorrange = (cmin, cmax)
        )

        for ω in ω_guides
            vlines!(ax, [log10(ω)], linestyle = :dash, linewidth = 2)
        end

        ax.title = ttl
        ax.xlabel = "log₁₀(ω)"
        ax.ylabel = "perturbation class"
        ax.yticks = (1:3, ["slow", "fast", "cross"])
        return hm
    end

    ax1 = Axis(fig[2, 1])
    hm1 = heatmap_panel!(ax1, HM["homogeneous"], "homogeneous T", ω_guides, logωs, cmin, cmax)

    ax2 = Axis(fig[2, 2])
    hm2 = heatmap_panel!(ax2, HM["aligned"], "aligned T", ω_guides, logωs, cmin, cmax)

    ax3 = Axis(fig[2, 3])
    hm3 = heatmap_panel!(ax3, HM["misaligned"], "misaligned T", ω_guides, logωs, cmin, cmax)

    Colorbar(
        fig[2, 4],
        hm1,
        label = "log₁₀ structured sensitivity"
    )

    display(fig)
end