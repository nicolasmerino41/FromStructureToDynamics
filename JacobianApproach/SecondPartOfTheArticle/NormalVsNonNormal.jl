using LinearAlgebra
using Statistics
using CairoMakie
using Printf

# ============================================================
# Non-normality proof of concept with matched eigenvalues
# ------------------------------------------------------------
# We compare:
#   1) a normal oscillatory matrix
#   2) a non-normal oscillatory matrix
#
# Both have exactly the same eigenvalues:
#   repeated blocks with eigenvalues -a ± i b
#
# The non-normal matrix is obtained by adding strictly
# block-upper-triangular couplings, which preserve eigenvalues
# but break normality.
#
# This makes the comparison much sharper than chain/feedforward:
# any difference in S(ω) comes from geometry, not from different
# oscillatory scales or different spectral abscissa.
#
# Homogeneous timescales:
#   T = I
#
# Resolvent:
#   R(ω) = (iωI - A)^(-1)
#
# Intrinsic sensitivity:
#   S(ω) = ||R(ω)||_2
# ============================================================

# -------------------------
# User parameters
# -------------------------
const NBLOCKS = 5                  # total dimension = 2*NBLOCKS
const N = 2 * NBLOCKS
const A_DAMP = 0.5                 # real part of eigenvalues = -A_DAMP
const B_OSC  = 1.0                 # imaginary part magnitude
const COUPLING = 0.8               # strength of non-normal couplings
const OMEGAS = exp.(range(log(1e-2), log(1e2), length = 400))

# -------------------------
# Helpers
# -------------------------
function spectral_abscissa(A::AbstractMatrix{<:Real})
    maximum(real.(eigvals(Matrix(A))))
end

function is_normal(A::AbstractMatrix{<:Real}; atol::Float64 = 1e-10)
    norm(A * A' - A' * A) < atol
end

function intrinsic_spectrum(A::AbstractMatrix{<:Real}, ωs::AbstractVector{<:Real})
    n = size(A, 1)
    Icomplex = Matrix{ComplexF64}(I, n, n)
    Ac = ComplexF64.(A)
    out = zeros(Float64, length(ωs))

    for (k, ω) in pairs(ωs)
        F = factorize(im * ω .* Icomplex - Ac)
        R = F \ Icomplex
        out[k] = opnorm(R, 2)
    end
    return out
end

function block_rotation(a::Float64, b::Float64)
    # eigenvalues = -a ± i b
    [ -a  -b
       b  -a ]
end

# -------------------------
# Build normal baseline
# ------------------------------------------------------------
# Block-diagonal matrix of identical damped-rotation blocks.
# This matrix is normal.
# -------------------------
function build_normal_oscillatory(nblocks::Int, a::Float64, b::Float64)
    A = zeros(Float64, 2nblocks, 2nblocks)
    B = block_rotation(a, b)

    for k in 1:nblocks
        idx = 2k-1:2k
        A[idx, idx] .= B
    end
    return A
end

# -------------------------
# Build non-normal matrix with same eigenvalues
# ------------------------------------------------------------
# Start from the same block diagonal, then add couplings above
# the block diagonal only. Because the resulting matrix is block
# upper triangular, its eigenvalues are unchanged.
#
# This preserves oscillatory frequencies and spectral abscissa,
# while making the matrix non-normal.
# -------------------------
function build_nonnormal_same_eigs(nblocks::Int, a::Float64, b::Float64, γ::Float64)
    A = build_normal_oscillatory(nblocks, a, b)

    # Add strictly block-upper-triangular couplings
    # Here we connect each block to the next one.
    C = γ .* [1.0  0.0;
              0.0  1.0]

    for k in 1:nblocks-1
        src = 2k-1:2k
        dst = 2(k+1)-1:2(k+1)
        A[src, dst] .= C
    end

    return A
end

# -------------------------
# Build matrices
# -------------------------
A_normal = build_normal_oscillatory(NBLOCKS, A_DAMP, B_OSC)
A_nonnormal = build_nonnormal_same_eigs(NBLOCKS, A_DAMP, B_OSC, COUPLING)

# Diagnostics
eig_normal = eigvals(A_normal)
eig_nonnormal = eigvals(A_nonnormal)

α_normal = spectral_abscissa(A_normal)
α_nonnormal = spectral_abscissa(A_nonnormal)

normal_flag = is_normal(A_normal)
nonnormal_flag = is_normal(A_nonnormal)

S_normal = intrinsic_spectrum(A_normal, OMEGAS)
S_nonnormal = intrinsic_spectrum(A_nonnormal, OMEGAS)

ω_peak_normal = OMEGAS[argmax(S_normal)]
ω_peak_nonnormal = OMEGAS[argmax(S_nonnormal)]

# -------------------------
# Plot
# -------------------------
begin
    fig = Figure(size = (1600, 950))

    Label(fig[0, :], "Non-normality proof of concept with matched eigenvalues", fontsize = 24)

    # Matrix heatmaps
    axA1 = Axis(fig[1, 1],
        title = "Normal oscillatory matrix",
        xlabel = "Species j", ylabel = "Species i",
        aspect = DataAspect(), titlesize = 20)
    heatmap!(axA1, A_normal)

    axA2 = Axis(fig[1, 2],
        title = "Non-normal matrix (same eigenvalues)",
        xlabel = "Species j", ylabel = "Species i",
        aspect = DataAspect(), titlesize = 20)
    heatmap!(axA2, A_nonnormal)

    # Intrinsic sensitivity spectra
    ax1 = Axis(
        fig[2, 1:2],
        xlabel = "Frequency ω",
        ylabel = "S(ω)",
        xscale = log10,
        yscale = log10,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 17,
        ylabelsize = 17
    )

    lines!(ax1, OMEGAS, S_normal, linewidth = 3, label = "normal")
    lines!(ax1, OMEGAS, S_nonnormal, linewidth = 3, label = "non-normal")

    vlines!(ax1, [ω_peak_normal], linestyle = :dot)
    vlines!(ax1, [ω_peak_nonnormal], linestyle = :dot)

    axislegend(ax1, position = :rt, labelsize = 18)
    ax1.xgridvisible = false
    ax1.ygridvisible = false

    # Eigenvalue plot
    ax2 = Axis(
        fig[1, 3],
        title = "Eigenvalues",
        xlabel = "Re(λ)", ylabel = "Im(λ)",
        titlesize = 20
    )

    scatter!(ax2, real.(eig_normal), imag.(eig_normal), markersize = 12, label = "normal")
    scatter!(ax2, real.(eig_nonnormal), imag.(eig_nonnormal), markersize = 8, label = "non-normal")
    axislegend(ax2, position = :rb)

    # Diagnostics text
    txt = """
    Construction:
    - normal baseline: block-diagonal damped rotations
    - non-normal comparison: same blocks + strictly block-upper couplings

    Shared eigenvalues:
    λ = -a ± i b
    with a = $(A_DAMP), b = $(B_OSC)

    Diagnostics:
    spectral abscissa (normal)     = $(round(α_normal, digits=6))
    spectral abscissa (non-normal) = $(round(α_nonnormal, digits=6))

    is normal?      baseline   = $(normal_flag)
    is normal?      comparison = $(nonnormal_flag)

    Peak frequencies:
    ω_peak normal     = $(round(ω_peak_normal, digits=4))
    ω_peak non-normal = $(round(ω_peak_nonnormal, digits=4))

    Interpretation:
    - both matrices have the same eigenvalues,
    so they share the same oscillatory scales and asymptotic stability
    - any difference in S(ω) is therefore due to geometry / non-normality
    """
    Label(fig[2, 3], txt, tellwidth = false, fontsize = 16, justification = :left)

    display(fig)
end