using LinearAlgebra
using Statistics
using CairoMakie
using Printf

# ============================================================
# Normal vs non-normal comparison with the SAME eigenvalues
# ------------------------------------------------------------
# Goal:
#   Show that in a normal system:
#     - imaginary parts locate the characteristic frequencies
#     - real parts set the amplification scale
#
#   while in a non-normal system with the SAME eigenvalues:
#     - peak height is no longer predicted by eigenvalues alone
#     - peak location can also shift away from the naive reading
#
# Construction:
#   1) Normal baseline: block-diagonal damped rotation blocks
#   2) Non-normal comparison: same diagonal blocks + strictly
#      upper block couplings, which preserve eigenvalues
#
# Homogeneous timescales:
#   T = I
#
# Resolvent:
#   R(ω) = (im*ω*I - A)^(-1)
#
# Intrinsic sensitivity:
#   S(ω) = ||R(ω)||_2
# ============================================================

# -------------------------
# User parameters
# -------------------------
const A1 = 0.40          # damping of block 1
const B1 = 0.80          # oscillation frequency of block 1
const A2 = 0.45          # damping of block 2
const B2 = 1.25          # oscillation frequency of block 2
const REPEAT_BLOCKS = 3  # total dimension = 2 * (2*REPEAT_BLOCKS)

const OMEGAS = exp.(range(log(1e-2), log(1e2), length = 1200))

# Coupling strengths for the non-normal system
# Strong cross-family coupling, weaker within-family coupling
const Γ12 = 2.0
const Γ11 = 0.35
const Γ22 = 0.35

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
    Ac = ComplexF64.(A)
    Icomplex = Matrix{ComplexF64}(I, n, n)
    vals = zeros(Float64, length(ωs))

    for (k, ω) in pairs(ωs)
        F = factorize(im * ω .* Icomplex - Ac)
        R = F \ Icomplex
        vals[k] = opnorm(R, 2)
    end
    return vals
end

function block_rotation(a::Float64, b::Float64)
    # eigenvalues = -a ± i b
    [ -a  -b
       b  -a ]
end

function block_diag(blocks::Vector{Matrix{Float64}})
    n = sum(size(B, 1) for B in blocks)
    A = zeros(Float64, n, n)
    idx = 1
    for B in blocks
        m = size(B, 1)
        A[idx:idx+m-1, idx:idx+m-1] .= B
        idx += m
    end
    return A
end

function repeated_blocks(a::Float64, b::Float64, r::Int)
    [block_rotation(a, b) for _ in 1:r]
end

# -------------------------
# Build normal matrix
# -------------------------
function build_normal_multifrequency(a1, b1, a2, b2, r)
    blocks = vcat(repeated_blocks(a1, b1, r), repeated_blocks(a2, b2, r))
    return block_diag(blocks)
end
function offdiag_frobenius(A::AbstractMatrix{<:Real})
    B = copy(Matrix(A))
    for i in 1:size(B, 1)
        B[i, i] = 0.0
    end
    return norm(B)
end
# -------------------------
# Build non-normal matrix with same eigenvalues
# -------------------------
# Because the matrix remains block upper triangular with the same
# diagonal blocks, its eigenvalues are unchanged exactly.
#
# We couple:
#   - low-frequency blocks among themselves
#   - high-frequency blocks among themselves
#   - low-frequency blocks to high-frequency blocks
# This creates non-normal modal interaction.
# -------------------------
# -------------------------
# -------------------------
# Helper: indices of block p
# -------------------------
function block_inds(p::Int)
    return (2p - 1):(2p)
end

# -------------------------
# Random 2x2 coupling block
# -------------------------
function random_coupling_block(scale::Float64)
    return scale .* randn(2, 2)
end

# -------------------------
# Build a random block-upper-triangular non-normal matrix
# with exactly the same eigenvalues as the normal one
# -------------------------
function build_random_nonnormal_same_eigs(a1, b1, a2, b2, r;
                                          scale = 2.0,
                                          density = 1.0)
    blocks = vcat(repeated_blocks(a1, b1, r), repeated_blocks(a2, b2, r))
    A = block_diag(blocks)

    nblocks = 2r

    for p in 1:nblocks-1
        for q in p+1:nblocks
            if rand() < density
                A[block_inds(p), block_inds(q)] .= random_coupling_block(scale)
            end
        end
    end

    return A
end

# -------------------------
# Search for a non-normal matrix whose peak is farthest
# from the normal peak
# -------------------------
function build_far_peak_nonnormal(a1, b1, a2, b2, r, ωs;
                                  ntrials = 400,
                                  scale_min = 0.5,
                                  scale_max = 8.0,
                                  density = 1.0,
                                  target = :either)
    A_normal = build_normal_multifrequency(a1, b1, a2, b2, r)
    S_normal = intrinsic_spectrum(A_normal, ωs)
    ω_peak_normal = ωs[argmax(S_normal)]

    best_A = nothing
    best_S = nothing
    best_ω_peak = NaN
    best_score = -Inf

    for _ in 1:ntrials
        scale = exp(rand() * (log(scale_max) - log(scale_min)) + log(scale_min))

        A_try = build_random_nonnormal_same_eigs(a1, b1, a2, b2, r;
                                                 scale = scale,
                                                 density = density)

        S_try = intrinsic_spectrum(A_try, ωs)
        ω_peak_try = ωs[argmax(S_try)]

        score = if target == :left
            ω_peak_try < ω_peak_normal ? log(ω_peak_normal / ω_peak_try) : -Inf
        elseif target == :right
            ω_peak_try > ω_peak_normal ? log(ω_peak_try / ω_peak_normal) : -Inf
        else
            abs(log(ω_peak_try / ω_peak_normal))
        end

        if score > best_score
            best_score = score
            best_A = copy(A_try)
            best_S = copy(S_try)
            best_ω_peak = ω_peak_try
        end
    end

    return best_A, best_S, ω_peak_normal, best_ω_peak, best_score
end
# -------------------------
# Naive normal-case prediction from eigenvalues
# ------------------------------------------------------------
# For a normal matrix:
#   ||R(ω)|| = 1 / min_j |iω - λ_j|
#
# This gives the exact profile in the normal case and a useful
# naive eigenvalue-only benchmark for the non-normal case.
# -------------------------
function eigenvalue_only_profile(eigs::AbstractVector{ComplexF64}, ωs::AbstractVector{<:Real})
    vals = zeros(Float64, length(ωs))
    for (k, ω) in pairs(ωs)
        d = minimum(abs.(im * ω .- eigs))
        vals[k] = 1 / d
    end
    vals
end

# -------------------------
# Build matrices
# -------------------------
A_normal = build_normal_multifrequency(A1, B1, A2, B2, REPEAT_BLOCKS)

A_nonnormal, S_nonnormal, ω_peak_normal_search, ω_peak_nonnormal_search, best_score =
    build_far_peak_nonnormal(
        A1, B1, A2, B2, REPEAT_BLOCKS, OMEGAS;
        ntrials = 600,
        scale_min = 0.5,
        scale_max = 10.0,
        density = 1.0,
        target = :right   # use :left or :right if you want a forced direction
    )
# -------------------------
# Diagnostics
# -------------------------
# -------------------------
# Diagnostics
# -------------------------
eigs_normal = ComplexF64.(eigvals(A_normal))
eigs_nonnormal = ComplexF64.(eigvals(A_nonnormal))

α_normal = spectral_abscissa(A_normal)
α_nonnormal = spectral_abscissa(A_nonnormal)

normal_flag = is_normal(A_normal)
nonnormal_flag = is_normal(A_nonnormal)

fro_off_normal = offdiag_frobenius(A_normal)
fro_off_nonnormal = offdiag_frobenius(A_nonnormal)

nnz_off_normal = count(!iszero, A_normal) - size(A_normal, 1)
nnz_off_nonnormal = count(!iszero, A_nonnormal) - size(A_nonnormal, 1)

@assert maximum(abs.(sort(eigs_normal, by=x->(real(x), imag(x))) .-
                     sort(eigs_nonnormal, by=x->(real(x), imag(x))))) < 1e-8

# -------------------------
# Spectra
# -------------------------
S_normal = intrinsic_spectrum(A_normal, OMEGAS)
# S_nonnormal already computed by the search
S_eig = eigenvalue_only_profile(ComplexF64.(eigvals(A_normal)), OMEGAS)

ω_peak_normal = OMEGAS[argmax(S_normal)]
ω_peak_nonnormal = OMEGAS[argmax(S_nonnormal)]
ω_peak_eig = OMEGAS[argmax(S_eig)]

peak_normal = maximum(S_normal)
peak_nonnormal = maximum(S_nonnormal)
peak_eig = maximum(S_eig)

# -------------------------
# Plot
# -------------------------
# -------------------------
# Shared color scaling for heatmaps
# -------------------------
vmin = min(minimum(A_normal), minimum(A_nonnormal))
vmax = max(maximum(A_normal), maximum(A_nonnormal))

# symmetric around zero (important for signed interactions)
vabs = max(abs(vmin), abs(vmax))
colorrange = (-vabs, vabs)
begin
    fig = Figure(size = (1800, 1050))
    Label(fig[0, :], "Normal vs non-normal systems with identical eigenvalues", fontsize = 24)

    # Matrices
    axA1 = Axis(fig[1, 1],
        title = "Normal matrix",
        xlabel = "Species j", ylabel = "Species i",
        aspect = DataAspect(), titlesize = 20)
    hm1 = heatmap!(
        axA1, A_normal,
        colormap = :balance,
        colorrange = colorrange
    )

    axA2 = Axis(fig[1, 3],
        title = "Non-normal matrix (same eigenvalues)",
        xlabel = "Species j", ylabel = "Species i",
        aspect = DataAspect(), titlesize = 20)
    hm2 = heatmap!(
        axA2, A_nonnormal,
        colormap = :balance,
        colorrange = colorrange
    )
    Colorbar(
        fig[1, 2],
        hm1,
        label = "Interaction strength"
    )

    # Eigenvalues
    axE = Axis(fig[1, 4],
        title = "Eigenvalues",
        xlabel = "Re(λ)", ylabel = "Im(λ)",
        titlesize = 20)
    scatter!(axE, real.(eigs_normal), imag.(eigs_normal), markersize = 12, label = "normal")
    scatter!(axE, real.(eigs_nonnormal), imag.(eigs_nonnormal), markersize = 7, label = "non-normal")
    axislegend(axE, position = :cb)
    # colsize!(fig.layout, 4, Relative(0.05))
    # colsize!(fig.layout, 2, Relative(0.005))
    # Spectra
    ax1 = Axis(
        fig[2, 1:3],
        xlabel = "Frequency ω",
        ylabel = "S(ω)",
        xscale = log10,
        yscale = log10,
        xticklabelsize = 16,
        yticklabelsize = 16,
        xlabelsize = 17,
        ylabelsize = 17
    )

    lines!(ax1, OMEGAS, S_eig, linewidth = 2, linestyle = :dash, color = :black, label = "eigenvalue prediction")
    lines!(ax1, OMEGAS, S_normal, linewidth = 3, color = :dodgerblue, label = "normal")
    lines!(ax1, OMEGAS, S_nonnormal, linewidth = 3, color = :crimson, label = "non-normal")

    # imaginary parts of eigenvalues
    vlines!(ax1, [B1], color = :gray40, linestyle = :dot, linewidth = 2)

    # observed peaks
    # vlines!(ax1, [ω_peak_normal], color = :dodgerblue, linestyle = :dash, linewidth = 2)
    # vlines!(ax1, [ω_peak_nonnormal], color = :crimson, linestyle = :dash, linewidth = 2)

    axislegend(ax1, position = :lb, labelsize = 18)
    ax1.xgridvisible = false
    ax1.ygridvisible = false

    # Diagnostics text
    txt = """
    Construction
    - normal baseline: block-diagonal damped rotations
    - non-normal comparison: same diagonal blocks + sparse strictly upper block couplings

    Shared eigenvalues
    - λ = -$A1 ± i$B1   repeated $REPEAT_BLOCKS times
    - λ = -$A2 ± i$B2   repeated $REPEAT_BLOCKS times

    Diagnostics
    - spectral abscissa (normal)     = $(round(α_normal, digits=6))
    - spectral abscissa (non-normal) = $(round(α_nonnormal, digits=6))

    - is normal? baseline   = $(normal_flag)
    - is normal? comparison = $(nonnormal_flag)

    - offdiag Frobenius (normal)     = $(round(fro_off_normal, digits=6))
    - offdiag Frobenius (non-normal) = $(round(fro_off_nonnormal, digits=6))

    - offdiag nonzeros (normal)      = $(nnz_off_normal)
    - offdiag nonzeros (non-normal)  = $(nnz_off_nonnormal)

    Peak locations
    - eigenvalue-only benchmark peak = $(round(ω_peak_eig, digits=4))
    - normal system peak             = $(round(ω_peak_normal, digits=4))
    - non-normal system peak         = $(round(ω_peak_nonnormal, digits=4))

    Peak heights
    - eigenvalue-only benchmark      = $(round(peak_eig, digits=4))
    - normal system                  = $(round(peak_normal, digits=4))
    - non-normal system              = $(round(peak_nonnormal, digits=4))

    """
    Label(fig[2, 4], txt, tellwidth = false, fontsize = 16, justification = :left)

    display(fig)
end