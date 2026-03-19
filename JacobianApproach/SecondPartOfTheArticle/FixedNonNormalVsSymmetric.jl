using LinearAlgebra
using Statistics
using CairoMakie

# ============================================================
# Non-normality proof of concept with matched baselines
# ------------------------------------------------------------
# We compare:
#   1) a symmetric chain (normal)
#   2) a feedforward chain (non-normal)
#
# and explicitly match them on:
#   - off-diagonal Frobenius norm
#   - spectral abscissa
#
# so that any remaining difference is attributable mainly
# to interaction geometry / non-normality.
#
# Homogeneous timescales:
#   T = I
#
# Resolvent:
#   R(ω) = (im*ω*I - A)^(-1)
#
# Intrinsic sensitivity:
#   S(ω) = ||R(ω)||_2
#
# Structured sensitivity:
#   S_P(ω) = ||R(ω) P R(ω)||_2
# ============================================================

# -------------------------
# User parameters
# -------------------------
const N = 10
const TARGET_OFFDIAG_FROB = 3.0
const TARGET_ABSCISSA = -0.5
const Q = 0.2
const OMEGAS = exp.(range(log(1e-2), log(1e2), length = 300))

# -------------------------
# Helpers
# -------------------------
function spectral_abscissa(A::AbstractMatrix{<:Real})
    maximum(real.(eigvals(Matrix(A))))
end

function offdiag_frobenius(A::AbstractMatrix{<:Real})
    B = copy(Matrix(A))
    for i in 1:size(B, 1)
        B[i, i] = 0.0
    end
    norm(B)
end

function weighted_degree(A::AbstractMatrix{<:Real})
    vec(sum(abs.(A), dims = 1)) .+ vec(sum(abs.(A), dims = 2))
end

function centrality_classes(c::AbstractVector{<:Real}; q::Float64 = 0.2)
    n = length(c)
    m = max(2, round(Int, q * n))
    p = sortperm(c)
    low = sort(p[1:m])
    high = sort(p[end-m+1:end])
    return low, high
end

function perturbation_operator_class(n::Int, C::AbstractVector{<:Integer})
    M = zeros(Float64, n, n)
    Cset = Set(C)
    @inbounds for i in 1:n, j in 1:n
        if i != j && (i in Cset || j in Cset)
            M[i, j] = 1.0
        end
    end
    M / norm(M)
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
    out
end

function structured_spectrum(A::AbstractMatrix{<:Real},
                             P::AbstractMatrix{<:Real},
                             ωs::AbstractVector{<:Real})
    n = size(A, 1)
    Icomplex = Matrix{ComplexF64}(I, n, n)
    Ac = ComplexF64.(A)
    Pc = ComplexF64.(P)
    out = zeros(Float64, length(ωs))

    for (k, ω) in pairs(ωs)
        F = factorize(im * ω .* Icomplex - Ac)
        R = F \ Icomplex
        out[k] = opnorm(R * Pc * R, 2)
    end
    out
end

# -------------------------
# Base off-diagonal structures
# -------------------------
function symmetric_chain_offdiag(n::Int)
    A = zeros(Float64, n, n)
    for i in 1:n-1
        A[i, i+1] = 1.0
        A[i+1, i] = 1.0
    end
    A
end

function feedforward_chain_offdiag(n::Int)
    A = zeros(Float64, n, n)
    for i in 1:n-1
        A[i, i+1] = 1.0
    end
    A
end

# -------------------------
# Matching procedure
# -------------------------
function scale_offdiag_to_frobenius(B::AbstractMatrix{<:Real}, target_frob::Float64)
    current = offdiag_frobenius(B)
    current == 0 && error("Off-diagonal Frobenius norm is zero.")
    return (target_frob / current) .* Matrix(B)
end

function add_diagonal_to_target_abscissa(B::AbstractMatrix{<:Real}, target_abscissa::Float64)
    # Build A = B - dI such that spectral_abscissa(A) = target_abscissa
    αB = spectral_abscissa(B)
    d = αB - target_abscissa
    A = Matrix(B) - d .* I
    return A, d
end

# -------------------------
# Build matched matrices
# -------------------------
Bsym0 = symmetric_chain_offdiag(N)
Bff0  = feedforward_chain_offdiag(N)

Bsym = scale_offdiag_to_frobenius(Bsym0, TARGET_OFFDIAG_FROB)
Bff  = scale_offdiag_to_frobenius(Bff0, TARGET_OFFDIAG_FROB)

Asym, d_sym = add_diagonal_to_target_abscissa(Bsym, TARGET_ABSCISSA)
Aff,  d_ff  = add_diagonal_to_target_abscissa(Bff, TARGET_ABSCISSA)

# Diagnostics
α_sym = spectral_abscissa(Asym)
α_ff  = spectral_abscissa(Aff)

fro_sym = offdiag_frobenius(Asym)
fro_ff  = offdiag_frobenius(Aff)

# Centrality classes and structured perturbations
c_sym = weighted_degree(Asym)
c_ff  = weighted_degree(Aff)

low_sym, high_sym = centrality_classes(c_sym; q = Q)
low_ff,  high_ff  = centrality_classes(c_ff; q = Q)

P_high_sym = perturbation_operator_class(N, high_sym)
P_high_ff  = perturbation_operator_class(N, high_ff)

# Spectra
S_sym = intrinsic_spectrum(Asym, OMEGAS)
S_ff  = intrinsic_spectrum(Aff, OMEGAS)

SP_sym = structured_spectrum(Asym, P_high_sym, OMEGAS)
SP_ff  = structured_spectrum(Aff, P_high_ff, OMEGAS)

# Proper structural worst-case baseline
S2_sym = S_sym .^ 2
S2_ff  = S_ff .^ 2

# Common y-limits
all_intrinsic = vcat(S_sym, S_ff)
all_struct = vcat(SP_sym, SP_ff, S2_sym, S2_ff)

intr_ymin = minimum(all_intrinsic[all_intrinsic .> 0])
intr_ymax = maximum(all_intrinsic)

struct_ymin = minimum(all_struct[all_struct .> 0])
struct_ymax = maximum(all_struct)

# -------------------------
# Plot
# -------------------------
begin
    fig = Figure(size = (1550, 1020))
    Label(fig[0, :], "Non-normality proof of concept with matched spectral abscissa and off-diagonal norm", fontsize = 24)

    # Matrices
    axA1 = Axis(fig[1, 1], title = "Symmetric chain (normal)", xlabel = "j", ylabel = "i", aspect = DataAspect())
    heatmap!(axA1, Asym)

    axA2 = Axis(fig[1, 2], title = "Feedforward chain (non-normal)", xlabel = "j", ylabel = "i", aspect = DataAspect())
    heatmap!(axA2, Aff)

    # Intrinsic spectra
    ax1 = Axis(
        fig[2, 1:2],
        title = "Intrinsic spectra  S(ω) = ||R(ω)||₂",
        xlabel = "frequency ω",
        ylabel = "S(ω)",
        xscale = log10,
        yscale = log10
    )
    lines!(ax1, OMEGAS, S_sym, linewidth = 3, label = "symmetric")
    lines!(ax1, OMEGAS, S_ff, linewidth = 3, label = "feedforward")
    ylims!(ax1, intr_ymin, intr_ymax)
    axislegend(ax1, position = :rb)

    # Structured spectra
    ax2 = Axis(
        fig[3, 1:2],
        title = "Structured spectra for high-centrality perturbation class",
        xlabel = "frequency ω",
        ylabel = "S_P(ω) = ||R(ω) P R(ω)||₂",
        xscale = log10,
        yscale = log10
    )
    lines!(ax2, OMEGAS, SP_sym, linewidth = 3, label = "symmetric")
    lines!(ax2, OMEGAS, SP_ff, linewidth = 3, label = "feedforward")
    ylims!(ax2, struct_ymin, struct_ymax)
    axislegend(ax2, position = :rb)

    # Proper structural baseline
    ax3 = Axis(
        fig[4, 1:2],
        title = "Proper structural worst-case baseline  ||R(ω)||₂²",
        xlabel = "frequency ω",
        ylabel = "structural baseline",
        xscale = log10,
        yscale = log10
    )
    lines!(ax3, OMEGAS, S2_sym, linewidth = 3, label = "symmetric: ||R||²")
    lines!(ax3, OMEGAS, S2_ff, linewidth = 3, label = "feedforward: ||R||²")
    lines!(ax3, OMEGAS, SP_sym, linewidth = 2, linestyle = :dash, label = "symmetric: S_P")
    lines!(ax3, OMEGAS, SP_ff, linewidth = 2, linestyle = :dash, label = "feedforward: S_P")
    ylims!(ax3, struct_ymin, struct_ymax)
    axislegend(ax3, position = :rb)

    # Diagnostics text
    txt = """
    Construction:
    1) build off-diagonal chain structures
    2) scale both to the same off-diagonal Frobenius norm
    3) choose diagonal damping separately so both have the same spectral abscissa

    Targets:
    offdiag Frobenius norm = $(round(TARGET_OFFDIAG_FROB, digits=4))
    spectral abscissa      = $(round(TARGET_ABSCISSA, digits=4))

    Diagnostics:
    symmetric spectral abscissa   = $(round(α_sym, digits=6))
    feedforward spectral abscissa = $(round(α_ff, digits=6))

    symmetric offdiag Frobenius   = $(round(fro_sym, digits=6))
    feedforward offdiag Frobenius = $(round(fro_ff, digits=6))

    symmetric diagonal damping    = $(round(d_sym, digits=6))
    feedforward diagonal damping  = $(round(d_ff, digits=6))

    Interpretation:
    the matrices are now comparable in
        - total off-diagonal interaction scale
        - distance to instability

    so remaining differences are attributable mainly
    to normal vs non-normal interaction geometry.
    """
    Label(fig[1:4, 3], txt, tellwidth = false, fontsize = 16, justification = :left)

    display(fig)
end