using LinearAlgebra
using Statistics
using CairoMakie
using Printf

# ============================================================
# Completely different-profile visualization with weak dynamics
# ------------------------------------------------------------
# Two independent systems:
#   - original  : dominant LOW-frequency resonance
#   - modified  : dominant HIGH-frequency resonance
#
# Two animations:
#   1) original weak  -> modified strong
#        force near modified peak, through high-frequency entrance
#
#   2) original strong -> modified weak
#        force near original peak, through low-frequency entrance
#
# Goal:
#   - very different profiles
#   - weak system should show only small, damped oscillations
#   - strong system should show large, sustained oscillations
#
# Layout (3 x 2):
#   row 1: forcing | forcing
#   row 2: original dynamics | modified dynamics
#   row 3: original ||R(iω)||₂ | modified ||R(iω)||₂
# ============================================================

# -------------------------
# Parameters
# -------------------------
const OMEGAS = exp.(range(log(0.05), log(10.0), length = 1600))
const DT = 0.02
const FORCE_START = 8.0
const FORCE_AMPLITUDE = 0.65
const OUTDIR = "animations_completely_different_profiles"
const SHOWN_SPECIES = [3, 5, 7, 9, 11]

mkpath(OUTDIR)

# ============================================================
# Linear algebra helpers
# ============================================================
function spectral_abscissa(A::AbstractMatrix{<:Real})
    maximum(real.(eigvals(Matrix(A))))
end

function stabilize_if_needed(A::AbstractMatrix{<:Real}; margin::Float64 = 0.22)
    B = Matrix{Float64}(A)
    α = spectral_abscissa(B)
    if α >= -margin
        shift = α + margin
        B -= shift * I
    end
    return B
end

function resolvent(A::AbstractMatrix{<:Real}, ω::Real)
    n = size(A, 1)
    Icomplex = Matrix{ComplexF64}(I, n, n)
    Ac = ComplexF64.(A)
    F = factorize(im * ω .* Icomplex - Ac)
    return F \ Icomplex
end

function resolvent_profile(A::AbstractMatrix{<:Real}, ωs::AbstractVector{<:Real})
    vals = zeros(Float64, length(ωs))
    for (k, ω) in pairs(ωs)
        vals[k] = opnorm(resolvent(A, ω), 2)
    end
    return vals
end

# ============================================================
# Oscillator building blocks
# ============================================================
function oscillator_block(ω::Float64, d::Float64)
    [
        -d   -ω
         ω   -d
    ]
end

function insert_block!(A::AbstractMatrix{<:Real}, block::AbstractMatrix{<:Real}, idx::Int)
    r = 2idx - 1
    A[r:r+1, r:r+1] .= block
    return A
end

function add_chain_couplings!(
    A::AbstractMatrix{<:Real};
    c12 = 0.0, c23 = 0.0, c34 = 0.0, c45 = 0.0, c56 = 0.0,
    long13 = 0.0, long24 = 0.0, long35 = 0.0, long46 = 0.0,
    fb21 = 0.0, fb32 = 0.0, fb43 = 0.0, fb54 = 0.0, fb65 = 0.0
)
    # mode 1 -> 2
    A[1, 3] = c12
    A[2, 3] = 0.40 * c12
    A[2, 4] = 0.20 * c12

    # mode 2 -> 3
    A[3, 5] = c23
    A[4, 5] = 0.40 * c23
    A[4, 6] = 0.20 * c23

    # mode 3 -> 4
    A[5, 7] = c34
    A[6, 7] = 0.40 * c34
    A[6, 8] = 0.20 * c34

    # mode 4 -> 5
    A[7, 9] = c45
    A[8, 9] = 0.40 * c45
    A[8, 10] = 0.20 * c45

    # mode 5 -> 6
    A[9, 11] = c56
    A[10, 11] = 0.40 * c56
    A[10, 12] = 0.20 * c56

    # long-range feedforward
    A[1, 5] = long13
    A[3, 7] = long24
    A[5, 9] = long35
    A[7, 11] = long46

    # weak feedback
    A[3, 1] = fb21
    A[5, 3] = fb32
    A[7, 5] = fb43
    A[9, 7] = fb54
    A[11, 9] = fb65

    return A
end

# ============================================================
# Two deliberately different systems
# ============================================================
function build_original_system()
    # One dominant low-frequency peak around ~0.36
    # Everything else is more damped so the background stays low.
    freqs = [0.20, 0.36, 0.70, 1.20, 2.10, 3.40]
    damps = [0.30, 0.020, 0.34, 0.42, 0.50, 0.55]

    A = zeros(Float64, 12, 12)
    for k in 1:6
        insert_block!(A, oscillator_block(freqs[k], damps[k]), k)
    end

    add_chain_couplings!(
        A;
        c12 = 1.10,
        c23 = 0.35,
        c34 = 0.08,
        c45 = 0.02,
        c56 = 0.00,
        long13 = 0.14,
        long24 = 0.02,
        long35 = 0.00,
        long46 = 0.00,
        fb21 = 0.00,
        fb32 = -0.03,
        fb43 = 0.00,
        fb54 = 0.00,
        fb65 = 0.00
    )

    return stabilize_if_needed(A; margin = 0.24)
end

function build_modified_system()
    # Dominant high-frequency resonance around ~5.25,
    # but distributed across several neighboring modes rather than
    # concentrated almost entirely on species 9.
    freqs = [1.50, 2.30, 3.20, 4.10, 5.25, 6.90]
    damps = [0.55, 0.48, 0.22, 0.11, 0.020, 0.16]

    A = zeros(Float64, 12, 12)
    for k in 1:6
        insert_block!(A, oscillator_block(freqs[k], damps[k]), k)
    end

    add_chain_couplings!(
        A;
        c12 = 0.00,
        c23 = 0.12,
        c34 = 0.55,
        c45 = 0.95,
        c56 = 0.85,
        long13 = 0.00,
        long24 = 0.06,
        long35 = 0.26,
        long46 = 0.22,
        fb21 = 0.00,
        fb32 = 0.00,
        fb43 = -0.05,
        fb54 = -0.06,
        fb65 = 0.00
    )

    # Reduce pure concentration onto species 9, increase spread.
    A[9, 7]   = 0.42
    A[10, 7]  = 0.24
    A[10, 8]  = 0.14

    A[7, 5]   = 0.62
    A[8, 5]   = 0.22
    A[8, 6]   = 0.10

    A[11, 9]  = 0.38
    A[12, 9]  = 0.20
    A[12, 10] = 0.10

    # Stronger cross-talk toward neighboring visible species
    A[9, 5]   = 0.42
    A[11, 7]  = 0.46
    A[7, 9]   = -0.06
    A[5, 9]   = 0.18
    A[11, 5]  = 0.22
    A[5, 11]  = 0.16

    return stabilize_if_needed(A; margin = 0.24)
end

# ============================================================
# Frequency choice
# ============================================================
function choose_main_peak_in_band(S::AbstractVector{<:Real}, ωs::AbstractVector{<:Real}, band::Tuple{Float64,Float64})
    idxs = findall((ωs .>= band[1]) .& (ωs .<= band[2]))
    isempty(idxs) && error("No frequencies in band $(band)")
    return idxs[argmax(S[idxs])]
end

# ============================================================
# Dynamics
# ============================================================
function simulate_forced_system(
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    ω::Float64;
    dt::Float64 = DT,
    forcing_amplitude::Float64 = FORCE_AMPLITUDE,
    forcing_start::Float64 = FORCE_START
)
    # Enough time for slow forcing to show many oscillations
    period = 2π / ω
    tmax = max(50.0, forcing_start + 18period)
    ts = collect(0:dt:tmax)

    n = size(A, 1)
    X = zeros(Float64, n, length(ts))

    forcing_at_time(t) = t < forcing_start ? zeros(Float64, n) :
                         forcing_amplitude * sin(ω * (t - forcing_start)) * b

    f(x, t) = A * x + forcing_at_time(t)

    x = zeros(Float64, n)
    for k in 1:length(ts)-1
        t = ts[k]
        k1 = f(x, t)
        k2 = f(x .+ 0.5dt .* k1, t + 0.5dt)
        k3 = f(x .+ 0.5dt .* k2, t + 0.5dt)
        k4 = f(x .+ dt .* k3, t + dt)
        x = x .+ (dt / 6.0) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        X[:, k+1] .= x
    end

    forcing_signal = [
        t < forcing_start ? 0.0 :
        forcing_amplitude * sin(ω * (t - forcing_start))
        for t in ts
    ]

    return ts, X, forcing_signal
end

# ============================================================
# Axis helpers
# ============================================================
function forcing_ylim(f::AbstractVector)
    m = maximum(abs, f)
    m = max(m, 1.0)
    return (-1.08m, 1.08m)
end

function shared_profile_ylim(S1::AbstractVector, S2::AbstractVector)
    ymin = minimum(vcat(S1, S2))
    ymax = maximum(vcat(S1, S2))
    ymin = max(ymin, 1e-4)
    ymax = max(ymax, 10ymin)
    return (0.95ymin, 1.05ymax)
end

# Shared y-axis within each case, but clipped robustly so the strong panel
# doesn't make the weak oscillations disappear completely.
function robust_state_ylim(X1::AbstractMatrix, X2::AbstractMatrix; q::Float64 = 0.980)
    vals = abs.(vcat(vec(X1), vec(X2)))
    m = quantile(vals, q)
    m = max(m, 0.08)
    return (-1.08m, 1.08m)
end

# ============================================================
# Animation
# ============================================================
function animate_case(
    filepath::String,
    title_text::String,
    ωforce::Float64,
    ts::AbstractVector{<:Real},
    X_orig::AbstractMatrix{<:Real},
    X_mod::AbstractMatrix{<:Real},
    forcing_signal::AbstractVector{<:Real},
    ωs::AbstractVector{<:Real},
    S_orig::AbstractVector{<:Real},
    S_mod::AbstractVector{<:Real};
    shown_species = SHOWN_SPECIES
)
    fig = Figure(size = (1600, 1100))
    Label(fig[0, :], title_text, fontsize = 24)

    y_force = forcing_ylim(forcing_signal)
    y_state = robust_state_ylim(X_orig[shown_species, :], X_mod[shown_species, :])
    y_prof = shared_profile_ylim(S_orig, S_mod)

    # Top row
    ax_force_left = Axis(
        fig[1, 1],
        title = @sprintf("Environmental forcing   (ω = %.3f)", ωforce),
        xlabel = "time",
        ylabel = "u(t)"
    )
    xlims!(ax_force_left, first(ts), last(ts))
    ylims!(ax_force_left, y_force...)
    ax_force_left.xgridvisible = false
    ax_force_left.ygridvisible = false

    ax_force_right = Axis(
        fig[1, 2],
        title = @sprintf("Environmental forcing   (ω = %.3f)", ωforce),
        xlabel = "time",
        ylabel = "u(t)"
    )
    xlims!(ax_force_right, first(ts), last(ts))
    ylims!(ax_force_right, y_force...)
    ax_force_right.xgridvisible = false
    ax_force_right.ygridvisible = false

    # Middle row
    ax_orig = Axis(
        fig[2, 1],
        title = "Community dynamics — original",
        xlabel = "time",
        ylabel = "state"
    )
    xlims!(ax_orig, first(ts), last(ts))
    ylims!(ax_orig, y_state...)
    ax_orig.xgridvisible = false
    ax_orig.ygridvisible = false

    ax_mod = Axis(
        fig[2, 2],
        title = "Community dynamics — modified",
        xlabel = "time",
        ylabel = "state"
    )
    xlims!(ax_mod, first(ts), last(ts))
    ylims!(ax_mod, y_state...)
    ax_mod.xgridvisible = false
    ax_mod.ygridvisible = false

    # Bottom row
    ax_prof_orig = Axis(
        fig[3, 1],
        title = "Original sensitivity profile   ||R(iω)||₂",
        xlabel = "frequency ω",
        ylabel = "||R(iω)||₂",
        xscale = log10,
        yscale = log10
    )
    ylims!(ax_prof_orig, y_prof...)
    ax_prof_orig.xgridvisible = false
    ax_prof_orig.ygridvisible = false

    ax_prof_mod = Axis(
        fig[3, 2],
        title = "Modified sensitivity profile   ||R(iω)||₂",
        xlabel = "frequency ω",
        ylabel = "||R(iω)||₂",
        xscale = log10,
        yscale = log10
    )
    ylims!(ax_prof_mod, y_prof...)
    ax_prof_mod.xgridvisible = false
    ax_prof_mod.ygridvisible = false

    idxω = argmin(abs.(ωs .- ωforce))

    lines!(ax_prof_orig, ωs, S_orig, linewidth = 3, color = :black)
    vlines!(ax_prof_orig, [ωforce], color = :dodgerblue, linestyle = :dash, linewidth = 3)
    scatter!(ax_prof_orig, [ωforce], [S_orig[idxω]], color = :black, markersize = 12)

    lines!(ax_prof_mod, ωs, S_mod, linewidth = 3, color = :crimson)
    vlines!(ax_prof_mod, [ωforce], color = :dodgerblue, linestyle = :dash, linewidth = 3)
    scatter!(ax_prof_mod, [ωforce], [S_mod[idxω]], color = :crimson, markersize = 12)

    # Animated traces
    t_idx = Observable(1)
    ts_now = lift(i -> ts[1:i], t_idx)
    f_now = lift(i -> forcing_signal[1:i], t_idx)

    lines!(ax_force_left, ts_now, f_now, linewidth = 3, color = :dodgerblue)
    lines!(ax_force_right, ts_now, f_now, linewidth = 3, color = :dodgerblue)

    colors = [:darkorange, :seagreen, :purple, :goldenrod, :steelblue]

    for (k, sp) in enumerate(shown_species)
        x_now = lift(i -> X_orig[sp, 1:i], t_idx)
        lines!(ax_orig, ts_now, x_now, linewidth = 2.7, color = colors[k], label = "species $sp")
    end
    ylims!(ax_orig, (-0.5, 0.5))
    # axislegend(ax_orig, position = :rb)

    for (k, sp) in enumerate(shown_species)
        x_now = lift(i -> X_mod[sp, 1:i], t_idx)
        lines!(ax_mod, ts_now, x_now, linewidth = 2.7, color = colors[k], label = "species $sp")
    end
    ylims!(ax_mod, (-0.5, 0.5))
    # axislegend(ax_mod, position = :rb)

    frame_step = 10
    frame_indices = 1:frame_step:length(ts)

    record(fig, filepath, frame_indices; framerate = 30) do i
        t_idx[] = i
    end
end

# ============================================================
# Main
# ============================================================
A_orig = build_original_system()
A_mod  = build_modified_system()

S_orig = resolvent_profile(A_orig, OMEGAS)
S_mod  = resolvent_profile(A_mod, OMEGAS)

# Strong->weak:
# original dominant low-frequency peak
idx_orig_peak = choose_main_peak_in_band(S_orig, OMEGAS, (0.28, 0.45))
ω_orig_peak = OMEGAS[idx_orig_peak]

# Weak->strong:
# modified dominant high-frequency peak
idx_mod_peak = choose_main_peak_in_band(S_mod, OMEGAS, (4.80, 5.70))
ω_mod_peak = OMEGAS[idx_mod_peak]

println(@sprintf(
    "Original peak case (strong -> weak): ω = %.4f | original = %.4f | modified = %.4f | ratio = %.3f",
    ω_orig_peak, S_orig[idx_orig_peak], S_mod[idx_orig_peak],
    S_orig[idx_orig_peak] / (S_mod[idx_orig_peak] + 1e-12)
))

println(@sprintf(
    "Modified peak case (weak -> strong): ω = %.4f | original = %.4f | modified = %.4f | ratio = %.3f",
    ω_mod_peak, S_orig[idx_mod_peak], S_mod[idx_mod_peak],
    S_mod[idx_mod_peak] / (S_orig[idx_mod_peak] + 1e-12)
))

# ------------------------------------------------------------
# Case-specific forcing vectors
# ------------------------------------------------------------
# These are the key fix:
#   - low-frequency forcing enters the low chain
#   - high-frequency forcing enters the high chain
# so the weak system remains much more damped in time.
# ------------------------------------------------------------
# For original strong / modified weak
b_low = zeros(Float64, 12)
b_low[1] = 1.00
b_low[3] = 0.18
b_low[5] = 0.05

# For modified strong / original weak
# For modified strong / original weak
# Spread the forcing over the high-frequency neighborhood,
# instead of injecting almost only into species 9.
# For modified strong / original weak
# Broad excitation of the high-frequency neighborhood
# For modified strong / original weak
# Deliberately avoid over-forcing species 9 directly.
b_high = zeros(Float64, 12)
b_high[5]  = 0.55
b_high[7]  = 1.00
b_high[8]  = 0.35
b_high[9]  = 0.32
b_high[10] = 0.30
b_high[11] = 0.95
b_high[12] = 0.28

# Simulations
ts_modpeak, X_modpeak_orig, f_modpeak = simulate_forced_system(A_orig, b_high, ω_mod_peak)
_,          X_modpeak_mod,  _         = simulate_forced_system(A_mod,  b_high, ω_mod_peak)

ts_origpeak, X_origpeak_orig, f_origpeak = simulate_forced_system(A_orig, b_low, ω_orig_peak)
_,           X_origpeak_mod,  _          = simulate_forced_system(A_mod,  b_low, ω_orig_peak)

# Animation 1: original weak -> modified strong
animate_case(
    joinpath(OUTDIR, "01_weak_to_strong.mp4"),
    @sprintf("Original weak → modified strong at the modified peak frequency   (ω = %.3f)", ω_mod_peak),
    ω_mod_peak,
    ts_modpeak,
    X_modpeak_orig,
    X_modpeak_mod,
    f_modpeak,
    OMEGAS,
    S_orig,
    S_mod
)

# Animation 2: original strong -> modified weak
animate_case(
    joinpath(OUTDIR, "02_strong_to_weak.mp4"),
    @sprintf("Original strong → modified weak at the original peak frequency   (ω = %.3f)", ω_orig_peak),
    ω_orig_peak,
    ts_origpeak,
    X_origpeak_orig,
    X_origpeak_mod,
    f_origpeak,
    OMEGAS,
    S_orig,
    S_mod
)

println()
println("Saved animations to: $(abspath(OUTDIR))")
println("Files:")
println("  - 01_weak_to_strong.mp4")
println("  - 02_strong_to_weak.mp4")