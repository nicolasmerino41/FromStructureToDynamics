using LinearAlgebra
using Statistics
using CairoMakie
using Printf

# ============================================================
# Frequency-resolved response visualization
# ------------------------------------------------------------
# Goal:
#   Show visually what it means for a system to be sensitive
#   or not at a given forcing frequency, and how a structural
#   modification changes that response.
#
# We animate 4 cases:
#   1) Original system, weak-sensitivity frequency
#   2) Modified system, weak-sensitivity frequency
#   3) Original system, high-sensitivity frequency
#   4) Modified system, high-sensitivity frequency
#
# Model:
#   dx/dt = A x + b sin(ω t)
#
# Structural modification:
#   A_mod = A + ε P
#
# We also display:
#   S(ω) = ||(iωI - A)^(-1)||_2
# ============================================================

# -------------------------
# User parameters
# -------------------------
const OMEGAS = exp.(range(log(1e-2), log(1e2), length = 600))
const DT = 0.01
const TMAX = 40.0
const FORCE_AMPLITUDE = 1.0
const EPS_P = 0.18
const OUTDIR = "animations_resolvent_demo"

mkpath(OUTDIR)

# -------------------------
# Helpers
# -------------------------
function spectral_abscissa(A::AbstractMatrix{<:Real})
    maximum(real.(eigvals(Matrix(A))))
end

function intrinsic_spectrum(A::AbstractMatrix{<:Real}, ωs::AbstractVector{<:Real})
    n = size(A, 1)
    Icomplex = Matrix{ComplexF64}(I, n, n)
    Ac = ComplexF64.(A)
    vals = zeros(Float64, length(ωs))

    for (k, ω) in pairs(ωs)
        F = factorize(im * ω .* Icomplex - Ac)
        R = F \ Icomplex
        vals[k] = opnorm(R, 2)
    end
    return vals
end

function simulate_forced_system(
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    ω::Float64;
    dt::Float64 = DT,
    tmax::Float64 = TMAX,
    forcing_amplitude::Float64 = FORCE_AMPLITUDE,
    forcing_start::Float64 = 8.0
)
    n = size(A, 1)
    ts = collect(0:dt:tmax)
    X = zeros(Float64, n, length(ts))

    function forcing_at_time(t)
        if t < forcing_start
            return zeros(Float64, n)
        else
            return forcing_amplitude * sin(ω * (t - forcing_start)) * b
        end
    end

    function f(x, t)
        return A * x + forcing_at_time(t)
    end

    x = zeros(Float64, n)

    for k in 1:length(ts)-1
        t = ts[k]
        k1 = f(x, t)
        k2 = f(x .+ 0.5dt .* k1, t + 0.5dt)
        k3 = f(x .+ 0.5dt .* k2, t + 0.5dt)
        k4 = f(x .+ dt .* k3, t + dt)
        x = x .+ (dt/6.0) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        X[:, k+1] .= x
    end

    forcing_signal = [t < forcing_start ? 0.0 :
                      forcing_amplitude * sin(ω * (t - forcing_start)) for t in ts]

    return ts, X, forcing_signal
end

function choose_demo_frequencies(ωs::AbstractVector{<:Real}, S::AbstractVector{<:Real})
    # weak frequency: low sensitivity but not at extreme edge
    # choose from lower third of spectrum values
    cutoff = quantile(S, 0.25)
    weak_candidates = findall(S .<= cutoff)

    # prefer a frequency away from very edges
    midmask = findall((ωs .> 0.08) .& (ωs .< 8.0))
    weak_candidates = intersect(weak_candidates, midmask)
    isempty(weak_candidates) && (weak_candidates = midmask)
    weak_idx = weak_candidates[round(Int, length(weak_candidates) * 0.6)]

    peak_idx = argmax(S)

    return ωs[weak_idx], ωs[peak_idx], weak_idx, peak_idx
end

function perturbation_operator_links_of_species(A::AbstractMatrix{<:Real}, species::Vector{Int}; tol::Float64 = 0.0)
    n = size(A, 1)
    M = zeros(Float64, n, n)
    Sset = Set(species)

    @inbounds for i in 1:n, j in 1:n
        if i != j && abs(A[i, j]) > tol && (i in Sset || j in Sset)
            M[i, j] = 1.0
        end
    end

    nf = norm(M)
    nf == 0 && error("Perturbation mask has zero Frobenius norm.")
    return M / nf
end

function make_demo_matrix()
    # A small stable non-normal oscillatory system
    # built from two damped oscillatory blocks plus directional couplings
    A = [
        -0.45  -1.00   0.55   0.00   0.00   0.00;
         1.00  -0.45   0.00   0.45   0.00   0.00;
         0.00   0.00  -0.55  -0.75   0.70   0.00;
         0.00   0.00   0.75  -0.55   0.00   0.60;
         0.00   0.00   0.00   0.00  -0.60  -1.35;
         0.30   0.00   0.00   0.00   1.35  -0.60
    ]

    # shift if needed so it is safely stable
    α = spectral_abscissa(A)
    if α >= -0.15
        A .-= (α + 0.2) .* I
    end
    return A
end

function response_ylim(X1::AbstractMatrix, X2::AbstractMatrix, X3::AbstractMatrix, X4::AbstractMatrix)
    m = maximum(abs, vcat(vec(X1), vec(X2), vec(X3), vec(X4)))
    m = max(m, 0.5)
    return (-1.1m, 1.1m)
end

function forcing_ylim(f1, f2, f3, f4)
    m = maximum(abs, vcat(f1, f2, f3, f4))
    m = max(m, 1.0)
    return (-1.1m, 1.1m)
end

function animate_case(
    filepath::String,
    title_text::String,
    A::AbstractMatrix{<:Real},
    Sω::AbstractVector{<:Real},
    ωs::AbstractVector{<:Real},
    ωforce::Float64,
    ts::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    forcing_signal::AbstractVector{<:Real},
    resp_ylim::Tuple{Float64,Float64},
    forc_ylim::Tuple{Float64,Float64};
    shown_species = [1, 3, 5]
)
    fig = Figure(size = (1450, 850))
    Label(fig[0, :], title_text, fontsize = 26)

    # top: forcing
    ax_forcing = Axis(
        fig[1, 1:2],
        title = "External forcing",
        xlabel = "time",
        ylabel = "forcing"
    )
    ylims!(ax_forcing, forc_ylim...)
    xlims!(ax_forcing, first(ts), last(ts))
    ax_forcing.xgridvisible = false
    ax_forcing.ygridvisible = false

    # bottom-left: species responses
    ax_resp = Axis(
        fig[2, 1:2],
        title = "Community response",
        xlabel = "time",
        ylabel = "state"
    )
    ylims!(ax_resp, resp_ylim...)
    xlims!(ax_resp, first(ts), last(ts))
    ax_resp.xgridvisible = false
    ax_resp.ygridvisible = false

    # right: sensitivity profile
    ax_prof = Axis(
        fig[1:2, 3],
        title = "Intrinsic sensitivity profile",
        xlabel = "frequency ω",
        ylabel = "S(ω)",
        xscale = log10,
        yscale = log10
    )
    ax_prof.xgridvisible = false
    ax_prof.ygridvisible = false

    lines!(ax_prof, ωs, Sω, linewidth = 3, color = :black)
    vlines!(ax_prof, [ωforce], color = :crimson, linestyle = :dash, linewidth = 3)

    text!(
        ax_prof, ωforce, maximum(Sω) / 1.4,
        text = @sprintf("ω = %.3f", ωforce),
        align = (:left, :center),
        fontsize = 18,
        color = :crimson
    )

    # animated observables
    t_idx = Observable(1)

    ts_now = lift(i -> ts[1:i], t_idx)
    forcing_now = lift(i -> forcing_signal[1:i], t_idx)

    lines!(ax_forcing, ts_now, forcing_now, linewidth = 3, color = :dodgerblue)

    colors = [:darkorange, :seagreen, :purple]
    for (k, sp) in enumerate(shown_species)
        x_now = lift(i -> X[sp, 1:i], t_idx)
        lines!(ax_resp, ts_now, x_now, linewidth = 3, color = colors[k], label = "species $sp")
    end
    axislegend(ax_resp, position = :rb)

    frame_step = 4
    frame_indices = 1:frame_step:length(ts)

    record(fig, filepath, frame_indices; framerate = 40) do i
        t_idx[] = i
    end
end

# -------------------------
# Build community and structural change
# -------------------------
A = make_demo_matrix()

# choose a structural perturbation class:
# links incident to species 1 and 2
P = perturbation_operator_links_of_species(A, [1, 2])

A_mod = A .+ EPS_P .* P

# Keep modified system stable if necessary
α_mod = spectral_abscissa(A_mod)
if α_mod >= -0.05
    A_mod .-= (α_mod + 0.1) .* I
end

# forcing direction
b = zeros(Float64, size(A, 1))
b[1] = 1.0
b[4] = 0.6

# -------------------------
# Compute profiles and choose frequencies
# -------------------------
S_A = intrinsic_spectrum(A, OMEGAS)
S_Amod = intrinsic_spectrum(A_mod, OMEGAS)

ω_weak, ω_peak, _, _ = choose_demo_frequencies(OMEGAS, S_A)

println(@sprintf("Chosen weak frequency: %.4f", ω_weak))
println(@sprintf("Chosen peak frequency: %.4f", ω_peak))

# -------------------------
# Simulate the four cases
# -------------------------
ts1, X1, f1 = simulate_forced_system(A, b, 10.0; forcing_start = 8.0)
ts2, X2, f2 = simulate_forced_system(A_mod, b, 10.0; forcing_start = 8.0)
ts3, X3, f3 = simulate_forced_system(A, b, ω_peak; forcing_start = 8.0)
ts4, X4, f4 = simulate_forced_system(A_mod, b, ω_peak; forcing_start = 8.0)

resp_ylim = response_ylim(X1, X2, X3, X4)
forc_ylim = forcing_ylim(f1, f2, f3, f4)

# -------------------------
# Record animations
# -------------------------
animate_case(
    joinpath(OUTDIR, "01_original_weak_frequency.mp4"),
    @sprintf("Original structure — forcing at weak-sensitivity frequency (ω = %.3f)", ω_weak),
    A, S_A, OMEGAS, 10.0, ts1, X1, f1, resp_ylim, forc_ylim
)

animate_case(
    joinpath(OUTDIR, "02_modified_weak_frequency.mp4"),
    @sprintf("Modified structure — forcing at weak-sensitivity frequency (ω = %.3f)", ω_weak),
    A_mod, S_Amod, OMEGAS, 10.0, ts2, X2, f2, resp_ylim, forc_ylim
)

animate_case(
    joinpath(OUTDIR, "03_original_peak_frequency.mp4"),
    @sprintf("Original structure — forcing near peak-sensitivity frequency (ω = %.3f)", ω_peak),
    A, S_A, OMEGAS, ω_peak, ts3, X3, f3, resp_ylim, forc_ylim
)

animate_case(
    joinpath(OUTDIR, "04_modified_peak_frequency.mp4"),
    @sprintf("Modified structure — forcing near peak-sensitivity frequency (ω = %.3f)", ω_peak),
    A_mod, S_Amod, OMEGAS, ω_peak, ts4, X4, f4, resp_ylim, forc_ylim
)

println("\nSaved animations to: $(abspath(OUTDIR))")
println("Files:")
println("  - 01_original_weak_frequency.mp4")
println("  - 02_modified_weak_frequency.mp4")
println("  - 03_original_peak_frequency.mp4")
println("  - 04_modified_peak_frequency.mp4")

##################################################
# ---------- helper index sets
midmask = (OMEGAS .> 0.08) .& (OMEGAS .< 8.0)

qA_lo = quantile(S_A, 0.15)
qA_hi  = quantile(S_A, 0.85)
qAm_lo = quantile(S_Amod, 0.25)
qAm_hi = quantile(S_Amod, 0.90)

# ---------- Case 1: weak -> weak
cand1 = findall((S_A .<= qA_lo) .& (S_Amod .<= qAm_lo) .& midmask)
idx_case1 = isempty(cand1) ? argmin((S_A .+ S_Amod) .+ 1e6 .* .!midmask) :
                             cand1[argmin(S_A[cand1] .+ S_Amod[cand1])]
ω_case1 = OMEGAS[idx_case1]

ts1, X1, f1 = simulate_forced_system(A,     b, ω_case1; forcing_start = 4.0)
ts2, X2, f2 = simulate_forced_system(A_mod, b, ω_case1; forcing_start = 4.0)

animate_case(joinpath(OUTDIR, "case1_weak_to_weak_original.mp4"),
    @sprintf("Case 1 — original: weak frequency stays weak (ω = %.3f)", ω_case1),
    A, S_A, OMEGAS, ω_case1, ts1, X1, f1, resp_ylim, forc_ylim)

animate_case(joinpath(OUTDIR, "case1_weak_to_weak_modified.mp4"),
    @sprintf("Case 1 — modified: weak frequency stays weak (ω = %.3f)", ω_case1),
    A_mod, S_Amod, OMEGAS, ω_case1, ts2, X2, f2, resp_ylim, forc_ylim)

# ---------- Case 2: weak -> strong  (strictly weak in original)
midmask = (OMEGAS .> 0.08) .& (OMEGAS .< 8.0)

qA_lo  = quantile(S_A, 0.25)
qAm_hi = quantile(S_Amod, 0.85)

weak_orig = findall((S_A .<= qA_lo) .& midmask)

if isempty(weak_orig)
    weak_orig = findall(midmask)
    @warn "No weak original frequencies found under the chosen threshold; falling back to all mid-range frequencies."
end

cand2 = [i for i in weak_orig if S_Amod[i] >= qAm_hi]

if !isempty(cand2)
    idx_case2 = cand2[argmax(S_Amod[cand2])]
else
    gains = S_Amod[weak_orig] ./ (S_A[weak_orig] .+ 1e-12)
    idx_case2 = weak_orig[argmax(gains)]
    @warn "No genuine weak→strong frequency found; using the weak original frequency with the largest uplift instead."
end

ω_case2 = OMEGAS[idx_case2]

println(@sprintf(
    "Case 2: ω = %.4f, S_A = %.4f, S_Amod = %.4f, ratio = %.3f",
    ω_case2, S_A[idx_case2], S_Amod[idx_case2], S_Amod[idx_case2] / S_A[idx_case2]
))

ts3, X3, f3 = simulate_forced_system(A,     b, ω_case2; forcing_start = 4.0)
ts4, X4, f4 = simulate_forced_system(A_mod, b, ω_case2; forcing_start = 4.0)

animate_case(joinpath(OUTDIR, "case2_weak_to_strong_original.mp4"),
    @sprintf("Case 2 — original: weak frequency (ω = %.3f)", ω_case2),
    A, S_A, OMEGAS, ω_case2, ts3, X3, f3, resp_ylim, forc_ylim)

animate_case(joinpath(OUTDIR, "case2_weak_to_strong_modified.mp4"),
    @sprintf("Case 2 — modified: same frequency becomes strong (ω = %.3f)", ω_case2),
    A_mod, S_Amod, OMEGAS, ω_case2, ts4, X4, f4, resp_ylim, forc_ylim)

# ---------- Case 3: weak -> weaker
cand3 = findall((S_A .<= qA_lo) .& (S_Amod .< S_A) .& midmask)
idx_case3 = isempty(cand3) ? argmax((S_A ./ (S_Amod .+ 1e-12)) .- 1e6 .* .!midmask) :
                             cand3[argmax(S_A[cand3] ./ (S_Amod[cand3] .+ 1e-12))]
ω_case3 = OMEGAS[idx_case3]

ts5, X5, f5 = simulate_forced_system(A,     b, ω_case3; forcing_start = 4.0)
ts6, X6, f6 = simulate_forced_system(A_mod, b, ω_case3; forcing_start = 4.0)

animate_case(joinpath(OUTDIR, "case3_weak_to_weaker_original.mp4"),
    @sprintf("Case 3 — original: weak frequency (ω = %.3f)", ω_case3),
    A, S_A, OMEGAS, ω_case3, ts5, X5, f5, resp_ylim, forc_ylim)

animate_case(joinpath(OUTDIR, "case3_weak_to_weaker_modified.mp4"),
    @sprintf("Case 3 — modified: same weak frequency becomes even weaker (ω = %.3f)", ω_case3),
    A_mod, S_Amod, OMEGAS, ω_case3, ts6, X6, f6, resp_ylim, forc_ylim)

# ---------- Case 4: strong -> strong
cand4 = findall((S_A .>= qA_hi) .& (S_Amod .>= qAm_hi) .& midmask)
idx_case4 = isempty(cand4) ? argmax(min.(S_A, S_Amod) .- 1e6 .* .!midmask) :
                             cand4[argmax(min.(S_A[cand4], S_Amod[cand4]))]
ω_case4 = OMEGAS[idx_case4]

ts7, X7, f7 = simulate_forced_system(A,     b, ω_case4; forcing_start = 4.0)
ts8, X8, f8 = simulate_forced_system(A_mod, b, ω_case4; forcing_start = 4.0)

animate_case(joinpath(OUTDIR, "case4_strong_to_strong_original.mp4"),
    @sprintf("Case 4 — original: strong frequency remains strong (ω = %.3f)", ω_case4),
    A, S_A, OMEGAS, ω_case4, ts7, X7, f7, resp_ylim, forc_ylim)

animate_case(joinpath(OUTDIR, "case4_strong_to_strong_modified.mp4"),
    @sprintf("Case 4 — modified: same frequency remains strong (ω = %.3f)", ω_case4),
    A_mod, S_Amod, OMEGAS, ω_case4, ts8, X8, f8, resp_ylim, forc_ylim)

# ---------- Case 5: strong -> stronger
cand5 = findall((S_A .>= qA_hi) .& (S_Amod .> S_A) .& midmask)
idx_case5 = isempty(cand5) ? argmax((S_Amod ./ (S_A .+ 1e-12)) .- 1e6 .* .!midmask) :
                             cand5[argmax(S_Amod[cand5] ./ (S_A[cand5] .+ 1e-12))]
ω_case5 = OMEGAS[idx_case5]

ts9, X9, f9 = simulate_forced_system(A,     b, ω_case5; forcing_start = 4.0)
ts10, X10, f10 = simulate_forced_system(A_mod, b, ω_case5; forcing_start = 4.0)

animate_case(joinpath(OUTDIR, "case5_strong_to_stronger_original.mp4"),
    @sprintf("Case 5 — original: strong frequency (ω = %.3f)", ω_case5),
    A, S_A, OMEGAS, ω_case5, ts9, X9, f9, resp_ylim, forc_ylim)

animate_case(joinpath(OUTDIR, "case5_strong_to_stronger_modified.mp4"),
    @sprintf("Case 5 — modified: same frequency becomes stronger (ω = %.3f)", ω_case5),
    A_mod, S_Amod, OMEGAS, ω_case5, ts10, X10, f10, resp_ylim, forc_ylim)

# ---------- Case 6: strong -> weaker
# ---------- Case 6: strong -> weaker  (strictly strong in original)
midmask = (OMEGAS .> 0.08) .& (OMEGAS .< 8.0)

qA_hi  = quantile(S_A, 0.85)
qAm_lo = quantile(S_Amod, 0.25)

strong_orig = findall((S_A .>= qA_hi) .& midmask)

# genuine strong->weaker candidates
cand6 = [i for i in strong_orig if S_Amod[i] <= qAm_lo]

if !isempty(cand6)
    # among true strong->weaker cases, pick the one with the largest drop
    idx_case6 = cand6[argmax(S_A[cand6] ./ (S_Amod[cand6] .+ 1e-12))]
else
    # no genuine strong->weaker exists; stay within strong original frequencies
    # and pick the one with the largest relative decrease
    idx_case6 = strong_orig[argmax(S_A[strong_orig] ./ (S_Amod[strong_orig] .+ 1e-12))]
    @warn "No genuine strong→weaker frequency found; using the strong original frequency with the largest drop instead."
end

ω_case6 = OMEGAS[idx_case6]

println(@sprintf(
    "Case 6: ω = %.4f, S_A = %.4f, S_Amod = %.4f, ratio = %.3f",
    ω_case6, S_A[idx_case6], S_Amod[idx_case6], S_Amod[idx_case6] / S_A[idx_case6]
))

ts11, X11, f11 = simulate_forced_system(A,     b, ω_case6; forcing_start = 4.0)
ts12, X12, f12 = simulate_forced_system(A_mod, b, ω_case6; forcing_start = 4.0)

animate_case(joinpath(OUTDIR, "case6_strong_to_weaker_original.mp4"),
    @sprintf("Case 6 — original: strong frequency (ω = %.3f)", ω_case6),
    A, S_A, OMEGAS, ω_case6, ts11, X11, f11, resp_ylim, forc_ylim)

animate_case(joinpath(OUTDIR, "case6_strong_to_weaker_modified.mp4"),
    @sprintf("Case 6 — modified: same frequency becomes weaker (ω = %.3f)", ω_case6),
    A_mod, S_Amod, OMEGAS, ω_case6, ts12, X12, f12, resp_ylim, forc_ylim)