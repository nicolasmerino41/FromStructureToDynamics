# ============================================================
# claim3_timescale_landscape_makie.jl
#
# Preliminary approach for Claim 3:
#   Structural sensitivity spans a landscape over admissible T.
#
# Admissible set:
#   log-timescales live in a centered 3D space (for 4 species),
#   bounded by a ball of radius rho_max.
#
# Summaries:
#   - peak of the RPR profile
#   - area of the RPR profile over log10(ω)
#
# Plots:
#   For each summary metric:
#     1. histogram over sampled admissible T
#     2. 2D section through homogeneous, best, and worst directions
#     3. profiles for homogeneous / mean-level / best / worst
#
# Makie only
# plotting code inside begin ... end, ending with display(fig)
# ============================================================

using LinearAlgebra
using Statistics
using Random
using CairoMakie
using Printf

# ------------------------------------------------------------
# Helpers: dynamics
# ------------------------------------------------------------
resolvent(A, T, ω) = inv(im * ω * T - A)

structured_sensitivity(A, T, P, ω) = opnorm(resolvent(A, T, ω) * P * resolvent(A, T, ω), 2)

function structured_profile(A, T, P, ωs)
    [structured_sensitivity(A, T, P, ω) for ω in ωs]
end

function trapz(x, y)
    s = 0.0
    for i in 1:length(x)-1
        s += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return s
end

peak_metric(profile_vals) = maximum(profile_vals)

function area_metric(profile_vals, logωs)
    trapz(logωs, profile_vals)
end

# ------------------------------------------------------------
# Helpers: model and perturbation class
# ------------------------------------------------------------
function coupled_blocks(α1, β1, α2, β2, ε)
    [
        -α1  -β1   ε    0.0;
         β1  -α1   0.0  ε;
         ε    0.0 -α2  -β2;
         0.0  ε    β2  -α2
    ]
end

# ------------------------------------------------------------
# Helpers: centered log-timescale space
# ------------------------------------------------------------

"""
Return an orthonormal basis B for the subspace of R^n with zero sum.
Then any centered log-timescale vector x can be written x = B*c,
where c is a coordinate vector of length n-1.
"""
function centered_basis(n::Int)
    M = Matrix{Float64}(I, n, n-1)
    for j in 1:n-1
        M[n, j] = -1.0
    end
    F = qr(M).Q
    B = Matrix(F[:, 1:n-1])
    return B
end

"""
Map coordinates c in centered log-timescale space to a diagonal T.
tau0 sets the geometric-mean timescale.
"""
function coords_to_T(c::AbstractVector, B::AbstractMatrix; tau0=1.0)
    x = B * c
    τ = tau0 .* exp.(x)
    return Diagonal(τ)
end

"""
Sample uniformly from a ball of radius rho in R^dim.
"""
function sample_ball(rng::AbstractRNG, dim::Int, rho::Float64)
    v = randn(rng, dim)
    v ./= norm(v)
    r = rho * rand(rng)^(1/dim)
    return r * v
end

"""
Sample N points in a ball of radius rho in R^dim.
"""
function sample_ball_points(rng::AbstractRNG, N::Int, dim::Int, rho::Float64)
    pts = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        pts[i] = sample_ball(rng, dim, rho)
    end
    return pts
end

# ------------------------------------------------------------
# Helpers: landscape analysis
# ------------------------------------------------------------

"""
Compute profiles and summary metrics for a collection of coordinate points.
"""
function evaluate_landscape(A, P, ωs, logωs, B, coords_list; tau0=1.0)
    N = length(coords_list)

    profiles = Vector{Vector{Float64}}(undef, N)
    peaks = zeros(N)
    areas = zeros(N)

    for i in 1:N
        T = coords_to_T(coords_list[i], B; tau0=tau0)
        prof = structured_profile(A, T, P, ωs)
        profiles[i] = prof
        peaks[i] = peak_metric(prof)
        areas[i] = area_metric(prof, logωs)
    end

    return profiles, peaks, areas
end

"""
Choose a 2D plane in coordinate space containing the origin and
spanned by the best and worst directions.
If best and worst are nearly collinear, complete with any orthogonal direction.
"""
function plane_from_best_worst(c_best::Vector{Float64}, c_worst::Vector{Float64})
    e1 = c_best / norm(c_best)

    v2 = c_worst - dot(c_worst, e1) * e1
    if norm(v2) < 1e-8
        # fallback: pick arbitrary direction orthogonal to e1
        candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        found = false
        for v in candidates
            v2 = v - dot(v, e1) * e1
            if norm(v2) > 1e-8
                found = true
                break
            end
        end
        !found && error("Could not construct plane basis.")
    end

    e2 = v2 / norm(v2)
    return e1, e2
end

"""
Project coordinate vector c onto plane basis (e1,e2).
Returns plane coordinates (a,b) and orthogonal residual norm.
"""
function project_to_plane(c::Vector{Float64}, e1::Vector{Float64}, e2::Vector{Float64})
    a = dot(c, e1)
    b = dot(c, e2)
    residual = norm(c - a*e1 - b*e2)
    return a, b, residual
end

"""
Evaluate summary metric on a grid over the disk section in the chosen plane.
metric_symbol ∈ (:peak, :area)
"""
function evaluate_section(A, P, ωs, logωs, B, e1, e2; rho=1.4, gridsize=181, tau0=1.0, metric_symbol=:peak)
    xs = range(-rho, rho, length=gridsize)
    ys = range(-rho, rho, length=gridsize)

    Z = fill(NaN, gridsize, gridsize)

    for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
        if x^2 + y^2 <= rho^2
            c = x .* e1 .+ y .* e2
            T = coords_to_T(c, B; tau0=tau0)
            prof = structured_profile(A, T, P, ωs)
            Z[iy, ix] = metric_symbol == :peak ? peak_metric(prof) : area_metric(prof, logωs)
        end
    end

    return collect(xs), collect(ys), Z
end

"""
Given a section landscape Z and a target value m, find a representative
point on the section whose value is closest to m.
"""
function representative_point_on_section(xs, ys, Z, target)
    best_err = Inf
    best_xy = (0.0, 0.0)
    best_val = NaN

    for ix in eachindex(xs), iy in eachindex(ys)
        z = Z[iy, ix]
        if !isnan(z)
            err = abs(z - target)
            if err < best_err
                best_err = err
                best_xy = (xs[ix], ys[iy])
                best_val = z
            end
        end
    end

    return best_xy, best_val
end

# ------------------------------------------------------------
# Model setup
# ------------------------------------------------------------

# 4-species system
A = coupled_blocks(
    0.24, 0.85,   # slow block
    0.20, 2.10,   # fast block
    0.14          # weak coupling
)

println("Eigenvalues of A:")
println(sort(eigvals(A), by = x -> imag(x)))

# Choose one structural perturbation class for Claim 3.
# This is the object whose induced sensitivity landscape we study.
# You can swap this for Pslow, Pfast, Pcross, or an aggregate.
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

# Chosen structural change:
Pstruct = Pcross

# ------------------------------------------------------------
# Timescale-space setup
# ------------------------------------------------------------

n = 4
B = centered_basis(n)   # 4 x 3 basis
dim = n - 1             # 3D centered log-timescale space

tau0 = 1.0              # geometric-mean timescale
rho_max = 1.35          # extent of admissible heterogeneity
rng = MersenneTwister(42)

# Frequency grid
ωs = 10 .^ range(-3, 2, length=900)
logωs = log10.(ωs)

# Sample admissible T-space
Nsamp = 2500
coords_samples = sample_ball_points(rng, Nsamp, dim, rho_max)

# Add homogeneous point explicitly as first item
c_ref = zeros(dim)
coords_all = vcat([c_ref], coords_samples)

profiles, peaks, areas = evaluate_landscape(A, Pstruct, ωs, logωs, B, coords_all; tau0=tau0)

# Indexing
idx_ref = 1
sample_range = 2:length(coords_all)

# Mean-case values over admissible set (including the reference is negligible either way)
peak_mean = mean(peaks[sample_range])
area_mean = mean(areas[sample_range])

# Best/worst
idx_peak_best = argmax(peaks[sample_range]) + 1
idx_peak_worst = argmin(peaks[sample_range]) + 1  # for clarity: "best"=largest sensitivity, "best shielding" is min
# We will rename below to avoid confusion.
idx_peak_max = idx_peak_best
idx_peak_min = idx_peak_worst

idx_area_max = argmax(areas[sample_range]) + 1
idx_area_min = argmin(areas[sample_range]) + 1

c_peak_max = coords_all[idx_peak_max]
c_peak_min = coords_all[idx_peak_min]

c_area_max = coords_all[idx_area_max]
c_area_min = coords_all[idx_area_min]

println("\nPeak metric:")
@printf("reference (homogeneous) = %.6f\n", peaks[idx_ref])
@printf("mean-case               = %.6f\n", peak_mean)
@printf("worst-case (max)        = %.6f\n", peaks[idx_peak_max])
@printf("best-case  (min)        = %.6f\n", peaks[idx_peak_min])

println("\nArea metric:")
@printf("reference (homogeneous) = %.6f\n", areas[idx_ref])
@printf("mean-case               = %.6f\n", area_mean)
@printf("worst-case (max)        = %.6f\n", areas[idx_area_max])
@printf("best-case  (min)        = %.6f\n", areas[idx_area_min])

# ------------------------------------------------------------
# Build 2D sections for peak and area
# ------------------------------------------------------------
# Peak section: plane spanned by max and min directions
e1_peak, e2_peak = plane_from_best_worst(c_peak_max, c_peak_min)
xs_peak, ys_peak, Z_peak = evaluate_section(A, Pstruct, ωs, logωs, B, e1_peak, e2_peak;
    rho=rho_max, gridsize=181, tau0=tau0, metric_symbol=:peak)

# Area section: plane spanned by max and min directions
e1_area, e2_area = plane_from_best_worst(c_area_max, c_area_min)
xs_area, ys_area, Z_area = evaluate_section(A, Pstruct, ωs, logωs, B, e1_area, e2_area;
    rho=rho_max, gridsize=181, tau0=tau0, metric_symbol=:area)

# Representative mean-level points on each section
(mean_xy_peak, mean_val_peak_on_section) = representative_point_on_section(xs_peak, ys_peak, Z_peak, peak_mean)
(mean_xy_area, mean_val_area_on_section) = representative_point_on_section(xs_area, ys_area, Z_area, area_mean)

# Coordinates of selected cases in each section
peak_ref_xy = (0.0, 0.0)
peak_max_xy = project_to_plane(c_peak_max, e1_peak, e2_peak)[1:2]
peak_min_xy = project_to_plane(c_peak_min, e1_peak, e2_peak)[1:2]

area_ref_xy = (0.0, 0.0)
area_max_xy = project_to_plane(c_area_max, e1_area, e2_area)[1:2]
area_min_xy = project_to_plane(c_area_min, e1_area, e2_area)[1:2]

# Recover T and profiles for representative mean-level points
c_peak_meanrep = mean_xy_peak[1] .* e1_peak .+ mean_xy_peak[2] .* e2_peak
T_peak_meanrep = coords_to_T(c_peak_meanrep, B; tau0=tau0)
profile_peak_meanrep = structured_profile(A, T_peak_meanrep, Pstruct, ωs)

c_area_meanrep = mean_xy_area[1] .* e1_area .+ mean_xy_area[2] .* e2_area
T_area_meanrep = coords_to_T(c_area_meanrep, B; tau0=tau0)
profile_area_meanrep = structured_profile(A, T_area_meanrep, Pstruct, ωs)

# Profiles for selected cases
profile_ref = profiles[idx_ref]
profile_peak_max = profiles[idx_peak_max]
profile_peak_min = profiles[idx_peak_min]

profile_area_max = profiles[idx_area_max]
profile_area_min = profiles[idx_area_min]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
begin
    fig = Figure(size = (1600, 980))

    # =========================
    # Row 1: peak metric
    # =========================

    # Histogram
    ax11 = Axis(
        fig[1, 1],
        xlabel = "peak of RPR profile",
        ylabel = "count",
        title = "Peak metric across admissible timescale configurations"
    )

    hist!(ax11, peaks[sample_range], bins = 40)
    vlines!(ax11, [peaks[idx_ref]], linestyle = :dash, linewidth = 3, label = "reference")
    vlines!(ax11, [peak_mean], linestyle = :dot, linewidth = 3, label = "mean-case")
    vlines!(ax11, [peaks[idx_peak_max]], linewidth = 3, label = "worst-case")
    vlines!(ax11, [peaks[idx_peak_min]], linewidth = 3, label = "best-case")
    axislegend(ax11, position = :rt)

    # 2D section
    ax12 = Axis(
        fig[1, 2],
        xlabel = "section coordinate 1",
        ylabel = "section coordinate 2",
        title = "Peak landscape on best–worst section",
        aspect = DataAspect()
    )

    hm12 = heatmap!(ax12, xs_peak, ys_peak, Z_peak)
    # Mask circle boundary visually
    θ = range(0, 2π, length=400)
    lines!(ax12, rho_max .* cos.(θ), rho_max .* sin.(θ), linewidth = 2)

    scatter!(ax12, [peak_ref_xy[1]], [peak_ref_xy[2]], markersize = 16, label = "reference")
    scatter!(ax12, [mean_xy_peak[1]], [mean_xy_peak[2]], markersize = 16, label = "mean-level rep.")
    scatter!(ax12, [peak_max_xy[1]], [peak_max_xy[2]], markersize = 16, label = "worst-case")
    scatter!(ax12, [peak_min_xy[1]], [peak_min_xy[2]], markersize = 16, label = "best-case")
    axislegend(ax12, position = :rb)

    # Profiles
    ax13 = Axis(
        fig[1, 3],
        xlabel = "ω",
        ylabel = "‖S(ω) P S(ω)‖₂",
        title = "RPR profiles: peak-selected cases",
        xscale = log10,
        yscale = log10
    )

    lines!(ax13, ωs, profile_ref, linewidth = 3, label = "reference")
    lines!(ax13, ωs, profile_peak_meanrep, linewidth = 3, label = "mean-level rep.")
    lines!(ax13, ωs, profile_peak_max, linewidth = 3, label = "worst-case")
    lines!(ax13, ωs, profile_peak_min, linewidth = 3, label = "best-case")
    axislegend(ax13, position = :lb)

    Colorbar(fig[1, 4], hm12, label = "peak")

    # =========================
    # Row 2: area metric
    # =========================

    # Histogram
    ax21 = Axis(
        fig[2, 1],
        xlabel = "area of RPR profile",
        ylabel = "count",
        title = "Area metric across admissible timescale configurations"
    )

    hist!(ax21, areas[sample_range], bins = 40)
    vlines!(ax21, [areas[idx_ref]], linestyle = :dash, linewidth = 3, label = "reference")
    vlines!(ax21, [area_mean], linestyle = :dot, linewidth = 3, label = "mean-case")
    vlines!(ax21, [areas[idx_area_max]], linewidth = 3, label = "worst-case")
    vlines!(ax21, [areas[idx_area_min]], linewidth = 3, label = "best-case")
    axislegend(ax21, position = :rt)

    # 2D section
    ax22 = Axis(
        fig[2, 2],
        xlabel = "section coordinate 1",
        ylabel = "section coordinate 2",
        title = "Area landscape on best–worst section",
        aspect = DataAspect()
    )

    hm22 = heatmap!(ax22, xs_area, ys_area, Z_area)
    lines!(ax22, rho_max .* cos.(θ), rho_max .* sin.(θ), linewidth = 2)

    scatter!(ax22, [area_ref_xy[1]], [area_ref_xy[2]], markersize = 16, label = "reference")
    scatter!(ax22, [mean_xy_area[1]], [mean_xy_area[2]], markersize = 16, label = "mean-level rep.")
    scatter!(ax22, [area_max_xy[1]], [area_max_xy[2]], markersize = 16, label = "worst-case")
    scatter!(ax22, [area_min_xy[1]], [area_min_xy[2]], markersize = 16, label = "best-case")
    axislegend(ax22, position = :rb)

    # Profiles
    ax23 = Axis(
        fig[2, 3],
        xlabel = "ω",
        ylabel = "‖S(ω) P S(ω)‖₂",
        title = "RPR profiles: area-selected cases",
        xscale = log10,
        yscale = log10
    )

    lines!(ax23, ωs, profile_ref, linewidth = 3, label = "reference")
    lines!(ax23, ωs, profile_area_meanrep, linewidth = 3, label = "mean-level rep.")
    lines!(ax23, ωs, profile_area_max, linewidth = 3, label = "worst-case")
    lines!(ax23, ωs, profile_area_min, linewidth = 3, label = "best-case")
    axislegend(ax23, position = :lb)

    Colorbar(fig[2, 4], hm22, label = "area")

    display(fig)
end