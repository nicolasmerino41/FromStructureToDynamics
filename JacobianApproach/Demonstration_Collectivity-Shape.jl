##############################################################
# SCRIPT 2 — Full integrated pipeline demonstration (ALL cases)
##############################################################
using LinearAlgebra, Random, Statistics
using CairoMakie

# ------------------------------------------------------------
# Assumed available in your environment:
#   jacobian(A,u)
#   rewire_pairs_preserving_values(A; rng, random_targets=true)
#   rho_match
#   compute_rmed_series_stable(J,u,t_vals)
#   random_u
#   build_trophic_ER   (or similar trophic builder)
#
# This script uses *your* versions of these functions.
# ------------------------------------------------------------

# ------------------------------------------------------------
# CANONICAL MATRIX BUILDERS (standalone)
# ------------------------------------------------------------

function build_symmetric_MF(S; a=0.2)
    A = fill(a, S, S)
    for i in 1:S
        A[i,i] = 0
    end
    return (A + A')/2
end

function build_skew_MF(S; a=0.2)
    A = zeros(S,S)
    for i in 1:S, j in i+1:S
        A[i,j] =  a
        A[j,i] = -a
    end
    return A
end

function build_triangular(S; a=0.2)
    A = zeros(S,S)
    for i in 1:S, j in i+1:S
        A[i,j] = a
    end
    return A
end

function build_random_matrix(S; a=0.2, rng=Xoshiro(7))
    A = a * randn(rng, S, S)
    for i in 1:S
        A[i,i] = 0
    end
    return A
end

# synthetic trophic mixture (standalone fallback if needed)
function build_trophic_mixed(S; rng=Xoshiro(1234), a=0.2)
    A = zeros(S,S)

    # feed-forward chain backbone
    for i in 1:S, j in i+1:S
        A[i,j] = a*rand(rng)
    end

    # predator–prey skew pairs
    for _ in 1:round(Int, S/3)
        i, j = rand(rng, 1:S, 2)
        A[i,j] =  a
        A[j,i] = -a
    end

    # noise
    A .+= 0.2a*randn(rng, S, S)
    A[diagind(A)] .= 0

    return A
end

# ------------------------------------------------------------
# NON-NORMALITY METRIC
# ------------------------------------------------------------
nonnorm(A) = norm(A'A - A*A')
rhoA(A)    = maximum(abs.(eigvals(Matrix(A))))

# ------------------------------------------------------------
# EXPERIMENT PARAMETERS
# ------------------------------------------------------------
S         = 50
rng       = Xoshiro(2024)
t_vals    = 10 .^ range(-2,2; length=40)
u         = random_u(S; mean=1.0, cv=0.5, rng=rng)

# ------------------------------------------------------------
# DEFINE ALL TEST CASES
# ------------------------------------------------------------
canonical_cases = Dict(
    "Symmetric MF"  => build_symmetric_MF(S),
    "Skew MF"       => build_skew_MF(S),
    "Triangular"    => build_triangular(S),
    "Random"        => build_random_matrix(S, rng=rng),
    "Trophic mix"   => build_trophic_mixed(S, rng=rng)
)

# ------------------------------------------------------------
# RUN PIPELINE FOR EACH CASE
# ------------------------------------------------------------
results = Dict()

for (name, A) in canonical_cases

    println("\\n====================")
    println("Case: $name")
    println("====================")

    # baseline
    J0  = jacobian(A,u)
    f0  = compute_rmed_series_stable(J0,u,t_vals)

    # raw rewiring
    A_raw = rewire_pairs_preserving_values(A; rng=rng, random_targets=true, preserving_pairs=false)
    J_raw = jacobian(A_raw,u)
    f_raw = compute_rmed_series_stable(J_raw,u,t_vals)

    # rho-match
    ρA = rhoA(A)
    A_rmatch = rho_match(A_raw, ρA)
    J_rmatch = jacobian(A_rmatch,u)
    f_rmatch = compute_rmed_series_stable(J_rmatch,u,t_vals)

    # diagnostics
    N0      = nonnorm(A)
    N_raw   = nonnorm(A_raw)
    N_rmatch= nonnorm(A_rmatch)

    println("  non-normality baseline:  $N0")
    println("  non-normality raw:       $N_raw")
    println("  non-normality r-match:   $N_rmatch")

    # deltas
    delta_raw    = abs.(f_raw    .- f0)
    delta_rmatch = abs.(f_rmatch .- f0)

    results[name] = (
        A=A, A_raw=A_raw, A_rmatch=A_rmatch,
        J0=J0, J_raw=J_raw, J_rmatch=J_rmatch,
        f0=f0, f_raw=f_raw, f_rmatch=f_rmatch,
        delta_raw=delta_raw, delta_rmatch=delta_rmatch,
        nonnorm=(N0, N_raw, N_rmatch),
        rho=(ρA, rhoA(A_raw), rhoA(A_rmatch)),
        t=t_vals
    )
end

# ------------------------------------------------------------
# PLOT FOR EACH CASE
# ------------------------------------------------------------
for (name, R) in results

    fig = Figure(; size=(1100,800))
    ax1 = Axis(fig[1,1], xscale=log10,
               xlabel="t", ylabel="|Δ r̃_med(t)|",
               title="Collectivity–Shape Demonstration: $name")

    lines!(ax1, R[:t], R[:delta_raw],
           color=:crimson, linewidth=3, label="raw rewiring")

    lines!(ax1, R[:t], R[:delta_rmatch],
           color=:forestgreen, linewidth=3, label="ρ-matched")

    axislegend(ax1, position=:lt)

    ax2 = Axis(fig[2,1], xscale=log10,
               xlabel="t", ylabel="r̃_med(t)")

    lines!(ax2, R[:t], R[:f0],         color=:black, linewidth=2, label="baseline")
    lines!(ax2, R[:t], R[:f_raw],      color=:crimson, linewidth=2, linestyle=:dash, label="raw")
    lines!(ax2, R[:t], R[:f_rmatch],   color=:forestgreen, linewidth=2, linestyle=:dash, label="ρ-match")

    axislegend(ax2, position=:rb)

    display(fig)
end
