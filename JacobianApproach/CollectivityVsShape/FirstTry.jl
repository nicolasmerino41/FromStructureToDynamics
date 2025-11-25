using LinearAlgebra, Statistics, Random, CairoMakie

"""
    spectral_radius(A) -> Φ

Spectral radius Φ = maximum |λ_i| of matrix A.
Use this as your "collectivity" Φ(A).
"""
function spectral_radius(A::AbstractMatrix)
    vals = eigvals(Matrix(A))
    return maximum(abs.(vals))
end

"""
    leading_real_eigenvalue(J) -> Float64

Largest real part among eigenvalues of J.
This is (minus) asymptotic resilience in your setting.
"""
function leading_real_eigenvalue(J::AbstractMatrix)
    vals = eigvals(Matrix(J))
    return maximum(real.(vals))
end

"""
    henrici_departure(J) -> Float64

Henrici's departure from normality:

    η(J) = sqrt( ||J||_F^2 - sum_i |λ_i|^2 )

η(J) = 0 for normal matrices, > 0 for non-normal ones.
"""
function henrici_departure(J::AbstractMatrix)
    Jm = Matrix(J)
    F2 = sum(abs2, Jm)
    vals = eigvals(Jm)
    diff = F2 - sum(abs2, vals)
    # Numerical noise can make diff slightly negative
    diff < 0 && (diff = max(diff, 0.0))
    return sqrt(diff)
end

"""
    numerical_abscissa(J) -> Float64

Maximum real eigenvalue of H = (J + J')/2, the Hermitian part.
This is the "reactivity" in Arnoldi's terminology.
"""
function numerical_abscissa(J::AbstractMatrix)
    Jm = Matrix(J)
    H = (Jm + Jm') / 2
    vals = eigvals(H)
    return maximum(real.(vals))
end

"""
    max_singular_expJ(J, t) -> Float64

Largest singular value of exp(J t).
Measures worst-case transient amplification at time t.
"""
function max_singular_expJ(J::AbstractMatrix, t::Real)
    Jm = Matrix(J)
    M = exp(t * Jm)
    s = svdvals(M)
    return maximum(s)
end

"""
    rmed_time_series(J; tmax=10.0, nt=200, n_dirs=500)

Compute r_med(t) for a given Jacobian J.

Returns:
- ts        :: Vector{Float64}    times
- rmed      :: Vector{Float64}    r_med(t) at each time
- log_meds  :: Vector{Float64}    log of median norm (for debugging)

Implementation:
- Draw n_dirs random unit vectors in R^n
- For each t, compute x(t) = exp(J t) u, take norms, then median
- Approximate r_med(t) = - d/dt log(median norm) by finite differences
"""
function rmed_time_series(J::AbstractMatrix; tmax=10.0, nt=200, n_dirs=500)
    Jm = Matrix(J)
    n = size(Jm, 1)
    ts = collect(range(0.0, tmax; length=nt))

    # Random unit perturbation directions (columns)
    U = randn(n, n_dirs)
    for k in 1:n_dirs
        U[:, k] ./= norm(U[:, k])
    end

    log_meds = zeros(Float64, nt)

    for (i, t) in enumerate(ts)
        M = exp(t * Jm)
        X = M * U                    # n x n_dirs
        norms = sqrt.(sum(abs2, X; dims=1))  # 1 x n_dirs
        log_meds[i] = log(median(vec(norms)))
    end

    # Finite-difference derivative of log_meds
    rmed = similar(log_meds)
    # Forward at first point
    rmed[1] = -(log_meds[2] - log_meds[1]) / (ts[2] - ts[1])
    # Central differences
    for i in 2:(nt-1)
        rmed[i] = -(log_meds[i+1] - log_meds[i-1]) / (ts[i+1] - ts[i-1])
    end
    # Backward at last point
    rmed[end] = -(log_meds[end] - log_meds[end-1]) / (ts[end] - ts[end-1])

    return ts, rmed, log_meds
end

"""
    rewire_offdiagonal(J; rng=Random.GLOBAL_RNG)

Randomly permutes all off-diagonal entries of J; diagonal entries are kept intact.

- Preserves the multiset of off-diagonal values (including zeros)
- Preserves the diagonal exactly
"""
function rewire_offdiagonal(J::AbstractMatrix; rng=Random.GLOBAL_RNG)
    Jm = Matrix(J)
    n, m = size(Jm)
    n == m || error("J must be square")

    D = diagm(0 => diag(Jm))  # preserves diagonal

    # Collect off-diagonal entries
    offvals = Float64[]
    for i in 1:n, j in 1:n
        i == j && continue
        push!(offvals, Jm[i, j])
    end

    shuffle!(rng, offvals)

    K = copy(D)
    idx = 1
    for i in 1:n, j in 1:n
        i == j && continue
        K[i, j] = offvals[idx]
        idx += 1
    end

    return K
end

"""
    scramble_eigenvectors(J; rng=Random.GLOBAL_RNG)

Constructs J_scr = V * Λ * V⁻¹ with:

- Λ = Diagonal(eigvals(J))
- V = random invertible matrix

This preserves the eigenvalues of J but randomizes eigenvectors.
J_scr will generally be complex if J has complex eigenvalues.
"""
function scramble_eigenvectors(J::AbstractMatrix; rng=Random.GLOBAL_RNG)
    Jm = Matrix(J)
    n, m = size(Jm)
    n == m || error("J must be square")

    vals = eigvals(Jm)
    Λ = Diagonal(vals)

    # Random invertible matrix V
    V = randn(rng, n, n)
    # Very crude safeguard against near-singular V
    while abs(det(V)) < 1e-6
        V = randn(rng, n, n)
    end
    Vinv = inv(V)

    Jscr = V * Λ * Vinv
    return Jscr
end

"""
    scale_to_match_collectivity(A, target_Φ)

Return A_scaled = α * A so that spectral_radius(A_scaled) ≈ target_Φ.

If current Φ(A) == 0, returns A unchanged and prints a warning.
"""
function scale_to_match_collectivity(A::AbstractMatrix, target_Φ::Real)
    Am = Matrix(A)
    Φ = spectral_radius(Am)
    if Φ == 0
        @warn "scale_to_match_collectivity: current Φ = 0, cannot rescale."
        return Am
    end
    α = target_Φ / Φ
    return α .* Am
end

"""
    trophic_levels(B) -> s

Compute trophic levels s_i for each species i from adjacency matrix B.

Assumes B[i, j] > 0 means j consumes i (i -> j).
Basal species (no prey, column sum == 0) get s_i = 1 by definition.
For others: s_j = 1 + (1/k_j) * sum_i B[i, j] * s_i

Returns:
- s :: Vector{Float64} of length n
"""
function trophic_levels(B::AbstractMatrix)
    Bm = Matrix(B)
    n, m = size(Bm)
    n == m || error("B must be square")
    # k_in[j] = number (or weight) of prey for species j
    k_in = vec(sum(Bm; dims=1))
    s = ones(Float64, n)  # initialize all s_j ~ 1

    # Build linear system M s = b representing the above equations
    M = zeros(Float64, n, n)
    b = ones(Float64, n)

    for j in 1:n
        if k_in[j] == 0
            # basal: s_j = 1
            M[j, j] = 1.0
            b[j] = 1.0
        else
            M[j, j] = 1.0
            for i in 1:n
                if Bm[i, j] != 0.0
                    M[j, i] -= Bm[i, j] / k_in[j]
                end
            end
            b[j] = 1.0
        end
    end

    s = M \ b
    return s
end

"""
    trophic_coherence(B) -> q

Given adjacency matrix B (i -> j meaning j consumes i):

1. Compute trophic levels s.
2. For each edge (i, j) with B[i, j] != 0, define x_ij = s_j - s_i.
3. q = sqrt( mean( (x_ij - 1)^2 ) ).

Smaller q = more coherent trophic structure.
"""
function trophic_coherence(B::AbstractMatrix)
    Bm = Matrix(B)
    n, m = size(Bm)
    n == m || error("B must be square")
    s = trophic_levels(Bm)
    xs = Float64[]

    for i in 1:n, j in 1:n
        if Bm[i, j] != 0.0
            x = s[j] - s[i]
            push!(xs, x)
        end
    end

    if isempty(xs)
        return NaN  # no edges, coherence undefined
    end

    q = sqrt(mean((xs .- 1.0).^2))
    return q
end

"""
    compute_network_metrics(name, J; A=nothing, B_for_trophic=nothing, t_trans=1.0)

Return a NamedTuple with key structural/dynamical metrics:

- :name
- :Φ             (collectivity from A if provided, else missing)
- :henrici       (departure from normality of J)
- :num_abscissa  (numerical abscissa of J)
- :leading_real  (leading real part of eigenvalues of J)
- :trophic_q     (trophic coherence q(B) if B given, else missing)
- :max_sing_exp  (largest singular value of exp(J*t_trans))
"""
function compute_network_metrics(
        name::AbstractString,
        J::AbstractMatrix;
        A::Union{AbstractMatrix,Nothing}=nothing,
        B_for_trophic::Union{AbstractMatrix,Nothing}=nothing,
        t_trans::Real=1.0
    )

    Φ = isnothing(A) ? missing : spectral_radius(A)
    η = henrici_departure(J)
    w = numerical_abscissa(J)
    λ_dom = leading_real_eigenvalue(J)
    q = isnothing(B_for_trophic) ? missing : trophic_coherence(B_for_trophic)
    σexp = max_singular_expJ(J, t_trans)

    return (
        name = String(name),
        Φ = Φ,
        henrici = η,
        num_abscissa = w,
        leading_real = λ_dom,
        trophic_q = q,
        max_sing_exp = σexp,
    )
end

"""
    plot_rmed_curves(ts, r_series, labels; outfile="rmed_curves.png")

- ts       :: Vector{Float64}
- r_series :: Vector of Vector{Float64} (one r_med curve per network)
- labels   :: Vector of strings same length as r_series
"""
function plot_rmed_curves(ts, r_series, labels; outfile="rmed_curves.png")
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "t", ylabel = "r_med(t)")

    for (r, lab) in zip(r_series, labels)
        lines!(ax, ts, r, label = lab)
    end

    axislegend(ax)
    # save(outfile, fig)
    display(fig)
end

"""
    plot_scatter(xs, ys; xlabel="", ylabel="", outfile="scatter.png")

Simple scatter plot of ys vs xs using CairoMakie.
"""
function plot_scatter(xs, ys; xlabel="", ylabel="", outfile="scatter.png")
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = xlabel, ylabel = ylabel)
    scatter!(ax, xs, ys)
    # save(outfile, fig)
    display(fig)
end

"""
    regress_delta_rpeak(dΦ, dη, dλ, dq, dr_peak)

Perform simple linear regression:

    Δr_peak ≈ β0 + β1 ΔΦ + β2 Δη + β3 Δλ_dom + β4 Δq

Arguments are vectors of equal length.
Returns vector β of length 5.
"""
function regress_delta_rpeak(dΦ, dη, dλ, dq, dr_peak)
    n = length(dr_peak)
    @assert length(dΦ) == n
    @assert length(dη) == n
    @assert length(dλ) == n
    @assert length(dq) == n

    X = hcat(ones(n), dΦ, dη, dλ, dq)
    β = X \ dr_peak   # least-squares solution
    return β
end

# ============================
# Example usage / pipeline
# ============================
function generate_synthetic_J(n::Int; rng=Random.GLOBAL_RNG)
    # Simple synthetic community matrix:
    # - diagonal negative (self-regulation)
    # - off-diagonal sparse random
    D = -0.5 .- rand(rng, n)          # random negative diagonals
    J = zeros(Float64, n, n)
    for i in 1:n
        J[i, i] = D[i]
    end
    # Off-diagonal interactions (sparse)
    p = 0.1
    for i in 1:n, j in 1:n
        i == j && continue
        if rand(rng) < p
            J[i, j] = 0.1 * (2rand(rng) - 1)   # small random interaction
        end
    end
    return J
end

"""
    example_pipeline()

- Generate a base J
- Create rewired and eigenvector-scrambled versions
- Compute r_med(t) for each
- Compute structural metrics
- Make a couple of plots
"""
function example_pipeline()
    rng = MersenneTwister(42)
    n = 20

    # Base matrix
    J_base = generate_synthetic_J(n; rng=rng)

    # Rewired (diag fixed)
    J_rewired = rewire_offdiagonal(J_base; rng=rng)

    # Eigenvector-scrambled (may be complex)
    J_scrambled = scramble_eigenvectors(J_base; rng=rng)

    # For this synthetic example, just use sign structure as "B"
    B_base = Float64.(J_base .> 0.0)

    # Metrics
    m_base = compute_network_metrics("base", J_base; B_for_trophic=B_base)
    m_rewired = compute_network_metrics("rewired", J_rewired; B_for_trophic=B_base)
    m_scrambled = compute_network_metrics("scrambled", J_scrambled; B_for_trophic=B_base)

    println("Base metrics:      ", m_base)
    println("Rewired metrics:   ", m_rewired)
    println("Scrambled metrics: ", m_scrambled)

    # r_med(t) curves
    tmax = 10.0
    nt = 200
    n_dirs = 500

    ts, r_base, _ = rmed_time_series(J_base; tmax=tmax, nt=nt, n_dirs=n_dirs)
    _,  r_rew,  _ = rmed_time_series(J_rewired; tmax=tmax, nt=nt, n_dirs=n_dirs)
    _,  r_scr,  _ = rmed_time_series(J_scrambled; tmax=tmax, nt=nt, n_dirs=n_dirs)

    # Plot r_med curves
    plot_rmed_curves(ts,
                     [r_base, r_rew, r_scr],
                     ["base", "rewired", "scrambled"];
                     outfile = "rmed_example.png")

    # For a crude "peak Δr_med" comparison between base and rewired
    Δr_rew = r_rew .- r_base
    Δr_scr = r_scr .- r_base
    idx_peak_rew = argmax(abs.(Δr_rew))
    idx_peak_scr = argmax(abs.(Δr_scr))

    println("Peak |Δr_med| (rewired):   ", Δr_rew[idx_peak_rew], " at t = ", ts[idx_peak_rew])
    println("Peak |Δr_med| (scrambled): ", Δr_scr[idx_peak_scr], " at t = ", ts[idx_peak_scr])

    # Example scatter: Δhenrici vs Δr_peak (for these two manipulations)
    Δη = [m_rewired.henrici - m_base.henrici,
          m_scrambled.henrici - m_base.henrici]
    Δr_peak = [Δr_rew[idx_peak_rew], Δr_scr[idx_peak_scr]]

    plot_scatter(Δη, Δr_peak;
                 xlabel = "Δ Henrici (non-normality)",
                 ylabel = "Δ r_med_peak",
                 outfile = "scatter_non_normality_vs_drpeak.png")
end

# Call the example if this file is run as a script
example_pipeline()
