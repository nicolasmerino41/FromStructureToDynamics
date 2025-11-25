#############################
# ensemble_foodweb_analysis.jl
#############################
using LinearAlgebra
using Statistics
using Random
using CairoMakie

# -------------------------------------------------------------
# 1. Basic linear algebra helpers
# -------------------------------------------------------------
"Spectral radius Φ(A) = max |λ_i|."
function spectral_radius(A::AbstractMatrix)
    vals = eigvals(Matrix(A))
    return maximum(abs.(vals))
end

"Largest real part among eigenvalues of J."
function leading_real_eigenvalue(J::AbstractMatrix)
    vals = eigvals(Matrix(J))
    return maximum(real.(vals))
end

"""
Henrici's departure from normality:

    η(J) = sqrt( ||J||_F^2 - sum_i |λ_i|^2 )
"""
function henrici_departure(J::AbstractMatrix)
    Jm = Matrix(J)
    F2 = sum(abs2, Jm)
    vals = eigvals(Jm)
    diff = F2 - sum(abs2, vals)
    diff < 0 && (diff = max(diff, 0.0))  # clamp small negatives
    return sqrt(diff)
end

"Numerical abscissa (reactivity): max Re eigenvalue of (J + J')/2."
function numerical_abscissa(J::AbstractMatrix)
    Jm = Matrix(J)
    H = (Jm + Jm') / 2
    vals = eigvals(H)
    return maximum(real.(vals))
end

"Max singular value of exp(J t) – worst–case transient amplification at time t."
function max_singular_expJ(J::AbstractMatrix, t::Real)
    Jm = Matrix(J)
    M = exp(t * Jm)
    s = svdvals(M)
    return maximum(s)
end

# -------------------------------------------------------------
# 2. r_med(t) computation
# -------------------------------------------------------------
"""
    rmed_time_series(J; tmax=10.0, nt=200, n_dirs=500)

Monte Carlo estimate of r_med(t):

- draw n_dirs random unit vectors u
- x(t) = exp(J t) u
- take median ||x(t)|| over directions
- r_med(t) = - d/dt log median(||x(t)||)
"""
"""
    rmed_time_series(J; tmax=10.0, nt=200, n_dirs=500,
                     perturbation=:uniform, biomass=nothing)

Compute r_med(t) with two options for how random perturbations are generated:

• perturbation = :uniform  → standard unit-norm random directions
• perturbation = :biomass → perturbations scaled by biomass vector

In the biomassed case, you must supply `biomass::Vector` of length n.
Each perturbation direction u is drawn as:
      u_i ∝ biomass[i] * randn()
Then normalized so that  ∑ (u_i^2 / biomass[i]^2) = 1 
→ ensures perturbations have unit norm **in biomass-weighted space**.
"""
function rmed_time_series(
    J::AbstractMatrix;
    tmax::Real = 10.0,
    nt::Int = 200,
    n_dirs::Int = 500,
    perturbation::Symbol = :biomass,  # :uniform or :biomass
    biomass::Union{Nothing,AbstractVector} = nothing
)
    Jm = Matrix(J)
    n = size(Jm, 1)
    ts = collect(range(0.0, tmax; length=nt))

    # ----------------------------------------------------
    # Generate perturbation directions
    # ----------------------------------------------------
    U = zeros(Float64, n, n_dirs)

    if perturbation === :uniform
        # Standard unit directions in Euclidean metric
        for k in 1:n_dirs
            u = randn(n)
            U[:, k] = u / norm(u)
        end

    elseif perturbation === :biomass
        isnothing(biomass) && error("biomass vector must be provided for perturbation=:biomass")
        length(biomass) == n || error("biomass vector must match size of J")

        # Construct biomass-weighted directions:
        # u_i = biomass[i] * randn()   and normalize in weighted norm:
        #    norm² = ∑ (u_i^2 / biomass[i]^2)
        for k in 1:n_dirs
            u = biomass .* randn(n)             # weighted noise
            w = sqrt(sum((u ./ biomass).^2))    # weighted norm
            U[:, k] = u ./ w                    # enforce unit weighted-norm
        end

    else
        error("Unknown perturbation type. Use :uniform or :biomass")
    end

    # ----------------------------------------------------
    # Time evolution and median norm
    # ----------------------------------------------------
    log_meds = zeros(Float64, nt)

    for (i, t) in enumerate(ts)
        M = exp(t * Jm)
        X = M * U

        if perturbation === :uniform
            norms = sqrt.(sum(abs2, X; dims=1))

        elseif perturbation === :biomass
            # Weighted norm, but keep it real:
            norms = sqrt.(sum(abs2.(X) ./ (biomass .^ 2); dims=1))
        end
        log_meds[i] = log(median(vec(norms)))
    end

    # ----------------------------------------------------
    # Compute r_med(t) = -d/dt [log median(norm)]
    # ----------------------------------------------------
    rmed = similar(log_meds)
    rmed[1] = -(log_meds[2] - log_meds[1]) / (ts[2] - ts[1])
    for i in 2:(nt - 1)
        rmed[i] = -(log_meds[i+1] - log_meds[i-1]) / (ts[i+1] - ts[i-1])
    end
    rmed[end] = -(log_meds[end] - log_meds[end-1]) / (ts[end] - ts[end-1])

    return ts, rmed, log_meds
end

# -------------------------------------------------------------
# 3. Rewiring and eigenvector scrambling
# -------------------------------------------------------------
"Randomly permute off–diagonal entries, keep diagonal intact."
function rewire_offdiagonal(J::AbstractMatrix; rng=Random.GLOBAL_RNG)
    Jm = Matrix(J)
    n, m = size(Jm)
    n == m || error("J must be square")

    D = diagm(0 => diag(Jm))   # keep diagonal

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
Scramble eigenvectors, keep eigenvalues:

J_scr = V Λ V⁻¹, with Λ from eigvals(J) and random invertible V.

Note: J_scr will usually be complex if J has complex eigenvalues.
"""
function scramble_eigenvectors(J::AbstractMatrix; rng=Random.GLOBAL_RNG)
    Jm = Matrix(J)
    n, m = size(Jm)
    n == m || error("J must be square")

    vals = eigvals(Jm)
    Λ = Diagonal(vals)

    V = randn(rng, n, n)
    while abs(det(V)) < 1e-6
        V = randn(rng, n, n)
    end
    Vinv = inv(V)

    return V * Λ * Vinv
end

"Extract interaction matrix A as the off–diagonal part of J."
function extract_A(J::AbstractMatrix)
    Jm = Matrix(J)
    n = size(Jm, 1)
    return Jm .- diagm(0 => diag(Jm))  # zero out diagonal
end

# -------------------------------------------------------------
# 4. Trophic coherence – B is recomputed for every J
# -------------------------------------------------------------
"""
    build_B(J; mode=:any_nonzero)

Construct adjacency matrix B from J.

- mode = :any_nonzero  → edge if J[i,j] != 0
- mode = :positive     → edge if J[i,j] > 0
- mode = :negative     → edge if J[i,j] < 0
Adapt this to your own predator–prey convention.
"""
function build_B(J::AbstractMatrix; mode::Symbol = :any_nonzero)
    Jm = Matrix(J)
    n, m = size(Jm)
    n == m || error("J must be square")
    B = zeros(Float64, n, n)

    for i in 1:n, j in 1:n
        i == j && continue
        v = real(Jm[i, j])
        if mode === :any_nonzero
            if v != 0
                B[i, j] = 1.0
            end
        elseif mode === :positive
            if v > 0
                B[i, j] = 1.0
            end
        elseif mode === :negative
            if v < 0
                B[i, j] = 1.0
            end
        else
            error("Unknown mode $mode")
        end
    end
    return B
end

"Compute trophic levels s_i from adjacency B (i -> j meaning j consumes i)."
function trophic_levels(B::AbstractMatrix)
    Bm = Matrix(B)
    n, m = size(Bm)
    n == m || error("B must be square")

    k_in = vec(sum(Bm; dims=1))
    s = ones(Float64, n)

    M = zeros(Float64, n, n)
    b = ones(Float64, n)

    for j in 1:n
        if k_in[j] == 0
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

    ϵ = 1e-8
    return (M + ϵ*I) \ b
end

"""
    trophic_coherence(B) -> q

q = sqrt( mean( (s_j - s_i - 1)^2 ) over all edges (i,j) ).
Smaller q means more coherent trophic structure.
"""
function trophic_coherence(B::AbstractMatrix)
    Bm = Matrix(B)
    n, m = size(Bm)
    n == m || error("B must be square")
    s = trophic_levels(Bm)
    xs = Float64[]

    for i in 1:n, j in 1:n
        if Bm[i, j] != 0.0
            push!(xs, s[j] - s[i])
        end
    end

    isempty(xs) && return NaN
    return sqrt(mean((xs .- 1.0).^2))
end

# -------------------------------------------------------------
# 5. Per–network metrics (B recomputed per J)
# -------------------------------------------------------------
"""
    compute_network_metrics(name, J; B_mode=:any_nonzero, t_trans=1.0)

Returns NamedTuple with:
- :name
- :Φ             (spectral radius of A = off-diagonal(J))
- :henrici
- :num_abscissa
- :leading_real
- :trophic_q     (NaN if J is complex)
- :max_sing_exp  (max singular value of exp(J t_trans))
"""
function compute_network_metrics(
        name::AbstractString,
        J::AbstractMatrix;
        B_mode::Symbol = :any_nonzero,
        t_trans::Real = 1.0
    )

    Jm = Matrix(J)
    A = extract_A(Jm)
    Φ = spectral_radius(A)
    η = henrici_departure(Jm)
    w = numerical_abscissa(Jm)
    λ_dom = leading_real_eigenvalue(Jm)
    if eltype(Jm) <: Real
        B = build_B(Jm; mode=B_mode)
        q = trophic_coherence(B)
    else
        q = NaN
    end
    σexp = max_singular_expJ(Jm, t_trans)

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

# -------------------------------------------------------------
# 6. Results container and regression
# -------------------------------------------------------------
"Create a Named Dict-of-vectors to hold ensemble results."
function init_results()
    return Dict(
        :variant  => String[],
        :base_name => String[],
        :Δr_peak  => Float64[],
        :ΔΦ       => Float64[],
        :Δη       => Float64[],
        :Δλ       => Float64[],
        :Δq       => Float64[],
        :Δdiag    => Float64[],
        :ΔH       => Float64[],
    )
end

"""
    regress_delta_rpeak(results)

Multiple linear regression:

Δr_peak ~ 1 + ΔΦ + Δη + Δλ + Δq + Δdiag + ΔH

Returns vector β of length 7.
"""
function regress_delta_rpeak(results::Dict)
    Δr    = results[:Δr_peak]
    ΔΦ    = results[:ΔΦ]
    Δη    = results[:Δη]
    Δλ    = results[:Δλ]
    Δq    = results[:Δq]
    Δdiag = results[:Δdiag]
    ΔH    = results[:ΔH]

    n = length(Δr)
    X = hcat(ones(n), ΔΦ, Δη, Δλ, Δq, Δdiag, ΔH)

    keep = BitVector(undef, n)
    for i in 1:n
        keep[i] = isfinite(Δr[i]) && all(isfinite, X[i, :])
    end

    Xf = X[keep, :]
    yf = Δr[keep]

    β = Xf \ yf
    return β
end

# -------------------------------------------------------------
# 7. Ensemble loop: rewiring & scrambling over many webs
# -------------------------------------------------------------
"""
    run_ensemble(base_Js, base_names;
                 n_rewires=10, n_scrambled=10,
                 tmax=10.0, nt=200, n_dirs=500,
                 t_window=(0.5,5.0),
                 B_mode=:any_nonzero)

base_Js    :: Vector of J matrices (your real community matrices)
base_names :: Vector of names (same length as base_Js)

For each base J, create n_rewires rewired matrices and
n_scrambled eigenvector-scrambled matrices.

For each variant, compute:
- Δr_peak   (difference in r_med inside chosen time window)
- ΔΦ, Δη, Δλ, Δq
- Δdiag (norm of diag difference)
- ΔH    (Frobenius norm of symmetric-part difference)

Returns:
- results :: Dict-of-vectors with all rows
- ts      :: time grid used for r_med(t)
"""
function run_ensemble(
        base_Js::Vector{<:AbstractMatrix},
        base_names::Vector{<:AbstractString};
        n_rewires::Int = 10,
        n_scrambled::Int = 10,
        tmax::Real = 10.0,
        nt::Int = 200,
        n_dirs::Int = 500,
        t_window::Tuple{Real,Real} = (0.5, 5.0),
        B_mode::Symbol = :any_nonzero,
        rng = MersenneTwister(42),
        store_rmed::Bool = true    # <<< NEW OPTION
    )

    length(base_Js) == length(base_names) ||
        error("base_Js and base_names must have same length")

    results = init_results()
    ts_global = nothing

    # NEW: container for full time-series
    results_rmed = Dict{Tuple{String,String}, Tuple{Vector{Float64}, Vector{Float64}}}()

    for (J_base, name) in zip(base_Js, base_names)
        Jb = Matrix(J_base)
        u = abs.(diag(Jb))              # biomass vector

        # ------------------------ Base ------------------------
        ts, r_base, _ = rmed_time_series(
            Jb; tmax=tmax, nt=nt, n_dirs=n_dirs,
            perturbation=:biomass, biomass=u
        )
        ts_global = ts

        if store_rmed
            results_rmed[(name, "base")] = (ts, r_base)
        end

        m_base = compute_network_metrics(name, Jb; B_mode=B_mode)
        diag_base = diag(Jb)
        H_base = (Jb + Jb') / 2
        inds = findall(t_window[1] .<= ts .<= t_window[2])

        # ------------------------ Rewired ------------------------
        for k in 1:n_rewires
            J_rew = rewire_offdiagonal(Jb; rng=rng)
            ts_rew, r_rew, _ = rmed_time_series(
                J_rew; tmax=tmax, nt=nt, n_dirs=n_dirs,
                perturbation=:biomass, biomass=u
            )
            if store_rmed
                results_rmed[(name, "rewired_$k")] = (ts_rew, r_rew)
            end

            Δr = r_rew .- r_base
            idx = !isempty(inds) ? inds[argmax(abs.(Δr[inds]))] : argmax(abs.(Δr))
            Δr_peak = Δr[idx]

            m_rew = compute_network_metrics(name, J_rew)
            ΔΦ = m_rew.Φ - m_base.Φ
            Δη = m_rew.henrici - m_base.henrici
            Δλ = m_rew.leading_real - m_base.leading_real
            Δq = m_rew.trophic_q - m_base.trophic_q
            Δdiag = norm(diag(J_rew) - diag_base)
            ΔH = norm((J_rew+J_rew')/2 - H_base)

            push!(results[:variant], "rewired")
            push!(results[:base_name], name)
            push!(results[:Δr_peak], Δr_peak)
            push!(results[:ΔΦ], ΔΦ)
            push!(results[:Δη], Δη)
            push!(results[:Δλ], Δλ)
            push!(results[:Δq], Δq)
            push!(results[:Δdiag], Δdiag)
            push!(results[:ΔH], ΔH)
        end

        # ------------------------ Scrambled ------------------------
        for k in 1:n_scrambled
            J_scr = scramble_eigenvectors(Jb; rng=rng)
            ts_scr, r_scr, _ = rmed_time_series(
                J_scr; tmax=tmax, nt=nt, n_dirs=n_dirs,
                perturbation=:biomass, biomass=u
            )
            if store_rmed
                results_rmed[(name, "scrambled_$k")] = (ts_scr, r_scr)
            end

            Δr = r_scr .- r_base
            idx = !isempty(inds) ? inds[argmax(abs.(Δr[inds]))] : argmax(abs.(Δr))
            Δr_peak = Δr[idx]

            m_scr = compute_network_metrics(name, J_scr)
            ΔΦ = m_scr.Φ - m_base.Φ
            Δη = m_scr.henrici - m_base.henrici
            Δλ = m_scr.leading_real - m_base.leading_real
            Δq = m_scr.trophic_q - m_base.trophic_q
            Δdiag = norm(diag(J_scr) - diag_base)
            ΔH = norm((Matrix(J_scr)+Matrix(J_scr)')/2 - H_base)

            push!(results[:variant], "scrambled")
            push!(results[:base_name], name)
            push!(results[:Δr_peak], Δr_peak)
            push!(results[:ΔΦ], ΔΦ)
            push!(results[:Δη], Δη)
            push!(results[:Δλ], Δλ)
            push!(results[:Δq], Δq)
            push!(results[:Δdiag], Δdiag)
            push!(results[:ΔH], ΔH)
        end
    end

    return (results=results, ts=ts_global, results_rmed=results_rmed)
end

# -------------------------------------------------------------
# 8. Makie plotting helpers for regression-style plots
# -------------------------------------------------------------
"Scatter Δr_peak vs chosen predictor, coloured by variant."
function plot_scatter_by_variant(results::Dict, x_key::Symbol;
                                 xlabel::String = "",
                                 outfile::String = "scatter.png")

    xs = copy(results[x_key])
    ys = copy(results[:Δr_peak])
    labels = copy(results[:variant])

    # ----- Filter Δq > 5.0 safely -----
    if x_key == :Δq
        keep = map(x -> x ≤ 5.0, xs)
        xs = xs[keep]
        ys = ys[keep]
        labels = labels[keep]
    end

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = xlabel, ylabel = "Δ r_med_peak")

    uniq = unique(labels)
    for lab in uniq
        idx = findall(==(lab), labels)
        scatter!(ax, xs[idx], ys[idx], label = lab)
    end

    axislegend(ax)
    display(fig)
end

# -------------------------------------------------------------
# 9. Example stub for plugging in your real food webs
# -------------------------------------------------------------
"""
Replace this with code that loads your real J matrices.
Here it's just a tiny random example.
"""
function load_real_foodwebs_stub()
    rng = MersenneTwister(123)
    n = 20
    function random_J()
        D = -0.5 .- rand(rng, n)
        J = zeros(Float64, n, n)
        for i in 1:n
            J[i, i] = D[i]
        end
        p = 0.1
        for i in 1:n, j in 1:n
            i == j && continue
            if rand(rng) < p
                J[i, j] = 0.1 * (2rand(rng) - 1)
            end
        end
        return J
    end
    base_Js = [random_J(), random_J()]
    base_names = ["web1", "web2"]
    return base_Js, base_names
end

# -------------------------------------------------------------
# REALISTIC FOOD WEB BUILDER (replace your stub)
# -------------------------------------------------------------
"""
Generate multiple realistic trophic food webs.

Features:
  • S species, grouped into TL trophic levels
  • Feeding only allowed from level k to k+1 (but optional omnivory)
  • Interaction strength decays with trophic efficiency
  • Negative diagonals for self-regulation (Jacobian-ready)
  • Add omnivory & competition stochastically
Return:
    base_Js :: Vector{Matrix}
    base_names :: Vector{String}
"""
function build_realistic_foodwebs(; S=40, TL=4,
        conn=0.15,                   # overall connectance
        trophic_eff=0.3,             # decay factor per trophic level (energy loss)
        omnivory_prob=0.15,          # chance of feeding across >1 level
        competition_prob=0.10,       # within-level competition
        diag_range=(-1.2, -0.3),     # self-regulation diagonal range
        rng=MersenneTwister(1234),   # RNG seed
        reps=5                       # number of webs to build
    )

    # assign trophic levels (roughly equal sizes)
    species = collect(1:S)
    levels  = rand(rng, 1:TL, S)     # random assignment of TLs

    # jit vector to assign names of webs
    base_Js    = Vector{Matrix{Float64}}(undef, reps)
    base_names = Vector{String}(undef, reps)

    for r in 1:reps
        J = zeros(Float64, S, S)

        # 1) diagonal self-regulation
        for i in species
            J[i,i] = rand(rng, diag_range[1]:0.01:diag_range[2])
        end

        # 2) predator-prey edges
        for i in species, j in species
            i == j && continue
            li, lj = levels[i], levels[j]

            # consume only if higher trophic level
            if li < lj
                if rand(rng) < conn
                    Δ = lj - li        # trophic level difference
                    eff = trophic_eff^Δ
                    val = eff * (rand(rng) < 0.5 ? +1 : -1)  # random sign
                    J[i,j] = val       # predator (+) receives benefit
                    J[j,i] = -val      # prey (-) loses energy
                end
            end

            # 3) Omnivory: allow feeding over multiple levels occasionally
            if li < lj && rand(rng) < omnivory_prob
                J[i,j] += trophic_eff^(lj-li) * (rand(rng) < 0.5 ? +1 : -1)
                J[j,i] -= J[i,j]
            end

            # 4) Competition: if same trophic level
            if li == lj && rand(rng) < competition_prob
                cval = 0.05 * (rand(rng) < 0.5 ? +1 : -1)
                J[i,j] += cval
                J[j,i] += cval      # mutual negative effect (symmetric)
            end
        end

        base_Js[r]    = J
        base_names[r] = "real_web_$(r)"
    end

    return base_Js, base_names
end

# Run an example ensemble if file executed directly
# base_Js, base_names = load_real_foodwebs_stub()
base_Js, base_names = build_realistic_foodwebs()

ens = run_ensemble(base_Js, base_names;
                    n_rewires = 5,
                    n_scrambled = 5,
                    tmax = 10.0,
                    nt = 200,
                    n_dirs = 300,
                    t_window = (0.5, 5.0),
                    B_mode = :any_nonzero)

results = ens.results

println("Regression coefficients β (Δr_peak ~ 1 + ΔΦ + Δη + Δλ + Δq + Δdiag + ΔH):")
β = regress_delta_rpeak(results)
println(β)

# Example plots: Δr_peak vs Δη (non-normality) and vs ΔΦ and vs Δq
plot_scatter_by_variant(results, :Δη;
                        xlabel = "Δ Henrici (non-normality) Real Food Webs",
                        outfile = "scatter_Deta_vs_Drpeak.png")

plot_scatter_by_variant(results, :ΔΦ;
                        xlabel = "Δ Φ (collectivity) Real Food Webs",
                        outfile = "scatter_DPhi_vs_Drpeak.png")

plot_scatter_by_variant(results, :Δq;
                        xlabel = "Δ trophic coherence q Real Food Webs",
                        outfile = "scatter_Dq_vs_Drpeak.png")

plot_scatter_by_variant(results, :ΔH;
                        xlabel = "ΔH Real Food Webs",
                        outfile = "scatter_Drpeak_vs_Drpeak.png")


(ts_base, r_base) = ens.results_rmed["real_web_1", "base"];
(ts_rew,  r_rew)  = ens.results_rmed["real_web_1", "rewired_1"];
(ts_scr,  r_scr)  = ens.results_rmed["real_web_1", "scrambled_3"];

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "time", ylabel = "Rmed")
    lines!(ax, ts_base, r_base; color=:black)
    lines!(ax, ts_rew,  r_rew;  color=:blue)
    lines!(ax, ts_scr,  r_scr;  color=:orange)
    display(fig)
end