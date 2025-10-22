
########################  Jacobian-only pipeline: single community  ########################
using Random, LinearAlgebra, Statistics, Distributions
# ---------------------- Core gLV pieces ----------------------
# J(u*) = Diag(u*) * (A - I)
jacobian_at(u::AbstractVector, A::AbstractMatrix) = Diagonal(u) * (A - I)
resilience(J::AbstractMatrix) = maximum(real, eigvals(J))
reactivity(J::AbstractMatrix) = maximum(real, eigvals((J + J')/2))

# ---------------------- Community builder (disordered trophic) ----------------------
"""
build_random_trophic(S; conn, mean_abs, mag_cv, rng)

- Undirected pairs {i,j} are selected with probability ≈ conn.
- For each active pair we draw a magnitude m>0 and set (A[i,j], A[j,i]) = (+m,-m) or (-m,+m).
- No self-links. Mean(|A_ij|) ≈ mean_abs, CV ≈ mag_cv.
"""
function build_random_trophic(S::Int; conn::Float64=0.10,
                                mean_abs::Float64=0.1, mag_cv::Float64=0.6,
                                rng=Random.default_rng())
    A = zeros(Float64, S, S)
    pairs = [(i,j) for i in 1:S for j in (i+1):S]
    E = clamp(round(Int, conn * S*(S-1)/2), 0, length(pairs))
    sel = rand(rng, 1:length(pairs), E)

    # lognormal parameters matching mean_abs & cv
    σ = sqrt(log(1 + mag_cv^2))
    μ = log(mean_abs) - σ^2/2

    for idx in sel
        i,j = pairs[idx]
        m = rand(rng, LogNormal(μ, σ))
        if rand(rng) < 0.5
            A[i,j] =  m;  A[j,i] = -m
        else
            A[i,j] = -m;  A[j,i] =  m
        end
    end
    return A
end

"""
    build_random_nontrophic(S; conn=0.10, mean_abs=0.10, mag_cv=0.60, rng=Random.default_rng())

Return an S×S interaction matrix A for a *directed*, fully random (non-trophic) system.

- For each i ≠ j, with probability `conn` set
      A[i,j] = sign * magnitude
  where `sign ∈ {+1, -1}` (equiprobable) and
        `magnitude ~ LogNormal(μ, σ)` calibrated so that E[|A[i,j]|] = `mean_abs`
        and CV(|A[i,j]|) = `mag_cv`.

- Diagonal entries are zero.
- Each direction (i→j and j→i) is sampled independently, so symmetric and antisymmetric
  parts appear naturally without any pairing constraint.
"""
function build_random_nontrophic(S::Int;
    conn::Float64 = 0.10,
    mean_abs::Float64 = 0.10,
    mag_cv::Float64 = 0.60,
    rng = Random.default_rng()
)
    A = zeros(Float64, S, S)

    # Calibrate LogNormal so its mean and CV match (for magnitudes)
    σ = sqrt(log(1 + mag_cv^2))
    μ = log(mean_abs) - σ^2/2
    LN = LogNormal(μ, σ)

    @inbounds for i in 1:S, j in 1:S
        if i != j && rand(rng) < conn
            m = rand(rng, LN)
            s = ifelse(rand(rng) < 0.5, -1.0, 1.0)
            A[i, j] = s * m
        end
    end
    return A
end

# --- tiny helper if you want to inspect what you got ---
mean_abs_nonzero(A) = mean(abs, A[A .!= 0])
directed_connectance(A) = count(!iszero, A) / (size(A,1)*(size(A,1)-1))

"""
random_u(S; mean, cv): positive lognormal u*
"""
function random_u(S::Int; mean::Float64=1.0, cv::Float64=0.5, rng=Random.default_rng())
    σ = sqrt(log(1 + cv^2))
    μ = log(mean) - σ^2/2
    return rand(rng, LogNormal(μ, σ), S)
end

"""
stabilize_shrink!(A, u*; margin, shrink, max_iter) -> (K, alpha, lambda_max)

Scale A in-place by a factor α ∈ (0,1] until max Re eig(J) ≤ -margin,
keeping u* fixed and setting K = (I - A)u* so that u* remains an exact equilibrium.
Returns K, the achieved α, and λ_max at stop.
"""
function stabilize_shrink!(A::Matrix{Float64}, u::Vector{Float64};
                           margin::Float64=0.05, shrink::Float64=0.85, max_iter::Int=60)
    α = 1.0
    for _ in 1:max_iter
        λmax = maximum(real, eigvals(jacobian_at(u, A)))
        if λmax <= -margin
            return (I - A) * u, α, λmax
        end
        A .*= shrink
        α *= shrink
    end
    # return whatever we ended with
    return (I - A) * u, α, maximum(real, eigvals(jacobian_at(u, A)))
end

# ---------------------- Jacobian-only operators ----------------------
"""
    build_J_from(alpha_off, Nstar) -> J

Given an off-diagonal interaction kernel `alpha_off` (same size as A, zeros on the diagonal)
and an equilibrium abundance vector `Nstar` (= u*), returns the Jacobian

    J = Diagonal(Nstar) * (alpha_off - I)

which is consistent with the gLV form du_i/dt = u_i (K_i - u_i + (A u)_i) and A having zero diagonal.
"""
function build_J_from(alpha_off::AbstractMatrix, Nstar::AbstractVector)
    S = size(alpha_off, 1)
    @assert size(alpha_off,2) == S "alpha_off must be square"
    @assert length(Nstar) == S      "Nstar length must match alpha_off size"

    # Ensure diagonal is exactly zero
    Atilde = copy(alpha_off)
    @inbounds for i in 1:S
        Atilde[i,i] = 0.0
    end

    return Diagonal(Nstar) * (Atilde - I)
end

# (Optional but recommended) make sure the operators that produce alpha_off always zero the diagonal:

"""
Row-mean operator on |alpha| preserving signs on existing edges.
Returns an off-diagonal matrix with zeros on the diagonal.
"""
function op_rowmean_alpha(alpha::AbstractMatrix)
    S = size(alpha,1)
    out = zeros(eltype(alpha), S, S)
    @inbounds for i in 1:S
        mags = Float64[]
        for j in 1:S
            if i != j && alpha[i,j] != 0.0
                push!(mags, abs(alpha[i,j]))
            end
        end
        if !isempty(mags)
            m = mean(mags)
            for j in 1:S
                if i != j && alpha[i,j] != 0.0
                    out[i,j] = sign(alpha[i,j]) * m
                end
            end
        end
    end
    return out
end

"""
Uniform N operator: replaces N* with its mean while keeping length S.
"""
uniform_N(N::AbstractVector) = fill(mean(N), length(N))

"""
Hard-threshold the off-diagonal magnitudes at τ, preserving signs.
Entries with |alpha_ij| < τ are set to 0 (off-diagonal only).
"""
function op_threshold_alpha(alpha::AbstractMatrix; τ::Real)
    S = size(alpha,1)
    out = zeros(eltype(alpha), S, S)
    @inbounds for i in 1:S, j in 1:S
        if i != j
            a = alpha[i,j]
            if abs(a) >= τ
                out[i,j] = a
            end
        end
    end
    return out
end

# ---------------------- One-shot pipeline ----------------------
function one_run(; S=150, conn=0.10, mean_abs=0.10, mag_cv=0.60,
                 u_mean=1.0, u_cv=0.5, margin=0.05, seed=42,
                 trophic=false)

    rng = MersenneTwister(seed)

    # (A, u*, K) for the FULL system
    if trophic
        A      = build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
    else
        A      = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
    end
    u_star = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

    # shrink A to make J(u*,A) stable, keep u* fixed; set K=(I-A)u*
    K, α, λmax = stabilize_shrink!(A, u_star; margin=margin)

    # Full Jacobian and its "kernel" alpha_off (= A with zeroed diagonal)
    J   = jacobian_at(u_star, A)
    αoff = copy(A); @inbounds for i in 1:S; αoff[i,i] = 0.0; end
    N   = u_star  # row weights in J

    # --- Jacobian-only simplifications ---
    α1 = op_rowmean_alpha(αoff)
    J1 = build_J_from(α1, N)

    # step2
    allmags = [abs(αoff[i,j]) for i in 1:S, j in 1:S if i!=j && αoff[i,j]!=0.0]
    τ = isempty(allmags) ? 0.0 : quantile(allmags, 0.20)   # drop weakest 20%
    αt = op_threshold_alpha(αoff; τ=τ)
    Jth = build_J_from(αt, N)
    
    #step3
    Nu = uniform_N(N)
    Ju = build_J_from(αoff, Nu)

    # step 4
    Jt = build_J_from(α1, Nu)

    # off-diagonal Frobenius norm
    offdiag_norm(M) = begin
        S=size(M,1); acc=0.0
        @inbounds for i in 1:S, j in 1:S
            if i!=j; acc += M[i,j]^2; end
        end
        sqrt(acc)
    end
    # mean absolute magnitude over *nonzero* off-diagonals
    mean_abs_offdiag_nonzero(M) = begin
        S=size(M,1); vals=Float64[]
        @inbounds for i in 1:S, j in 1:S
            if i!=j && M[i,j]!=0.0; push!(vals, abs(M[i,j])); end
        end
        isempty(vals) ? 0.0 : mean(vals)
    end
    println("||J_full||_off  = ", offdiag_norm(J))
    println("||J_rowm||_off  = ", offdiag_norm(J1))
    println("||J_thres||_off  = ", offdiag_norm(Jth))
    println("||J_unif||_off  = ", offdiag_norm(Ju))
    println("||J_both||_off  = ", offdiag_norm(Jt))
    println("mean|J_unif| offdiag nz = ", mean_abs_offdiag_nonzero(Ju))
    println("mean U* = ", mean(u_star))
    return (;
        S, conn, mean_abs, mag_cv,
        shrink_alpha = α, lambda_max = λmax,
        IS_actual = mean(abs, A[A .!= 0]),             # realized IS after shrink
        u_mean = mean(u_star), u_cv = std(u_star)/mean(u_star),
        res_full = resilience(J),  rea_full = reactivity(J),
        res_rowm = resilience(J1), rea_rowm = reactivity(J1),
        res_thres = resilience(Jth), rea_thres = reactivity(Jth),
        res_unif = resilience(Ju), rea_unif = reactivity(Ju),
        res_both = resilience(Jt), rea_both = reactivity(Jt)
    )
end

# ---------------------- Run once and print ----------------------
stats = one_run(; S=150, conn=0.10, mean_abs=0.10, mag_cv=0.60,
                 u_mean=1.0, u_cv=0.5, margin=0.05, seed=123,
                 trophic=true)

println("=== Baseline (built once) ===")
println("S=$(stats.S), conn=$(stats.conn), shrink_alpha=$(round(stats.shrink_alpha, digits=3))")
println("λ_max(J_full)=$(round(stats.lambda_max, digits=4))  (≤ -margin?)")
println("Realized IS (mean |A_ij| over nonzeros) = $(round(stats.IS_actual, digits=4))")
println("u*: mean=$(round(stats.u_mean, digits=3)), cv=$(round(stats.u_cv, digits=3))\n")

println("=== Resilience / Reactivity (Jacobian-only simplifications) ===")
println("Full:   res=$(round(stats.res_full, digits=4)),  rea=$(round(stats.rea_full, digits=4))")
println("RowMean:res=$(round(stats.res_rowm, digits=4)),  rea=$(round(stats.rea_rowm, digits=4))")
println("Thres:  res=$(round(stats.res_thres, digits=4)),  rea=$(round(stats.rea_thres, digits=4))")
println("Uniform:res=$(round(stats.res_unif, digits=4)),  rea=$(round(stats.rea_unif, digits=4))")
println("Both:   res=$(round(stats.res_both, digits=4)),  rea=$(round(stats.rea_both, digits=4))")