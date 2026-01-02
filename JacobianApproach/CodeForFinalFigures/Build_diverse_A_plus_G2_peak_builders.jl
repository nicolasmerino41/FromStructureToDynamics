using Random, LinearAlgebra, Statistics, Distributions

# ---------- utilities ----------
spectral_abscissa(J::AbstractMatrix) = maximum(real.(eigen(J).values))

"Extract off-diagonal part (keeps diagonal = 0)."
function offdiag_part(M::AbstractMatrix)
    S = size(M,1)
    O = copy(Matrix(M))
    for i in 1:S
        O[i,i] = 0.0
    end
    return O
end

"""
Stabilize J by shrinking its off-diagonal part:
J = -Diagonal(u) + s*O, shrink s until α(J) < -margin
"""
function stabilize_by_shrinking_offdiag(O::AbstractMatrix, u::AbstractVector;
                                        s0::Real=1.0, margin::Real=1e-3,
                                        max_shrinks::Int=40)
    S = length(u)
    @assert size(O,1)==S && size(O,2)==S
    @assert all(u .> 0)

    s = float(s0)
    J = -Diagonal(u) + s * O
    α = spectral_abscissa(J)

    k = 0
    while !(isfinite(α) && α < -margin) && k < max_shrinks
        s *= 0.5
        J = -Diagonal(u) + s * O
        α = spectral_abscissa(J)
        k += 1
    end
    return J, s, α
end

# ---------- family builders (return OFFDIAGONAL matrix O) ----------
"Near-symmetric / weakly non-normal."
function O_symmetric_like(S; p=0.05, rng=Random.default_rng())
    B = zeros(Float64, S, S)
    for i in 1:S, j in i+1:S
        if rand(rng) < p
            v = randn(rng)
            B[i,j] = v
            B[j,i] = v
        end
    end
    return B
end

"Generic asymmetric random."
function O_asymmetric(S; p=0.05, rng=Random.default_rng())
    B = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        i == j && continue
        rand(rng) < p && (B[i,j] = randn(rng))
    end
    return B
end

"Strongly non-normal feedforward (upper triangular)."
function O_feedforward(S; p=0.05, rng=Random.default_rng())
    B = zeros(Float64, S, S)
    for i in 1:S-1, j in i+1:S
        rand(rng) < p && (B[i,j] = randn(rng))
    end
    return B
end

"Jordan-like chain (extreme non-normality)."
function O_jordan(S)
    B = zeros(Float64, S, S)
    for i in 1:S-1
        B[i, i+1] = 1.0
    end
    return B
end

"Block (modular) feedforward: modules with feedforward between modules."
function O_block_feedforward(S; nblocks=4, pin=0.08, pout=0.02, rng=Random.default_rng())
    B = zeros(Float64, S, S)
    # simple contiguous blocks
    cuts = round.(Int, range(0, S; length=nblocks+1))
    cuts[1] = 0; cuts[end] = S
    blocks = [(cuts[k]+1):(cuts[k+1]) for k in 1:nblocks]

    # within-block (mildly random)
    for b in 1:nblocks
        idx = blocks[b]
        for i in idx, j in idx
            i == j && continue
            rand(rng) < pin && (B[i,j] = randn(rng))
        end
    end

    # feedforward between blocks (b -> b+1..)
    for b1 in 1:nblocks-1
        for b2 in b1+1:nblocks
            for i in blocks[b1], j in blocks[b2]
                rand(rng) < pout && (B[i,j] = randn(rng))
            end
        end
    end

    # remove diagonal just in case
    for i in 1:S
        B[i,i] = 0.0
    end
    return B
end

# ---------- κ metric (optional, but useful for checking range) ----------
function mode_kappas(J::AbstractMatrix)
    S = size(J,1)
    try
        F = eigen(J)
        V = F.vectors
        Y = inv(V)'                      # Y'V = I
        κ = [norm(V[:,i]) * norm(Y[:,i]) for i in 1:S]
        return κ
    catch
        return fill(NaN, S)
    end
end

function kappa_mean_max(κ::AbstractVector)
    vals = filter(x -> isfinite(x) && x > 0, κ)
    isempty(vals) && return (NaN, NaN)
    return (mean(vals), maximum(vals))
end

# ============================================================
# THE BUILDER YOU WANT: returns A (diag(A)=0) + metadata
# ============================================================
"""
Build a random community A with broad range of non-normality/conditioning.

Inputs:
- u: fixed positive timescales (kept unchanged)
- family_probs: probabilities for each family
- amp_log10_range: amplitude sampled as 10^(Uniform(low, high)) applied to O
- stabilize: shrink off-diagonal amplitude until J stable (α < -margin)
Returns:
- A (diag=0)
- meta NamedTuple (family, amp_draw, amp_used, alpha, kmean, kmax)
"""
function build_A_diverse(u::AbstractVector;
    family_probs = Dict(
        :symmetric_like => 0.20,
        :asymmetric     => 0.25,
        :feedforward    => 0.25,
        :jordan         => 0.15,
        :block_ff       => 0.15
    ),
    p::Real=0.05,              # base connectance knob used in several families
    nblocks::Int=4,
    amp_log10_range = (-2.0, 2.0),   # draw amplitude from 1e-2 to 1e2
    stabilize::Bool=true,
    margin::Real=1e-3,
    rng=Random.default_rng()
)
    S = length(u)
    @assert S > 1
    @assert all(u .> 0)

    # sample family
    fams = collect(keys(family_probs))
    probs = normalize([family_probs[f] for f in fams], 1)
    fam = rand(rng, Distributions.Categorical(probs)) |> i -> fams[i]

    # build raw off-diagonal structure O (unit scale)
    O = if fam == :symmetric_like
        O_symmetric_like(S; p=p, rng=rng)
    elseif fam == :asymmetric
        O_asymmetric(S; p=p, rng=rng)
    elseif fam == :feedforward
        O_feedforward(S; p=p, rng=rng)
    elseif fam == :jordan
        O_jordan(S)
    elseif fam == :block_ff
        O_block_feedforward(S; nblocks=nblocks, pin=2p, pout=p/2, rng=rng)
    else
        error("Unknown family $fam")
    end

    # random amplitude (broad range)
    lo, hi = amp_log10_range
    amp_draw = 10.0^(rand(rng)*(hi-lo) + lo)
    O *= amp_draw

    # stabilize (optional but recommended for your exp(tJ) computations)
    if stabilize
        J, amp_used, α = stabilize_by_shrinking_offdiag(O, u; s0=1.0, margin=margin)
        # convert to A
        Dinv = Diagonal(1.0 ./ u)
        A = Dinv * J + I
        # enforce diag(A)=0 exactly
        for i in 1:S
            A[i,i] = 0.0
        end
        κ, _ = mode_kappas(J)
        kmean, kmax, ksum = kappa_mean_max_sum(κ)
        return A, (family=fam, amp_draw=amp_draw, amp_used=amp_used, alpha=α, kmean=kmean, kmax=kmax)
    else
        # no stabilization: just build J directly
        J = -Diagonal(u) + O
        α = spectral_abscissa(J)
        Dinv = Diagonal(1.0 ./ u)
        A = Dinv * J + I
        for i in 1:S
            A[i,i] = 0.0
        end
        κ, _ = mode_kappas(J)
        kmean, kmax = kappa_mean_max(κ)
        return A, (family=fam, amp_draw=amp_draw, amp_used=1.0, alpha=α, kmean=kmean, kmax=kmax)
    end
end

# ============================================================
# Propagator-aligned non-normality / amplification metrics
# ============================================================

"""
Peak propagator gain over a time grid:
G2_peak = max_t ||exp(J t)||_2^2  (square of largest singular value)
Returns (G2_peak, t_at_peak).
"""
function peak_singular_gain(J::AbstractMatrix, tvals::AbstractVector)
    best = -Inf
    tbest = float(tvals[1])
    for t in tvals
        E = exp(float(t) * J)
        g2 = opnorm(E)^2               # spectral norm squared = σ_max(E)^2
        if isfinite(g2) && g2 > best
            best = g2
            tbest = float(t)
        end
    end
    return best, tbest
end

"""
Peak transient growth *relative to asymptotic decay* is sometimes approximated by
max_t ||exp(J t)||_2 (no square). This returns the unsquared gain.
"""
function peak_singular_gain_unsquared(J::AbstractMatrix, tvals::AbstractVector)
    best = -Inf
    tbest = float(tvals[1])
    for t in tvals
        E = exp(float(t) * J)
        g = opnorm(E)
        if isfinite(g) && g > best
            best = g
            tbest = float(t)
        end
    end
    return best, tbest
end

"""
Optional: a cheaper proxy that avoids exp(Jt) (good for big S):
numerical abscissa ω(J) = λ_max((J + J')/2).
If ω(J) > 0, there exists instantaneous growth; bigger ω often correlates with transient gain.
"""
function numerical_abscissa(J::AbstractMatrix)
    return maximum(eigen(Symmetric((J + J')/2)).values)
end

# ============================================================
# How to integrate into your pipeline (minimal edits)
# ============================================================
# Inside your run_pipeline loop, after you have J and Jrew:
#
#    G2_o, tG_o = peak_singular_gain(J, tvals)
#    G2_r, tG_r = peak_singular_gain(Jrew, tvals)
#    ΔG2 = abs(G2_o - G2_r) / G2_o
#
# Store G2_o, G2_r, ΔG2 as new arrays and scatter ΔG2 vs your response.

# ============================================================
# Example patch for your existing run_pipeline
# ============================================================

# Add these arrays near your other allocations:
#   G2_orig = fill(NaN, n)
#   G2_rew  = fill(NaN, n)
#   tG_orig = fill(NaN, n)
#   tG_rew  = fill(NaN, n)

# Then inside the loop after J and Jrew are built (and pass resilience check):
#   G2_orig[k], tG_orig[k] = peak_singular_gain(J, tvals)
#   G2_rew[k],  tG_rew[k]  = peak_singular_gain(Jrew, tvals)

# After loop:
#   ΔG2 = abs.(G2_orig .- G2_rew) ./ G2_orig

# Return ΔG2 and optionally the peak times.

