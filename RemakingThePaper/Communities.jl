###############
# Dependencies
###############
using Random, LinearAlgebra, Statistics
using Distributions, StatsBase
using DataFrames, Serialization
using CairoMakie
using Base.Threads
using DifferentialEquations, SparseArrays

############################
# Data structures & helpers
############################
"""
A compact container for one community.
"""
struct Community
    S::Int
    R::Int
    C::Int
    A::Matrix{Float64}
    K::Vector{Float64}
    ustar::Vector{Float64}
    r_block::Vector{Int}
    c_block::Vector{Int}
    params::NamedTuple
    metrics::NamedTuple
end

# Utility: coefficient of variation (guards against zero-mean)
_cv(x::AbstractVector) = (isempty(x) || mean(x) == 0.0) ? NaN : std(x) / mean(x)

# Utility: maximum real part of eigenvalues of a matrix
lambda_max_real(M::AbstractMatrix) = maximum(real, eigvals(M))

#############################################
# Block assignments (balanced by construction)
#############################################
function balanced_blocks(n::Int, blocks::Int)
    b = repeat(1:blocks, inner=ceil(Int, n/blocks))[1:n]
    return b
end

#############################################
# Degree-corrected bipartite topology generator
#############################################
"""
build_topology(S, R; conn=0.1, cv_cons=2.0, cv_res=1.0, modularity=0.3, blocks=2, IS=0.2)

Returns:
- A :: SxS interaction matrix (bipartite CR+RC only; A[c,r]>0, A[r,c]<0)
- r_block, c_block :: block labels for resources and consumers
"""
function build_topology(S::Int, R::Int;
    conn::Float64=0.1,
    cv_cons::Float64=2.0,
    cv_res::Float64=1.0,
    modularity::Float64=0.3,
    blocks::Int=2,
    IS::Float64=0.2,
    rng=Random.default_rng()
)
    @assert 0 < R < S "R must be in 1..S-1"
    C = S - R
    A = zeros(Float64, S, S)

    E_target = conn * R * C

    # Heterogeneity weights (lognormal with mean 1)
    sigma_c = sqrt(log(1 + cv_cons^2)); mu_c = -sigma_c^2/2
    sigma_r = sqrt(log(1 + cv_res^2)); mu_r = -sigma_r^2/2
    wc = rand(rng, LogNormal(mu_c, sigma_c), C)
    wr = rand(rng, LogNormal(mu_r, sigma_r), R)

    c_block = balanced_blocks(C, blocks)
    r_block = balanced_blocks(R, blocks)
    delta = clamp(modularity, 0.0, 0.95)

    base_sum = 0.0
    factors = Matrix{Float64}(undef, C, R)
    for ic in 1:C, jr in 1:R
        same = (c_block[ic] == r_block[jr])
        factors[ic, jr] = same ? (1 + delta) : (1 - delta)
        base_sum += wc[ic] * wr[jr] * factors[ic, jr]
    end
    theta = (E_target <= 0 || base_sum == 0) ? 0.0 : E_target / base_sum

    for ic in 1:C, jr in 1:R
        p = clamp(theta * wc[ic] * wr[jr] * factors[ic, jr], 0.0, 0.999)
        if rand(rng) < p
            i = R + ic
            j = jr
            wpos = abs(rand(rng, Normal(0.0, IS)))
            wneg = abs(rand(rng, Normal(0.0, IS)))
            A[i, j] = wpos
            A[j, i] = -wneg
        end
    end

    return A, r_block, c_block
end

##########################################
# Choose a positive equilibrium u*
##########################################
function choose_equilibrium(S::Int, R::Int;
    u_mean::Float64=1.0,
    u_cv_res::Float64=0.5,
    u_cv_cons::Float64=0.7,
    cons_scale::Float64=1.3,
    rng=Random.default_rng()
)
    C = S - R
    function lognorm_vec(n, cv)
        sigma = sqrt(log(1 + cv^2))
        mu = log(u_mean) - sigma^2/2
        return rand(rng, LogNormal(mu, sigma), n)
    end
    ures = lognorm_vec(R, u_cv_res)
    ucon = lognorm_vec(C, u_cv_cons) .* cons_scale
    return vcat(ures, ucon)
end

##############################################################
# Feasibility and stability via rescaling
##############################################################
function make_feasible!(A::AbstractMatrix, ustar::AbstractVector)
    K = (I - A) * ustar
    return K
end

jacobian_at_equilibrium(A::AbstractMatrix, u::AbstractVector) = Diagonal(u) * (A - I)

function stabilize!(A::Matrix{Float64}, u::Vector{Float64}, K::Vector{Float64};
    margin::Float64=0.05, max_iter::Int=30, shrink::Float64=0.8
)
    alpha = 1.0
    for _ in 1:max_iter
        J = jacobian_at_equilibrium(A, u)
        lambda_val = lambda_max_real(J)
        if lambda_val <= -margin
            return alpha, lambda_val
        end
        A .*= shrink
        alpha *= shrink
        K .= (I - A) * u
    end
    J = jacobian_at_equilibrium(A, u)
    lambda_val = lambda_max_real(J)
    return alpha, lambda_val
end

########################################
# Structural metrics for a community
########################################
function degree_vectors(A::AbstractMatrix, R::Int)
    S = size(A,1); C = S - R
    deg_cons_out = zeros(Int, C)
    deg_res_in   = zeros(Int, R)
    for ic in 1:C
        i = R + ic
        deg_cons_out[ic] = count(!iszero, A[i, 1:R])
    end
    for jr in 1:R
        deg_res_in[jr] = count(!iszero, A[(R+1):S, jr])
    end
    return deg_cons_out, deg_res_in
end

function connectance(A::AbstractMatrix, R::Int)
    S = size(A,1); C = S - R
    E = 0
    for ic in 1:C, jr in 1:R
        if A[R+ic, jr] != 0.0
            E += 1
        end
    end
    return E / (R*C)
end

function within_fraction(A::AbstractMatrix, R::Int, r_block::Vector{Int}, c_block::Vector{Int})
    S = size(A,1); C = S - R
    tot = 0; win = 0
    for ic in 1:C, jr in 1:R
        if A[R+ic, jr] != 0.0
            tot += 1
            win += (c_block[ic] == r_block[jr]) ? 1 : 0
        end
    end
    return tot == 0 ? NaN : win / tot
end

function sigma_nonzero(A::AbstractMatrix)
    S = size(A,1)
    offs = Float64[]
    for i in 1:S, j in 1:S
        if i != j && A[i,j] != 0.0
            push!(offs, A[i,j])
        end
    end
    return isempty(offs) ? 0.0 : std(offs)
end

function summarize_metrics(A::AbstractMatrix, u::AbstractVector, R::Int,
    r_block::Vector{Int}, c_block::Vector{Int}; alpha::Float64, params::NamedTuple)

    S = size(A,1)
    C = S - R
    deg_c, deg_r = degree_vectors(A, R)
    conn = connectance(A, R)
    within = within_fraction(A, R, r_block, c_block)
    J = jacobian_at_equilibrium(A, u)
    lambda_val = lambda_max_real(J)
    sigma_val = sigma_nonzero(A)

    return (
        conn = conn,
        deg_cv_cons_out = _cv(deg_c),
        deg_cv_res_in   = _cv(deg_r),
        within_fraction = within,
        mean_u = mean(u),
        cv_u   = _cv(u),
        sigma_nonzero = sigma_val,
        alpha = alpha,
        lambda_max = lambda_val,
        S = S,
        R = R,
        C = C,
        cv_cons = params.cv_cons, cv_res = params.cv_res,
        modularity = params.modularity, blocks = params.blocks,
        IS = params.IS, margin = params.margin
    )
end

########################################
# Build a single community
########################################
function build_community(S::Int, R::Int;
    conn::Float64=0.1,
    cv_cons::Float64=2.0,
    cv_res::Float64=1.0,
    modularity::Float64=0.3,
    blocks::Int=2,
    IS::Float64=0.2,
    u_mean::Float64=1.0,
    u_cv_res::Float64=0.5,
    u_cv_cons::Float64=0.7,
    cons_scale::Float64=1.3,
    margin::Float64=0.05,
    rng=Random.default_rng()
)
    A, r_block, c_block = build_topology(S, R;
        conn=conn, cv_cons=cv_cons, cv_res=cv_res,
        modularity=modularity, blocks=blocks, IS=IS, rng=rng)

    ustar = choose_equilibrium(S, R;
        u_mean=u_mean, u_cv_res=u_cv_res, u_cv_cons=u_cv_cons,
        cons_scale=cons_scale, rng=rng)

    K = make_feasible!(A, ustar)

    alpha, lambda_val = stabilize!(A, ustar, K; margin=margin)
    params = (conn=conn, cv_cons=cv_cons, cv_res=cv_res, modularity=modularity,
              blocks=blocks, IS=IS, u_mean=u_mean, u_cv_res=u_cv_res,
              u_cv_cons=u_cv_cons, cons_scale=cons_scale, margin=margin)

    metrics = summarize_metrics(A, ustar, R, r_block, c_block; alpha=alpha, params=params)
    return Community(S, R, S-R, A, K, ustar, r_block, c_block, params, metrics)
end

############################################################
# Generate a batch of communities
############################################################
@inline function _rng_for_index(seed::Int, idx::Int)
    x = (UInt64(seed) ⊻ 0x9e3779b97f4a7c15) ⊻ (UInt64(idx) * 0x9e3779b97f4a7c15)
    return MersenneTwister(Int(x % UInt64(typemax(Int))))
end

function generate_batch(N::Int;
    S::Int=120, R::Int=60,
    conn_range = (0.02, 0.35),
    cv_cons_range = (0.5, 3.0),
    cv_res_range  = (0.5, 2.5),
    modularity_range = (0.0, 0.8),
    blocks_choices = [2,3,4],
    IS_range = (0.05, 0.4),
    u_mean=1.0,
    u_cv_res_range=(0.2, 0.8),
    u_cv_cons_range=(0.3, 1.2),
    cons_scale_range=(1.0, 2.0),
    margin_range=(0.02, 0.15),
    seed::Int=42
)
    comms = Vector{Community}(undef, N)
    rows  = Vector{NamedTuple}(undef, N)

    @threads for n in 1:N
        rng = _rng_for_index(seed, n)

        conn       = rand(rng)*(conn_range[2]-conn_range[1]) + conn_range[1]
        cv_cons    = rand(rng)*(cv_cons_range[2]-cv_cons_range[1]) + cv_cons_range[1]
        cv_res     = rand(rng)*(cv_res_range[2]-cv_res_range[1])  + cv_res_range[1]
        modularity = rand(rng)*(modularity_range[2]-modularity_range[1]) + modularity_range[1]
        blocks     = rand(rng, blocks_choices)
        IS         = rand(rng)*(IS_range[2]-IS_range[1]) + IS_range[1]
        u_cv_res   = rand(rng)*(u_cv_res_range[2]-u_cv_res_range[1]) + u_cv_res_range[1]
        u_cv_cons  = rand(rng)*(u_cv_cons_range[2]-u_cv_cons_range[1]) + u_cv_cons_range[1]
        cons_scale = rand(rng)*(cons_scale_range[2]-cons_scale_range[1]) + cons_scale_range[1]
        margin     = rand(rng)*(margin_range[2]-margin_range[1]) + margin_range[1]

        comm = build_community(S, R;
            conn=conn, cv_cons=cv_cons, cv_res=cv_res,
            modularity=modularity, blocks=blocks, IS=IS,
            u_mean=u_mean, u_cv_res=u_cv_res, u_cv_cons=u_cv_cons,
            cons_scale=cons_scale, margin=margin, rng=rng)

        comms[n] = comm
        rows[n]  = comm.metrics
    end

    return comms, DataFrame(rows)
end