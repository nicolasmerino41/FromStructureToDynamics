
using Random, Statistics, LinearAlgebra, DataFrames
using DifferentialEquations
using CairoMakie
using Base.Threads

# Given A, K, ustar (from your full model) — do this ONCE
function full_jacobian(A::AbstractMatrix, ustar::AbstractVector)
    S = length(ustar)
    J = -Diagonal(ustar)                    # diagonal: -u*_i
    @inbounds for i in 1:S, j in 1:S
        if i != j && A[i,j] != 0.0
            J[i,j] = ustar[i] * A[i,j]     # off-diagonal: u*_i * A_ij
        end
    end
    return J
end

# Helpers to rebuild a modified J' from (alpha', N*') without touching A,K,u*
# alpha_off should contain ONLY off-diagonals (alpha'[i,i] unused)
function build_J_from(alpha_off::AbstractMatrix, Nstar::AbstractVector)
    S = length(Nstar)
    Jp = -Diagonal(Nstar)
    @inbounds for i in 1:S, j in 1:S
        if i != j && alpha_off[i,j] != 0.0
            Jp[i,j] = Nstar[i] * alpha_off[i,j]
        end
    end
    return Jp
end

# Example structure operators S(α)
# 1) Row-mean magnitude, sign preserved
function op_rowmean_alpha(alpha::AbstractMatrix)
    S = size(alpha,1)
    out = zeros(eltype(alpha), S, S)
    @inbounds for i in 1:S
        mags = [abs(alpha[i,j]) for j in 1:S if j!=i && alpha[i,j]!=0.0]
        if !isempty(mags)
            m = mean(mags)
            for j in 1:S
                if i!=j && alpha[i,j]!=0.0
                    out[i,j] = sign(alpha[i,j]) * m
                end
            end
        end
    end
    return out
end

# 2) Threshold weak links
function op_threshold_alpha(alpha::AbstractMatrix; tau::Float64)
    out = similar(alpha); fill!(out, 0.0)
    @inbounds for i in 1:size(alpha,1), j in 1:size(alpha,2)
        if i!=j && abs(alpha[i,j]) >= tau
            out[i,j] = alpha[i,j]
        end
    end
    return out
end

# Abundance operators A(N*)
uniform_N(Nstar) = fill(mean(Nstar), length(Nstar))
clip_N(Nstar; qlo=0.05, qhi=0.95) = clamp.(Nstar, quantile(Nstar,qlo), quantile(Nstar,qhi))

# Metrics from any J
resilience(J) = maximum(real, eigvals(J))
reactivity(J) = maximum(real, eigvals((J + J')/2))

# --- Pipeline example ---
# 0) From the full model
J  = full_jacobian(A, ustar)
α  = A                      # off-diagonal kernel; signs as in A
N  = ustar

# A) Structure-only simplification (row-mean)
α1 = op_rowmean_alpha(α)
J1 = build_J_from(α1, N)
res1, rea1 = resilience(J1), reactivity(J1)

# B) Abundance-only simplification (uniform N*)
Nu = uniform_N(N)
Ju = build_J_from(α, Nu)
resu, reau = resilience(Ju), reactivity(Ju)

# C) Both (threshold + uniform)
αt = op_threshold_alpha(α; tau=quantile(abs.(α[findall(!iszero, α)]), 0.2))
Jt = build_J_from(αt, Nu)
rest, reat = resilience(Jt), reactivity(Jt)

