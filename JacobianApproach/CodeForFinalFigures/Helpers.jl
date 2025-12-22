"""
Compute t95 from an rmed(t) curve using the implicit definition:
exp(-Ravg(t)*t) = target

Inputs:
- tvals  : vector of times (increasing)
- rmed   : vector rmed(t) at those times (same length)
Returns:
- t95 (Float64) or Inf if never crosses target.
"""
"""
Compute t95 from an rmed(t) curve via:
    exp(-rmed(t) * t) <= target

- Uses first finite crossing.
- Interpolates in log-space between bracketing points.
Returns Inf if it never crosses.
"""
function t95_from_rmed(tvals::AbstractVector, rmed::AbstractVector; target::Real=0.05)
    @assert length(tvals) == length(rmed)
    @assert target > 0 && target < 1

    n = length(tvals)

    # Find first index where y(t) <= target with finite y
    idx = 0
    y_prev = NaN
    t_prev = NaN

    for i in 1:n
        ti = float(tvals[i])
        ri = float(rmed[i])
        if !isfinite(ti) || !isfinite(ri) || ti <= 0
            continue
        end
        yi = exp(-ri * ti)
        if !isfinite(yi)
            continue
        end

        if yi <= target
            idx = i
            break
        end
        y_prev = yi
        t_prev = ti
    end

    idx == 0 && return Inf

    # If the first valid point already crosses
    ti = float(tvals[idx])
    ri = float(rmed[idx])
    yi = exp(-ri * ti)
    if !isfinite(y_prev) || !isfinite(t_prev)
        return ti
    end

    # Log-linear interpolation between (t_prev, y_prev) and (ti, yi)
    # Guard against degenerate segments
    if yi == y_prev || yi <= 0 || y_prev <= 0
        return ti
    end

    ℓ1 = log(y_prev)
    ℓ2 = log(yi)
    ℓt = log(float(target))

    if ℓ2 == ℓ1
        return ti
    end

    return t_prev + (ℓt - ℓ1) * (ti - t_prev) / (ℓ2 - ℓ1)
end

function median_return_rate(
    J::AbstractMatrix, u::AbstractVector;
    t::Real=0.01, perturbation::Symbol=:biomass
)
    S = size(J,1)
    if S == 0 || any(!isfinite, J)
        return NaN
    end
    E = exp(t*J)  # matrix exponential
    if perturbation === :uniform
        num = log(tr(E * transpose(E)))
        den = log(S)
    elseif perturbation === :biomass
        @assert u !== nothing "u is required for perturbation=:biomass"
        w = u .^ 2
        C = Diagonal(w)
        num = log(tr(E * C * transpose(E)))
        den = log(sum(w))
    else
        error("Unknown perturbation model: $perturbation")
    end
    return -(num - den) / (2*t)
end

function median_return_rate(J::AbstractMatrix, u::AbstractVector; t::Real=0.01, perturbation::Symbol=:biomass)
    S = size(J,1)
    if S == 0 || any(!isfinite, J)
        return NaN
    end

    E = exp(t*J)
    if any(!isfinite, E)          # <-- critical
        return NaN
    end

    if perturbation === :uniform
        T = tr(E * transpose(E))
        if !isfinite(T) || T <= 0
            return NaN
        end
        num = log(T)
        den = log(S)
    elseif perturbation === :biomass
        @assert u !== nothing
        w = u .^ 2
        C = Diagonal(w)
        T = tr(E * C * transpose(E))
        if !isfinite(T) || T <= 0
            return NaN
        end
        num = log(T)
        den = log(sum(w))
    else
        error("Unknown perturbation model: $perturbation")
    end

    r = -(num - den) / (2*t)
    return isfinite(r) ? r : NaN
end

"""
Partially reshuffle off-diagonal entries of J:
- pick m off-diagonal positions
- permute their values among themselves
Diagonal stays fixed.
"""
function partial_reshuffle_offdiagonal(J::AbstractMatrix; frac::Real=0.1, rng=Random.default_rng())
    S = size(J,1)
    @assert 0 ≤ frac ≤ 1
    J2 = copy(Matrix(J))

    # collect off-diagonal indices
    idxs = Tuple{Int,Int}[]
    for i in 1:S, j in 1:S
        i == j && continue
        push!(idxs, (i,j))
    end

    n_off = length(idxs)
    m = round(Int, frac * n_off)
    m == 0 && return J2

    chosen = rand(rng, 1:n_off, m)
    vals = [J2[idxs[k]...] for k in chosen]

    perm = randperm(rng, m)
    for (pos, k) in enumerate(chosen)
        (i,j) = idxs[k]
        J2[i,j] = vals[perm[pos]]
    end
    return J2
end
