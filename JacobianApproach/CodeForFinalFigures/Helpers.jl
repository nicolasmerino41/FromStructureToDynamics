"""
Compute t95 from an rmed(t) curve using the implicit definition:
exp(-Ravg(t)*t) = target

Inputs:
- tvals  : vector of times (increasing)
- rmed   : vector rmed(t) at those times (same length)
Returns:
- t95 (Float64) or Inf if never crosses target.
"""
function t95_from_rmed(tvals::AbstractVector, rmed::AbstractVector; target::Real=0.05)
    @assert length(tvals) == length(rmed)
    y = @. exp(-rmed * tvals)              # predicted remaining fraction ||x(t)||/||x(0)||
    idx = findfirst(y .<= target)          # first crossing (earliest t satisfying)
    isnothing(idx) && return Inf
    idx == 1 && return float(tvals[1])

    # linear interpolation between points idx-1 and idx
    t1, t2 = float(tvals[idx-1]), float(tvals[idx])
    y1, y2 = float(y[idx-1]), float(y[idx])

    # guard against degenerate segment
    if y2 == y1
        return t2
    end

    return t1 + (target - y1) * (t2 - t1) / (y2 - y1)
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
