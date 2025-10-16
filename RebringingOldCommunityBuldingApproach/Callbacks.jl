# ---------------------------------------------------------
# Fast, reusable callback constructor
# ---------------------------------------------------------
# One PositiveDomain() + one VectorContinuousCallback with S conditions.
# Same semantics as your per-species threshold callbacks.
function make_callbackset(S::Int, EXT::Float64)
    # condition: out[i] = u[i] - EXT  (event when u[i] crosses EXT)
    function vcond!(out, u, t, integrator)
        @inbounds @simd for i in 1:S
            out[i] = u[i] - EXT
        end
        nothing
    end
    # affect: zero the crossed species
    function vaffect!(integrator, idx)
        integrator.u[idx] = 0.0
        nothing
    end
    vcb = VectorContinuousCallback(vcond!, vaffect!, S)
    return CallbackSet(PositiveDomain(), vcb)
end

# ---------------------------------------------------------
# Build and save callback cache
# ---------------------------------------------------------
"""
    save_callback_cache(S_list::AbstractVector{<:Integer},
                        ext_list::AbstractVector{<:Real};
                        path::AbstractString = "callbacks_cache.jls")

Prebuilds CallbackSet(PositiveDomain() + VectorContinuousCallback) for every
(S, EXT) pair and serializes a Dict{Tuple{Int,Float64}, CallbackSet} to `path`.
"""
function save_callback_cache(S_list::AbstractVector{<:Integer},
                             ext_list::AbstractVector{<:Real};
                             path::AbstractString = "callbacks_cache.jls")
    cache = Dict{Tuple{Int,Float64}, CallbackSet}()
    for S in S_list, EXT in ext_list
        key = (Int(S), Float64(EXT))
        cache[key] = make_callbackset(S, EXT)
    end
    open(path, "w") do io
        serialize(io, cache)
    end
    println("Saved $(length(S_list)*length(ext_list)) callback sets to $path")
    return path
end

# ---------------------------------------------------------
# Load cache
# ---------------------------------------------------------
"""
    load_callback_cache(path::AbstractString = "callbacks_cache.jls")

Loads and returns the Dict{Tuple{Int,Float64}, CallbackSet}.
"""
function load_callback_cache(path::AbstractString = "callbacks_cache.jls")
    open(path, "r") do io
        return deserialize(io)
    end
end

# ---------------------------------------------------------
# Fetch a callback (with optional on-demand build)
# ---------------------------------------------------------
"""
    get_callback(S::Int, EXT::Real;
                 cache::Union{Nothing,Dict}=nothing,
                 path::Union{Nothing,AbstractString}="callbacks_cache.jls",
                 allow_build_if_missing::Bool=true)

Returns a CallbackSet for (S, EXT). If `cache` dict is provided, uses it.
Else tries to load from `path`. If missing and `allow_build_if_missing=true`,
it builds once with `make_callbackset` (but does not persist it).
"""
function get_callback(S::Int, EXT::Real;
                      cache::Union{Nothing,Dict}=nothing,
                      path::Union{Nothing,AbstractString}="callbacks_cache.jls",
                      allow_build_if_missing::Bool=true)
    key = (Int(S), Float64(EXT))
    if cache !== nothing
        return haskey(cache, key) ? cache[key] :
               (allow_build_if_missing ? make_callbackset(S, EXT) :
                error("Callback $(key) not in provided cache."))
    end
    if path !== nothing && isfile(path)
        dict = load_callback_cache(path)
        return haskey(dict, key) ? dict[key] :
               (allow_build_if_missing ? make_callbackset(S, EXT) :
                error("Callback $(key) not in cache file $path"))
    end
    return allow_build_if_missing ? make_callbackset(S, EXT) :
           error("No cache available and building is disabled.")
end

# Example usage (commented):
# S_list   = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
# ext_list = [1e-6]  # add more thresholds if you ever change it
# save_callback_cache(S_list, ext_list; path="RemakingThePaper/callbacks_cache.jls")
# cb = make_callbackset(120, 1e-6)
# @assert cb isa CallbackSet
