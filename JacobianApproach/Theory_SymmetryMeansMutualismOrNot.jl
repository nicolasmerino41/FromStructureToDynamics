using Random, Statistics, LinearAlgebra, DataFrames, CairoMakie

# ---------- Metrics ----------
resilience(J) = maximum(real, eigvals(J))
reactivity(J) = maximum(real, eigvals((J + J') / 2))

function species_return_rates(J::AbstractMatrix; t::Real = 0.01)
    S = size(J, 1)
    # If J contains any non-finite values, return NaNs
    if any(!isfinite, J)
        return fill(NaN, S)
    end

    # Compute exp(t * J)
    E = exp(t * J)

    # Compute the product exp(tJ) * (exp(tJ))'
    M = E * transpose(E)

    # Extract the diagonal and divide by 2*t
    rates = Vector{Float64}(undef, S)
    for i in 1:S
        rates[i] = (M[i,i] - 1.0) / (2*t)
    end
    return -rates # flip the sign to make a positive return rate?? Is this correct?
end

# Computes the median across species of the analytical species-level return-rates.
function median_return_rate(J::AbstractMatrix; t::Real=0.01)
    rates = species_return_rates(J; t=t)
    return median(rates)
end

# ---------- Modification operators ----------
# ---------------------------------------------------------------
# Step 1: Variance-preserving reshuffling (VAR-SHUF)
# ---------------------------------------------------------------
function op_reshuffle_alpha(α::AbstractMatrix; rng=Random.default_rng())
    S = size(α,1)
    nonzeros = [(i,j) for i in 1:S for j in 1:S if i!=j && α[i,j] != 0.0]
    values = [α[i,j] for (i,j) in nonzeros]
    perm = randperm(rng, length(values))
    α_new = zeros(Float64, S, S)
    for (k, (i,j)) in enumerate(nonzeros)
        α_new[i,j] = values[perm[k]]  # preserve sign and variance, randomize position
    end
    return α_new
end


function op_rowmean_alpha(α)
    S = size(α,1); out = zeros(Float64, S, S)
    for i in 1:S
        nonz = [abs(α[i,j]) for j in 1:S if i!=j && α[i,j]!=0.0]
        mi = isempty(nonz) ? 0.0 : mean(nonz)
        for j in 1:S
            if i != j && α[i,j]!=0.0
                out[i,j] = sign(α[i,j])*mi
            end
        end
    end
    out
end

function op_threshold_alpha(α; q=0.2)
    mags = [abs(α[i,j]) for i in 1:size(α,1), j in 1:size(α,2) if i!=j && α[i,j]!=0.0]
    τ = isempty(mags) ? 0.0 : quantile(mags, q)
    S = size(α,1); out = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        if i!=j && abs(α[i,j]) >= τ
            out[i,j] = α[i,j]
        end
    end
    out
end

uniform_u(u) = fill(mean(u), length(u))

function build_J_from(α::AbstractMatrix, u::AbstractVector)
    nonzero_idx = findall(!iszero, u)
    n = length(nonzero_idx)
    if n == 0
        return zeros(Float64, 0, 0)
    end
    αs = α[nonzero_idx, nonzero_idx]
    us = u[nonzero_idx]
    J = zeros(Float64, n, n)
    for i in 1:n
        J[i,i] = -us[i]
        for j in 1:n
            if i!=j && αs[i,j]!=0.0
                J[i,j] = us[i]*αs[i,j]
            end
        end
    end
    J
end

# ---------- Asymmetry interpolation ----------
function mix_asymmetry(A; η=0.0, mode=:intra)
    if mode == :mutualism
        # from trophic (antisym) to mutualistic (sym)
        Aη = (1 - η) * A + η * ((A + A') / 2)
    elseif mode == :intra
        # within trophic, add random symmetric noise of same magnitude
        S = size(A,1)
        B = randn(S,S)
        B = (B + B') / 2
        B .*= mean(abs.(A)) / mean(abs.(B))
        Aη = (1 - η) * A + η * B
    else
        error("mode must be :mutualism or :intra")
    end
    Aη
end

function build_trophic_with_symmetry(
    S; conn=0.1, mean_abs=0.1, mag_cv=0.6,
    rho_sym=0.0, rng=Random.default_rng()
)
    A = zeros(Float64, S, S)
    pairs = [(i,j) for i in 1:S for j in (i+1):S]
    E_target = clamp(round(Int, conn * length(pairs)), 0, length(pairs))
    σm = sqrt(log(1 + mag_cv^2))
    μm = log(mean_abs) - σm^2/2

    chosen = sample(rng, pairs, E_target; replace=false)
    for (i,j) in chosen
        m1 = rand(rng, LogNormal(μm, σm))
        m2 = rho_sym*m1 + (1-rho_sym)*rand(rng, LogNormal(μm, σm))
        # Random trophic direction
        if rand(rng) < 0.5
            A[i,j] =  m1; A[j,i] = -m2
        else
            A[i,j] = -m1; A[j,i] =  m2
        end
    end
    return A
end


# ---------- Main experiment ----------
function run_experiment(; S=80, conn=0.1, mean_abs=0.1, mag_cv=0.6, 
                         η_vals=0:0.2:1.0, rng=MersenneTwister(42))

    # Build a random antisymmetric base matrix
    A = zeros(Float64, S, S)
    pairs = [(i,j) for i in 1:S for j in (i+1):S]
    for (i,j) in pairs
        if rand(rng) < conn
            m = rand(rng, LogNormal(log(mean_abs)-0.5*log(1+mag_cv^2), sqrt(log(1+mag_cv^2))))
            if rand(rng) < 0.5
                A[i,j] =  m; A[j,i] = -m
            else
                A[i,j] = -m; A[j,i] =  m
            end
        end
    end

    u = rand(rng, LogNormal(0,0.5), S)
    α = A
    results = DataFrame(η=Float64[], mode=String[],
                        resilience=Float64[], reactivity=Float64[], mreturn=Float64[])

    for mode in [:mutualism, :intra]
        for η in η_vals
            Aη = 
            # Aη = mix_asymmetry(A; η=η, mode=mode)
            J = build_J_from(Aη, u)
            push!(results, (η, string(mode),
                resilience(J), reactivity(J), median_return_rate(J)))
        end
    end

    results
end

# ---------- Plot ----------
function plot_results(df)
    fig = Figure(size=(800,400))
    for (i, metric) in enumerate([:resilience, :reactivity, :mreturn])
        ax = Axis(fig[1,i],
            title=string(metric), xlabel="η (symmetry coefficient)", ylabel=string(metric))
        for mode in unique(df.mode)
            sub = filter(r -> r.mode==mode, eachrow(df))
            lines!(ax, [r.η for r in sub], [r[metric] for r in sub]
,
                   label=mode, linewidth=2)
            scatter!(ax, [r.η for r in sub], [r[metric] for r in sub])
        end
        axislegend(ax; position=:rb)
    end
    display(fig)
end

# Run + plot
df = run_experiment()
plot_results(df)

function plot_stability_metrics(df; title="Stability metrics vs symmetry coefficient")
    fig = Figure(size=(1000, 400))
    for (i, metric) in enumerate([:res_full, :rea_full, :mrr_full])
        ax = Axis(fig[1, i],
            title=replace(string(metric), "_full" => ""),
            xlabel="Symmetry coefficient (ρ)",
            ylabel=metric==:mrr_full ? "Median Return Rate" :
                    metric==:rea_full ? "Reactivity" : "Resilience")
        for mode in unique(df.mode)
            sub = filter(r -> r.mode == mode, eachrow(df))
            scatter!(ax, [r.rho_sym for r in sub], [getfield(r, metric) for r in sub];
                     label=mode, alpha=0.4, markersize=5)
        end
        axislegend(ax; position=:rb)
    end
    Label(fig[0, 1:3], title; fontsize=18, halign=:center)
    display(fig)
end
