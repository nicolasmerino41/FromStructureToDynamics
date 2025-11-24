###############################################################
#   Trophic Coherence Pipeline (Option A)
#   ----------------------------------------------------------
#   Builds trophic networks A across a coherence axis (q)
#   Provides operators:
#       (1) same_coherence(A)
#       (2) more_coherent(A)
#       (3) less_coherent(A)
#
#   Runs recovery analysis:
#       baseline, same, coherent↑, coherent↓, and φ-matched versions
#
#   Produces:
#       - R̃med(t) 8-panel grid
#       - ΔR̃med(t) 8-panel grid
#       - mid-time amplification vs q
#       - Δq vs ΔR̃med mid-time
###############################################################

using Random, LinearAlgebra, Statistics
using CairoMakie

#################################################################
# 1. TROPHIC COHERENCE (Johnson et al. 2014)
#################################################################

"""
    trophic_levels(A)
Compute trophic levels ℓ via Johnson et al. 2014 definition:
ℓ = (I - W)^(-1) * 1,   W_ij = A⁺_ij / k_i,
where A⁺ keeps only positive interaction directions (consumer→resource).
"""
function trophic_levels(A; ϵ=1e-6)
    S = size(A,1)
    Ap = max.(A,0)
    kin = vec(sum(Ap .> 0, dims=2))
    D = Diagonal((kin .+ ϵ))   # avoids division by zero
    W = Ap ./ (kin .+ ϵ)
    ℓ = (I - W) \ ones(S)
    return ℓ
end

"""
    trophic_coherence(A)
q = sqrt( (1/L) * Σ_{i→j} (ℓ_j - ℓ_i - 1)^2 )
"""
function trophic_coherence(A)
    ℓ = trophic_levels(A)
    diffs = Float64[]
    S = size(A,1)
    for i in 1:S, j in 1:S
        if A[i,j] > 0    # consumer i → resource j
            push!(diffs, (ℓ[j] - ℓ[i] - 1)^2)
        end
    end
    return isempty(diffs) ? 0.0 : sqrt(mean(diffs))
end


#################################################################
# 2. Build initial trophic networks across coherence axis
#################################################################

"""
    build_trophic_network(S; conn, mean_abs, mag_cv, θ)
Baseline generator:
    θ ≈ 1 → directional (coherent)
    θ ≈ 0 → loopy (incoherent)
Used only to *sample* networks with different coherence levels.
"""
function build_trophic_network(S; conn=0.1, mean_abs=0.5, mag_cv=0.5, θ=0.0, rng=Random.default_rng())
    A = zeros(Float64,S,S)
    K = round(Int, conn * S*(S-1)/2)
    pairs = [(i,j) for i=1:S-1 for j=i+1:S]
    chosen = sample(rng, pairs, K; replace=false)
    mags = rand(rng, LogNormal(log(mean_abs),mag_cv), K)

    for (k,(i,j)) in enumerate(chosen)
        m = mags[k]
        if rand(rng) < θ
            # directional (more coherent)
            if rand(rng) < 0.5
                A[i,j] = +m; A[j,i] = -m
            else
                A[i,j] = -m; A[j,i] = +m
            end
        else
            # loopy (incoherent)
            if rand(rng) < 0.5
                A[i,j] = +m; A[j,i] = +m/2
            else
                A[i,j] = -m; A[j,i] = -m/2
            end
        end
    end

    return A
end


#################################################################
# 3. Structural coherence operators
#################################################################

"""
    same_coherence(A)
Reshuffle magnitudes across edges, preserving:
    - sign structure
    - consumer/resource direction
    - approximate coherence
"""
function same_coherence(A; rng=Random.default_rng())
    S = size(A,1)
    Anew = zeros(Float64,S,S)
    mags = abs.(A[A .!= 0])
    mags = shuffle(rng, mags)

    idx = 1
    for i in 1:S, j in 1:S
        if A[i,j] != 0
            Anew[i,j] = sign(A[i,j]) * mags[idx]
            idx += 1
        end
    end

    return Anew
end


"""
    more_coherent(A)
Make matrix strictly feed-forward according to inferred trophic levels.
"""
function more_coherent(A; rng=Random.default_rng())
    S = size(A,1)
    ℓ = trophic_levels(A)
    ord = sortperm(ℓ)
    inv = invperm(ord)

    mags = abs.(A[A .!= 0])
    mags = shuffle(rng, mags)

    Anew = zeros(Float64,S,S)
    idx = 1

    for i in 1:S-1, j in i+1:S
        if idx > length(mags); break end
        m = mags[idx]; idx += 1

        # force direction by trophic order
        if inv[i] < inv[j]
            Anew[i,j] = +m; Anew[j,i] = -m
        else
            Anew[i,j] = -m; Anew[j,i] = +m
        end
    end

    return Anew
end


"""
    less_coherent(A)
Create maximum omnivory and reciprocal edges.
Always preserves predator-prey signs.
"""
function less_coherent(A; rng=Random.default_rng())
    S = size(A,1)
    mags = abs.(A[A .!= 0])
    mags = shuffle(rng, mags)

    Anew = zeros(Float64,S,S)
    idx = 1

    for i in 1:S-1, j in i+1:S
        if idx > length(mags); break end
        m = mags[idx]; idx += 1

        # random predator assignment
        if rand(rng) < 0.5
            Anew[i,j] = +m; Anew[j,i] = -m
        else
            Anew[i,j] = -m; Anew[j,i] = +m
        end

        # inject reciprocal loop
        if rand(rng) < 0.7
            Anew[i,j] = +m
            Anew[j,i] = +m/2
        end
    end

    return Anew
end


#################################################################
# 4. Compute recovery curves and diffs
#################################################################
function compute_all_curves(A; u, t_vals, rng)
    f0 = compute_rmed_series_stable(jacobian(A,u), u, t_vals)

    Asame = same_coherence(A; rng=rng)
    fsame = compute_rmed_series_stable(jacobian(Asame,u), u, t_vals)

    Aup   = more_coherent(A; rng=rng)
    fup   = compute_rmed_series_stable(jacobian(Aup,u), u, t_vals)

    Adown = less_coherent(A; rng=rng)
    fdown = compute_rmed_series_stable(jacobian(Adown,u), u, t_vals)

    ρA = maximum(abs.(eigvals(A)))

    Aupφ   = rho_match(Aup, ρA)
    Adownφ = rho_match(Adown, ρA)

    fupφ   = compute_rmed_series_stable(jacobian(Aupφ,u), u, t_vals)
    fdownφ = compute_rmed_series_stable(jacobian(Adownφ,u), u, t_vals)

    return (f0=f0, fsame=fsame, fup=fup, fdown=fdown, fupφ=fupφ, fdownφ=fdownφ)
end


#################################################################
# 5. Run the whole experiment across 8 coherence levels
#################################################################

function run_coherence_experiment(; S=50, conn=0.1, mean_abs=0.5, mag_cv=0.6,
    θs=range(0,1,length=8),
    reps=5,
    t_vals=10 .^ range(-2,2,length=40))

    rng = Xoshiro(2025)
    results = []

    for θ in θs
        println("θ = $(round(θ,digits=2))")

        # storage
        f0_bar = zeros(length(t_vals))
        fsame_bar = zeros(length(t_vals))
        fup_bar = zeros(length(t_vals))
        fdown_bar = zeros(length(t_vals))
        fupφ_bar = zeros(length(t_vals))
        fdownφ_bar= zeros(length(t_vals))

        qvals = Float64[]
        count = 0

        for r in 1:reps
            rA = Xoshiro(rand(rng,UInt64))
            A = build_trophic_network(S; conn=conn, mean_abs=mean_abs,
                                     mag_cv=mag_cv, θ=θ, rng=rA)
            qA = trophic_coherence(A)
            push!(qvals, qA)

            u = random_u(S; mean=1.0, cv=0.6, rng=rA)

            curves = compute_all_curves(A; u=u, t_vals=t_vals, rng=rA)

            f0_bar    .+= curves.f0
            fsame_bar .+= curves.fsame
            fup_bar   .+= curves.fup
            fdown_bar .+= curves.fdown
            fupφ_bar  .+= curves.fupφ
            fdownφ_bar.+= curves.fdownφ

            count += 1
        end

        results_θ = (
            θ = θ,
            q = mean(qvals),
            t = t_vals,
            f0    = f0_bar    ./ count,
            fsame = fsame_bar ./ count,
            fup   = fup_bar   ./ count,
            fdown = fdown_bar ./ count,
            fupφ  = fupφ_bar  ./ count,
            fdownφ= fdownφ_bar./ count
        )

        push!(results, results_θ)
    end

    return results
end


#################################################################
# 6. Plotting functions
#################################################################
function plot_rmed_grid(results)
    n = length(results)
    fig = Figure(size=(1400,1200))

    for (k,res) in enumerate(results)
        row = ceil(Int, k/3)
        col = k - (row-1)*3
        ax = Axis(fig[row,col], title="θ=$(round(res.θ,digits=2)), q=$(round(res.q,digits=2))",
                  xlabel="t", ylabel="R̃med", xscale=log10)

        t = res.t

        lines!(ax, t, res.f0,     color=:black, linewidth=2, label="baseline")
        lines!(ax, t, res.fsame,  color=:gray, linewidth=2, label="same")
        lines!(ax, t, res.fup,    color=:blue, linewidth=2, label="more coherent")
        lines!(ax, t, res.fdown,  color=:orange, linewidth=2, label="less coherent")
        lines!(ax, t, res.fupφ,   color=:blue, linewidth=2, linestyle=:dash, label="up φ")
        lines!(ax, t, res.fdownφ, color=:orange, linewidth=2, linestyle=:dash, label="down φ")

        if k == 1
            axislegend(ax, position=:lb, framevisible=false)
        end
    end

    display(fig)
end

function plot_delta_grid(results)
    n = length(results)
    fig = Figure(size=(1400,1200))

    for (k,res) in enumerate(results)
        row = ceil(Int, k/3)
        col = k - (row-1)*3
        ax = Axis(fig[row,col], title="ΔR̃med, θ=$(round(res.θ,digits=2))",
                  xlabel="t", ylabel="ΔR̃med", xscale=log10)
        t = res.t

        f0 = res.f0

        lines!(ax, t, res.fsame .- f0,  color=:gray, linewidth=2)
        lines!(ax, t, res.fup   .- f0,  color=:blue, linewidth=2)
        lines!(ax, t, res.fdown .- f0,  color=:orange, linewidth=2)
        lines!(ax, t, res.fupφ  .- f0,  color=:blue, linewidth=2, linestyle=:dash)
        lines!(ax, t, res.fdownφ.- f0,  color=:orange, linewidth=2, linestyle=:dash)

        hlines!(ax, [0], color=:black, linestyle=:dot)
    end

    display(fig)
end

function plot_midtime_vs_q(results)
    fig = Figure(size=(800,600))
    ax = Axis(fig[1,1], xlabel="Trophic coherence q", ylabel="Mid-time ΔR̃med")

    mids = i -> findall(results[1].t .>= 0.5 .&& results[1].t .<= 5)

    qs = [res.q for res in results]
    same = [mean(res.fsame[mids(res)] .- res.f0[mids(res)]) for res in results]
    up   = [mean(res.fup[mids(res)]   .- res.f0[mids(res)]) for res in results]
    down = [mean(res.fdown[mids(res)] .- res.f0[mids(res)]) for res in results]

    scatterlines!(ax, qs, same, color=:gray,   label="same")
    scatterlines!(ax, qs, up,   color=:blue,   label="more coherent")
    scatterlines!(ax, qs, down, color=:orange, label="less coherent")

    axislegend(ax, position=:rb, framevisible=false)
    display(fig)
end


#################################################################
# 7. Run everything
#################################################################
results = run_coherence_experiment()

fig1 = plot_rmed_grid(results)
fig2 = plot_delta_grid(results)
fig3 = plot_midtime_vs_q(results)

