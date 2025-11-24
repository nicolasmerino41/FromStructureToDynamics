###############################################################
#   Trophically Constrained Directionality–Loopiness Pipeline
#   ----------------------------------------------------------
#   This script builds:
#     1. A(θ): trophic networks across directionality spectrum
#     2. Operators:
#         - same_directionality(A)
#         - fully_directional(A)
#         - fully_loopy(A)
#     3. Recovery curves + mid-time amplification
#     4. φ-matched comparisons
#
#   Requires user-defined:
#       jacobian(A,u)
#       random_u(...)
#       compute_rmed_series_stable(J,u,t)
#       rho_match(A,ρ)
###############################################################
using Random, LinearAlgebra, Statistics
using CairoMakie

#################################################################
# 1. Trophic level inference (Option 2: infer from A)
#################################################################

"""
    infer_trophic_levels(A)

Compute trophic levels ℓ via modified Johnson et al. method:
ℓ = (D + ϵI - A⁺)⁻¹ * 1
Where A⁺ keeps only consumer->resource positive flows.

Returns levels::Vector and order::Vector (sorted indices).
"""
function infer_trophic_levels(A::AbstractMatrix; ϵ=1e-3)
    S = size(A,1)
    Ap = max.(A, 0)                      # positive flows only
    kin = vec(sum(Ap .> 0, dims=2))      # in-degree
    D = Diagonal(kin .+ ϵ)               # damping ensures invertible
    M = D - Ap
    ℓ = M \ ones(S)
    order = sortperm(ℓ)
    return ℓ, order
end


#################################################################
# 2. Trophic continuum generator: A(θ)
#################################################################
"""
    build_trophic_continuum(S; conn, mean_abs, θ, mag_cv, rng)

θ = 1 → highly directional (cascade-like)
θ = 0 → highly loopy (omnivorous)
"""
function build_trophic_continuum(
        S::Int;
        conn=0.1,
        mean_abs=0.5,
        θ::Float64,
        mag_cv=0.5,
        rng=Random.default_rng()
    )

    @assert 0 ≤ θ ≤ 1 "θ must be ∈ [0,1]"

    A = zeros(Float64, S, S)

    # how many unordered pairs to occupy
    K = round(Int, conn * S*(S-1)/2)

    # choose which unordered pairs get interactions
    pairs = [(i,j) for i in 1:S-1 for j in i+1:S]
    chosen = sample(rng, pairs, K; replace=false)

    # sample magnitudes
    mags = rand(rng, LogNormal(log(mean_abs),mag_cv), K)

    for (k,(i,j)) in enumerate(chosen)
        m = mags[k]

        if rand(rng) < θ
            # Directional structure
            # consumer → resource determined by random draw
            if rand(rng) < 0.5
                A[i,j] = +m
                A[j,i] = -m
            else
                A[i,j] = -m
                A[j,i] = +m
            end
        else
            # Loopy structure: reciprocal
            if rand(rng) < 0.5
                A[i,j] = +m
                A[j,i] = -m
            else
                A[i,j] = -m
                A[j,i] = +m
            end
            # With 50% add same-sign feedback to create loops
            if rand(rng) < 0.5
                A[i,j] = +m
                A[j,i] = +m/2
            end
        end
    end

    return A
end


#################################################################
# 3. Three trophically constrained operators
#################################################################

############### Operator A: same directionality ################
"""
    same_directionality(A)

Reshuffles interactions *without changing* the trophic mode
(feed-forward edges stay feed-forward; reciprocals stay reciprocals).
Predator–prey signs always +/–.
"""
function same_directionality(A::AbstractMatrix; rng=Random.default_rng())
    S = size(A,1)

    # infer trophic order
    ℓ, order = infer_trophic_levels(A)
    inv_order = invperm(order)

    # extract pair types relative to trophic order
    U = Vector{Tuple{Int,Int}}()   # feed-forward up (low→high)
    D = Vector{Tuple{Int,Int}}()   # feed-forward down (high→low)
    L = Vector{Tuple{Int,Int}}()   # loopy pairs

    for i in 1:S-1, j in i+1:S
        if A[i,j] == 0 && A[j,i] == 0
            continue
        end
        oi = inv_order[i]
        oj = inv_order[j]

        if sign(A[i,j]) == +1 && sign(A[j,i]) == -1
            # i consumes j
            if oi < oj
                push!(U,(i,j))
            else
                push!(D,(i,j))
            end
        elseif sign(A[i,j]) == -1 && sign(A[j,i]) == +1
            # j consumes i
            if oi < oj
                push!(D,(i,j))
            else
                push!(U,(i,j))
            end
        else
            push!(L,(i,j))
        end
    end

    # shuffle within each category
    Anew = zeros(Float64,S,S)

    function reassign!(pairs)
        K = length(pairs)
        mags = [abs(A[i,j]) for (i,j) in pairs]
        mags = shuffle(rng, mags)
        positions = shuffle(rng, pairs)
        for k in 1:K
            (i,j) = positions[k]
            m = mags[k]
            # preserve which direction was predator
            if A[i,j] > 0
                Anew[i,j] = +m
                Anew[j,i] = -m
            else
                Anew[i,j] = -m
                Anew[j,i] = +m
            end
        end
    end

    reassign!(U)
    reassign!(D)
    reassign!(L)

    return Anew
end


############### Operator B: fully directional ##################
"""
    fully_directional(A)

Uses trophic order to enforce that all edges flow upward in trophic level.
"""
function fully_directional(A::AbstractMatrix; rng=Random.default_rng())
    S = size(A,1)
    ℓ, order = infer_trophic_levels(A)
    inv_order = invperm(order)

    # Extract full magnitude distribution
    mags = abs.(A[A .!= 0])

    # new matrix
    Anew = zeros(Float64,S,S)

    # for each unordered pair, assign direction by trophic order
    k = 1
    Lm = length(mags)
    for i in 1:S-1, j in i+1:S
        if k > Lm; break end
        m = mags[k]; k += 1

        # enforce direction via trophic order
        oi = inv_order[i]
        oj = inv_order[j]

        if oi < oj
            Anew[i,j] = +m
            Anew[j,i] = -m
        else
            Anew[i,j] = -m
            Anew[j,i] = +m
        end
    end

    return Anew
end


############### Operator C: fully loopy ########################
"""
    fully_loopy(A)

Destroy directional order while preserving predator–prey ± structure.
Creates maximal omnivory + reciprocal interactions.
"""
function fully_loopy(A::AbstractMatrix; rng=Random.default_rng())
    S = size(A,1)
    Anew = zeros(Float64,S,S)

    # magnitude pool
    mags = abs.(A[A .!= 0])

    pairs = [(i,j) for i in 1:S-1 for j in i+1:S]
    chosen = sample(rng, pairs, length(mags); replace=true)

    for (k,(i,j)) in enumerate(chosen)
        m = mags[rand(rng,1:length(mags))]
        # random predator assignment
        if rand(rng) < 0.5
            Anew[i,j] = +m
            Anew[j,i] = -m
        else
            Anew[i,j] = -m
            Anew[j,i] = +m
        end

        # inject reciprocal with probability 0.7
        if rand(rng) < 0.7
            # flip sign to create loop
            Anew[i,j] = +m
            Anew[j,i] = +m/2
        end
    end

    return Anew
end


#################################################################
# 4. Recovery curves for baseline + 3 operators + φ-matching
#################################################################
function compute_all_curves(A; u, t_vals, rng)
    f0 = compute_rmed_series_stable(jacobian(A,u), u, t_vals)

    # Operator A
    Asame  = same_directionality(A; rng=rng)
    fsame  = compute_rmed_series_stable(jacobian(Asame,u), u, t_vals)

    # Operator B
    Adir   = fully_directional(A; rng=rng)
    fdir   = compute_rmed_series_stable(jacobian(Adir,u), u, t_vals)

    # Operator C
    Aloop  = fully_loopy(A; rng=rng)
    floop  = compute_rmed_series_stable(jacobian(Aloop,u), u, t_vals)

    ρA = maximum(abs.(eigvals(A)))

    # φ-matching
    Adirφ  = rho_match(Adir, ρA)
    Aloopφ = rho_match(Aloop, ρA)

    fdirφ  = compute_rmed_series_stable(jacobian(Adirφ,u), u, t_vals)
    floopφ = compute_rmed_series_stable(jacobian(Aloopφ,u), u, t_vals)

    return (
        f0=f0,
        fsame=fsame,
        fdir=fdir,
        floop=floop,
        fdirφ=fdirφ,
        floopφ=floopφ
    )
end


#################################################################
# 5. Experiment loop across θ
#################################################################
function run_experiment(; S=50, conn=0.1, mean_abs=0.5, mag_cv=0.6,
    θs=range(0,1,length=8), reps=10,
    t_vals=10 .^ range(-2,2,length=40))

    rng = Xoshiro(2025)
    results = []

    for θ in θs
        println("θ = $(round(θ,digits=2))")

        mids_same = Float64[]
        mids_dir  = Float64[]
        mids_loop = Float64[]
        mids_dirφ = Float64[]
        mids_loopφ= Float64[]

        for r in 1:reps
            rA = Xoshiro(rand(rng,UInt64))
            A = build_trophic_continuum(S; conn=conn, mean_abs=mean_abs,
                                        θ=θ, mag_cv=mag_cv, rng=rA)
            u = random_u(S; mean=1.0, cv=0.6, rng=rA)

            curves = compute_all_curves(A; u=u, t_vals=t_vals, rng=rA)

            Δsame  = abs.(curves.fsame  .- curves.f0)
            Δdir   = abs.(curves.fdir   .- curves.f0)
            Δloop  = abs.(curves.floop  .- curves.f0)
            Δdirφ  = abs.(curves.fdirφ  .- curves.f0)
            Δloopφ = abs.(curves.floopφ .- curves.f0)

            # mid-time indices
            mid = findall(t_vals .>= 0.5 .&& t_vals .<= 5)

            push!(mids_same, mean(Δsame[mid]))
            push!(mids_dir,  mean(Δdir[mid]))
            push!(mids_loop, mean(Δloop[mid]))
            push!(mids_dirφ, mean(Δdirφ[mid]))
            push!(mids_loopφ,mean(Δloopφ[mid]))
        end

        push!(results, (
            θ=θ,
            same=mean(mids_same),
            dir =mean(mids_dir),
            loop=mean(mids_loop),
            dirφ=mean(mids_dirφ),
            loopφ=mean(mids_loopφ)
        ))
    end

    return results
end


#################################################################
# 6. Plotting
#################################################################
function plot_results(results)
    fig = Figure(size=(1100,600))
    ax = Axis(fig[1,1], xlabel="Directionality θ", ylabel="Mid-time amplification")

    θs = [r.θ for r in results]

    lines!(ax, θs, [r.same for r in results],
        color=:gray, linewidth=3, label="same-directionality")

    # lines!(ax, θs, [r.dir for r in results],
    #     color=:blue, linewidth=3, label="fully directional")

    # lines!(ax, θs, [r.loop for r in results],
    #     color=:orange, linewidth=3, label="fully loopy")

    # lines!(ax, θs, [r.dirφ for r in results],
    #     color=:blue, linewidth=3, linestyle=:dash, label="dir φ-matched")

    # lines!(ax, θs, [r.loopφ for r in results],
    #     color=:orange, linewidth=3, linestyle=:dash, label="loop φ-matched")

    axislegend(ax)
    display(fig)
end

results = run_experiment()
fig = plot_results(results)

########### PLOTTING RMED ACROSS TIME ############
function run_experiment_series(; S=50, conn=0.1, mean_abs=0.5, mag_cv=0.6,
    θs=range(0,1,length=8), reps=5,
    t_vals=10 .^ range(-2,2,length=40))

    rng = Xoshiro(2023)
    results = []

    for θ in θs
        println("θ = $(round(θ,digits=2))")

        # To store averages
        f0_bar     = zeros(length(t_vals))
        fsame_bar  = zeros(length(t_vals))
        fdir_bar   = zeros(length(t_vals))
        floop_bar  = zeros(length(t_vals))
        fdirφ_bar  = zeros(length(t_vals))
        floopφ_bar = zeros(length(t_vals))

        count = 0

        for r in 1:reps
            rA = Xoshiro(rand(rng,UInt64))
            A = build_trophic_continuum(S; conn=conn, mean_abs=mean_abs,
                                        θ=θ, mag_cv=mag_cv, rng=rA)
            u = random_u(S; mean=1.0, cv=0.6, rng=rA)

            curves = compute_all_curves(A; u=u, t_vals=t_vals, rng=rA)
            count += 1

            f0_bar     .+= curves.f0
            fsame_bar  .+= curves.fsame
            fdir_bar   .+= curves.fdir
            floop_bar  .+= curves.floop
            fdirφ_bar  .+= curves.fdirφ
            floopφ_bar .+= curves.floopφ
        end

        push!(results, (
            θ=θ,
            t_vals=t_vals,
            f0     = f0_bar    ./ count,
            fsame  = fsame_bar ./ count,
            fdir   = fdir_bar  ./ count,
            floop  = floop_bar ./ count,
            fdirφ  = fdirφ_bar ./ count,
            floopφ = floopφ_bar ./ count
        ))
    end

    return results
end

function plot_rmed_grid(results)
    n = length(results)

    ############################################################
    ## FIGURE 1 — Raw r̃med(t) curves
    ############################################################
    fig1 = Figure(size=(1400, 1200))

    for (k,res) in enumerate(results)
        row = ceil(Int, k/3)
        col = k - (row-1)*3

        ax = Axis(fig1[row, col],
                  title="θ = $(round(res.θ, digits=2))",
                  xlabel="t",
                  ylabel="r̃ₘₑd",
                  xscale=log10)

        t = res.t_vals

        lines!(ax, t, res.f0,     color=:black,  linewidth=2, label="baseline")
        lines!(ax, t, res.fsame,  color=:gray,   linewidth=2, label="same-dir")
        lines!(ax, t, res.fdir,   color=:blue,   linewidth=2, label="dir")
        lines!(ax, t, res.floop,  color=:orange, linewidth=2, label="loop")
        lines!(ax, t, res.fdirφ,  color=:blue,   linewidth=2, linestyle=:dash, label="dir φ")
        lines!(ax, t, res.floopφ, color=:orange, linewidth=2, linestyle=:dash, label="loop φ")

        axislegend(ax, position=:rb)
    end


    ############################################################
    ## FIGURE 2 — Δ r̃med(t) curves (signed differences)
    ############################################################
    fig2 = Figure(size=(1400, 1200))

    for (k,res) in enumerate(results)
        row = ceil(Int, k/3)
        col = k - (row-1)*3

        ax = Axis(
            fig2[row, col],
            title="Δ r̃ₘₑd(t) — θ = $(round(res.θ, digits=2))",
            xlabel="t",
            ylabel="Δ r̃ₘₑd",
            xscale=log10
            )

        t = res.t_vals
        f0 = res.f0

        # signed differences
        Δsame  = res.fsame  .- f0
        Δdir   = res.fdir   .- f0
        Δloop  = res.floop  .- f0
        Δdirφ  = res.fdirφ  .- f0
        Δloopφ = res.floopφ .- f0

        lines!(ax, t, Δsame,  color=:gray,   linewidth=2, label="same-dir")
        lines!(ax, t, Δdir,   color=:blue,   linewidth=2, label="dir")
        lines!(ax, t, Δloop,  color=:orange, linewidth=2, label="loop")
        lines!(ax, t, Δdirφ,  color=:blue,   linewidth=2, linestyle=:dash, label="dir φ")
        lines!(ax, t, Δloopφ, color=:orange, linewidth=2, linestyle=:dash, label="loop φ")

        # reference line Δ=0
        hlines!(ax, [0], color=:black, linestyle=:dot, linewidth=1)

        axislegend(ax, position=:rb)
    end

    display(fig1)
    display(fig2)
end

results_series = run_experiment_series()

gridfig = plot_rmed_grid(results_series)
