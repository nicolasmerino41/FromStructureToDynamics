# -------------------------------------------------------------
# REALISTIC FOOD WEB BUILDER (replace your stub)
# -------------------------------------------------------------
"""
Generate multiple realistic trophic food webs.

Features:
  • S species, grouped into TL trophic levels
  • Feeding only allowed from level k to k+1 (but optional omnivory)
  • Interaction strength decays with trophic efficiency
  • Negative diagonals for self-regulation (Jacobian-ready)
  • Add omnivory & competition stochastically
Return:
    base_Js :: Vector{Matrix}
    base_names :: Vector{String}
"""
function build_realistic_foodwebs(; S=40, TL=4,
        conn=0.15,                   # overall connectance
        trophic_eff=0.3,             # decay factor per trophic level (energy loss)
        omnivory_prob=0.15,          # chance of feeding across >1 level
        competition_prob=0.10,       # within-level competition
        diag_range=(-1.2, -0.3),     # self-regulation diagonal range
        rng=MersenneTwister(1234),   # RNG seed
        reps=5                       # number of webs to build
    )

    # assign trophic levels (roughly equal sizes)
    species = collect(1:S)
    levels  = rand(rng, 1:TL, S)     # random assignment of TLs

    # jit vector to assign names of webs
    base_Js    = Vector{Matrix{Float64}}(undef, reps)
    base_names = Vector{String}(undef, reps)

    for r in 1:reps
        J = zeros(Float64, S, S)

        # 1) diagonal self-regulation
        for i in species
            J[i,i] = rand(rng, diag_range[1]:0.01:diag_range[2])
        end

        # 2) predator-prey edges
        for i in species, j in species
            i == j && continue
            li, lj = levels[i], levels[j]

            # consume only if higher trophic level
            if li < lj
                if rand(rng) < conn
                    Δ = lj - li        # trophic level difference
                    eff = trophic_eff^Δ
                    val = eff * (rand(rng) < 0.5 ? +1 : -1)  # random sign
                    J[i,j] = val       # predator (+) receives benefit
                    J[j,i] = -val      # prey (-) loses energy
                end
            end

            # 3) Omnivory: allow feeding over multiple levels occasionally
            if li < lj && rand(rng) < omnivory_prob
                J[i,j] += trophic_eff^(lj-li) * (rand(rng) < 0.5 ? +1 : -1)
                J[j,i] -= J[i,j]
            end

            # 4) Competition: if same trophic level
            if li == lj && rand(rng) < competition_prob
                cval = 0.05 * (rand(rng) < 0.5 ? +1 : -1)
                J[i,j] += cval
                J[j,i] += cval      # mutual negative effect (symmetric)
            end
        end

        base_Js[r]    = J
        base_names[r] = "real_web_$(r)"
    end

    return base_Js, base_names
end
