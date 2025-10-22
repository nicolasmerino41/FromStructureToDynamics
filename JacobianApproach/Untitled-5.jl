
using Random, Statistics, LinearAlgebra, DataFrames, Distributions

function simulate_heterogeneity(; S=80, conn=0.1, mean_abs=0.1, mag_cv=0.6, reps=200, seed=42)
    rng = MersenneTwister(seed)
    results = DataFrame(u_cv=Float64[], resilience=Float64[], min_u=Float64[], diff_res_min_u=Float64[], res_over_min_u=Float64[])
    for cv in 0.01:0.01:1.0
        for r in 1:reps
            A = build_random_predation(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv, rng=rng)
            u = random_u(S; mean=1.0, cv=cv, rng=rng)
            J = jacobian(A,u)
            res = resilience(J)
            push!(results, (u_cv=std(u)/mean(u), resilience=res, min_u=minimum(u), diff_res_min_u=-res-minimum(u), res_over_min_u=-res/minimum(u)))
        end
    end
    results
end

proof = simulate_heterogeneity(S=120, conn=0.15, mean_abs=0.1, mag_cv=0.6, reps=100)
println(first(proof,5))
println("\nMean resilience vs CV:")
println(combine(groupby(proof, :u_cv), :resilience => mean))

begin
    using CairoMakie
    fig = Figure(size=(1100,650))
    ax = Axis(fig[1,1], xlabel="CV(u)", ylabel="Resilience (Re λmax)", title="Effect of abundance heterogeneity on stability")
    ax2 = Axis(fig[1,2], xlabel="CV(u)", ylabel="Resilience - min u", title="Effect of abundance heterogeneity on stability")
    ax3 = Axis(fig[2,1], xlabel="CV(u)", ylabel="Resilience over min u ", title="Effect of abundance heterogeneity on stability")
    ax4 = Axis(fig[2,2], xlabel="min(u)", ylabel="Resilience (Re λmax)", title="Effect of abundance heterogeneity on stability")
    
    proof_group = combine(groupby(proof,:u_cv), :resilience => mean => :res_mean)
    
    scatter!(ax, proof.u_cv, proof.resilience, color=(:black,0.25))
    scatter!(ax2, proof.u_cv, proof.diff_res_min_u, color=(:black,0.25))
    scatter!(ax3, proof.u_cv, proof.res_over_min_u, color=(:black,0.25))
    scatter!(ax4, proof.min_u, proof.resilience, color=(:black,0.25))
    
    # lines!(ax, proof_group.u_cv, -proof_group.res_mean, linewidth=3)
    display(fig)
end
