function build_PPM(S, B, T)
    # start with basal species 1:B → no prey
    A = zeros(Int, S, S)
    s = ones(Float64, S)     # trophic levels

    for i in B+1:S
        prey = rand(1:i-1)           # first prey
        A[i, prey] = 1
        s[i] = s[prey] + 1           # provisional trophic level

        # now preferential preying
        for j in 1:i-1
            if rand() < exp(-abs(s[prey] - s[j]) / T)
                A[i,j] = 1
            end
        end
        # recalc trophic level from Johnson et al. eq. [1] page 6:
        s[i] = 1 + sum(A[i,j]*s[j] for j in 1:i-1) / max(sum(A[i,1:i-1]),1)
    end
    return A, s
end

function trophic_incoherence(A, s)
    xs = Float64[]
    for i in 1:size(A,1), j in 1:size(A,2)
        if A[i,j] == 1
            push!(xs, s[i] - s[j])
        end
    end
    return sqrt(mean(xs.^2) - 1)   # Johnson defines q = sqrt(<x²> - 1)
end

Ts = range(0.01, 2.0, length=40)

lambda_max = zeros(length(Ts))
rhos        = zeros(length(Ts))
qvals      = zeros(length(Ts))

for (k,T) in pairs(Ts)
    S = 50
    A, s = build_PPM(S, 5, T)
    J   = η * A - A' - D
    λ   = eigvals(J)
    lambda_max[k] = maximum(real.(λ))
    rhos[k]        = spectral_radius(J)
    qvals[k]      = trophic_incoherence(A, s)
end

begin
    f = Figure(; size=(1100,800))
    ax = Axis(f[1,1], xlabel="T (trophic coherence)", ylabel="max Re(λ)")
    lines!(ax, Ts, lambda_max)

    ax2 = Axis(f[1,1], yaxisposition=:right, ylabel="spectral radius ρ(J)")
    # linkx!(ax2, ax)
    lines!(ax2, Ts, rhos, color=:red)

    ax3 = Axis(f[2,1], xlabel="T", ylabel="q (incoherence)")
    lines!(ax3, Ts, qvals)

    display(f)
end

idx_crit = findfirst(x -> x > 0, lambda_max)
T_crit = isnothing(idx_crit) ? nothing : Ts[idx_crit]

