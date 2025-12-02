# Compute spectral radius: max absolute value of eigenvalues
function spectral_radius(M)
    λ = eigvals(M)
    return maximum(abs.(λ))
end

a = 1.0  # same magnitude as before

L = zeros(Float64, S, S)
for i in 2:S
    for j in 1:i-1
        L[i,j] = a          # negative effects in the lower triangle
    end
end

U_pos = zeros(Float64, S, S)
for i in 1:S-1
    for j in i+1:S
        U_pos[i,j] = -a       # positive upper interactions
    end
end

epsilon = range(0.0, 2.0, length=100)
lambda_max = similar(collect(epsilon))
rhos = similar(lambda_max)

for (k, ε) in pairs(epsilon)
    A = L + ε .* U_pos      # base = lower-only, add upper scaled by ε
    J = A - D               # same negative diagonal from u
    λ = eigvals(J)
    lambda_max[k] = maximum(real.(λ))
    rhos[k] = spectral_radius(J)
end

idx_crit = findfirst(x -> x > 0, lambda_max)
ε_crit = idx_crit === nothing ? nothing : epsilon[idx_crit]
begin
        
    f = Figure(; size = (600, 400))
    ax = Axis(f[1, 1],
        xlabel = "ε (trophic efficiency)",
        ylabel = "max Re(λ)",
        title = "Stability vs trophic efficiency"
    )

    lines!(ax, epsilon, lambda_max, label = "max Re(λ)")
    hlines!(ax, [0.0], linestyle = :dash)

    # second axis sharing x but with right y-axis
    ax2 = Axis(f[1, 1],
        yaxisposition = :right,
        ylabel = "ρ(J)",
        xticklabelsvisible = false,
        xticksvisible = false
    )

    # linkx!(ax2, ax)
    lines!(ax2, epsilon, rhos, color = :red, label = "spectral radius ρ(J)")

    display(f)
end