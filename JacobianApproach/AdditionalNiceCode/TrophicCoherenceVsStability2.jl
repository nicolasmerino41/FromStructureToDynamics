α = 0.5
Ss = round.(Int, exp.(range(log(30), log(600), length=12)))
T_val = 0.25   # choose a coherence level (Johnson shows several)
η     = 0.2    # Johnson’s efficiency parameter

lambda_max_S = zeros(length(Ss))
rho_S        = zeros(length(Ss))

for (k,S) in pairs(Ss)
    # compute number of links from K = S^α
    K = S^α
    L = round(Int, K * S)

    # number of basal species (fixed fraction, Johnson uses ~0.25 S)
    B = max(2, round(Int, 0.25*S))

    # build network with our PPM model
    A, s = build_PPM(S, B, T_val)

    # interaction matrix (Johnson p.6): W = ηA - Aᵀ
    W = η .* A .- A'

    u = fill(1.0, S)
    W = jacobian(W, u)
    # stability measure: leading eigenvalue real part
    λ = eigvals(W)
    lambda_max_S[k] = maximum(real.(λ))

    # spectral radius
    rho_S[k] = spectral_radius(W)
end

begin
    f = Figure()

    ax = Axis(f[1,1], xlabel="S", ylabel="max Re(λ)", title="Complexity–stability scaling")
    lines!(ax, Ss, lambda_max_S)

    ax2 = Axis(f[1,1], yaxisposition=:right, ylabel="ρ(W)")
    # linkx!(ax2, ax)
    lines!(ax2, Ss, rho_S, color=:red)

    display(f)
end



Tsweep = [10.0, 0.5, 0.3, 0.2, 0.01]
lambda_T = Dict{Float64, Vector{Float64}}()
rho_T     = Dict{Float64, Vector{Float64}}()

for T in Tsweep
    lambda_T[T] = zeros(length(Ss))
    rho_T[T]    = zeros(length(Ss))
end
for T in Tsweep
    for (k,S) in pairs(Ss)

        K = S^α
        L = round(Int, K*S)
        B = max(2, round(Int, 0.25*S))

        A, s = build_PPM(S, B, T)
        W = η .* A .- A'

        u = fill(1.0, S)
        W = jacobian(W, u)

        λ = eigvals(W)
        lambda_T[T][k] = maximum(real.(λ))
        rho_T[T][k]    = spectral_radius(W)

    end
end

begin
    f = Figure()

    ax = Axis(f[1,1],
        xlabel = "S (species richness)",  
        ylabel = "max Re(λ)",
        title  = "Complexity–stability scaling across T values"
    )

    for T in Tsweep
        lines!(ax, Ss, lambda_T[T], label = "T = $T")
    end

    axislegend(ax)

    display(f)
end