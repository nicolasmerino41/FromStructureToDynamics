
# -----------------------------
# Community generation
# -----------------------------
function random_u(S; mean=1.0, cv=0.5, rng=Random.default_rng())
    sigma = sqrt(log(1 + cv^2))
    mu = log(mean) - sigma^2/2
    rand(rng, LogNormal(mu, sigma), S)
end

function random_interaction_matrix(S::Int, connectance::Real; σ::Real=1.0, rng=Random.default_rng())
    A = zeros(Float64, S, S)
    for i in 1:S, j in 1:S
        if i != j && rand(rng) < connectance
            A[i,j] = rand(rng, Normal(0, σ))
        end
    end
    return A
end

jacobian(A,u) = Diagonal(u) * (A - I)

# -----------------------------
# Fix A: match stability margin
# -----------------------------
function spectral_abscissa(J::AbstractMatrix)
    vals = eigen(J).values
    return maximum(real.(vals))
end

function shift_to_margin(J::AbstractMatrix; γ::Real=0.1)
    α = spectral_abscissa(J)
    c = α + γ
    Jshift = J - c * I
    return Jshift, c, α
end

# -----------------------------
# Fix B: partial rewiring
# -----------------------------
function partial_reshuffle_offdiagonal(J::AbstractMatrix; frac::Real=0.1, rng=Random.default_rng())
    S = size(J,1)
    @assert 0 ≤ frac ≤ 1
    J2 = copy(Matrix(J))

    idxs = Tuple{Int,Int}[]
    for i in 1:S, j in 1:S
        i == j && continue
        push!(idxs, (i,j))
    end

    n_off = length(idxs)
    m = round(Int, frac * n_off)
    m == 0 && return J2

    chosen = rand(rng, 1:n_off, m)
    vals = [J2[idxs[k]...] for k in chosen]

    perm = randperm(rng, m)
    for (pos, k) in enumerate(chosen)
        (i,j) = idxs[k]
        J2[i,j] = vals[perm[pos]]
    end
    return J2
end

# -----------------------------
# κ metrics (mean/max) from left/right eigenvectors
# -----------------------------
function mode_kappas(J::AbstractMatrix)
    S = size(J,1)
    try
        F = eigen(J)
        V = F.vectors
        Vinv = inv(V)
        Y = Vinv'  # columns are left eigenvectors with Y'V = I
        κ = [norm(V[:,i]) * norm(Y[:,i]) for i in 1:S]
        return κ
    catch
        return fill(NaN, S)
    end
end

function kappa_mean_max(κ::AbstractVector)
    vals = filter(x -> isfinite(x) && x > 0, κ)
    isempty(vals) && return (NaN, NaN)
    return (mean(vals), maximum(vals))
end

# -----------------------------
# T∞ via Lyapunov / controllability Gramian trace
# T∞ = tr(X)/S, where JX + XJ' = -I
# -----------------------------
function transient_T_infty(J::AbstractMatrix)
    S = size(J,1)
    Q = Matrix{Float64}(I, S, S)
    try
        X = lyap(J, Q)        # JX + XJ' + Q = 0
        T = tr(X) / S
        return (isfinite(T) && T > 0) ? T : NaN
    catch
        return NaN
    end
end

# -----------------------------
# Time-domain comparison: ||x(t) - xrew(t)|| / ||x0||
# (Uses explicit matrix exponential; for S=120 this can be heavy)
# -----------------------------
function diff_trajectory_norms(J::AbstractMatrix, Jrew::AbstractMatrix, x0::AbstractVector, tvals::AbstractVector)
    x0n = norm(x0)
    d = similar(tvals, Float64)
    a = similar(tvals, Float64)
    b = similar(tvals, Float64)
    for (k, t) in enumerate(tvals)
        E  = exp(t * J)
        Er = exp(t * Jrew)
        x  = E  * x0
        xr = Er * x0
        d[k] = norm(x - xr) / x0n
        a[k] = norm(x) / x0n
        b[k] = norm(xr) / x0n
    end
    return d, a, b
end

# -----------------------------
# Main pipeline: generate many cases, then plot 9 random ones
# -----------------------------
function run_and_plot_9(;
    S::Int=80,
    connectance::Real=0.05,
    n::Int=50,
    u_mean::Real=1.0,
    u_cv::Real=0.5,
    σA::Real=0.25,
    seed::Int=1234,
    rew_frac::Real=0.10,
    γ::Real=0.10,
    tvals = 10 .^ range(log10(0.01), log10(100.0); length=40),
    do_you_shift_to_margin = false
)
    rng = MersenneTwister(seed)

    Js   = Vector{Matrix{Float64}}(undef, n)
    Jrs  = Vector{Matrix{Float64}}(undef, n)
    ΔT   = fill(NaN, n)
    Δkμ  = fill(NaN, n)
    ΔkM  = fill(NaN, n)
    Tinf = fill(NaN, n)

    for k in 1:n

        # Random
        # A = random_interaction_matrix(S, connectance; σ=σA, rng=rng)
        
        # Trophic
        
        # PPM
        b = PPMBuilder()
        set!(b; S=S, B=0.25S, L=connectance*S^2, T=0.01, η=0.2)
        net = build(b)
        A = net.A
        # --- turn topology into signed interaction matrix W ---
        A = build_interaction_matrix(A;
            mag_abs=σA,
            mag_cv=σA,
            corr_aij_aji=0.0,
            rng=rng
        )
        
        u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
        J = Matrix(jacobian(A, u))

        Jrew = partial_reshuffle_offdiagonal(J; frac=rew_frac, rng=rng)

        if do_you_shift_to_margin
            # Fix A: force same stability margin for both
            Jc,  _, _ = shift_to_margin(J;    γ=γ)
            Jrc, _, _ = shift_to_margin(Jrew; γ=γ)
        else
            Jc  = J
            Jrc = Jrew
        end

        Js[k]  = Jc
        Jrs[k] = Jrc

        T1 = transient_T_infty(Jc)
        T2 = transient_T_infty(Jrc)

        Tinf[k] = T1
        ΔT[k]   = abs(T1 - T2)

        kμ1, kM1 = kappa_mean_max(mode_kappas(Jc))
        kμ2, kM2 = kappa_mean_max(mode_kappas(Jrc))
        Δkμ[k] = abs(kμ1 - kμ2)
        ΔkM[k] = abs(kM1 - kM2)
    end

    good = findall(i -> all(isfinite, Js[i]) && all(isfinite, Jrs[i]), 1:n)
    if length(good) < 9
        error("Not enough valid cases.")
    end

    pick = sort(rand(rng, good, 9); by = i -> Tinf[i])

    # ---------------------------------------------------
    # Generate trajectories ONCE
    # ---------------------------------------------------
    traj = Dict{Int, Vector{Float64}}()
    ylow  = Inf
    yhigh = -Inf

    for idx in pick
        x0 = randn(rng, S)
        d, _, _ = diff_trajectory_norms(Js[idx], Jrs[idx], x0, tvals)

        traj[idx] = d
        ylow  = min(ylow,  minimum(d))
        yhigh = max(yhigh, maximum(d))
    end

    # Optional small padding (prevents visual clipping)
    pad = 0.05 * (yhigh - ylow)
    ylow  -= pad
    yhigh += pad

    # ---------------------------------------------------
    # Plot
    # ---------------------------------------------------
    fig = Figure(size=(1200, 900))

    for (p, idx) in enumerate(pick)
        r = (p - 1) ÷ 3 + 1
        c = (p - 1) % 3 + 1

        ax = Axis(
            fig[r, c];
            xscale = log10,
            xlabel = "t",
            ylabel = "‖x(t) - xrew(t)‖ / ‖x0‖",
            title = "case $idx  (T∞=$(round(Tinf[idx], sigdigits=3)))"
        )

        lines!(ax, tvals, traj[idx], linewidth=2)
        ylims!(ax, ylow, yhigh)
    end

    display(fig)

    return (
        Js = Js,
        Jrs = Jrs,
        Tinf = Tinf,
        ΔT = ΔT,
        Δkmean = Δkμ,
        Δkmax = ΔkM,
        pick = pick,
        fig = fig
    )
end

# -----------------------------
# Run
# -----------------------------
tvals = 10 .^ range(log10(0.01), log10(100.0); length=40)

out = run_and_plot_9(
    S=80,                 # bump to 120 if runtime is ok
    connectance=0.05,
    n=60,
    σA=0.25,
    rew_frac=0.10,        # Fix B amplitude; try 0.01, 0.05, 0.10, 0.25
    γ=0.10,               # Fix A margin; try 0.05, 0.10, 0.20
    tvals=tvals,
    do_you_shift_to_margin=true
)