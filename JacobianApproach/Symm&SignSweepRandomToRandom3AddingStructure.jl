function build_ER_degcv(S::Int, conn::Float64, mean_abs::Float64, mag_cv::Float64,
                        rho_mag::Float64, rho_sign::Float64, deg_cv::Float64;
                        rng::AbstractRNG=MersenneTwister(42))

    @assert 0.0 ≤ conn ≤ 1.0
    @assert mean_abs > 0
    @assert mag_cv ≥ 0
    @assert 0.0 ≤ rho_mag < 1.0 "use < 1 for numerical stability"
    @assert 0.0 ≤ rho_sign ≤ 1.0
    @assert deg_cv ≥ 0

    # Lognormal parameters for magnitudes
    σ2 = log(1 + mag_cv^2)
    σ  = sqrt(σ2)
    μ  = log(mean_abs) - σ2/2
    LN = LogNormal(μ, σ)

    # Heterogeneous out-degree propensities
    # (mean 1, CV = deg_cv, normalized to preserve global conn)
    if deg_cv > 0
        raw = rand(rng, LogNormal(-0.5 * log(1 + deg_cv^2), sqrt(log(1 + deg_cv^2))), S)
        w = raw ./ mean(raw)
    else
        w = ones(S)
    end

    # Magnitude correlation
    Lmag = cholesky(Symmetric([1.0 rho_mag; rho_mag 1.0])).L
    stdN = Normal()

    A = zeros(Float64, S, S)
    for i in 1:S-1, j in i+1:S
        # Adjusted connectance per source
        p_ij = conn * w[i]
        p_ji = conn * w[j]

        if rand(rng) < p_ij
            z = randn(rng, 2)
            z .= Lmag * z
            m1 = quantile(LN, cdf(stdN, z[1]))
            m2 = quantile(LN, cdf(stdN, z[2]))
            s1 = ifelse(rand(rng) < 0.5, 1.0, -1.0)
            s2 = ifelse(rand(rng) < rho_sign, -s1, s1)
            A[i,j] = s1 * m1
            A[j,i] = s2 * m2
        elseif rand(rng) < p_ji
            m  = rand(rng, LN)
            s  = ifelse(rand(rng) < 0.5, 1.0, -1.0)
            A[j,i] = s * m
        end
    end
    return A
end

function run_axis_grid(; axis::Symbol, levels::AbstractVector, reps::Int=50,
                       t_vals::AbstractVector,
                       S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.5, mag_cv::Float64=0.60,
                       u_mean::Float64=1.0, u_cv::Float64=0.6,
                       IS_target::Float64=0.5, seed::Int=20251110)

    base = UInt(seed)
    nthreads_used = Threads.nthreads()
    buckets = [NamedTuple[] for _ in 1:nthreads_used]
    combos = 1:length(levels)

    @threads for idx in combos
        tid = threadid()
        local_rows = buckets[tid]

        lvl_sign  = (axis == :sign  || axis == :both) ? levels[idx] : 1.0
        lvl_degcv = (axis == :degcv || axis == :both) ? levels[idx] : 0.0

        rng_iter = Random.Xoshiro(base ⊻ UInt(idx*7919) ⊻ UInt(tid*4099))

        nt  = length(t_vals)
        n   = zeros(Int,      nt)
        sx  = zeros(Float64,  nt)
        sy  = zeros(Float64,  nt)
        sxx = zeros(Float64,  nt)
        syy = zeros(Float64,  nt)
        sxy = zeros(Float64,  nt)
        sad = zeros(Float64,  nt)

        # tiny closure to compute the full R̃med series using a cached Schur
        series_from_Ju = let tvals = t_vals
            function (J::AbstractMatrix{<:Real}, u::AbstractVector{<:Real})
                F = schur(Matrix{Float64}(J))           # J = Z*T*Z'
                Z, T = F.Z, F.T
                w = u .^ 2
                s = sqrt.(w)
                logW = log(sum(w))
                out = Vector{Float64}(undef, length(tvals))
                @inbounds for (k,t) in pairs(tvals)
                    Et = exp(t .* T)                     # uses real Schur blocks
                    E  = Z * Et * Z'
                    # weight rows by sqrt(w)
                    @views E .= E .* reshape(s, 1, :)
                    out[k] = -(log(tr(E*E')) - logW) / (2t)
                end
                out
            end
        end

        for rep in 1:reps
            rng = Random.Xoshiro(rand(rng_iter, UInt))

            # draw two independent TRdeg communities (same knobs) and scale to IS_target
            A0 = build_ER_degcv(S, conn, mean_abs, mag_cv, 0.0, lvl_sign, lvl_degcv; rng=rng)
            A1 = build_ER_degcv(S, conn, mean_abs, mag_cv, 0.0, lvl_sign, lvl_degcv; rng=rng)

            is0 = realized_IS(A0); is1 = realized_IS(A1)
            (is0 == 0 || is1 == 0) && continue
            A0 .*= IS_target / is0
            A1 .*= IS_target / is1

            u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

            J0 = jacobian(A0, u)
            J1 = jacobian(A1, u)

            f = series_from_Ju(J0, u)   # length nt
            g = series_from_Ju(J1, u)

            @inbounds for k in 1:nt
                x = f[k]; y = g[k]
                if isfinite(x) && isfinite(y)
                    n[k]   += 1
                    sx[k]  += x
                    sy[k]  += y
                    sxx[k] += x*x
                    syy[k] += y*y
                    sxy[k] += x*y
                    sad[k] += abs(y - x)
                end
            end
        end

        epsd = 1e-12
        for k in 1:nt
            t = t_vals[k]
            if n[k] == 0
                push!(local_rows, (; axis, case=idx, deg_cv=lvl_degcv, rho_sign=lvl_sign,
                                   t, r2=0.0, absdiff=NaN))
                continue
            end
            ybar = sy[k] / n[k]
            sse  = syy[k] + sxx[k] - 2*sxy[k]                 # ∑(y-x)^2
            sst  = max(syy[k] - n[k]*ybar*ybar, epsd)         # ∑(y-ȳ)^2
            r2   = max(0.0, 1.0 - sse/sst)
            ad   = sad[k] / n[k]
            push!(local_rows, (; axis, case=idx, deg_cv=lvl_degcv, rho_sign=lvl_sign, t, r2, absdiff=ad))
        end

        buckets[tid] = local_rows
    end

    DataFrame(vcat(buckets...))
end

function plot_axis_grid(df::DataFrame, axis::Symbol; title::String, absdiff::Bool=false)
    # Determine layout (3x3 grid)
    levels = sort(unique(df.case))
    ncases = length(levels)
    ncols = 3
    nrows = ceil(Int, ncases / ncols)

    # Create figure
    fig = Figure(size=(1100, 900))
    Label(fig[0, 1:3], title; fontsize=20, font=:bold, halign=:left)

    # Loop through all cases
    for (k, cidx) in enumerate(levels)
        sub = df[(df.axis .== axis) .& (df.case .== cidx), :]
        isempty(sub) && continue

        # Extract parameter values for labeling
        degcv_val  = hasproperty(sub, :deg_cv)   ? first(unique(sub.deg_cv))   : 0.0
        sign_val   = hasproperty(sub, :rho_sign) ? first(unique(sub.rho_sign)) : 0.0

        r, c = divrem(k-1, ncols)
        ax = Axis(fig[r+1, c+1];
                  xscale=log10,
                  xlabel="t",
                  ylabel=absdiff ? "|ΔRmed|" : (c == 0 ? "R²" : ""),
                  limits=((minimum(df.t), maximum(df.t)), (-0.05, 1.05)))

        # Dynamic subplot titles based on axis type
        if axis == :degcv
            ax.title = @sprintf("deg_cv = %.2f", degcv_val)
        elseif axis == :sign
            ax.title = @sprintf("ρsign = %.2f", sign_val)
        elseif axis == :both
            ax.title = @sprintf("deg_cv = %.2f, ρsign = %.2f", degcv_val, sign_val)
        else
            ax.title = "case $cidx"
        end

        sort!(sub, :t)
        if absdiff
            lines!(ax, sub.t, sub.absdiff; linewidth=2, color=:steelblue)
            scatter!(ax, sub.t, sub.absdiff; color=:steelblue)
        else
            lines!(ax, sub.t, sub.r2; linewidth=2, color=:steelblue)
            scatter!(ax, sub.t, sub.r2; color=:steelblue)
        end
    end

    display(fig)
end

levels_deg = range(0.00, 1.50; length=9)
levels_sig = range(0.00, 1.00; length=9)

df_degcv = run_axis_grid(; axis=:degcv, levels=levels_deg,  reps=50, t_vals=t_vals, S=S, conn=conn,
                         mean_abs=mean_abs, mag_cv=mag_cv, u_mean=u_mean, u_cv=u_cv,
                         IS_target=IS_target, seed=seed)

df_sign  = run_axis_grid(; axis=:sign,  levels=levels_sig, reps=50, t_vals=t_vals, S=S, conn=conn,
                         mean_abs=mean_abs, mag_cv=mag_cv, u_mean=u_mean, u_cv=u_cv,
                         IS_target=IS_target, seed=seed)

# “both” varies them together; keep sign in bounds
df_both  = run_axis_grid(; axis=:both,  levels=levels_sig, reps=50, t_vals=t_vals, S=S, conn=conn,
                         mean_abs=mean_abs, mag_cv=mag_cv, u_mean=u_mean, u_cv=u_cv,
                         IS_target=IS_target, seed=seed)

fig_deg  = plot_axis_grid(df_degcv, :degcv; title="Grid 1 — Degree heterogeneity only (|ΔRmed|) Recycling pairs", absdiff=true)
fig_deg  = plot_axis_grid(df_degcv, :degcv; title="Grid 1 — Degree heterogeneity only (R²) Recycling pairs", absdiff=false)

fig_sign = plot_axis_grid(df_sign,  :sign;  title="Grid 2 — Sign antisymmetry only (|ΔRmed|) Recycling pairs", absdiff=true)
fig_sign = plot_axis_grid(df_sign,  :sign;  title="Grid 2 — Sign antisymmetry only (R²) Recycling pairs", absdiff=false)

fig_both = plot_axis_grid(df_both,  :both;  title="Grid 3 — Degree + Sign combined (|ΔRmed|) Recycling pairs", absdiff=true)
fig_both = plot_axis_grid(df_both,  :both;  title="Grid 3 — Degree + Sign combined (R²) Recycling pairs", absdiff=false)
