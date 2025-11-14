# ----- predictors (match your derivations) -----
# small-t predictor:
#   :uniform  ->  -mean(diag(J))
#   :biomass  ->  -sum(J_ii * u_i^2) / sum(u_i^2)
small_t_predictor(J::AbstractMatrix, u::AbstractVector; perturbation::Symbol=:biomass) =
    perturbation === :uniform  ?  -mean(diag(J)) :
    perturbation === :biomass  ?  -sum(diag(J) .* (u.^2)) / sum(u.^2) :
    error("Unknown perturbation: $perturbation")

# "edge" predictor (large-t): resilience = -max real eigenvalue
resilience(J::AbstractMatrix) = -maximum(real(eigvals(J)))

# convenience: one community, built exactly as you asked
function build_one_community(; S::Int, conn::Float64, mean_abs::Float64, mag_cv::Float64,
                              rho_sym::Float64, u_mean::Float64, u_cv::Float64,
                              t_vals::AbstractVector, perturbation::Symbol=:biomass,
                              rng::AbstractRNG=Random.default_rng())
    # A, u exactly as in your pipeline
    A = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs,
                                mag_cv=mag_cv, rho_sym=rho_sym, rng=rng)
    u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)

    # Jacobian via your function (diagonal determined by u exactly as before)
    J = jacobian(A, u)

    # series and endpoints
    rmed = [median_return_rate(J, u; t=t, perturbation=perturbation) for t in t_vals]
    p_small = small_t_predictor(J, u; perturbation=perturbation)
    rho = resilience(J)

    return (; A, u, J, rmed, p_small, rho)
end

"""
edge_predictor(J)
Large-time asymptotic for R̃med(t): R̃med(t→∞) ≈ resilience = -max(real(eigvals(J))).
"""
function edge_predictor(J::AbstractMatrix)
    λmax = maximum(real.(eigvals(J)))
    return -λmax
end

# ---------- 2) Ensemble runner ----------
# build N communities; return tidy DF for plotting + a DF with predictors
function build_ensemble(; N::Int=60, S::Int=120, conn::Float64=0.10, mean_abs::Float64=0.50,
                         mag_cv::Float64=0.60, rho_sym::Float64=1.0, u_mean::Float64=1.0,
                         u_cv::Float64=0.6, t_vals=10 .^ range(-2, 2; length=40),
                         perturbation::Symbol=:biomass, seed::Int=20251111)

    rng0 = Random.Xoshiro(seed)
    rows = NamedTuple[]
    preds = NamedTuple[]

    for i in 1:N
        rng = Random.Xoshiro(rand(rng0, UInt64))
        C = build_one_community(; S, conn, mean_abs, mag_cv, rho_sym, u_mean, u_cv,
                                t_vals, perturbation, rng)
        # time series rows
        for (k, t) in pairs(t_vals)
            push!(rows, (; id=i, t=t, rmed=C.rmed[k]))
        end
        # predictors per community
        push!(preds, (; id=i, small=C.p_small, edge=C.rho))
    end

    return DataFrame(rows), DataFrame(preds)
end

# ---------- 3) Diagnostics over t ----------
"""
diagnostics_over_t(df, preds, t_vals)
For each t:
- corr_small(t) = corr( rmed_k(t), small_k )
- corr_edge(t)  = corr( rmed_k(t), edge_k  )
- var_rmed(t)   = var( rmed_k(t) )
Also: regress rmed on both predictors to get residual variance (optional, commented).
"""
function diagnostics_over_t(df::DataFrame, preds::DataFrame, t_vals)
    out = NamedTuple[]
    byid = groupby(df, :id)
    # make a wide matrix R (N × T) and vectors s, e
    ids = sort(unique(df.id))
    N = length(ids); T = length(t_vals)
    R = Matrix{Float64}(undef, N, T)
    for (i, id) in enumerate(ids)
        sub = df[(df.id .== id), :]
        sort!(sub, :t)
        R[i, :] = sub.rmed
    end
    ps = preds[sortperm(preds.id), :]
    s = ps.small; e = ps.edge

    for (j, t) in enumerate(t_vals)
        r = R[:, j]
        cs = cor(r, s)
        ce = cor(r, e)
        vr = var(r)
        # Optional: residual variance after linear regression on both predictors
        # X = hcat(fill(1.0, N), s, e)
        # β = X \ r
        # resid = r .- X*β
        # vres = var(resid)
        push!(out, (; t, corr_small=cs, corr_edge=ce, var_rmed=vr)) #, var_resid=vres))
    end
    return DataFrame(out)
end

# ---------- 4) Plots ----------
"""
plot_diagnostics(df, preds, t_vals; title)
Panels:
(A) spaghetti of R̃med(t) (thin) with mean ± sd band (thick)
(B) correlations corr_small(t) and corr_edge(t) vs t
(C) variance Var[R̃med(t)] vs t
(D) endpoint scatters: small-t (t_min) vs predictor and large-t (t_max) vs resilience
"""
function plot_diagnostics(df::DataFrame, preds::DataFrame, t_vals; title="R̃med diagnostics (uniform)")
    ids = sort(unique(df.id))
    N = length(ids); T = length(t_vals)

    # assemble matrix
    R = Matrix{Float64}(undef, N, T)
    for (i, id) in enumerate(ids)
        sub = df[(df.id .== id), :]
        sort!(sub, :t)
        R[i, :] = sub.rmed
    end
    μ = vec(mean(R, dims=1))
    σ = vec(std(R, dims=1))

    di = diagnostics_over_t(df, preds, t_vals)

    fig = Figure(size=(1150, 750))
    Label(fig[0,1:3], title; fontsize=20, font=:bold, halign=:left)

    # (A) R̃med(t) ensemble
    axA = Axis(fig[1,1]; title="R̃med(t) across communities", xscale=log10, xlabel="t", ylabel="R̃med")
    for i in 1:N
        lines!(axA, t_vals, R[i, :]; color=(0.3,0.5,0.8,0.2))
    end
    lines!(axA, t_vals, μ; color=:steelblue, linewidth=2)
    band!(axA, t_vals, μ .- σ, μ .+ σ; color=(0.3,0.5,0.8,0.2))

    # (B) correlations vs t
    axB = Axis(fig[1,2]; title="Correlation with predictors", xscale=log10, xlabel="t", ylabel="corr")
    lines!(axB, di.t, di.corr_small; color=:darkgreen, linewidth=2, label="corr(small-t predictor)")
    lines!(axB, di.t, di.corr_edge;  color=:crimson,   linewidth=2, label="corr(edge predictor)")
    axislegend(axB; position=:lb, framevisible=false)

    # (C) variance vs t
    axC = Axis(fig[1,3]; title="Across-community variance", xscale=log10, xlabel="t", ylabel="Var[R̃med(t)]")
    lines!(axC, di.t, di.var_rmed; color=:gray35, linewidth=2)

    # (D1) small-t scatter
    t_small = first(t_vals)
    r_small = R[:, 1]
    axD1 = Axis(fig[2,1]; title="Small-t: R̃med vs small-t predictor", xlabel="−mean(diag J)", ylabel="R̃med(t_min)")
    scatter!(axD1, preds.small, r_small; color=:darkgreen)
    # trend
    a = [ones(N) preds.small] \ r_small
    xs = range(minimum(preds.small), maximum(preds.small); length=100)
    lines!(axD1, xs, a[1] .+ a[2].*xs; color=:darkgreen, linewidth=2)

    # (D2) large-t scatter
    t_large = last(t_vals)
    r_large = R[:, end]
    axD2 = Axis(fig[2,2]; title="Large-t: R̃med vs resilience", xlabel="resilience (−max Re λ)", ylabel="R̃med(t_max)")
    scatter!(axD2, preds.edge, r_large; color=:crimson)
    b = [ones(N) preds.edge] \ r_large
    xs2 = range(minimum(preds.edge), maximum(preds.edge); length=100)
    lines!(axD2, xs2, b[1] .+ b[2].*xs2; color=:crimson, linewidth=2)

    # (D3) correlation summaries at endpoints
    axD3 = Axis(fig[2,3]; title="Endpoint correlations", ylabel="corr", xticks=(1:2, ["small-t","large-t"]))
    corr_small_end = cor(r_small, preds.small)
    corr_edge_end  = cor(r_large, preds.edge)
    scatter!(axD3, [1,2], [corr_small_end, corr_edge_end]; markersize=12, color=[:darkgreen, :crimson])
    hlines!(axD3, 0.0; color=:gray70, linestyle=:dash)

    display(fig)
end

t_vals = 10 .^ range(-2, 2; length=40)

df, preds = build_ensemble(
    ; N=80, S=120, conn=0.10, mean_abs=0.5, mag_cv=0.60,
    rho_sym=1.0, u_mean=1.0, u_cv=0.6,
    t_vals=t_vals, perturbation=:uniform, seed=42
)

# 2) Plot the diagnostics
plot_diagnostics(df, preds, t_vals; title="R̃med diagnostics — trophic ER (perturbation=:uniform)")