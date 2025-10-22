"""
add_community_diagnostics!(df; k_consumer_cutoff=0.015)

Adds columns to `df` describing *pre-modification* fragility:
- gap_A            = 1 - spectral_radius(A)          (distance-to-pole)
- resolvent_norm   = opnorm(inv(I - A))               (|| (I-A)^(-1) ||_2)
- cond_IminusA     = cond(I - A)                      (κ2(I-A))
- u_min            = minimum(B_eq)                    (rarest species)
- u_cv             = std(B_eq) / mean(B_eq)           (SAD heterogeneity)
- partner_leverage = mean_i[ var(contrib_i) * (1/deg_i) ] over consumers
  where contrib_i = A[i, resources] .* B_eq[resources] (only nonzero links)
"""
function add_community_diagnostics!(df::DataFrame; k_consumer_cutoff::Float64=0.01)
    N = nrow(df)
    gap_A          = Vector{Float64}(undef, N)
    resolvent_norm = Vector{Float64}(undef, N)
    cond_IminusA   = Vector{Float64}(undef, N)
    u_min          = Vector{Float64}(undef, N)
    u_cv           = Vector{Float64}(undef, N)
    partner_lev    = Vector{Float64}(undef, N)

    for (i,row) in enumerate(eachrow(df))
        K, A = row.p_final
        u    = row.B_eq
        S    = length(K)

        # distance to pole & conditioning
        λ = eigvals(A)
        ρ = maximum(abs, λ)
        gap_A[i] = 1.0 - ρ
        try
            M = I - A
            resolvent_norm[i] = opnorm(inv(M)) # 2-norm
            cond_IminusA[i]   = cond(M)        # κ2
        catch
            resolvent_norm[i] = Inf
            cond_IminusA[i]   = Inf
        end

        # SAD lower tail / heterogeneity
        u_min[i] = minimum(u)
        m = mean(u)
        u_cv[i]  = m == 0 ? NaN : std(u) / m

        # partner leverage (consumers only)
        # identify consumers via K threshold
        consumers = findall(k -> k <= k_consumer_cutoff, K)
        resources = findall(k -> k  > k_consumer_cutoff, K)

        levs = Float64[]
        for ci in consumers
            # nonzero links to resources
            nz = findall(j -> A[ci,j] != 0.0, resources)
            if isempty(nz); continue; end
            # contributions consumer receives from those resources at equilibrium
            contrib = [A[ci, resources[j]] * u[resources[j]] for j in nz]
            v = length(contrib) > 1 ? var(contrib) : 0.0
            deg = length(nz)
            push!(levs, v * (1/deg))
        end
        partner_lev[i] = isempty(levs) ? NaN : mean(levs)
    end

    df.gap_A          = gap_A
    df.resolvent_norm = resolvent_norm
    df.cond_IminusA   = cond_IminusA
    df.u_min          = u_min
    df.u_cv           = u_cv
    df.partner_leverage = partner_lev
    return df
end

"""
add_predictability_errors!(df; step=1, metrics=[:resilience,:reactivity,:rt_pulse,:after_press],
                           relative=true)

Adds columns:
  err_<metric>      (absolute or relative error for given step w.r.t. _full)
  err_mean_all      mean over listed metrics
"""
function add_predictability_errors!(df::DataFrame;
    step::Int=1,
    metrics=[:resilience, :reactivity, :rt_pulse, :after_press],
    relative::Bool=true
)
    eps = 1e-6
    for m in metrics
        fullco = Symbol("$(m)_full")
        stepco = Symbol("$(m)_S$(step)")
        # @assert fullco in names(df) "Missing $(fullco)"
        # @assert stepco in names(df) "Missing $(stepco)"

        err = relative ?
            abs.(df[!, stepco] .- df[!, fullco]) ./ (abs.(df[!, fullco]) .+ eps) :
            abs.(df[!, stepco] .- df[!, fullco])

        df[!, Symbol("err_$(m)")] = err
    end
    # mean across included metrics
    errcols = Symbol.("err_" .* String.(metrics))
    df.err_mean_all = [mean(collect(skipmissing(row[errcols]))) for row in eachrow(df)]
    return df
end

# small helpers (same style you already use)
trimmed_mean(v::AbstractVector{<:Real}, trim::Float64=0.10) = begin
    n = length(v)
    n == 0 && return NaN
    t = clamp(floor(Int, trim*n), 0, max(0, div(n-1,2)))
    vs = sort!(collect(v))
    vs = vs[(t+1):(n-t)]
    mean(vs)
end

function moving_avg(v::AbstractVector{<:Real}, k::Int=5)
    k ≤ 1 && return collect(v)
    n = length(v); out = similar(collect(v))
    h = k ÷ 2
    for i in 1:n
        lo = max(1, i-h); hi = min(n, i+h)
        out[i] = mean(@view v[lo:hi])
    end
    out
end

"""
plot_error_vs_diagnostics(df;
    diagnostics=[:gap_A,:u_min,:u_cv,:partner_leverage],
    metrics=[:resilience,:reactivity,:rt_pulse,:after_press],
    binning=:quantile, n_bins=12, trim_frac=0.10, smooth_window=5,
    error_bars=false, resolution=(1100,800), px_per_unit=6.0)

Rows = metrics, Cols = diagnostics. Each panel: binned mean error (trimmed, smoothed).
"""
function plot_error_vs_diagnostics(df::DataFrame;
    diagnostics = [:gap_A, :u_min, :u_cv, :partner_leverage],
    metrics     = [:resilience, :reactivity, :rt_pulse, :after_press],
    metric_names = Dict(:resilience=>"Resilience", :reactivity=>"Reactivity",
                        :rt_pulse=>"Return time", :after_press=>"Persistence"),
    diag_names   = Dict(:gap_A=>"1 − ρ(A)", :u_min=>"min(B_eq)", :u_cv=>"CV(B_eq)",
                        :partner_leverage=>"Partner leverage (consumers)"),
    binning::Symbol = :quantile,           # :quantile or :equal
    n_bins::Int = 12,
    trim_frac::Float64 = 0.10,
    smooth_window::Int = 5,
    error_bars::Bool = false,
    resolution=(1100, 800),
    px_per_unit=6.0,
    save_plot=false,
    filename="Figures/diagnostic_thresholds.png"
)
    # sanity: make sure diagnostic columns exist
    # for d in diagnostics
    #     @assert d in names(df) "Missing diagnostic column $(d). Run add_community_diagnostics! first."
    # end
    # # and error columns
    # for m in metrics
    #     @assert Symbol("err_$(m)") in names(df) "Missing err_$(m). Run add_predictability_errors! first."
    # end

    fig = Figure(size=resolution)
    colors = [:black]  # single line per panel (clean)
    Label(fig[0, :], "Diagnostic thresholds step 2", fontsize = 20, font = :bold, halign = :left)

    for (mi, m) in enumerate(metrics), (di, d) in enumerate(diagnostics)
        ax = Axis(fig[mi, di];
                  title = "$(get(metric_names, m, String(m))) vs $(get(diag_names, d, String(d)))",
                  xlabel= get(diag_names, d, String(d)),
                  ylabel= mi==1 ? "Prediction error" : "",
                  xgridvisible=false, ygridvisible=false)

        x = df[!, d]
        y = df[!, Symbol("err_$(m)")]
        mask = isfinite.(x) .& isfinite.(y)
        x, y = x[mask], y[mask]
        if isempty(x)
            continue
        end

        mx, my, sy = Float64[], Float64[], Float64[]
        if binning == :quantile
            ord = sortperm(x)
            n   = length(x)
            chunk = max(1, fld(n, n_bins))
            for b in 1:n_bins
                lo = (b-1)*chunk + 1
                hi = (b==n_bins) ? n : min(b*chunk, n)
                if lo > hi; continue; end
                idx = ord[lo:hi]
                push!(mx, mean(x[idx]))
                push!(my, trimmed_mean(y[idx], trim_frac))
                push!(sy, std(y[idx]))
            end
        else
            xmin, xmax = minimum(x), maximum(x)
            edges = range(xmin, xmax, length=n_bins+1)
            bix = searchsortedlast.(Ref(edges), x)
            for b in 1:n_bins
                idx = findall(bix .== b)
                if !isempty(idx)
                    push!(mx, mean(x[idx]))
                    push!(my, trimmed_mean(y[idx], trim_frac))
                    push!(sy, std(y[idx]))
                end
            end
        end

        # smooth
        smx = moving_avg(mx, smooth_window)
        smy = moving_avg(my, smooth_window)
        ssy = moving_avg(sy, smooth_window)

        lines!(ax, smx, smy; color=colors[1], linewidth=2.5)
        if error_bars
            errorbars!(ax, smx, smy, ssy; color=colors[1])
        end
    end

    display(fig)
    save_plot && save(filename, fig; px_per_unit=px_per_unit)
end

# 0) Start from your sim_results or stable_sim_results
df = sim_results  # or stable_sim_results

# 1) Add diagnostics (community construction side)
add_community_diagnostics!(df; k_consumer_cutoff=0.01)

# 2) Add predictability errors for step 1 (soft rewiring)
add_predictability_errors!(
    df; step=2,
    metrics=[:resilience,:reactivity,:resilienceE, :reactivityE,:rt_pulse,:after_press],
    relative=true
)

# 3) Visualise thresholds (quantile-binned, trimmed, smoothed)
plot_error_vs_diagnostics(
    df;
    diagnostics=[:gap_A, :u_min, :u_cv, :partner_leverage],
    metrics=[:resilience,:reactivity,:rt_pulse,:after_press],
    binning=:quantile, n_bins=12, trim_frac=0.10, smooth_window=5,
    error_bars=false, resolution=(1100,800), save_plot=true,
    filename="Figures/diagnostic_thresholds.png"
)