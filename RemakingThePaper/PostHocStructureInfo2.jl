# ---------- tiny helpers ----------
_cv(x::AbstractVector) = (isempty(x) || mean(x)==0) ? NaN : std(x)/mean(x)

# degree vectors from your bipartite A (consumers = rows R+1:S to resource cols 1:R)
function _degree_vectors(A, R::Int)
    S = size(A,1); C = S - R
    # consumer out-degree to resources
    degc = Vector{Int}(undef, C)
    @inbounds for ic in 1:C
        degc[ic] = count(!iszero, A[R+ic, 1:R])
    end
    # resource in-degree from consumers
    degr = Vector{Int}(undef, R)
    @inbounds for jr in 1:R
        degr[jr] = count(!iszero, A[(R+1):S, jr])
    end
    return degc, degr
end

##############################################
# Plotting: sample abundance & degree dists
##############################################
"""
    plot_sample_distributions_from_results(df; nsample=9, seed=nothing)

Randomly samples up to 9 rows (communities) from the results DataFrame and plots:
- A 3×3 grid of abundance histograms of `B_eq`
- A 3×3 grid of degree histograms (consumer out-degree, resource in-degree) from `A` in `p_final`

Returns (fig_abund, fig_degrees).
"""
function plot_sample_distributions_from_results(df::DataFrame; nsample::Int=9, seed=nothing)
    n = nrow(df)
    n == 0 && error("DataFrame is empty.")
    rng = MersenneTwister(seed === nothing ? rand(1:typemax(Int)) : seed)
    ns = min(nsample, n)
    idxs = sort!(rand(rng, 1:n, ns))

    # --- Helper functions ---
    _cv(x) = (isempty(x) || mean(x) == 0) ? NaN : std(x) / mean(x)
    function _degree_vectors(A::AbstractMatrix, R::Int)
        S = size(A, 1); C = S - R
        deg_cons = [count(!iszero, A[R + ic, 1:R]) for ic in 1:C]
        deg_res  = [count(!iszero, A[R+1:S, jr]) for jr in 1:R]
        return deg_cons, deg_res
    end

    # --- Abundances: 3x3 grid ---
    ncols = 3
    nrows = ceil(Int, ns / ncols)
    fig_abund = Figure(; size = (1100, 720))
    Label(fig_abund[0, 1:ncols], "Species Abundance Distributions", fontsize = 20, font = :bold, halign = :left)
    for (k, ri) in enumerate(idxs)
        row = df[ri, :]
        u = row.B_eq :: AbstractVector
        r = div(k - 1, ncols) + 1
        c = ((k - 1) % ncols) + 1
        ax = Axis(fig_abund[r, c];
            xlabel = "Abundance u",
            ylabel = "Count",
            title  = "Comm $(ri) (mean=$(round(mean(u), digits=2)), cv=$(round(_cv(u), digits=2)))"
        )
        hist!(ax, u; bins = 30)
    end

    # --- Degrees: 3x3 grid ---
    fig_degrees = Figure(; size = (1100, 720))
    Label(fig_degrees[0, 1:3], "Degree Distributions", fontsize = 20, font = :bold, halign = :left)
    for (k, ri) in enumerate(idxs)
        row = df[ri, :]
        R  = Int(row.R)
        K, A = row.p_final
        degc, degr = _degree_vectors(A, R)

        r = div(k - 1, ncols) + 1
        c = ((k - 1) % ncols) + 1
        ax = Axis(fig_degrees[r, c];
            xlabel = "Degree k",
            ylabel = "Count",
            title  = "Comm $(ri): cv_cons=$(round(_cv(degc), digits=2)), cv_res=$(round(_cv(degr), digits=2))"
        )

        maxc = maximum(degc; init=0)
        maxr = maximum(degr; init=0)
        bins_c = collect(0:(max(maxc, maxr) + 1))
        hist!(ax, degc; bins=bins_c, color=(RGBAf(70/255, 130/255, 180/255, 0.5)), label="Consumers")
        hist!(ax, degr; bins=bins_c, color=(RGBAf(1.0, 0.55, 0.0, 0.5)), label="Resources")
        axislegend(ax; position=:rt, framevisible=false)
    end

    display(fig_abund)
    display(fig_degrees)
    return fig_abund, fig_degrees
end

#########################################################
# Structural range report (post-hoc on results DF)
#########################################################
"""
    report_structure_range_from_results(df)

Prints min | median | max for realized structural metrics across rows of `df`.
Uses the **realized** columns produced by RunSimulations:
- :conn
- :deg_cv_cons_out_realized
- :deg_cv_res_in_realized
- :realized_mod
- :sigma_over_min_d_full
- :lambda_max_full

Ignores missing metrics gracefully if columns aren’t present.
"""
function report_structure_range_from_results(df::DataFrame)
    function mm3(v)
        x = collect(skipmissing(v))
        isempty(x) && return (NaN, NaN, NaN)
        return (minimum(x), median(x), maximum(x))
    end

    metrics = [
        (:conn,                        "connectance"),
        (:deg_cv_cons_out_realized,    "deg_cv_cons_out"),
        (:deg_cv_res_in_realized,      "deg_cv_res_in"),
        (:realized_mod,                "within_fraction"),
        (:sigma_over_min_d_full,       "sigma_over_min_d"),
        (:lambda_max_full,             "lambda_max")
    ]

    println("=== Structural Range (min | median | max) over $(nrow(df)) communities ===")
    for (sym, label) in metrics
        if sym in names(df)
            mn, md, mx = mm3(df[!, sym])
            println(rpad(label, 22), ": ",
                round(mn, digits=3), " | ",
                round(md, digits=3), " | ",
                round(mx, digits=3))
        else
            @warn "Column $(sym) not found; skipping in report."
        end
    end
    return nothing
end

# 1) sample plots
plot_sample_distributions_from_results(sim_results; nsample=6)

# 2) realized-structure range summary (prints to REPL)
report_structure_range_from_results(sim_results)
