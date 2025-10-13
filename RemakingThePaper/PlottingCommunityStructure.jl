##############################################
# Plotting
##############################################
function plot_sample_distributions(comms::Vector{Community}; nsample::Int=6, seed=nothing)
    rng = MersenneTwister(seed)
    ns = min(nsample, length(comms))
    idxs = rand(rng, 1:length(comms), ns)
    if isnothing(seed)
        seed = rand(1:typemax(Int))
    end

    ncols = 3
    nrows = ceil(Int, ns / ncols)
    fig_abund = Figure(; size = (1100, 300 * nrows))
    for (k, idx) in enumerate(idxs)
        row = div(k - 1, ncols) + 1
        col = ((k - 1) % ncols) + 1
        u = comms[idx].ustar
        ax = Axis(fig_abund[row, col];
            xlabel = "u",
            ylabel = "count",
            title = "Comm $(idx) (mean=$(round(mean(u), digits=2)), cv=$(round(_cv(u), digits=2)))"
        )
        hist!(ax, u; bins = 30)
    end

    fig_degrees = Figure(; size = (1100, 220 * ns))
    for (r, idx) in enumerate(idxs)
        A = comms[idx].A; R = comms[idx].R
        degc, degr = degree_vectors(A, R)
        ax1 = Axis(fig_degrees[r, 1];
            xlabel = "k_out (consumers)",
            ylabel = "count",
            title = "Comm $(idx) cv_cons=$(round(_cv(degc), digits=2))"
        )
        ax2 = Axis(fig_degrees[r, 2];
            xlabel = "k_in (resources)",
            ylabel = "count",
            title = "Comm $(idx) cv_res=$(round(_cv(degr), digits=2))"
        )
        bins_c = 0:(maximum(degc) + 1)
        bins_r = 0:(maximum(degr) + 1)
        hist!(ax1, degc; bins = bins_c)
        hist!(ax2, degr; bins = bins_r)
    end
    display(fig_abund)
    display(fig_degrees)
    return fig_abund, fig_degrees
end

#########################################################
# Structural range report
#########################################################
function report_structure_range(df::DataFrame)
    function mm3(x)
        x = skipmissing(x)
        return (minimum(x), median(x), maximum(x))
    end
    metrics = [:conn, :deg_cv_cons_out, :deg_cv_res_in, :within_fraction, :sigma_nonzero, :lambda_max]
    println("=== Structural Range (min | median | max) over $(nrow(df)) communities ===")
    for m in metrics
        if hasproperty(df, m)
            (mn, md, mx) = mm3(df[!, m])
            println(rpad(string(m), 22), ": ",
                    round(mn,digits=3), " | ", round(md,digits=3), " | ", round(mx,digits=3))
        end
    end
end

#################
# Example driver
#################
comms, summary = generate_batch(400; S=200, R=120, seed=123)
report_structure_range(summary)
fig1, fig2 = plot_sample_distributions(comms; nsample=6)