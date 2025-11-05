using CairoMakie, DataFrames, Statistics, Printf, ColorSchemes
###############################
# 1) Correlation grid
###############################
function plot_correlations(
    df::DataFrame;
    steps=["reshuf", "thr", "row", "uni", "rarer", "rew"],
    metrics=[:res, :rea, :rmed, :long_rmed],
    title="Correlations across stability metrics"
)
    labels = Dict(
        :res=>"Resilience",
        :rea=>"Reactivity",
        :rmed=>"Rmed (t=0.01)",
        :long_rmed=>"Rmed (t=5.0)"
    )
    colors = [:steelblue, :orangered, :seagreen, :purple]

    fig = Figure(size=(1100, 725))
    Label(fig[0, 2:6], title; fontsize=18, font=:bold, halign=:left)

    for (mi, m) in enumerate(metrics)
        xname = Symbol(m, :_full)
        for (si, s) in enumerate(steps)
            yname = Symbol(m, :_, s)

            xs = df[!, xname]
            ys = df[!, yname]
            x, y = Float64[], Float64[]

            @inbounds for i in eachindex(xs)
                xi, yi = xs[i], ys[i]
                if xi isa Real && yi isa Real && isfinite(xi) && isfinite(yi)
                    push!(x, float(xi)); push!(y, float(yi))
                end
            end

            ax = Axis(fig[mi, si];
                title="$(labels[m]) — $(s)",
                xgridvisible=false, ygridvisible=false,
                xticklabelsize=10, yticklabelsize=10,
                xlabelsize=12, ylabelsize=12, titlesize=11,
                xlabel=string(xname), ylabel=string(yname)
            )

            if isempty(x)
                continue
            end

            mn, mx = extrema(vcat(x, y))
            pad = 0.05 * (mx - mn)
            mn -= pad; mx += pad
            limits!(ax, (mn, mx), (mn, mx))

            scatter!(ax, x, y; color=colors[mi], markersize=3, alpha=0.4)
            lines!(ax, [mn, mx], [mn, mx]; color=:black, linestyle=:dash)

            μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
            r2 = sst == 0 ? NaN : 1 - ssr/sst
            isfinite(r2) && text!(ax, "R²=$(round(r2, digits=3))";
                                  position=(mx, mn), align=(:right,:bottom))
        end
    end
    display(fig)
end

###############################
# 2) R² vs t summary
###############################
function plot_rmed_time_summary(
    df_t::DataFrame;
    title = "Predictability of r̃med vs t"
)
    fig = Figure(size=(800, 450))
    ax = Axis(
        fig[1,1];
        xscale=log10, xlabel="t", ylabel="R² (vs full)",
        title=title
    )

    # reconstruct long form from per-t r_* columns if needed
    steps = [:row, :thr, :reshuf, :rew, :ushuf, :rarer]
    rows = NamedTuple[]
    for s in steps
        col = Symbol(:r_, s)
        for sub in groupby(df_t, :t)
            x = collect(skipmissing(sub.r_full))
            y = collect(skipmissing(sub[!, col]))
            if length(x) ≥ 3 && var(x) > 0
                μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
                r2 = sst == 0 ? 0.0 : max(0.0, 1 - ssr/sst)
                r2 = max(r2, 0.0)
                push!(rows, (; step=string(s), t=sub.t[1], r2))
            end
        end
    end
    df_r2 = DataFrame(rows)

    for s in unique(df_r2.step)
        sub = df_r2[df_r2.step .== s, :]
        sort!(sub, :t)
        lines!(ax, sub.t, sub.r2; label=s)
    end

    axislegend(ax; position=:lb)
    display(fig)
    return df_r2
end

###############################
# 3) R² vs t by IS (progressive)
###############################
function r2_by_IS(df_t::DataFrame; steps::Vector{Symbol}, min_n::Int=4)
    # handle string column names
    if "IS_real" in names(df_t) && !(:IS_real in names(df_t))
        rename!(df_t, Symbol.(names(df_t)))
    end

    df_t_round = transform(df_t, :IS_real => ByRow(x -> round(x, digits=2)) => :IS_real_q)

    rows = NamedTuple[]
    for step in steps
        col_step = Symbol(:r_, step)
        @assert hasproperty(df_t_round, col_step) "missing $(col_step)"
        for sub in groupby(df_t_round, [:t, :IS_real_q])
            x = collect(skipmissing(sub.r_full))
            y = collect(skipmissing(sub[!, col_step]))
            n = min(length(x), length(y))
            if n ≥ min_n && var(x) > eps()
                μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
                r2 = sst == 0 ? 0.0 : max(0.0, 1 - ssr/sst)
                push!(rows, (; step=String(step), t=sub.t[1],
                              IS_target=sub.IS_real_q[1], r2, n))
            end
        end
    end
    DataFrame(rows)
end

function plot_step_grid_by_IS_progressive(df_t::DataFrame;
        # steps::Vector{Symbol} = [:row, :thr, :reshuf, :rew, :ushuf, :rarer],
        steps::Vector{Symbol} = [:reshuf, :rew, :ushuf,:row],
        cmap = :viridis,
        title::String = "Predictability of r̃med vs t — progressive IS palette")

    df = r2_by_IS(df_t; steps=steps)
    IS_list = sort(unique(df.IS_target))
    col = cgrad(cmap, length(IS_list), categorical=true)
    leg_labels = [@sprintf("IS=%.2f", v) for v in IS_list]

    fig = Figure(size=(1100, 575))
    Label(fig[0, 1:3], title; fontsize=18, font=:bold, halign=:center)

    for (i, step) in enumerate(steps)
        r = (i - 1) ÷ 3 + 1
        c = (i - 1) % 3 + 1
        ax = Axis(fig[r, c];
                  xscale=log10, xlabel="t", ylabel=(c == 1 ? "R² (vs full)" : ""),
                  title=uppercase(string(step)),
                  limits=((nothing, nothing), (-0.05, 1.05)))

        dstep = df[df.step .== String(step), :]
        isempty(dstep) && continue

        for (k, IS) in enumerate(IS_list)
            sub = dstep[dstep.IS_target .== IS, :]
            isempty(sub) && continue
            sort!(sub, :t)
            lines!(ax, sub.t, sub.r2; color=col[k], label=leg_labels[k])
        end

        if i == 1
            axislegend(ax; position=:rc, framevisible=false, nbanks=1)
        end
    end

    display(fig)
end

###############################
# 4) Example usage
###############################
# correlation grid:
plot_correlations(df_main_uniform)

# R² vs t:
A = plot_rmed_time_summary(
    df_t_uniform;
    title="Predictability of r̃med vs t (rmed non weighted)",
)

# R² vs t by step and IS:
df_t_B_lowIScv = filter(row -> row.mag_cv == 0.01, df_t_B)
df_t_B_medIScv = filter(row -> row.mag_cv == 0.75, df_t_B)
df_t_B_highIScv = filter(row -> row.mag_cv == 1.5, df_t_B)

df_t_shortGoodBio_rhoOne = filter(row -> row.rho_sym == 1.0, df_t_shortGoodBio)
fig = plot_step_grid_by_IS_progressive(
    df_t_shortGoodBio_rhoOne; cmap=:viridis,
    title="Predictability of r̃med vs t BIOMASS (rmed weighted), ρ=1.0"
)