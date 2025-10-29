using Statistics, DataFrames, CairoMakie, Printf, Colors

# --- R² helper ---
function r2_to_identity(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == 0 && return (NaN, NaN, NaN)
    μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
    if sst == 0; return ssr == 0 ? (1.0, 1.0, 0.0) : (0.0, NaN, NaN); end
    β = [x ones(n)] \ y
    return (1 - ssr/sst, β[1], β[2])
end

function r2_by_IS(df_raw::DataFrame; steps::Vector{Symbol}, min_n::Int=4)
    rows = NamedTuple[]
    for step in steps
        col_step = Symbol(:r_, step)
        @assert hasproperty(df_raw, col_step) "df_raw missing $(col_step)"
        for sub in groupby(df_raw, [:t, :IS_target])
            x = collect(skipmissing(sub.r_full))
            y = collect(skipmissing(sub[!, col_step]))
            n = min(length(x), length(y))
            if n ≥ min_n && var(x) > eps() && var(y) > eps()
                r2, slope, intercept = r2_to_identity(x, y)
                r2 = max(0.0, r2)
                push!(rows, (;
                    step=String(step),
                    t=only(unique(sub.t)),
                    IS_target=only(unique(sub.IS_target)),
                    r2=max(r2,0.0), slope, intercept, n
                ))
            end
        end
    end
    DataFrame(rows)
end

"""
plot_step_grid_by_IS_progressive(df_raw; steps, cmap)

2×3 grid (one panel per step). Lines are **IS** with a progressive palette.
`cmap` can be any Makie colormap symbol, e.g. :viridis, :plasma, :magma, :inferno.
"""
function plot_step_grid_by_IS_progressive(df_raw::DataFrame;
        steps::Vector{Symbol} = [:row, :thr, :reshuf, :rew, :ushuf, :rarer],
        cmap = :viridis,
        title::String = "Predictability of r̃med vs t — progressive IS palette")

    df = r2_by_IS(df_raw; steps=steps)

    # global, sorted IS list for consistent color mapping across panels
    IS_list = sort(unique(df.IS_target))
    col = cgrad(:viridis, length(IS_list), categorical=true)

    # legend labels/colors
    leg_labels = [@sprintf("IS=%.2f", v) for v in IS_list]
    leg_colors = [col[i] for i in 1:length(IS_list)]

    fig = Figure(size=(1100, 640))
    Label(fig[0, 1:3], title; fontsize=18, font=:bold, halign=:center)

    for (i, step) in enumerate(steps)
        r = (i - 1) ÷ 3 + 1
        c = (i - 1) % 3 + 1
        ax = Axis(fig[r, c];
                  xscale=log10, xlabel="t", ylabel=(c == 1 ? "R² (vs full)" : ""),
                  title=uppercase(string(step)), limits=((nothing, nothing), (-0.1, 1.05)),
                  ygridvisible=true, xgridvisible=false)

        dstep = df[df.step .== String(step), :]
        isempty(dstep) && continue

        for (k, IS) in enumerate(IS_list)
            sub = dstep[dstep.IS_target .== IS, :]
            isempty(sub) && continue
            sort!(sub, :t)
            lines!(ax, sub.t, sub.r2; color=col[k], label=leg_labels[k])
        end

        if i == 1
            axislegend(ax; position=:lb, framevisible=false, nbanks=2)
        end
    end

    display(fig)
end

# Usage:
fig = plot_step_grid_by_IS_progressive(df_raw; cmap=:magma)
