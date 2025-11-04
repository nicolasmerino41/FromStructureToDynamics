using CairoMakie, Statistics, DataFrames

# Robust R² to identity (y vs x)
_r2_to_identity(x::AbstractVector, y::AbstractVector) = begin
    n = length(x); n==0 && return NaN
    μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
    r2 = sst == 0 ? (ssr == 0 ? 1.0 : 0.0) : 1 - ssr/sst
    max(r2, 0.0)
end

"""
plot_steps_vs_t_with_IS_halo(dt;
    steps = ["row","thr","reshuf","rew","ushuf","rarer"],
    IS_baseline = 0.5, IS_width_center = 0.05, IS_width_halo = 0.20,
    title = "Predictability of r̃_med vs t (with IS halo)",
    fname = "steps_vs_t_with_IS_halo.png")

- dt: joined time table (has :t, :IS_real, :r_full, :r_row, :r_thr, ...).
- For each t and each step, computes R²(step vs full) on three IS bands:
    center: [b - w_c, b + w_c]
    low   : [b - w_h, b - w_c)
    high  : (b + w_c, b + w_h]
- Plots center line; halo = ribbon between low & high curves.
"""
function plot_steps_vs_t_with_IS_halo(dt::DataFrame;
        steps = ["row","thr","reshuf","rew","ushuf","rarer"],
        IS_baseline = 0.5, IS_width_center = 0.05, IS_width_halo = 0.20,
        title = "Predictability of r̃_med vs t (with IS halo)",
        fname = "steps_vs_t_with_IS_halo.png")

    # @assert all(Symbol("r_", s) ∈ names(dt) for s in steps) "dt is missing r_<step> columns"
    # @assert (:r_full ∈ names(dt)) "dt must contain :r_full"
    # @assert (:IS_real ∈ names(dt)) "dt must contain :IS_real"

    # Clamp band edges to observed IS range
    # ISmin, ISmax = minimum(dt.IS_real), maximum(dt.IS_real)
    b  = isnothing(IS_baseline) ? median(dt.IS_real) : IS_baseline
    ISmin, ISmax = minimum(dt.IS_real), maximum(dt.IS_real)

    lc = clamp(b - IS_width_center, ISmin, ISmax)
    rc = clamp(b + IS_width_center, ISmin, ISmax)
    ll = clamp(b - IS_width_halo,   ISmin, ISmax)
    rr = clamp(b + IS_width_halo,   ISmin, ISmax)

    # b  = IS_baseline
    # lc = clamp(b - IS_width_center, ISmin, ISmax)
    # rc = clamp(b + IS_width_center, ISmin, ISmax)
    # ll = clamp(b - IS_width_halo,   ISmin, ISmax)
    # rr = clamp(b + IS_width_halo,   ISmin, ISmax)

    tvals = sort(unique(dt.t))
    cols = Makie.resample_cmap(:magma, length(steps)+2)[2:end-1]

    fig = Figure(size=(1050, 620))
    ax  = Axis(fig[1,1];
        xscale=log10, xlabel="t (log)", ylabel="R² vs FULL",
        limits=((minimum(tvals), maximum(tvals)), (-0.02, 1.02)),
        title=title)

    for (k, s) in enumerate(steps)
        ys_center = Float64[]; ys_low = Float64[]; ys_high = Float64[]
        for tt in tvals
            sub_t = dt[dt.t .== tt, :]
            # center
            subC = sub_t[(sub_t.IS_real .>= lc) .& (sub_t.IS_real .<= rc), :]
            # low & high; allow empty => NaN
            subL = sub_t[(sub_t.IS_real .>= ll) .& (sub_t.IS_real .<  lc), :]
            subH = sub_t[(sub_t.IS_real .>  rc) .& (sub_t.IS_real .<= rr), :]

            push!(ys_center, _r2_to_identity(subC.r_full, subC[!, Symbol("r_", s)]))
            push!(ys_low,    _r2_to_identity(subL.r_full, subL[!, Symbol("r_", s)]))
            push!(ys_high,   _r2_to_identity(subH.r_full, subH[!, Symbol("r_", s)]))
        end

        # If low/high missing at some t, fall back to center to avoid gaps
        ylo  = [isfinite(v) ? v : ys_center[i] for (i,v) in enumerate(ys_low)]
        yhi  = [isfinite(v) ? v : ys_center[i] for (i,v) in enumerate(ys_high)]
        c    = cols[k]

        # Halo ribbon (min↔max of low/high at each t)
        y1 = map(min, ylo, yhi)
        y2 = map(max, ylo, yhi)
        # @show s, count(isfinite, ys_center), count(isfinite, ys_low), count(isfinite, ys_high)
        band!(ax, tvals, y1, y2; color=(c, 0.18))

        # Center line
        lines!(ax, tvals, ys_center; color=c, linewidth=2.8, label=uppercase(s))
    end

    # Legend & a small caption for the halo bands
    axislegend(ax; position=:lb, framevisible=false, nbanks=3)
    text!(
        ax,
        @sprintf(
            "baseline IS≈%.2f (line); halo: IS∈[%.2f,%.2f] vs [%.2f,%.2f]",
            b, ll, lc, rc, rr
        );
        position = (0.98, 0.05),
        space = :relative,
        fontsize = 10,
        # halign = :right
    )
    # @show nrow(dt)
    # @show extrema(dt.t)
    # @show names(dt)
    # @show any(ismissing, dt.IS_real)
    # @show minimum(dt.IS_real), maximum(dt.IS_real)

    save(fname, fig, px_per_unit=2)
    println("Saved $(fname)")
    display(fig)
end

# --- Example usage ---
# For biomass-weighted:
plot_steps_vs_t_with_IS_halo(
    dt_bio; IS_baseline=nothing, IS_width_center=0.05, IS_width_halo=0.20,
    title="Biomass pulse: R²(step vs FULL) across t with IS halo",
    fname="steps_vs_t_with_IS_halo_biomass.png")

# For uniform pulse:
plot_steps_vs_t_with_IS_halo(
    dt_uni; IS_baseline=nothing, IS_width_center=0.05, IS_width_halo=0.20,
    title="Uniform pulse: R²(step vs FULL) across t with IS halo",
    fname="steps_vs_t_with_IS_halo_uniform.png"
)
