############################
# Robust Branch Weights + Plots (one-file runner)
############################
using DataFrames, Statistics, Printf
using CairoMakie

# ----------------------------------------------------
# 0) Input table: pick dt_bio, dt_uni, or dt (in that order)
# ----------------------------------------------------
local_dt = let
    if @isdefined(dt_bio)
        dt_bio
    elseif @isdefined(dt_uni)
        dt_uni
    elseif @isdefined(dt)
        dt
    else
        error("Please define one of: dt_bio, dt_uni, or dt before running this script.")
    end
end

# ----------------------------------------------------
# 1) Column helpers (Symbol/String safe) + small utils
# ----------------------------------------------------
_hascol(df::AbstractDataFrame, name::Union{Symbol,String}) = begin
    nms = Set(string.(names(df)))
    string(name) in nms
end

_pick_first_col(df::AbstractDataFrame, candidates) = begin
    for c in candidates
        if _hascol(df, c); return Symbol(c); end
    end
    error("None of the expected columns found among: $(candidates). Available: $(names(df))")
end

# clamp to [0,1]
_r2_to_identity(x::AbstractVector, y::AbstractVector) = begin
    n = length(x)
    n == 0 && return NaN
    μy = mean(y)
    sst = sum((y .- μy).^2)
    ssr = sum((y .- x).^2)
    r2 = sst == 0 ? (ssr == 0 ? 1.0 : 0.0) : 1 - ssr/sst
    max(min(r2, 1.0), 0.0)
end

# nearest available times in data to requested targets
function _nearest_times(all_t::AbstractVector{<:Real}, targets::Vector{<:Real})
    [all_t[argmin(abs.(all_t .- τ))] for τ in targets]
end

# small neighborhood expander in (u_cv, rho) space to guarantee ≥ min_n rows per cell
function _cell_subset_with_neighbors(d::DataFrame, ucv_col::Symbol, rho_col::Symbol;
                                     ucv0::Real, rho0::Real, t0::Real,
                                     min_n::Int=8, max_steps::Int=6,
                                     du0::Real=0.0, dr0::Real=0.0, du_step::Real=0.1, dr_step::Real=0.1)
    step = 0
    sub = d[(d.t .== t0) .& (d[!, ucv_col] .== ucv0) .& (d[!, rho_col] .== rho0), :]
    while nrow(sub) < min_n && step <= max_steps
        du = du0 + step*du_step
        dr = dr0 + step*dr_step
        sub = d[(d.t .== t0) .&
                (abs.(d[!, ucv_col] .- ucv0) .<= du .+ 1e-12) .&
                (abs.(d[!, rho_col] .- rho0) .<= dr .+ 1e-12), :]
        step += 1
    end
    return sub
end

# safe Δ from R²; if degenerate, returns 0 loss (matches "no info loss")
_safe_delta(x::AbstractVector, y::AbstractVector) = begin
    r2 = _r2_to_identity(x, y)
    isfinite(r2) ? (1 - r2) : 0.0
end

# ----------------------------------------------------
# 2) Robust branch weights (TS=ushuf, IS=reshuf, TOP=rew)
#    - auto time alignment
#    - auto IS center
#    - adaptive per-cell neighborhoods so n >= min_n
# ----------------------------------------------------
"""
compute_branch_weights_robust(dt;
    t_targets=[0.01,0.5,5.0],
    IS_center=:auto, IS_halfwidth=0.10,
    percell_min_n=8, max_neighbor_steps=6,
    du_step=0.15, dr_step=0.15)

Branches:
  TS  := USHUF     → Δ_TS  = 1 - R²(full, ushuf)
  IS  := RESHUF    → Δ_IS  = 1 - R²(full, reshuf)
  TOP := REWIRING  → Δ_TOP = 1 - R²(full, rew)

Returns DataFrame with W_TS,W_IS,W_TOP, Itot, n. No masking flags used by plots.
"""
function compute_branch_weights_robust(dt::DataFrame;
        t_targets = [0.01, 0.5, 5.0],
        IS_center::Union{Real,Symbol} = :auto,
        IS_halfwidth::Real = 0.10,
        percell_min_n::Int = 8,
        max_neighbor_steps::Int = 6,
        du_step::Real = 0.15,
        dr_step::Real = 0.15)

    # --- resolve columns
    rho_col = _pick_first_col(dt, [:rho, :rho_sym])
    ucv_col = _pick_first_col(dt, [:u_cv_target, :u_cv])
    is_col  = _pick_first_col(dt, [:IS_real, :IS, :IS_target])

    for c in [:t, :r_full, :r_ushuf, :r_reshuf, :r_rew]
        @assert _hascol(dt, c) "Missing column $(c)"
    end

    # --- pick times
    all_t = sort(unique(dt[!, :t]))
    t_vals = _nearest_times(all_t, t_targets) |> unique |> sort

    # --- IS window
    IS_ctr = IS_center === :auto ? median(skipmissing(dt[!, is_col])) : Float64(IS_center)
    loIS, hiIS = IS_ctr - IS_halfwidth, IS_ctr + IS_halfwidth
    inIS = (dt[!, is_col] .>= loIS) .& (dt[!, is_col] .<= hiIS)
    d = dt[inIS .& in.(dt.t, Ref(t_vals)), :]
    @info @sprintf("Using IS ∈ [%.3f, %.3f] (center=%.3f); kept %d/%d rows; times = %s",
                   loIS, hiIS, IS_ctr, nrow(d), nrow(dt), string(t_vals))

    # --- axes from retained data
    uvals = sort(unique(d[!, ucv_col]))
    rvals = sort(unique(d[!, rho_col]))

    rows = NamedTuple[]
    for t0 in t_vals, ucv in uvals, rho in rvals
        # adaptive neighborhood to guarantee percell_min_n
        sub = _cell_subset_with_neighbors(d, ucv_col, rho_col;
                                          ucv0=ucv, rho0=rho, t0=t0,
                                          min_n=percell_min_n, max_steps=max_neighbor_steps,
                                          du_step=du_step, dr_step=dr_step)

        n = nrow(sub)
        if n == 0
            # This should almost never happen; still return equal weights
            push!(rows, (; t=t0, u_cv=ucv, rho,
                         W_TS=1/3, W_IS=1/3, W_TOP=1/3,
                         Itot=0.0, n=0))
            continue
        end

        # single-step ablation losses
        Δ_TS  = _safe_delta(sub.r_full,   sub.r_ushuf)     # time-scales
        Δ_IS  = _safe_delta(sub.r_full,   sub.r_reshuf)    # IS arrangement
        Δ_TOP = _safe_delta(sub.r_full,   sub.r_rew)       # topology

        Itot = max(Δ_TS + Δ_IS + Δ_TOP, 0.0)
        if Itot <= 0
            W_TS = W_IS = W_TOP = 1/3
        else
            W_TS  = Δ_TS  / Itot
            W_IS  = Δ_IS  / Itot
            W_TOP = Δ_TOP / Itot
        end

        push!(rows, (; t=t0, u_cv=ucv, rho, W_TS, W_IS, W_TOP, Itot, n))
    end
    return DataFrame(rows)
end

# ----------------------------------------------------
# 3) Plotting — no greyscale masking, auto-fill tiny gaps
# ----------------------------------------------------
function _grid_from_weights(bw::DataFrame, t0::Real)
    d = bw[bw.t .== t0, :]
    uvals = sort(unique(d.u_cv))
    rvals = sort(unique(d.rho))
    # 3 matrices; fill with NaN first
    Wts  = fill(NaN, length(uvals), length(rvals))
    Wis  = fill(NaN, length(uvals), length(rvals))
    Wtop = fill(NaN, length(uvals), length(rvals))
    It   = fill(NaN, length(uvals), length(rvals))

    uix = Dict(u=>i for (i,u) in enumerate(uvals))
    rix = Dict(r=>j for (j,r) in enumerate(rvals))

    for row in eachrow(d)
        i = uix[row.u_cv]; j = rix[row.rho]
        Wts[i,j]  = row.W_TS
        Wis[i,j]  = row.W_IS
        Wtop[i,j] = row.W_TOP
        It[i,j]   = row.Itot
    end

    # simple NaN fill: replace with local median, else global median
    function fill_nans!(M)
        if all(isfinite, M); return M; end
        gmed = 0.3333
        for i in eachindex(M)
            if !isfinite(M[i])
                M[i] = gmed
            end
        end
        return M
    end
    fill_nans!(Wts); fill_nans!(Wis); fill_nans!(Wtop); fill_nans!(It)

    return uvals, rvals, Wts, Wis, Wtop, It
end

function plot_phase_triptych_nomask(bw::DataFrame; title::String="Branch weights", fname_prefix::String="phase_triptych")
    for t0 in sort(unique(bw.t))
        uvals, rvals, Wts, Wis, Wtop, It = _grid_from_weights(bw, t0)

        fig = Figure(size=(1100, 525))
        Label(fig[0,1:3], @sprintf("%s — t≈%.3g", title, t0); fontsize=18, font=:bold, halign=:left)

        ax1 = Axis(fig[1,1], title="TS (u shuffled)", xlabel="ρ", ylabel="u_cv")
        ax2 = Axis(fig[1,2], title="IS (pair reshuffle)", xlabel="ρ", ylabel="")
        ax3 = Axis(fig[1,3], title="TOP (rewiring)", xlabel="ρ", ylabel="")

        hm1 = heatmap!(ax1, rvals, uvals, Wts; colormap=:viridis, colorrange=(0,1))
        hm2 = heatmap!(ax2, rvals, uvals, Wis; colormap=:viridis, colorrange=(0,1))
        hm3 = heatmap!(ax3, rvals, uvals, Wtop; colormap=:viridis, colorrange=(0,1))

        # --- inside plot_phase_triptych_nomask ---
        cb1 = Colorbar(fig[2,1], hm1; label="weight", vertical=false)
        cb2 = Colorbar(fig[2,2], hm2; label="weight", vertical=false)
        cb3 = Colorbar(fig[2,3], hm3; label="weight", vertical=false)

        cb1.width = Relative(0.9)
        cb2.width = Relative(0.9)
        cb3.width = Relative(0.9)
        cb1.height = 10
        cb2.height = 10
        cb3.height = 10

        save(@sprintf("%s_t=%.3g.png", fname_prefix, t0), fig, px_per_unit=2)
        display(fig)
    end
end

# HSV dominance map (hue=winner; saturation=margin; value=scaled Itot)
function plot_phase_rgb_nomask(bw::DataFrame;
        title::String="Dominance (TS=red, IS=blue, TOP=yellow)",
        fname_prefix::String="phase_rgb")

    col_TS  = RGBf(1, 0, 0)   # red
    col_IS  = RGBf(0, 0, 1)   # blue
    col_TOP = RGBf(1, 1, 0)   # yellow

    for t0 in sort(unique(bw.t))
        uvals, rvals, Wts, Wis, Wtop, It = _grid_from_weights(bw, t0)

        img = Array{RGBf}(undef, size(Wts))

        for i in eachindex(Wts)
            wts, wis, wtop = Wts[i], Wis[i], Wtop[i]
            k = argmax((wts, wis, wtop))
            col = k == 1 ? col_TS : (k == 2 ? col_IS : col_TOP)

            # weight by total info: fade to black if Itot small
            # soften darks: lift the black floor
            val_raw = clamp(It[i] / quantile(vec(It), 0.8), 0, 1)
            val = 0.25 + 0.75 * val_raw   # ensures darkest points = 25% brightness

            img[i] = RGBf(val * col.r, val * col.g, val * col.b)

            if !isfinite(It[i])
                img[i] = RGBf(0.3, 0.3, 0.3)   # neutral gray for missing
            else
                val = clamp(It[i] / quantile(vec(It), 0.8), 0, 1)
                img[i] = RGBf(val*col.r, val*col.g, val*col.b)
            end
        end

        fig = Figure(size=(650, 450))
        Label(fig[0,1], @sprintf("%s — t≈%.3g", title, t0);
              fontsize=18, font=:bold, halign=:left)
        ax = Axis(fig[1,1], xlabel="ρ", ylabel="u_cv")

        image!(ax, rvals[1] .. rvals[end], uvals[1] .. uvals[end], img)

        save(@sprintf("%s_t=%.3g.png", fname_prefix, t0), fig, px_per_unit=2)
        display(fig)
    end
end

# ----------------------------------------------------
# 4) RUN: compute + save + plot (no grey, no empty bins)
# ----------------------------------------------------
# Choose your target times; script will snap to nearest available
t_targets = [0.01, 0.5, 5.0]

bw = compute_branch_weights_robust(local_dt;
    t_targets=t_targets,
    IS_center=:auto,         # center on your data automatically
    IS_halfwidth=0.10,       # reasonably wide; you can tighten later
    percell_min_n=8,         # guarantees n ≥ 8 per cell via neighborhood expansion
    max_neighbor_steps=6,
    du_step=0.15, dr_step=0.15)

# Save the table (optional)
# CSV.write("branch_weights.csv", bw; writeheader=true)

# Plots (no masking, no greys)
plot_phase_triptych_nomask(bw; title="Branch weights (TS=ushuf, IS=reshuf, TOP=rew)", fname_prefix="phase_triptych_nomask")
# plot_phase_hsv_nomask(bw; title="Dominance (TS/IS/TOP)", fname_prefix="phase_hsv_nomask")
plot_phase_rgb_nomask(
    bw;
    title ="Dominance (TS=red, IS=blue, TOP=yellow)",
    fname_prefix="phase_rgb"
)

println("Done. Saved: branch_weights.csv, phase_triptych_nomask_*.png, phase_hsv_nomask_*.png")
