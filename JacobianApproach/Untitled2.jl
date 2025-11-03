using DataFrames, Statistics

# --- quick peek (optional) ---
diagnose_dt(dt::DataFrame; k=10) = begin
    println("Columns: ", names(dt))
    if :t in names(dt)
        ts = sort(unique(dt.t))
        println("t range: ", (first(ts), last(ts)), "  count=", length(ts))
        println("first times: ", ts[1:min(k,end)])
    end
    for c in (:IS_real, :IS, :IS_target)
        if c in names(dt)
            xs = skipmissing(dt[!, c])
            isempty(xs) || println(string(c), " range: ", (minimum(xs), maximum(xs)))
        end
    end
    nothing
end

# --- R² to identity ---
_r2_to_identity(x::AbstractVector, y::AbstractVector) = begin
    n = length(x); n==0 && return NaN
    μy = mean(y); sst = sum((y .- μy).^2); ssr = sum((y .- x).^2)
    r2 = sst == 0 ? (ssr == 0 ? 1.0 : 0.0) : 1 - ssr/sst
    max(min(r2,1.0), 0.0)
end

# --- tiny, robust helpers (drop-in) -----------------------------------
_hascol(dt::DataFrame, c) =
    (c isa Symbol  && c in propertynames(dt)) ||
    (c isa AbstractString && Symbol(c) in propertynames(dt))

# returns the column name as a Symbol (first match), accepts String or Symbol candidates
_pick_first_col(dt::DataFrame, candidates) = begin
    for c in candidates
        if _hascol(dt, c)
            return c isa Symbol ? c : Symbol(c)
        end
    end
    error("None of the expected columns found among: $(candidates). Available: $(propertynames(dt))")
end

# boolean mask for |t - t0| ≤ t_tol * max(1, |t0|)
_time_match_mask(tv::AbstractVector{<:Real}, t0::Real, t_tol::Real) =
    abs.(tv .- t0) .<= (t_tol * max(1.0, abs(t0)))

# nearest time in log-space (good for log-spaced grids)
_nearest_log_time(t_all::AbstractVector{<:Real}, target::Real) = begin
    i = argmin(abs.(log10.(t_all) .- log10(target)))
    t_all[i]
end

"""
compute_branch_weights(dt;
    t_targets=[0.01,0.5,5.0], use_nearest=true, t_tol=1e-6,
    IS_center=:auto, IS_halfwidth=0.05,
    min_n=20, eps=1e-9, tau_total=0.05)

- Picks :rho or :rho_sym; :u_cv_target or :u_cv; IS among :IS_real/:IS/:IS_target.
- If use_nearest=true, replaces each target time by the nearest available time
  (in log-space). Otherwise, matches by tolerance t_tol.
"""
function compute_branch_weights(dt::DataFrame;
        t_targets = [0.01, 0.5, 5.0],
        use_nearest::Bool = true,
        t_tol::Float64 = 1e-6,
        IS_center::Union{Real,Symbol} = :auto,
        IS_halfwidth::Real = 0.05,
        min_n::Int = 20,
        eps::Float64 = 1e-9,
        tau_total::Float64 = 0.05)

    # --- resolve columns (Symbol-safe) ------------------------------------
    rho_col = _pick_first_col(dt, [:rho, :rho_sym, "rho", "rho_sym"])
    ucv_col = _pick_first_col(dt, [:u_cv_target, :u_cv, "u_cv_target", "u_cv"])
    is_col  = _pick_first_col(dt, [:IS_real, :IS, :IS_target, "IS_real", "IS", "IS_target"])
    t_col   = _pick_first_col(dt, [:t, "t"])

    @assert all(_hascol(dt, c) for c in [:r_full, :r_row, :r_thr, :r_reshuf, :r_rew, :r_ushuf,
                                        "r_full","r_row","r_thr","r_reshuf","r_rew","r_ushuf"]) "dt missing r_* columns"

    # choose times
    t_all = sort(unique(dt[!, t_col]))
    t_vals = use_nearest ? [ _nearest_log_time(t_all, τ) for τ in t_targets ] : copy(t_targets)
    println("Requested t targets: ", t_targets, "  -> using t values: ", t_vals)

    # IS window
    IS_ctr = IS_center === :auto ? median(skipmissing(dt[!, is_col])) :: Float64 : Float64(IS_center)
    loIS, hiIS = IS_ctr - IS_halfwidth, IS_ctr + IS_halfwidth
    inIS = (dt[!, is_col] .>= loIS) .& (dt[!, is_col] .<= hiIS)

    # time filter (tolerance if not using-nearest)
    any_t = falses(nrow(dt))
    if use_nearest
        # exact equality with the chosen nearest times
        for τ in t_vals
            any_t .|= (dt[!, t_col] .== τ)
        end
    else
        for τ in t_vals
            any_t .|= _time_match_mask(dt[!, t_col], τ, t_tol)
        end
    end

    d = dt[inIS .& any_t, :]
    println("IS column: ", is_col, " | IS window: [", round(loIS,digits=4), ", ", round(hiIS,digits=4),
            "] (center=", round(IS_ctr,digits=4), ")")
    println("Rows kept (IS & t): ", nrow(d))

    uvals = sort(unique(d[!, ucv_col]))
    rvals = sort(unique(d[!, rho_col]))

    rows = NamedTuple[]
    for τ in t_vals, ucv in uvals, rho in rvals
        mask_t = d.t .== τ
        sub = d[mask_t .& (d[!, ucv_col] .== ucv) .& (d[!, rho_col] .== rho), :]

        n = nrow(sub)
        if n == 0
            push!(rows, (; t=τ, u_cv=ucv, rho, W_TS=NaN, W_IS=NaN, W_TOP=NaN,
                           Itot=NaN, n=0, flag_lowinfo=true, flag_smalln=true))
            continue
        end

        Δ_TS  = 1 - _r2_to_identity(sub.r_full, sub.r_ushuf)
        Δ_ROW = 1 - _r2_to_identity(sub.r_full, sub.r_row)
        Δ_THR = 1 - _r2_to_identity(sub.r_full, sub.r_thr)
        Δ_REW = 1 - _r2_to_identity(sub.r_full, sub.r_rew)
        Δ_RSH = 1 - _r2_to_identity(sub.r_full, sub.r_reshuf)

        Δ_IS  = median((Δ_ROW, Δ_THR))
        Δ_TOP = max(Δ_REW, Δ_RSH)

        Itot = max(Δ_TS + Δ_IS + Δ_TOP, 0.0)
        W_TS  = Δ_TS  / (Itot + eps)
        W_IS  = Δ_IS  / (Itot + eps)
        W_TOP = Δ_TOP / (Itot + eps)

        push!(rows, (; t=τ, u_cv=ucv, rho, W_TS, W_IS, W_TOP,
                       Itot, n,
                       flag_lowinfo = Itot < tau_total,
                       flag_smalln  = n < min_n))
    end
    DataFrame(rows)
end

##########################
# Phase map — HSV variant
##########################
# fixed hues for branches (in degrees)
const H_TS  = 225  # blue-ish
const H_IS  = 45   # amber/gold
const H_TOP = 300  # plum

# map weights -> HSV color
function _weights_to_hsv(wts::NTuple{3,Float64}, v_scale::Float64)
    (wts[1] < 0 || wts[2] < 0 || wts[3] < 0) && return RGBf(0.9,0.9,0.9)
    W = collect(wts)
    ord = sortperm(W; rev=true)
    wmax, w2 = W[ord[1]], W[ord[2]]
    hue = ord[1] == 1 ? H_TS : ord[1] == 2 ? H_IS : H_TOP
    sat = clamp(wmax - w2, 0, 1)                 # dominance
    val = clamp(v_scale, 0.25, 1.0)              # brightness from Itot
    Colors.HSV(hue, sat, val) |> RGBf
end

"""
plot_phase_hsv(bw; title, fname_prefix="phase_hsv")

- bw: output of compute_branch_weights(...)
- Saves one PNG per time value with name "<prefix>_t=<t>.png"
"""
function plot_phase_hsv(bw::DataFrame; title::String="Dominance map", fname_prefix::String="phase_hsv")
    
    for t in sort(unique(bw.t))
        sub = bw[bw.t .== t, :]
        uvals = sort(unique(sub.u_cv))
        rvals = sort(unique(sub.rho))

        # build color grid
        C = Matrix{RGBf}(undef, length(rvals), length(uvals))
        # scale value (brightness) by Itot across this panel
        It = filter(isfinite, sub.Itot)
        vmin, vmax = isempty(It) ? (0.0, 1.0) : (quantile(It, 0.1), quantile(It, 0.9))
        span = max(vmax - vmin, 1e-9)

        for (i,r) in enumerate(rvals), (j,u) in enumerate(uvals)
            row = sub[(sub.u_cv .== u) .& (sub.rho .== r), :]
            if nrow(row) == 0 || row.flag_lowinfo[1] || row.flag_smalln[1] ||
               !all(isfinite, (row.W_TS[1], row.W_IS[1], row.W_TOP[1]))
                C[i,j] = RGBf(0.85,0.85,0.85) # gray
            else
                vscale = 0.35 + 0.6 * clamp((row.Itot[1] - vmin) / span, 0, 1)
                C[i,j] = _weights_to_hsv( (row.W_TS[1], row.W_IS[1], row.W_TOP[1]), vscale )
            end
        end

        fig = Figure(size=(900, 560))
        ax  = Axis(fig[1,1]; xlabel="u_cv", ylabel="ρ", title = @sprintf("%s — t=%.3g", title, t))
        heatmap!(ax, uvals, rvals, C')
        xlims!(ax, minimum(uvals), maximum(uvals))
        ylims!(ax, minimum(rvals), maximum(rvals))
        # legend swatches
        legend_colors = [RGBf(Colors.HSV(H_TS,1,0.9)), RGBf(Colors.HSV(H_IS,1,0.9)), RGBf(Colors.HSV(H_TOP,1,0.9)), RGBf(0.85,0.85,0.85)]
        legend_labels = ["TS (time-scales) wins", "IS (row spread) wins", "TOP (topology) wins", "Low info / small n"]
        # axislegend(ax, legend_colors, legend_labels; position=:rb, framevisible=false)
        handles = [
            scatter!(ax, [NaN], [NaN]; color=RGBf(Colors.HSV(H_TS,1,0.9))),
            scatter!(ax, [NaN], [NaN]; color=RGBf(Colors.HSV(H_IS,1,0.9))),
            scatter!(ax, [NaN], [NaN]; color=RGBf(Colors.HSV(H_TOP,1,0.9))),
            scatter!(ax, [NaN], [NaN]; color=RGBf(0.85,0.85,0.85))
        ]
        legend_labels = ["TS", "IS", "TOP", "Low info"]
        axislegend(ax, handles, legend_labels; position=:rb, framevisible=false)

        fname = @sprintf("%s_t=%.3g.png", fname_prefix, t)
        save(fname, fig, px_per_unit=2)
        @info "Saved $fname"
        display(fig)
    end
end

#################################
# Phase map — triptych of weights
#################################
function _winner(WTS, WIS, WTOP)
    vals = (WTS, WIS, WTOP)
    all(isfinite, vals) || return 0
    argmax(vals)
end

"""
plot_phase_triptych(bw; title, fname_prefix="phase_triptych")

- bw: output of compute_branch_weights(...)
- Produces one PNG per time with 3 panels (TS/IS/TOP weights on [0,1]).
"""
function plot_phase_triptych(bw::DataFrame; title::String="Branch weights", fname_prefix::String="phase_triptych")
    for t in sort(unique(bw.t))
        sub = bw[bw.t .== t, :]
        uvals = sort(unique(sub.u_cv))
        rvals = sort(unique(sub.rho))

        # grids
        WT  = fill(NaN, length(rvals), length(uvals))
        WI  = fill(NaN, length(rvals), length(uvals))
        WP  = fill(NaN, length(rvals), length(uvals))
        MSK = falses(length(rvals), length(uvals))  # mask for gray cells
        WIN = fill(0, length(rvals), length(uvals)) # winner id

        for (i,r) in enumerate(rvals), (j,u) in enumerate(uvals)
            row = sub[(sub.u_cv .== u) .& (sub.rho .== r), :]
            if nrow(row)==0 || row.flag_lowinfo[1] || row.flag_smalln[1] ||
               !all(isfinite, (row.W_TS[1], row.W_IS[1], row.W_TOP[1]))
                MSK[i,j] = true
            else
                WT[i,j] = row.W_TS[1]; WI[i,j] = row.W_IS[1]; WP[i,j] = row.W_TOP[1]
                WIN[i,j] = _winner(WT[i,j], WI[i,j], WP[i,j])
            end
        end

        fig = Figure(size=(1200, 520))
        suptitle = @sprintf("%s — t=%.3g", title, t)
        Label(fig[0,1:3], suptitle; fontsize=18, font=:bold)

        labs = ["TS (time-scales)", "IS (row spread)", "TOP (topology)"]
        grids = (WT, WI, WP)
        cmaps = (:blues, :amp, :magma)  # any Makie colormaps you like

        for p in 1:3
            ax = Axis(fig[1,p]; xlabel="u_cv", ylabel=(p==1 ? "ρ" : ""), title=labs[p], yticklabelsize=10)
            hm = heatmap!(
                ax, uvals, rvals, permutedims(grids[p]);
                colormap = cmaps[p], colorrange = (0, 1), interpolate = false
            )
            # gray overlay
            if any(MSK)
                overlay = fill(RGBAf(0.85,0.85,0.85,0.85), size(MSK)...)
                heatmap!(ax, uvals, rvals, permutedims(overlay); interpolate=false, colorrange=(0,1))
                # punch “holes” where data exist
                for (i,r) in enumerate(rvals), (j,u) in enumerate(uvals)
                    if !MSK[i,j]
                        # draw a tiny transparent rect on top to reveal the value below
                        rect!(ax, Point(u,r), 0, 0; color=:transparent)  # no-op; keeps grid consistent
                    end
                end
            end
            # Colorbar(fig[2,p], hm; label="weight", ticks=[0,0.5,1.0])
        end

        # dashed contours where each branch is the winner
        for (p,grid) in enumerate(grids)
            ax = content(fig[1,p])
            # draw a thin contour by scanning cells
            for (i,r) in enumerate(rvals), (j,u) in enumerate(uvals)
                if WIN[i,j] == p && !MSK[i,j]
                    # mark winners with a faint dot (contours are overkill for coarse grids)
                    scatter!(ax, [u], [r]; markersize=3, color=:black, strokewidth=0)
                end
            end
        end

        fname = @sprintf("%s_t=%.3g.png", fname_prefix, t)
        save(fname, fig, px_per_unit=2); @info "Saved $fname"
        display(fig)
    end
end

# (optional) quick look so you know what times/IS you actually have)
diagnose_dt(dt_bio)

# compute weights (uses nearest available times to 0.01, 0.5, 5.0 and median IS as center)
bw = compute_branch_weights(dt_bio;
    t_targets=[0.01, 0.5, 5.0],
    use_nearest=true,
    IS_center=:auto,
    IS_halfwidth=0.05,
    min_n=20)

# then plot as before
plot_phase_hsv(bw; title="Dominance map — biomass pulse", fname_prefix="phase_hsv_bio")
plot_phase_triptych(bw; title="Branch weights — biomass pulse", fname_prefix="phase_triptych_bio")
