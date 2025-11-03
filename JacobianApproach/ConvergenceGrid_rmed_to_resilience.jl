################ ConvergenceGrid_rmed_to_resilience.jl ################
using Random, Statistics, LinearAlgebra, CairoMakie, OrderedCollections

# --- One community draw (uses your builders) -------------------------
function _draw_comm(; S=120, conn=0.10, mean_abs=0.10, mag_cv=0.60,
                     degree_family=:uniform, deg_param=0.0, rho_sym=0.5,
                     u_mean=1.0, u_cv=0.8, seed=42)
    rng = Random.Xoshiro(seed)
    # Prefer niche; fallback to random_trophic if niche not defined
    A0, R = build_niche_trophic(S; conn, mean_abs, mag_cv, degree_family, deg_param, rho_sym, rng)

    baseIS = realized_IS(A0)
    β = baseIS > 0 ? mean_abs/baseIS : 1.0  # put A on the desired IS scale
    A = β .* A0
    u = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
    return (; A, u, rng)
end

# --- Pack all 7 variants (original + 6 steps) ------------------------
function _variants(A, u; rng, conn, mean_abs, mag_cv, rho_sym, q_thr=0.20, p_rarer=0.10)
    J_full = jacobian(A, u)
    α      = alpha_off_from(J_full, u)

    α_reshuf = op_reshuffle_alpha(α; rng=rng)
    α_row    = op_rowmean_alpha(α)
    α_thr    = op_threshold_alpha(α; q=q_thr)

    u_uni   = fill(mean(u), length(u))
    u_rarer = remove_rarest_species(u; p=p_rarer)

    J_reshuf = build_J_from(α_reshuf, u)
    J_row    = build_J_from(α_row,    u)
    J_thr    = build_J_from(α_thr,    u)
    J_uni    = build_J_from(α,        u_uni)
    J_rarer  = build_J_from(α,        u_rarer)

    A_rew0 = build_random_trophic_ER(size(A,1); conn=conn, mean_abs=mean_abs,
                                     mag_cv=mag_cv, rho_sym=rho_sym, rng=rng)
    βr = realized_IS(A_rew0)
    A_rew = (βr > 0 ? (mean_abs / βr) : 1.0) .* A_rew0
    J_rew = jacobian(A_rew, u)

    return OrderedDict(
        "ORIGINAL" => (J_full, u),
        "RESHUF"   => (J_reshuf, u),
        "ROW"      => (J_row,    u),
        "THR"      => (J_thr,    u),
        "UNI"      => (J_uni,    u_uni),
        "RARER"    => (J_rarer,  u_rarer),
        "REW"      => (J_rew,    u)
    )
end

# --- r̃med(t) curve and baseline for one J,u -------------------------
function _series(J, u; u_weighted_biomass = :biomass, tgrid = 10 .^ range(log10(0.01), log10(50.0); length=30))
    rmed = [median_return_rate(J, u; t=t, perturbation=u_weighted_biomass) for t in tgrid]
    base = -resilience(J)   # positive baseline to which r̃med(t) should converge
    return (; t=tgrid, rmed, base)
end

# --- Main: build one community and plot 7 panels ---------------------
"""
plot_convergence_grid(; S=120, conn=0.10, mean_abs=0.10, mag_cv=0.60,
                        degree_family=:uniform, deg_param=0.0, rho_sym=0.5,
                        u_mean=1.0, u_cv=0.8, seed=42,
                        q_thr=0.20, p_rarer=0.10,
                        tgrid = 10 .^ range(log10(0.01), log10(50.0); length=30))

Creates a 2×4 grid of panels:
  [ ORIGINAL | RESHUF | ROW | THR ]
  [    UNI   |  RARER | REW |  —  ]
Each panel shows r̃med(t) (biomass) and a dashed baseline at -resilience of that panel.
"""
function plot_convergence_grid(
    ; S=120, conn=0.10, mean_abs=0.10, mag_cv=0.60,
    degree_family=:uniform, deg_param=0.0, rho_sym=0.5,
    u_mean=1.0, u_cv=0.8, seed=42,
    q_thr=0.20, p_rarer=0.10,
    tgrid = 10 .^ range(log10(0.01), log10(50.0); length=30),
    u_weighted_biomass = :biomass
)

    C = _draw_comm(; S, conn, mean_abs, mag_cv, degree_family, deg_param, rho_sym,
                     u_mean, u_cv, seed)
    panels = _variants(C.A, C.u; rng=C.rng, conn, mean_abs, mag_cv, rho_sym, q_thr, p_rarer)

    fig = Figure(size = (1050, 520))
    title = "Convergence of Rmed to Resilience. deg=$degree_family, ρ=$rho_sym, conn=$conn, u_cv=$u_cv, weighted=$u_weighted_biomass"
    positions = Dict(
        "ORIGINAL" => (1,1), "RESHUF" => (1,2), "ROW" => (1,3), "THR" => (1,4),
        "UNI"      => (2,1), "RARER"  => (2,2), "REW" => (2,3)
    )
    real_base = _series(panels["ORIGINAL"][1], panels["ORIGINAL"][2]; u_weighted_biomass = u_weighted_biomass, tgrid = tgrid).base
    for (name, (Jp, up)) in panels
        (r, c) = positions[name]
        ax = Axis(fig[r, c];
                xscale = log10,
                xlabel = "t (log)",
                ylabel = "rate",
                title  = name)
        s = _series(Jp, filter(!iszero, up); u_weighted_biomass = u_weighted_biomass, tgrid = tgrid)
        lines!(ax, s.t, s.rmed)
        hlines!(ax, [s.base]; color = :black, linestyle = :dash)
        hlines!(ax, [real_base]; color = :gray35, linestyle = :dash)
    end
    fig[0, :] = Label(fig, title;
                  fontsize = 18,
                  font = :bold,
                  halign = :left,
                  tellheight = false)

    display(fig)
end

# Run demo (one community)
plot_convergence_grid(
    ; S=120, conn=0.3, mean_abs=0.10, mag_cv=0.60,
    degree_family=:lognormal, deg_param=2.0, rho_sym=0.5,
    u_mean=1.0, u_cv=2.0, seed=42,
    q_thr=0.20, p_rarer=0.10,
    tgrid = 10 .^ range(log10(0.01), log10(100.0); length=30),
    u_weighted_biomass = :biomass
)
#######################################################################
