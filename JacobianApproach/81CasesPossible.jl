###############################
# Scenario grid — R² vs time
#   (ushuf, reshuf_pair, rew)
###############################
using Random, Statistics
using CairoMakie
using Base.Threads

# ---------------- tiny helpers ----------------
# log-spaced time grid
logspace10(a::Real, b::Real, n::Int) = 10 .^ range(a, b; length=n)

# ---------------- core evaluation for one community draw ----------------
function _metrics_for_times(A, u, tgrid; rng, q_thresh=0.20, rho_sym=0.5)
    Jfull = jacobian(A, u)
    alpha = alpha_off_from(Jfull, u)

    # 1) ushuf
    u_sh = copy(u); shuffle!(rng, u_sh)
    J_ush = build_J_from(alpha, u_sh)

    # 2) reshuffle alpha (paired)
    alpha_rp = op_reshuffle_preserve_pairs(alpha; rng)
    # alpha_rp = op_reshuffle_alpha(alpha; rng)
    J_rp = build_J_from(alpha_rp, u)

    # 3) rewiring
    A_er = build_random_trophic_ER(size(A,1);
        conn=realized_connectance(A),
        mean_abs=realized_IS(A), mag_cv=0.60, rho_sym=rho_sym, rng)
    beta = realized_IS(A) / max(realized_IS(A_er), eps())
    J_rew = jacobian(beta .* A_er, u)

    full  = [median_return_rate(Jfull, u; t=t, perturbation=:biomass) for t in tgrid]
    ushuf = [median_return_rate(J_ush,  u_sh; t=t, perturbation=:biomass) for t in tgrid]
    rpair = [median_return_rate(J_rp,   u;    t=t, perturbation=:biomass) for t in tgrid]
    rew   = [median_return_rate(J_rew,  u;    t=t, perturbation=:biomass) for t in tgrid]
    return full, ushuf, rpair, rew
end

# ---------------- threaded scenario simulator ----------------
function simulate_scenario(; S::Int=120, conn::Float64=0.10,
        mean_abs::Float64=0.50, mag_cv::Float64=0.60, rho_sym::Float64=0.5,
        u_mean::Float64=1.0, u_cv::Float64=0.6,
        degree_family::Symbol=:lognormal, deg_param::Float64=0.5,
        IS_target::Float64=0.5, reps::Int=24,
        tgrid::Vector{Float64}=logspace10(-2, 2, 40), seed::Int=42)

    base_rng = Random.Xoshiro(seed)
    thread_rngs = [Random.Xoshiro(rand(base_rng, UInt64)) for _ in 1:nthreads()]

    F_threads = [ [Float64[] for _ in eachindex(tgrid)] for _ in 1:nthreads() ]
    U_threads = [ [Float64[] for _ in eachindex(tgrid)] for _ in 1:nthreads() ]
    P_threads = [ [Float64[] for _ in eachindex(tgrid)] for _ in 1:nthreads() ]
    W_threads = [ [Float64[] for _ in eachindex(tgrid)] for _ in 1:nthreads() ]

    Threads.@threads for r in 1:reps
        tid = threadid()
        rng = Random.Xoshiro(rand(thread_rngs[tid], UInt64))

        A0 = build_niche_trophic(S; conn, mean_abs, mag_cv, degree_family, deg_param, rho_sym, rng)
        baseIS = realized_IS(A0)
        if baseIS <= 0
            continue
        end
        A = (IS_target / baseIS) .* A0
        u = random_u(S; mean=u_mean, cv=u_cv, rng)

        full, ush, rp, rew = _metrics_for_times(A, u, tgrid; rng, rho_sym=rho_sym)

        for (i, _) in enumerate(tgrid)
            push!(F_threads[tid][i], full[i])
            push!(U_threads[tid][i], ush[i])
            push!(P_threads[tid][i], rp[i])
            push!(W_threads[tid][i], rew[i])
        end
    end

    F = [vcat([F_threads[t][i] for t in 1:nthreads()]...) for i in eachindex(tgrid)]
    U = [vcat([U_threads[t][i] for t in 1:nthreads()]...) for i in eachindex(tgrid)]
    P = [vcat([P_threads[t][i] for t in 1:nthreads()]...) for i in eachindex(tgrid)]
    W = [vcat([W_threads[t][i] for t in 1:nthreads()]...) for i in eachindex(tgrid)]

    R2_ush = [ _r2_to_identity(F[i], U[i]) for i in eachindex(tgrid) ]
    R2_rp  = [ _r2_to_identity(F[i], P[i]) for i in eachindex(tgrid) ]
    R2_rew = [ _r2_to_identity(F[i], W[i]) for i in eachindex(tgrid) ]

    return (tgrid=tgrid, R2_ushuf=R2_ush, R2_reshuf=R2_rp, R2_rew=R2_rew)
end

# ---------------- categorical scenario presets ----------------
const LEVELS = (
    u_cv   = (low=0.01,  med=1.0,  high=3.0),
    degcv  = (low=0.01,  med=0.8,  high=1.8),
    magcv  = (low=0.01,  med=0.6,  high=1.5),
    rho    = (low=0.01,  med=0.5,  high=1.0),
)

const SCENARIOS = [
    (; name="BASELINE (u_cv=low, deg_cv=low, mag_cv=low, rho=high)", u_cv=LEVELS.u_cv.low,  degcv=LEVELS.degcv.low,
        magcv=LEVELS.magcv.low, rho=LEVELS.rho.high),
    (; name="High u_cv, low others", u_cv=LEVELS.u_cv.high, degcv=LEVELS.degcv.low,
        magcv=LEVELS.magcv.low, rho=LEVELS.rho.high),
    (; name="High deg_cv, low others", u_cv=LEVELS.u_cv.low, degcv=LEVELS.degcv.high,
        magcv=LEVELS.magcv.low, rho=LEVELS.rho.high),
    (; name="High mag_cv, low others", u_cv=LEVELS.u_cv.low, degcv=LEVELS.degcv.low,
        magcv=LEVELS.magcv.high, rho=LEVELS.rho.high),
    (; name="Low rho, low others", u_cv=LEVELS.u_cv.low, degcv=LEVELS.degcv.low,
        magcv=LEVELS.magcv.low, rho=LEVELS.rho.low),
    (; name="OPPOSITE BASELINE", u_cv=LEVELS.u_cv.high, degcv=LEVELS.degcv.high,
        magcv=LEVELS.magcv.high, rho=LEVELS.rho.low),
    (; name="OPPOSITE BASELINE (keeping rho high)", u_cv=LEVELS.u_cv.high, degcv=LEVELS.degcv.high,
    magcv=LEVELS.magcv.high, rho=LEVELS.rho.high),
    (; name="Low u_cv, mid others", u_cv=LEVELS.u_cv.med, degcv=LEVELS.degcv.med,
    magcv=LEVELS.magcv.med, rho=LEVELS.rho.high),
    (; name="Low deg_cv, mid others", u_cv=LEVELS.u_cv.med, degcv=LEVELS.degcv.low,
        magcv=LEVELS.magcv.med, rho=LEVELS.rho.high),
    (; name="Low mag_cv, mid others", u_cv=LEVELS.u_cv.med, degcv=LEVELS.degcv.med,
        magcv=LEVELS.magcv.low, rho=LEVELS.rho.high),
    (; name="Low rho, mid others", u_cv=LEVELS.u_cv.med, degcv=LEVELS.degcv.med,
        magcv=LEVELS.magcv.med, rho=LEVELS.rho.low)
]

# ---------------- plotting ----------------
function plot_scenarios(summaries; ncol::Int=3, savepath::Union{Nothing,String}=nothing)
    n = length(summaries)
    nrow = cld(n, ncol)
    fig = Figure(size=(1200, 360*nrow))
    Label(fig[0,1:ncol], "Predictability (R² vs full) over time by scenario (RESHUF NOT PAIRED)";
          fontsize=22, font=:bold, halign=:left)

    for (k, (name, ts, r2u, r2p, r2w)) in enumerate(summaries)
        r = (k-1) ÷ ncol + 1
        c = (k-1) % ncol + 1
        ax = Axis(fig[r, c];
                  xscale=log10,
                  xlabel="time t", ylabel="R²",
                  title=name,
                  limits=((nothing, nothing), (-0.05, 1.05)))
        lines!(ax, ts, r2u,  label="ushuf")
        lines!(ax, ts, r2p,  label="reshuf(NOTpair)")
        lines!(ax, ts, r2w,  label="rew")
        if k == 1
            axislegend(ax; position=:lb, framevisible=false)
        end
        hlines!(ax, [0,1]; color=:gray, linewidth=0.5, linestyle=:dot)
    end

    if savepath !== nothing
        save(savepath, fig)
    end
    display(fig)
end

# ---------------- parallel demo runner ----------------
function run_demo(; reps=24, seed=20251105)
    tgrid = logspace10(-2, 2, 48)
    outs = Vector{NamedTuple}(undef, length(SCENARIOS))

    Threads.@threads for i in eachindex(SCENARIOS)
        sc = SCENARIOS[i]
        sim = simulate_scenario(; S=120, conn=0.10,
            mean_abs=0.50, mag_cv=sc.magcv, rho_sym=sc.rho,
            u_mean=1.0, u_cv=sc.u_cv, degree_family=:lognormal,
            deg_param=sc.degcv, IS_target=0.50, reps, tgrid, seed)
        outs[i] = (
            name = sc.name,
            tgrid = sim.tgrid,
            R2_ushuf = sim.R2_ushuf,
            R2_reshuf = sim.R2_reshuf,
            R2_rew = sim.R2_rew
        )
    end

    plot_scenarios(outs; ncol=4, savepath="JacobianApproach/FinalFigures/scenarios_r2_vs_time.png")
    println("Saved: scenarios_r2_vs_time.png")
end

# Uncomment to run
run_demo(; reps=24)
