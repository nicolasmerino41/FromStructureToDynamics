############## AntisymmetryContinuum.jl ##############
# Goal
# Build a *continuous* knob γ ∈ [0,1] controlling how much pairwise
# antisymmetry is preserved under two manipulations: RESHUFFLE and REWIRE.
# γ=1: fully preserve original pair blocks (magnitudes+opposite signs)
# γ=0: fully break pair antisymmetry (independent magnitudes, same signs)
#
# We show that predictability (vs FULL) of four metrics {res, rea, r̃med_s, r̃med_l}
# declines as γ↓ — i.e., the impact of RESHUFFLE/REWIRE is contingent on preserving
# antisymmetry. Strength scale is controlled by matching mean |α|.
#
# Assumes in scope (or `include("MainCode.jl")`):
#   build_random_trophic, random_u, jacobian, resilience, reactivity,
#   alpha_off_from, build_J_from, realized_IS, median_return_rate
########################################################
# ---------- helpers ---------------------------------------------------
metrics(J,u; t_short=0.01,t_long=0.50) = (
    res = resilience(J),
    rea = reactivity(J),
    rmed_s = median_return_rate(J,u; t=t_short, perturbation=:biomass),
    rmed_l = median_return_rate(J,u; t=t_long,  perturbation=:biomass)
)

r2_to_identity(x,y) = (μ=mean(y); sst=sum((y .- μ).^2); ssr=sum((y .- x).^2); sst==0 ? 1.0 : 1 - ssr/sst)

# mean |α| over nonzeros
function mean_abs_alpha(α)
    vals = [abs(α[i,j]) for i in 1:size(α,1), j in 1:size(α,2) if i!=j && α[i,j]!=0]
    return isempty(vals) ? 0.0 : mean(vals)
end

# scale α_new so that mean |α_new| matc hes that of α_ref
function renorm_alpha!(α_new, α_ref)
    m_ref = mean_abs_alpha(α_ref)
    m_new = mean_abs_alpha(α_new)
    if m_new > 0
        α_new .*= (m_ref / m_new)
    end
    return α_new
end

# diagnostics: sign antisymmetry and magnitude correlation across unordered pairs
function pair_diagnostics(α)
    S = size(α,1)
    signs = Float64[]
    mags1 = Float64[]; mags2 = Float64[]
    for i in 1:S, j in (i+1):S
        if α[i,j]!=0.0 || α[j,i]!=0.0
            push!(signs, sign(α[i,j]) == -sign(α[j,i]) ? 1.0 : 0.0)
            push!(mags1, abs(α[i,j]))
            push!(mags2, abs(α[j,i]))
        end
    end
    sign_anti = isempty(signs) ? NaN : mean(signs)
    mag_corr  = (isempty(mags1) || std(mags1)==0 || std(mags2)==0) ? NaN : cor(mags1, mags2)
    return (sign_anti=sign_anti, mag_corr=mag_corr)
end

# ---------- continuous antisymmetry operators on α --------------------
# γ∈[0,1] controls preservation level. For broken part, we:
#  - draw independent magnitudes from the global |α| pool
#  - set SAME signs on the two directions (maximal sign break)
function reshuffle_continuous(α; γ::Float64, rng=Random.default_rng())
    S = size(α,1)
    out = zeros(eltype(α), S, S)
    # pool of magnitudes for broken part
    pool = [abs(α[i,j]) for i in 1:S, j in 1:S if i!=j && α[i,j]!=0.0]
    pool_len = length(pool)
    # unordered pairs
    pairs = [(i,j) for i in 1:S for j in (i+1):S if α[i,j]!=0.0 || α[j,i]!=0.0]
    # indices to randomize blocks for the preserved subset
    perm_blocks = randperm(rng, length(pairs))
    # When preserving, we keep the original block at the SAME place (pure reshuffle keeps topology)
    for (k,(i,j)) in enumerate(pairs)
        if rand(rng) < γ
            # preserve this pair block exactly
            out[i,j] = α[i,j]; out[j,i] = α[j,i]
        else
            # break magnitude correlation but preserve opposite signs
            if pool_len == 0
                out[i,j] = 0.0; out[j,i] = 0.0
            else
                m1 = pool[rand(1:pool_len)]
                m2 = pool[rand(1:pool_len)]
                if rand(rng) < 0.5
                    out[i,j], out[j,i] =  m1, -m2
                else
                    out[i,j], out[j,i] = -m1,  m2
                end
            end
        end
    end
    return out
end

# REWIRE: keep the *count* of active unordered pairs; place results on new pairs
function rewire_continuous(α; γ::Float64, rng=Random.default_rng())
    S = size(α,1)
    out = zeros(eltype(α), S, S)
    # collect existing blocks and all destination pairs
    blocks = Tuple{eltype(α),eltype(α)}[]
    src_pairs = Tuple{Int,Int}[]
    for i in 1:S, j in (i+1):S
        if α[i,j]!=0.0 || α[j,i]!=0.0
            push!(blocks, (α[i,j], α[j,i]))
            push!(src_pairs, (i,j))
        end
    end
    K = length(blocks)
    allpairs = [(i,j) for i in 1:S for j in (i+1):S]
    # pick K destination pairs without replacement
    dest_idx = randperm(rng, length(allpairs))[1:K]
    # pool for broken case
    pool = [abs(α[i,j]) for i in 1:S, j in 1:S if i!=j && α[i,j]!=0.0]
    pool_len = length(pool)

    # decide which fraction is preserved as intact blocks
    K_pres = round(Int, γ*K)
    pres_idx = randperm(rng, K)[1:K_pres]
    pres_mask = falses(K); pres_mask[pres_idx] .= true

    for k in 1:K
        i,j = allpairs[dest_idx[k]]
        if pres_mask[k]
            # move intact block to a new pair → preserves pair antisymmetry
            out[i,j], out[j,i] = blocks[k]
        else
            # break antisymmetry at this new location
            if pool_len == 0
                out[i,j] = 0.0; out[j,i] = 0.0
            else
                m1 = pool[rand(1:pool_len)]
                m2 = pool[rand(1:pool_len)]
                if rand(rng) < 0.5
                    out[i,j], out[j,i] =  m1, -m2
                else
                    out[i,j], out[j,i] = -m1,  m2
                end
            end
        end
    end
    return out
end

# ---------- experiment runner ----------------------------------------
Base.@kwdef struct AntiContOpts
    S::Int = 120; conn::Float64 = 0.10; mean_abs::Float64 = 0.10; mag_cv::Float64 = 0.60
    degree_family::Symbol = :lognormal; deg_param::Float64 = 0.0
    rho_sym::Float64 = 1.0
    u_mean::Float64 = 1.0; u_cv::Float64 = 0.8
    IS_target::Float64 = 0.10      # fix realized IS of A before making α
    reps::Int = 120
    t_short::Float64 = 0.01; t_long::Float64 = 0.50
    gammas::Vector{Float64} = collect(range(0.0, 1.0; length=11))
    seed::Int = 20251027
end

function run_antisymmetry_continuum(opts::AntiContOpts)
    rng_global = Random.Xoshiro(opts.seed)
    rows = NamedTuple[]

    for r in 1:opts.reps
        rng = Random.Xoshiro(rand(rng_global, UInt64))
        # base trophic A and u
        A0 = build_niche_trophic(opts.S; conn=opts.conn, mean_abs=opts.mean_abs, mag_cv=opts.mag_cv,
                                  degree_family=opts.degree_family, deg_param=opts.deg_param,
                                  rho_sym=opts.rho_sym, rng=rng)
        base_IS = realized_IS(A0)
        β = base_IS>0 ? opts.IS_target/base_IS : 1.0
        A = β .* A0
        u = random_u(opts.S; mean=opts.u_mean, cv=opts.u_cv, rng=rng)

        # baseline J and α
        J_full = jacobian(A,u)
        mF = metrics(J_full,u; t_short=opts.t_short, t_long=opts.t_long)
        α0 = alpha_off_from(J_full, u)
        diag0 = pair_diagnostics(α0)

        for γ in opts.gammas
            # RESHUFFLE continuum (keep pair locations)
            α_re = reshuffle_continuous(α0; γ=γ, rng=rng)
            renorm_alpha!(α_re, α0)
            d_re = pair_diagnostics(α_re)
            J_re = build_J_from(α_re, u)
            m_re = metrics(J_re,u; t_short=opts.t_short, t_long=opts.t_long)

            # REWIRE continuum (move pair locations)
            α_rw = rewire_continuous(α0; γ=γ, rng=rng)
            renorm_alpha!(α_rw, α0)
            d_rw = pair_diagnostics(α_rw)
            J_rw = build_J_from(α_rw, u)
            m_rw = metrics(J_rw,u; t_short=opts.t_short, t_long=opts.t_long)

            push!(rows, (; rep=r, gamma=γ,
                           # baseline metrics
                           res_full=mF.res, rea_full=mF.rea, rmed_s_full=mF.rmed_s, rmed_l_full=mF.rmed_l,
                           signanti_full=diag0.sign_anti, magcorr_full=diag0.mag_corr,
                           # reshuffle
                           res_re=m_re.res, rea_re=m_re.rea, rmed_s_re=m_re.rmed_s, rmed_l_re=m_re.rmed_l,
                           signanti_re=d_re.sign_anti, magcorr_re=d_re.mag_corr,
                           # rewire
                           res_rw=m_rw.res, rea_rw=m_rw.rea, rmed_s_rw=m_rw.rmed_s, rmed_l_rw=m_rw.rmed_l,
                           signanti_rw=d_rw.sign_anti, magcorr_rw=d_rw.mag_corr))
        end
    end

    df = DataFrame(rows)

    # R² vs γ (per manipulation)
    function summarize(df; metric_sym::Symbol, col_step::Symbol)
        out = DataFrame(gamma=Float64[], r2=Float64[])
        for γ in sort(unique(df.gamma))
            sub = df[df.gamma .== γ, :]
            x = sub[!, Symbol(metric_sym, :_full)]
            y = sub[!, Symbol(metric_sym, :_, col_step)]
            r2 = r2_to_identity(collect(x), collect(y))
            push!(out, (; gamma=γ, r2=r2))
        end
        return out
    end

    summ = Dict(
        :reshuffle => (
            res  = summarize(df; metric_sym=:res,     col_step=:re),
            rea  = summarize(df; metric_sym=:rea,     col_step=:re),
            rmed_s = summarize(df; metric_sym=:rmed_s, col_step=:re),
            rmed_l = summarize(df; metric_sym=:rmed_l, col_step=:re)
        ),
        :rewire => (
            res  = summarize(df; metric_sym=:res,     col_step=:rw),
            rea  = summarize(df; metric_sym=:rea,     col_step=:rw),
            rmed_s = summarize(df; metric_sym=:rmed_s, col_step=:rw),
            rmed_l = summarize(df; metric_sym=:rmed_l, col_step=:rw)
        )
    )

    return df, summ
end

# ---------- plotting --------------------------------------------------
# Accepts either the Dict returned by run_antisymmetry_continuum
# or the raw df (and will rebuild the needed summaries on the fly).
function plot_r2_vs_gamma(input; metric::Union{Symbol,String}=:res, title::String="Predictability vs antisymmetry preservation γ")
    # Convert metric to Symbol
    msym = metric isa String ? Symbol(metric) : metric

    # Local helper to summarize from a raw df
    function _summarize_from_df(df::DataFrame, metric_sym::Symbol, col_step::Symbol)
        out = DataFrame(gamma=Float64[], r2=Float64[])
        for γ in sort(unique(df[!, :gamma]))
            sub = df[df[!, :gamma] .== γ, :]
            x = sub[!, Symbol(metric_sym, :_full)]
            y = sub[!, Symbol(metric_sym, :_, col_step)]
            push!(out, (; gamma=γ, r2=r2_to_identity(collect(x), collect(y))))
        end
        out
    end

    # Normalize input → build `summ` Dict with the expected shape
    summ = if input isa Dict
        input
    elseif input isa DataFrame
        Dict(
            :reshuffle => Dict(
                :res     => _summarize_from_df(input, :res,     :re),
                :rea     => _summarize_from_df(input, :rea,     :re),
                :rmed_s  => _summarize_from_df(input, :rmed_s,  :re),
                :rmed_l  => _summarize_from_df(input, :rmed_l,  :re)
            ),
            :rewire => Dict(
                :res     => _summarize_from_df(input, :res,     :rw),
                :rea     => _summarize_from_df(input, :rea,     :rw),
                :rmed_s  => _summarize_from_df(input, :rmed_s,  :rw),
                :rmed_l  => _summarize_from_df(input, :rmed_l,  :rw)
            )
        )
    else
        error("plot_r2_vs_gamma expects the Dict from run_antisymmetry_continuum or a DataFrame with :gamma and *_full/*_re/*_rw columns.")
    end

    # Plot
    fig = Figure(size=(950, 520))
    Label(fig[0, 1:2]; text=title)
    axes_titles = ["res","rea","r̃med_s","r̃med_l"]
    keys = [:res,:rea,:rmed_s,:rmed_l]
    for (i,k) in enumerate(keys)
        ax = Axis(fig[1,i]; title=axes_titles[i],
                  xlabel=(i==2 ? "γ (preserved pairs fraction)" : ""),
                  ylabel=(i==1 ? "R²" : ""), limits=((0,1),(-0.1,1.05)))
        # reshuffle
        d_re = summ[:reshuffle][k]
        lines!(ax, d_re[!, :gamma], d_re[!, :r2]; label="reshuffle")
        scatter!(ax, d_re[!, :gamma], d_re[!, :r2])
        # rewire
        d_rw = summ[:rewire][k]
        lines!(ax, d_rw[!, :gamma], d_rw[!, :r2]; label="rewire")
        scatter!(ax, d_rw[!, :gamma], d_rw[!, :r2])
        if i==1; axislegend(ax; position=:lb); end
    end
    display(fig)
end

function plot_antisymmetry_diagnostics(df::DataFrame; title="Pair antisymmetry diagnostics")
    # @assert :gamma in names(df) "The DataFrame passed to plot_antisymmetry_diagnostics must be the one returned by run_antisymmetry_continuum (it needs a :gamma column)."
    # @assert all(Ref.([:signanti_re, :magcorr_re, :signanti_rw, :magcorr_rw]) .∈ Ref(names(df))) "Missing diagnostics columns; pass df returned by run_antisymmetry_continuum."

    fig = Figure(size=(900, 380))
    ax1 = Axis(fig[1,1]; title="sign antisymmetry", xlabel="γ", ylabel="mean 1{sign opposite}", limits=((0,1),(-0.1,1.1)))
    ax2 = Axis(fig[1,2]; title="magnitude correlation", xlabel="γ", ylabel="corr(|α_ij|,|α_ji|)", limits=((0,1),(-0.1,1.1)))

    for (label, suf) in [("reshuffle","re"), ("rewire","rw")]
        grp = combine(groupby(df, :gamma),
                      :gamma => first => :gamma,
                      Symbol("signanti_"*suf) => mean => Symbol(label*"_signanti"),
                      Symbol("magcorr_"*suf)  => mean => Symbol(label*"_magcorr"))
        lines!(ax1, grp[!, :gamma], grp[!, Symbol(label*"_signanti")]; label=label)
        scatter!(ax1, grp[!, :gamma], grp[!, Symbol(label*"_signanti")])
        lines!(ax2, grp[!, :gamma], grp[!, Symbol(label*"_magcorr")]; label=label)
        scatter!(ax2, grp[!, :gamma], grp[!, Symbol(label*"_magcorr")])
    end
    axislegend(ax1; position=:lb)
    display(fig)
end

# ---------- example run ----------------------------------------------
opts = AntiContOpts()
df, summ = run_antisymmetry_continuum(opts)
# Clip all negative R² values to 0.0 across all metrics and manipulations
for manip in keys(summ)
    for metric in keys(summ[manip])
        dfmetric = summ[manip][metric]
        dfmetric.r2 .= clamp.(dfmetric.r2, 0.0, 1.0)
    end
end
plot_r2_vs_gamma(summ; title="Predictability vs antisymmetry preservation γ (initial ρ = 1.0)")
plot_antisymmetry_diagnostics(df)


