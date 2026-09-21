using LinearAlgebra, Printf, Test
using CairoMakie

# All outputs remain alongside this script, independent of the working directory.
const OUT = joinpath(@__DIR__, "outputs")
const K = 2.0
const EPSILON = 0.20       # Equal absolute changes, 10% of either baseline link.
const AMPLITUDE = 0.20     # Same forcing amplitude for every system/frequency.
const DEMO_FREQUENCIES = [0.5, 4.0] # Set in advance, either side of sqrt(K^2-1).
const COLORS = ["#D55E00", "#0072B2"]
const LABELS = ["A: weaken 2 → 1", "B: strengthen 1 → 2"]

function setup()
    # Two self-regulating species with predator-prey interaction signs.
    # x denotes local displacement, not raw abundance. T = I.
    A = [-1.0 -K; K -1.0]
    b = [1.0,0.0]          # Environment always acts on species 1 only.
    c = [0.0,1.0]          # Same focal species 2, every time.
    P = [[0.0 1.0; 0.0 0.0], [0.0 0.0; 1.0 0.0]]
    A,b,c,P
end

R(A,w) = (im*w*I-A) \ Matrix{Float64}(I,2,2)

function metrics(A,b,c,P,w,epsilon=EPSILON)
    r=R(A,w); h=r*P*r*b
    exact=(R(A+epsilon*P,w)-r)*b
    (; focal_prediction=AMPLITUDE*epsilon*abs(dot(c,h))/sqrt(2),
       community_prediction=AMPLITUDE*epsilon*norm(h)/sqrt(2),
       focal_exact=AMPLITUDE*abs(dot(c,exact))/sqrt(2),
       community_exact=AMPLITUDE*norm(exact)/sqrt(2))
end

function integrate_forcing(A,b,w,t)
    # Independent RK4 dynamics, starting from the reference equilibrium.
    X=zeros(2,length(t)); x=zeros(2)
    f(x,s)=A*x+AMPLITUDE*b*cos(w*s)
    for k in 1:length(t)-1
        h=t[k+1]-t[k]; s=t[k]
        k1=f(x,s); k2=f(x+h*k1/2,s+h/2)
        k3=f(x+h*k2/2,s+h/2); k4=f(x+h*k3,s+h)
        x += h*(k1+2k2+2k3+k4)/6
        X[:,k+1]=x
    end
    X
end

function rms_integral(y,t)
    # Trapezoidal time mean over an exact integer number of cycles.
    sqrt(sum(diff(t).*(y[1:end-1]+y[2:end])/2)/(t[end]-t[1]))
end

function simulation(A,b,c,Ps,w;epsilon=EPSILON,refinement=1)
    period=2pi/w; burn_cycles=ceil(Int,35/period)
    steps=refinement*max(800,ceil(Int,period/0.005))
    t=collect(range(0,(burn_cycles+4)*period;length=(burn_cycles+4)*steps+1))
    keep=(burn_cycles*steps+1):length(t)
    base=integrate_forcing(A,b,w,t)[:,keep]
    modified=[integrate_forcing(A+epsilon*P,b,w,t)[:,keep] for P in Ps]
    tk=t[keep]; differences=[x-base for x in modified]
    focal=[rms_integral(vec(d[2,:]).^2,tk) for d in differences]
    community=[rms_integral(vec(sum(abs2,d;dims=1)),tk) for d in differences]
    expected=real.((AMPLITUDE*R(A,w)*b).*transpose(exp.(im*w*tk)))
    ode_error=norm(base-expected)/norm(expected)
    (;w,t=tk,cycles=(tk.-tk[1])./period,base,modified,differences,focal,community,ode_error)
end

function checks(A,b,c,Ps,sims)
    @testset "Forcing specific predictions and simulations" begin
        @test maximum(real.(eigvals(A))) < 0
        for P in Ps
            @test maximum(real.(eigvals(A+EPSILON*P))) < 0
            @test count(!iszero,P) == 1
            @test norm(P) == 1
        end
        for w in [0.0,0.5,sqrt(3),2.0,4.0,10.0]
            ma=metrics(A,b,c,Ps[1],w); mb=metrics(A,b,c,Ps[2],w)
            @test ma.focal_prediction/mb.focal_prediction ≈ K^2/(1+w^2)
            @test ma.community_prediction/mb.community_prediction ≈ K/sqrt(1+w^2)
            for P in Ps
                h=1e-5
                derivative=(R(A+h*P,w)-R(A-h*P,w))/(2h)
                @test norm(derivative-R(A,w)*P*R(A,w)) < 1e-8
                @test norm(R(A,w)*P*R(A,w)*b) <= opnorm(R(A,w)*P*R(A,w))+1e-12
            end
        end
        for s in sims
            @test s.ode_error < 1e-7
            for l in 1:2
                m=metrics(A,b,c,Ps[l],s.w)
                @test isapprox(s.focal[l],m.focal_exact;rtol=1e-6)
                @test isapprox(s.community[l],m.community_exact;rtol=1e-6)
                # Validate prediction convergence, not just one finite change.
                small=metrics(A,b,c,Ps[l],s.w,EPSILON/10)
                @test abs(small.community_prediction/small.community_exact-1) < abs(m.community_prediction/m.community_exact-1)
            end
        end
        @test sims[1].focal[1] > sims[1].focal[2]
        @test sims[1].community[1] > sims[1].community[2]
        @test sims[2].focal[1] < sims[2].focal[2]
        @test sims[2].community[1] < sims[2].community[2]
        fine=simulation(A,b,c,Ps,DEMO_FREQUENCIES[2];refinement=2)
        @test all(isapprox.(fine.community,sims[2].community;rtol=1e-7))
    end
end

function save_data(A,b,c,Ps,ws,sims)
    mkpath(OUT)
    open(joinpath(OUT,"frequency_predictions.csv"),"w") do io
        println(io,"omega,modification,focal_predicted_rms,community_predicted_rms,focal_exact_rms,community_exact_rms")
        for w in ws, l in 1:2
            m=metrics(A,b,c,Ps[l],w)
            println(io,join((w,l,m.focal_prediction,m.community_prediction,m.focal_exact,m.community_exact),","))
        end
    end
    open(joinpath(OUT,"simulation_metrics.csv"),"w") do io
        println(io,"omega,modification,focal_simulated_rms,focal_predicted_rms,community_simulated_rms,community_predicted_rms,focal_relative_prediction_error,community_relative_prediction_error")
        for s in sims, l in 1:2
            m=metrics(A,b,c,Ps[l],s.w)
            println(io,join((s.w,l,s.focal[l],m.focal_prediction,s.community[l],m.community_prediction,
                abs(m.focal_prediction/s.focal[l]-1),abs(m.community_prediction/s.community[l]-1)),","))
        end
    end
    open(joinpath(OUT,"trajectories.csv"),"w") do io
        println(io,"omega,time,cycles_after_burn,original_species1,original_species2,A_species1,A_species2,B_species1,B_species2")
        for s in sims, k in eachindex(s.t)
            println(io,join((s.w,s.t[k],s.cycles[k],s.base[1,k],s.base[2,k],s.modified[1][1,k],s.modified[1][2,k],s.modified[2][1,k],s.modified[2][2,k]),","))
        end
    end
    open(joinpath(OUT,"results.txt"),"w") do io
        println(io,"Julia ",VERSION,"; CairoMakie ",Base.pkgversion(CairoMakie))
        println(io,"A = ",A,"; T=I; b=",b,"; focal species=2")
        println(io,"Forcing amplitude=",AMPLITUDE,"; absolute link increment=",EPSILON)
        println(io,"First-order crossover omega=sqrt(k^2-1)=",sqrt(K^2-1))
        println(io,"Predicted A/B focal RMS ratio = k^2/(1+omega^2)")
        println(io,"Predicted A/B community RMS ratio = k/sqrt(1+omega^2)")
        for s in sims
            println(io,"\nomega=",s.w,"; period=",2pi/s.w)
            println(io,"Simulated focal RMS A/B = ",s.focal,"; ratio=",s.focal[1]/s.focal[2])
            println(io,"Simulated community RMS A/B = ",s.community,"; ratio=",s.community[1]/s.community[2])
            for l in 1:2
                m=metrics(A,b,c,Ps[l],s.w)
                println(io,LABELS[l],": predicted focal=",m.focal_prediction,"; predicted community=",m.community_prediction,
                    "; relative prediction error=",abs(m.community_prediction/s.community[l]-1))
            end
            println(io,"Direct ODE baseline vs exact periodic solution relative error=",s.ode_error)
        end
        println(io,"\nThis is a designed transparent illustrative local linear model, not empirical or nonlinear validation.")
        println(io,"No search over networks, inputs, observables or frequencies; the chosen frequencies straddle an analytic crossover.")
    end
end

function make_figure(A,b,c,Ps,ws,sims)
    set_theme!(Theme(fontsize=15,Axis=(xgridvisible=false,ygridvisible=false,spinewidth=0.8)))
    fig=Figure(size=(1450,1350),figure_padding=30)
    Label(fig[0,1:2],"The same interaction change matters differently under slow and fast forcing",fontsize=25,font=:bold)
    Label(fig[1,1:2],"Two species • forcing on species 1 only • species 2 shown in both time-series panels • identical forcing amplitude",fontsize=15)
    Label(fig[2,1:2],"Original links: 2 → 1 = −2, 1 → 2 = +2.   Modification A: −2 → −1.8.   Modification B: +2 → +2.2.   Self-regulation = −1.",fontsize=14)
    ax1=Axis(fig[3,1],title="A  Prediction for the focal species",xlabel="Angular forcing frequency ω",ylabel="Predicted RMS departure of species 2",xscale=log10,yscale=log10)
    ax2=Axis(fig[3,2],title="B  Prediction for the whole community",xlabel="Angular forcing frequency ω",ylabel="Predicted community RMS distance",xscale=log10,yscale=log10)
    for l in 1:2
        ms=[metrics(A,b,c,Ps[l],w) for w in ws]
        lines!(ax1,ws,[m.focal_prediction for m in ms],color=COLORS[l],linewidth=3,label=LABELS[l])
        lines!(ax2,ws,[m.community_prediction for m in ms],color=COLORS[l],linewidth=3,label=LABELS[l])
    end
    for ax in [ax1,ax2]
        vlines!(ax,DEMO_FREQUENCIES,color=:gray65,linestyle=:dot)
        vlines!(ax,[sqrt(K^2-1)],color=:gray40,linestyle=:dash)
    end
    axislegend(ax1,position=:lb,framevisible=false,labelsize=12)
    axislegend(ax2,position=:lb,framevisible=false,labelsize=12)
    Label(fig[4,1:2],"Dotted lines: demonstration frequencies 0.5 and 4.   Dashed line: analytically predicted crossover at ω = √3.   All predictions use the actual forcing.",fontsize=13,color=:gray35)

    max_original=maximum(maximum(abs,x) for s in sims for x in [s.base,s.modified...])
    max_delta=maximum(maximum(abs,d[2,:]) for s in sims for d in s.differences)
    for (q,s) in enumerate(sims)
        g=GridLayout(); fig[5,q]=g
        title=@sprintf("%s  %s forcing: ω = %.1f",q==1 ? "C" : "D",q==1 ? "Slow" : "Fast",s.w)
        ax=Axis(g[1,1],title=title,ylabel="Species 2 displacement",xticklabelsvisible=false)
        da=Axis(g[2,1],xlabel="Time (model units; two cycles)",ylabel="Departure from original",ylabelsize=12,yticklabelsize=11)
        show=findall(s.cycles .<= 2.0+1e-9)
        tt=s.t[show].-s.t[1]
        lines!(ax,tt,s.base[2,show],color=:gray25,linewidth=3,label="Original")
        for l in 1:2
            lines!(ax,tt,s.modified[l][2,show],color=COLORS[l],linewidth=2.5,label=LABELS[l])
            lines!(da,tt,s.differences[l][2,show],color=COLORS[l],linewidth=2.5)
        end
        hlines!(da,[0],color=:gray75,linewidth=0.8)
        ylims!(ax,-1.08max_original,1.08max_original)
        ylims!(da,-1.12max_delta,1.12max_delta)
        linkxaxes!(ax,da)
        q==1 && axislegend(ax,position=:rt,labelsize=10,framevisible=true)
        rowsize!(g,2,Relative(0.30)); rowgap!(g,8)
    end
    Label(fig[6,1:2],"Direct simulations above: original and modified systems receive the same disturbance. Lower strips show their differences. Matching vertical scales across slow/fast panels.",fontsize=13,color=:gray35)
    for (q,field) in enumerate([:focal,:community])
        title=q==1 ? "E  Measured effect on the focal species" : "F  Measured effect on the whole community"
        ax=Axis(fig[7,q],title=title,ylabel=q==1 ? "Species 2 RMS departure" : "Community RMS distance",xticks=([1.5,4.5],["Slow forcing (ω = 0.5)","Fast forcing (ω = 4)"]))
        positions=[1.,2.,4.,5.]
        actual=[getproperty(s,field)[l] for s in sims for l in 1:2]
        predicted=[getproperty(metrics(A,b,c,Ps[l],s.w),q==1 ? :focal_prediction : :community_prediction) for s in sims for l in 1:2]
        barplot!(ax,positions,actual,color=repeat(COLORS,2),width=0.7)
        scatter!(ax,positions,predicted,marker=:diamond,color=:white,strokecolor=:black,strokewidth=1.5,markersize=14,label="First-order prediction")
        ylims!(ax,0,1.2maximum(vcat(actual,predicted)))
        axislegend(ax,position=:rt,labelsize=12,framevisible=false)
    end
    Label(fig[8,1:2],"Bars: RMS measured over four complete cycles after transients. Diamonds: forcing-specific first-order predictions.\nCommunity distance combines squared species departures before averaging over time; it is not total biomass.",fontsize=13,color=:gray35)
    Label(fig[9,1:2],"A minimal local linear example with predator–prey signs. Displacements and time are in model units. A 10% link change makes finite-change departures from the prediction visible.",fontsize=13,color=:gray35)
    rowgap!(fig.layout,15); colgap!(fig.layout,35)
    rowsize!(fig.layout,3,Relative(0.24)); rowsize!(fig.layout,5,Relative(0.33)); rowsize!(fig.layout,7,Relative(0.21))
    save(joinpath(OUT,"forcing_specific_effects.png"),fig,px_per_unit=1.7)
    save(joinpath(OUT,"forcing_specific_effects.pdf"),fig)
end

function main()
    A,b,c,Ps=setup()
    ws=10.0 .^ range(-1,1;length=400)
    sims=[simulation(A,b,c,Ps,w) for w in DEMO_FREQUENCIES]
    checks(A,b,c,Ps,sims)
    save_data(A,b,c,Ps,ws,sims)
    make_figure(A,b,c,Ps,ws,sims)
    print(read(joinpath(OUT,"results.txt"),String))
end

main()
