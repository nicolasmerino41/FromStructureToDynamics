using LinearAlgebra, Printf, Test
using CairoMakie

const OUT=joinpath(@__DIR__,"outputs")
const FRACTION=0.30
const EPS=2FRACTION
const AMP=0.20
const MIN_FREQUENCY=0.5 # User-selected lower bound for periodic demonstrations.
const COLORS=["#D55E00","#0072B2"]
const NAMES=["A: weaken 2 → 1","B: strengthen 1 → 2"]

function system(g=1.0,tau=2.0)
    # Original pair retained. Species 3 receives equal effects from both,
    # has slower relaxation, and has no feedback to the pair.
    A=[-1.0 -2.0 0.0;2.0 -1.0 0.0;g g -1.0]
    T=Diagonal([1.0,1.0,tau]); b=[1.0,0.0,0.0]
    Ps=[zeros(3,3),zeros(3,3)]; Ps[1][1,2]=1; Ps[2][2,1]=1
    A,T,b,Ps
end
res(A,T,w)=(im*w*T-A)\Matrix{Float64}(I,3,3)
function response(A,T,b,P,w,e=EPS)
    R=res(A,T,w)
    exact=AMP*(res(A+e*P,T,w)-R)*b
    first=AMP*e*R*P*R*b
    (;exact,first,community=norm(exact)/sqrt(2),focal=abs(exact[2])/sqrt(2),
      first_community=norm(first)/sqrt(2),first_focal=abs(first[2])/sqrt(2))
end

function goldenmax(f,a,b)
    r=(sqrt(5)-1)/2; c=b-r*(b-a); d=a+r*(b-a)
    fc=f(c); fd=f(d)
    for _ in 1:90
        if fc>fd
            b=d; d=c; fd=fc; c=b-r*(b-a); fc=f(c)
        else
            a=c; c=d; fc=fd; d=a+r*(b-a); fd=f(d)
        end
    end
    (a+b)/2
end

function peak(A,T,b,P;e=EPS,points=4001)
    # Scan includes zero; refine ALL resolved local maxima. No crossover input.
    f(w)=response(A,T,b,P,w,e).community
    upper=20.0; grid=collect(range(0,upper;length=points)); values=f.(grid)
    candidates=[0.0,upper]
    for k in 2:length(grid)-1
        if values[k]>=values[k-1] && values[k]>=values[k+1]
            push!(candidates,goldenmax(f,grid[k-1],grid[k+1]))
        end
    end
    w=candidates[argmax(f.(candidates))]
    # Resolvent identity + singular-value bound proves the omitted tail cannot win.
    minT=minimum(diag(T)); nA=opnorm(A); nAp=opnorm(A+e*P)
    @assert upper*minT>max(nA,nAp)
    tail=AMP*e/((upper*minT-nA)*(upper*minT-nAp))/sqrt(2)
    @assert tail<f(w)
    (;w,value=f(w),tail)
end

function gap_peak(A,T,b,Ps,l;points=4001)
    # Maximise the signed vertical gap, not either curve's height or their ratio.
    other=3-l
    f(w)=response(A,T,b,Ps[l],w).community-response(A,T,b,Ps[other],w).community
    upper=20.0; grid=collect(range(MIN_FREQUENCY,upper;length=points)); values=f.(grid)
    candidates=[MIN_FREQUENCY,upper]
    for k in 2:length(grid)-1
        if values[k]>=values[k-1] && values[k]>=values[k+1]
            push!(candidates,goldenmax(f,grid[k-1],grid[k+1]))
        end
    end
    w=candidates[argmax(f.(candidates))]
    minT=minimum(diag(T)); nA=opnorm(A); nAp=opnorm(A+EPS*Ps[l])
    @assert upper*minT>max(nA,nAp)
    # Signed advantage cannot exceed the favoured curve itself.
    tail=AMP*EPS/((upper*minT-nA)*(upper*minT-nAp))/sqrt(2)
    @assert tail<f(w)
    (;w,value=f(w),tail)
end

function integrate(A,T,b,w,t)
    J=T\A; q=T\b; X=zeros(3,length(t)); x=zeros(3)
    f(x,s)=J*x+(w==0 ? AMP/sqrt(2) : AMP*cos(w*s))*q
    for k in 1:length(t)-1
        h=t[k+1]-t[k]; s=t[k]
        a=f(x,s); bb=f(x+h*a/2,s+h/2); c=f(x+h*bb/2,s+h/2); d=f(x+h*c,s+h)
        x+=h*(a+2bb+2c+d)/6; X[:,k+1]=x
    end
    X
end
rms(y,t)=sqrt(sum(diff(t).*(y[1:end-1]+y[2:end])/2)/(t[end]-t[1]))
function simulate(A,T,b,Ps,w;refine=1)
    if w==0
        decay=minimum(-maximum(real.(eigvals(T\M))) for M in [A,(A+EPS*P for P in Ps)...])
        burn=35/decay; duration=burn+20/decay
        t=collect(range(0,duration;length=ceil(Int,duration/(0.005/refine))+1))
        original=integrate(A,T,b,w,t)
        modified=[integrate(A+EPS*P,T,b,w,t) for P in Ps]
        delta=[M-original for M in modified]; keep=findall(t .>= burn)
        focal=[rms(vec(D[2,keep]).^2,t[keep]) for D in delta]
        community=[rms(vec(sum(abs2,D[:,keep];dims=1)),t[keep]) for D in delta]
        return (;w,t,original,modified,delta,focal,community,keep)
    end
    period=2pi/w
    decay=minimum(-maximum(real.(eigvals(T\M))) for M in [A,(A+EPS*P for P in Ps)...])
    burn=ceil(Int,35/decay/period)
    steps=refine*max(1000,ceil(Int,period/0.005))
    t=collect(range(0,(burn+4)*period;length=(burn+4)*steps+1)); keep=burn*steps+1:length(t)
    original=integrate(A,T,b,w,t)[:,keep]
    modified=[integrate(A+EPS*P,T,b,w,t)[:,keep] for P in Ps]
    t=t[keep]; delta=[M-original for M in modified]
    focal=[rms(vec(D[2,:]).^2,t) for D in delta]
    community=[rms(vec(sum(abs2,D;dims=1)),t) for D in delta]
    (;w,t,original,modified,delta,focal,community,keep=collect(eachindex(t)))
end

function checks(A,T,b,Ps,peaks,sims)
    @testset "Maximum signed gaps and forcing specific response" begin
        @test maximum(real.(eigvals(T\A)))<0
        @test all(p.w>=MIN_FREQUENCY for p in peaks)
        for P in Ps, f in [0.1,0.2,0.3]
            @test maximum(real.(eigvals(T\(A+2f*P))))<0
        end
        for l in 1:2
            p2=gap_peak(A,T,b,Ps,l;points=8001)
            @test isapprox(peaks[l].w,p2.w;atol=1e-5)
            @test peaks[l].value>=response(A,T,b,Ps[l],MIN_FREQUENCY).community-response(A,T,b,Ps[3-l],MIN_FREQUENCY).community-1e-12
            # Check the complete visible band independently, including endpoints.
            @test all(peaks[l].value >= response(A,T,b,Ps[l],w).community-response(A,T,b,Ps[3-l],w).community-1e-10 for w in range(MIN_FREQUENCY,20;length=3001))
            for s in sims
                r=response(A,T,b,Ps[l],s.w)
                @test isapprox(s.focal[l],r.focal;rtol=1e-7)
                @test isapprox(s.community[l],r.community;rtol=1e-7)
                exacttrace=s.w==0 ? repeat(real.(r.exact)/sqrt(2),1,length(s.keep)) : real.(r.exact.*transpose(exp.(im*s.w*s.t[s.keep])))
                @test norm(exacttrace-s.delta[l][:,s.keep])/norm(exacttrace)<1e-7
            end
        end
        for w in [0.,0.5,1.,2.,4.],P in Ps
            h=1e-5; R=res(A,T,w)
            fd=(res(A+h*P,T,w)-res(A-h*P,T,w))/(2h)
            @test norm(fd-R*P*R)<1e-8
        end
        fine=simulate(A,T,b,Ps,peaks[2].w;refine=2)
        @test all(isapprox.(fine.community,sims[2].community;rtol=1e-7))
    end
end

function export_results(A,T,b,Ps,peaks,sims,ws)
    mkpath(OUT)
    open(joinpath(OUT,"profiles.csv"),"w") do io
        println(io,"omega,modification,exact_community_rms,exact_focal_rms,first_order_community_rms,first_order_focal_rms")
        for w in ws,l in 1:2
            r=response(A,T,b,Ps[l],w)
            println(io,join((w,l,r.community,r.focal,r.first_community,r.first_focal),","))
        end
    end
    open(joinpath(OUT,"amplitude_comparison.csv"),"w") do io
        println(io,"fraction,modification,own_peak_omega,peak_community_rms,peak_focal_rms,relative_complex_error,relative_community_rms_error,relative_focal_rms_error")
        for f in [0.1,0.2,0.3],l in 1:2
            p=peak(A,T,b,Ps[l];e=2f); r=response(A,T,b,Ps[l],p.w,2f)
            println(io,join((f,l,p.w,r.community,r.focal,norm(r.exact-r.first)/norm(r.exact),abs(r.first_community/r.community-1),abs(r.first_focal/r.focal-1)),","))
        end
    end
    open(joinpath(OUT,"parameter_sweep.csv"),"w") do io
        println(io,"incoming_strength,timescale3,modification,peak_omega,peak_community_rms,peak_focal_rms,species3_fraction_of_squared_departure")
        for g in [0.5,1.0,1.5],tau in [1.,2.,3.],l in 1:2
            AA,TT,bb,pp=system(g,tau); p=peak(AA,TT,bb,pp[l]); r=response(AA,TT,bb,pp[l],p.w)
            println(io,join((g,tau,l,p.w,r.community,r.focal,abs2(r.exact[3])/sum(abs2,r.exact)),","))
        end
    end
    open(joinpath(OUT,"trajectories.csv"),"w") do io
        println(io,"largest_advantage_of_modification,omega,time_from_display_start,species,original,modified_A,modified_B")
        for (q,s) in enumerate(sims),k in eachindex(s.t),i in 1:3
            println(io,join((q,s.w,s.t[k]-s.t[1],i,s.original[i,k],s.modified[1][i,k],s.modified[2][i,k]),","))
        end
    end
    open(joinpath(OUT,"summary.txt"),"w") do io
        println(io,"Julia ",VERSION,"; CairoMakie ",Base.pkgversion(CairoMakie))
        println(io,"A=",A,"; T=",diag(T),"; forcing b=",b,"; forcing amplitude=",AMP)
        println(io,"Equal interaction increments=",EPS," (",100FRACTION,"%). Focal species=2.")
        println(io,"Selection: maximize D_A-D_B, then D_B-D_A, using exact finite-change community RMS. Restricted to omega >= 0.5; no curve-peak or ratio selection.")
        println(io,"Equal forcing RMS=",AMP/sqrt(2),". Both panels use periodic input AMP*cos(omega*t), with identical amplitude and phase.")
        for (q,s) in enumerate(sims)
            println(io,"\nLargest advantage of modification ",q,": omega=",s.w,"; signed gap=",peaks[q].value,"; regime=",s.w==0 ? "constant forcing" : "periodic forcing")
            for l in 1:2
                r=response(A,T,b,Ps[l],s.w)
                println(io,NAMES[l],"; community RMS=",s.community[l],"; focal RMS=",s.focal[l],
                    "; species3 squared share=",abs2(r.exact[3])/sum(abs2,r.exact),
                    "; first-order complex error=",norm(r.exact-r.first)/norm(r.exact))
            end
        end
        println(io,"Both panels show two periodic cycles after transients. RMS uses four complete cycles.")
        println(io,"\nMain curves use exact finite-change resolvents, not first-order derivatives. This is exact only within the local linear model.")
        println(io,"Species 3 changes the output weighting but does not feed back to species 2. The extra species makes community and focal responses distinct, not independent.")
    end
end

function figure_main(A,T,b,Ps,peaks,sims,ws)
    set_theme!(Theme(fontsize=16,Axis=(xgridvisible=false,ygridvisible=false,spinewidth=0.8)))
    fig=Figure(size=(1500,1150),figure_padding=30)
    Label(fig[0,1:2],"Where modifications A and B differ most within ω ≥ 0.5",fontsize=26,font=:bold)
    Label(fig[1,1:2],"Same forced species 1 • same focal species 2 • equal forcing RMS • equal 30% interaction changes",fontsize=16)
    ax=Axis(fig[2,1],title="A  A slower third species receives both effects",limits=(-0.25,1.25,-0.2,1.2))
    hidedecorations!(ax); hidespines!(ax)
    points=[Point2f(0.15,0.85),Point2f(0.95,0.85),Point2f(0.55,0.15)]
    # Offset upper arrows identify two directional links unambiguously.
    arrows2d!(ax,[Point2f(.26,.94)],[Vec2f(.57,0)],color=COLORS[2],shaftwidth=3,tipwidth=12,tiplength=14)
    arrows2d!(ax,[Point2f(.84,.75)],[Vec2f(-.57,0)],color=COLORS[1],shaftwidth=3,tipwidth=12,tiplength=14)
    arrows2d!(ax,[Point2f(.22,.72),Point2f(.88,.72)],[Vec2f(.27,-.46),Vec2f(-.27,-.46)],color=:gray45,shaftwidth=2,tipwidth=11,tiplength=13)
    scatter!(ax,points,color=["#E6E6E6","#DDEBF4","#E7E5DA"],markersize=57,strokecolor=:gray30,strokewidth=1)
    for i in 1:3; text!(ax,points[i],text=string(i),align=(:center,:center),fontsize=21,font=:bold); end
    text!(ax,.55,1.06,text="B: +2 → +2.6",color=COLORS[2],align=(:center,:center),fontsize=15)
    text!(ax,.55,.67,text="A: −2 → −1.4",color=COLORS[1],align=(:center,:center),fontsize=15)
    text!(ax,.19,.4,text="+1",align=(:center,:center));text!(ax,.91,.4,text="+1",align=(:center,:center))
    text!(ax,.10,1.10,text="forced",fontsize=14,align=(:center,:center))
    text!(ax,1.02,1.10,text="focal",fontsize=14,align=(:center,:center))
    text!(ax,.55,-.035,text="τ₃ = 2; τ₁ = τ₂ = 1\nNo feedback from species 3",fontsize=14,align=(:center,:center))
    axp=Axis(fig[2,2],title="B  Largest vertical gaps for ω ≥ 0.5",xlabel="Angular forcing frequency ω",ylabel="RMS departure from original",limits=((0.,6.),nothing))
    for l in 1:2
        rr=[response(A,T,b,Ps[l],w) for w in ws]
        lines!(axp,ws,[r.community for r in rr],color=COLORS[l],linewidth=3,label=NAMES[l])
        lines!(axp,ws,[r.focal for r in rr],color=(COLORS[l],.7),linestyle=:dash,linewidth=2)
        y1=response(A,T,b,Ps[1],peaks[l].w).community
        y2=response(A,T,b,Ps[2],peaks[l].w).community
        lines!(axp,[peaks[l].w,peaks[l].w],[y1,y2],color=COLORS[l],linewidth=5)
        scatter!(axp,[peaks[l].w,peaks[l].w],[y1,y2],color=COLORS[l],markersize=10)
        vlines!(axp,[peaks[l].w],color=(COLORS[l],.4),linestyle=:dot)
    end
    axislegend(axp,position=:rt,labelsize=12,framevisible=false)
    Label(fig[3,1],"Self-regulation: Aᵢᵢ = −1. Species 3 has twice the relaxation timescale.\nAll species are represented in common scaled displacement units.",fontsize=13,color=:gray35)
    Label(fig[3,2],"Solid: community. Dashed: species 2. Segments: largest gaps for ω ≥ 0.5.\nOrange maximises D_A − D_B; blue maximises D_B − D_A.",fontsize=14,color=:gray35)
    original_limit=maximum(maximum(abs,M[2,:]) for s in sims for M in [s.original,s.modified...])
    delta_limit=maximum(maximum(abs,D[2,:]) for s in sims for D in s.delta)
    for (q,s) in enumerate(sims)
        g=GridLayout();fig[4,q]=g
        ax=Axis(g[1,1],title=@sprintf("%s  Largest %s advantage in range: ω = %.3f",q==1 ? "C" : "D",q==1 ? "A" : "B",s.w),ylabel="Species 2 displacement",xticklabelsvisible=false)
        da=Axis(g[2,1],xlabel=s.w==0 ? "Time since constant forcing starts" : "Time after transient (model units)",ylabel="Departure",ylabelsize=13)
        show=findall((s.t.-s.t[1]) .<= (s.w==0 ? 15.0 : 4pi/s.w)+1e-10); tt=s.t[show].-s.t[1]
        lines!(ax,tt,s.original[2,show],color=:gray25,linewidth=3,label="Original")
        for l in 1:2
            lines!(ax,tt,s.modified[l][2,show],color=COLORS[l],linewidth=2.5,label=l==1 ? "Modification A" : "Modification B")
            lines!(da,tt,s.delta[l][2,show],color=COLORS[l],linewidth=2.5)
        end
        hlines!(da,[0],color=:gray70,linewidth=1)
        ylims!(ax,-1.1original_limit,1.1original_limit);ylims!(da,-1.1delta_limit,1.1delta_limit)
        linkxaxes!(ax,da);rowsize!(g,2,Relative(.3));rowgap!(g,9)
        q==1 && axislegend(ax,position=:rt,labelsize=11,framevisible=true)
        Label(fig[5,q],@sprintf("Community RMS: A %.4f  |  B %.4f\nSpecies 2 RMS: A %.4f  |  B %.4f",s.community[1],s.community[2],s.focal[1],s.focal[2]),fontsize=16)
    end
    Label(fig[6,1:2],"Direct ODE simulations: identical forcing amplitude and phase; matching vertical scales. Both panels show established periodic responses.\nRMS uses four complete cycles after transients. Selection maximises community gaps for ω ≥ 0.5, not focal-species gaps.",fontsize=14,color=:gray35)
    Label(fig[7,1:2],"Illustrative local linear model. Exact finite-change predictions are used because a 30% modification produces substantial first-order error; see the separate diagnostic.",fontsize=13,color=:gray35)
    rowgap!(fig.layout,18);colgap!(fig.layout,36)
    rowsize!(fig.layout,2,Relative(.34));rowsize!(fig.layout,4,Relative(.39))
    save(joinpath(OUT,"peak_effects.png"),fig,px_per_unit=1.6)
    save(joinpath(OUT,"peak_effects.pdf"),fig)
end

function diagnostics(A,T,b,Ps)
    fig=Figure(size=(1200,500),figure_padding=25)
    Label(fig[0,1:2],"Larger structural changes: visibility versus first-order accuracy",fontsize=23,font=:bold)
    ax=Axis(fig[1,1],xlabel="Change relative to baseline link (%)",ylabel="Relative complex-response error (%)",title="Approximation error at each modification's own peak")
    ay=Axis(fig[1,2],xlabel="Change relative to baseline link (%)",ylabel="Peak community RMS departure",title="Magnitude of the actual finite-change effect")
    for l in 1:2
        fractions=[.1,.2,.3];rr=[response(A,T,b,Ps[l],peak(A,T,b,Ps[l];e=2f).w,2f) for f in fractions]
        scatterlines!(ax,100fractions,[100norm(r.exact-r.first)/norm(r.exact) for r in rr],color=COLORS[l],linewidth=3,label=NAMES[l])
        scatterlines!(ay,100fractions,[r.community for r in rr],color=COLORS[l],linewidth=3)
    end
    axislegend(ax,position=:lt,labelsize=12,framevisible=false)
    Label(fig[2,1:2],"Errors include phase and amplitude. Exact finite-change resolvents match direct integration; they do not validate a nonlinear ecological model.",fontsize=13)
    save(joinpath(OUT,"amplitude_diagnostic.png"),fig,px_per_unit=1.6)
    save(joinpath(OUT,"amplitude_diagnostic.pdf"),fig)
end

function main()
    A,T,b,Ps=system();peaks=[gap_peak(A,T,b,Ps,l) for l in 1:2]
    sims=[simulate(A,T,b,Ps,p.w) for p in peaks]
    checks(A,T,b,Ps,peaks,sims)
    ws=collect(range(0.,6;length=701)) # Plot full frequency range; selection still uses MIN_FREQUENCY.
    export_results(A,T,b,Ps,peaks,sims,ws)
    figure_main(A,T,b,Ps,peaks,sims,ws);diagnostics(A,T,b,Ps)
    print(read(joinpath(OUT,"summary.txt"),String))
end
main()
