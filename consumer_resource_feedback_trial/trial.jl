using LinearAlgebra, Statistics, Printf, Random, DelimitedFiles
using CairoMakie

# A reproducible exploratory study, not an empirical fit or a claim of generality.
const OUT = joinpath(@__DIR__, "outputs")
mkpath(OUT)
const BLUE = "#0072B2"
const ORANGE = "#D55E00"
const INK = "#263746"
const FRACTION = 0.10
const ENV_AMPLITUDE = 0.02
const BASE_Q = 0.40
const BASE_GAMMA = 0.20
const FREQ = 10 .^ range(log10(0.015), log10(4.0), length=241)
const GAMMAS = 10 .^ range(log10(0.02), log10(2.0), length=151)
const TAUS = 10 .^ range(-2, 2, length=181)
const QS = collect(range(0.12, 0.80, length=151))
set_theme!(Theme(font="Arial", fontsize=19, textcolor=INK,
    Axis=(xgridvisible=false, ygridvisible=false, titlesize=21,
          titlealign=:left, spinewidth=1, xticklabelsize=16, yticklabelsize=16),
    Legend=(labelsize=17, framevisible=false)))

"""
Logistic resource, consumer with optional saturating consumption:
Rdot = r R (1-R/K) (1+u) - a R C/(1+a h R)
Cdot = s [e a R/(1+a h R)-m] C.
s scales both consumer gain and loss (phenomenological consumer response speed).
Default r=K=1, e=0.5, m=0.2. h=0 is type I.
u is fractional resource-growth forcing, with fixed amplitude/variance.
"""
function model(a, s; h=0.0, r=1.0, K=1.0, e=0.5, m=0.2)
    R=m/(a*(e-m*h)); q=R/K
    C=r*(1-q)*(1+a*h*R)/a
    d=1+a*h*R
    J=[r*(1-2q)-a*C/d^2 -a*R/d; s*e*a*C/d^2 0.0]
    b=[r*R*(1-q), 0.0]
    stable=0<q<1 && C>0 && maximum(real.(eigvals(J))) < -1e-10
    (;a,s,h,r,K,e,m,R,C,q,J,b,stable)
end
function from_q(q,gamma; chi=0.0)
    r=1.0; m=0.2; e=0.5
    h=chi*e/m
    a=m/(q*(e-m*h))
    model(a,gamma*r/m;h)
end
modified(p; fraction=FRACTION)=model(p.a*(1-fraction),p.s;h=p.h,r=p.r,K=p.K,e=p.e,m=p.m)
response(p,w)=(im*w*I-p.J)\p.b
harmvar(p,w)=abs2.(response(p,w)).*(ENV_AMPLITUDE^2/2)

# Stationary covariance under unit-variance OU forcing, E[u(t)u(t+l)]=exp(-|l|/tau).
# The environmental variance is held fixed, not the noise-increment intensity.
function oucov(p,tau)
    L=zeros(3,3); L[1:2,1:2]=p.J; L[1:2,3]=p.b; L[3,3]=-1/tau
    Q=zeros(3,3); Q[3,3]=2/tau
    P=reshape(-(kron(Matrix{Float64}(I,3,3),L)+kron(L,Matrix{Float64}(I,3,3)))\vec(Q),3,3)
    (P+P')/2
end
ouvar(p,tau)=diag(oucov(p,tau))[1:2]
percent(x,y)=100*(y/x-1)
function csv(name, header, rows)
    open(joinpath(OUT,name),"w") do io
        println(io,join(header,",")); for row in rows; println(io,join(row,",")); end
    end
end
function savepng(name,f)
    save(joinpath(OUT,name*".png"),f,px_per_unit=1.7)
end

function derivatives(p)
    @assert p.h==0
    # Full derivative along the moving coexistence equilibrium.
    Jtotal=[p.r*p.q/p.a 0.0; p.s*p.e*p.r*p.q/p.a 0.0]
    Jdirect=[-p.C -p.R; p.s*p.e*p.C p.s*p.e*p.R]
    bprime=[-p.r*p.R/p.a*(1-2p.q),0.0]
    Jtotal,Jdirect,Jtotal-Jdirect,bprime
end
function signed_terms(p,w)
    H=inv(im*w*I-p.J); y=H*p.b
    jt,jd,je,db=derivatives(p)
    ys=[H*jd*y,H*je*y,H*db]
    # First-order percentage variance changes for a 10% attack-rate reduction.
    [100*2real(conj(y[1])*z[1])*(-FRACTION*p.a)/abs2(y[1]) for z in ys]
end

function rhs(p,z,t,w)
    R,C=z; u=ENV_AMPLITUDE*cos(w*t)
    consumption=p.a*R*C/(1+p.a*p.h*R)
    [p.r*R*(1-R/p.K)*(1+u)-consumption,
     p.s*(p.e*consumption-p.m*C)]
end
function simulate(p,w;dtmax=0.02)
    period=2pi/w; nper=ceil(Int,period/dtmax); dt=period/nper
    decay=-maximum(real.(eigvals(p.J)))
    burn=ceil(Int,40/decay/period); ncycles=8
    z=[p.R,p.C]; data=zeros(nper*ncycles,3)
    for k in 1:((burn+ncycles)*nper)
        t=(k-1)*dt
        k1=rhs(p,z,t,w); k2=rhs(p,z+dt*k1/2,t+dt/2,w)
        k3=rhs(p,z+dt*k2/2,t+dt/2,w); k4=rhs(p,z+dt*k3,t+dt,w)
        z+=dt*(k1+2k2+2k3+k4)/6
        @assert minimum(z)>0 && all(isfinite,z)
        if k>burn*nper
            n=k-burn*nper; data[n,:]=[n*dt,z[1],z[2]]
        end
    end
    av=vec(mean(data[:,2:3],dims=1)); v=vec(var(data[:,2:3],dims=1,corrected=false))
    (;data, av, v, period,dt)
end

function main()
    p=from_q(BASE_Q,BASE_GAMMA); pm=modified(p)
    @assert p.stable && pm.stable
    checks=String[]
    # Analytic type-I harmonic and OU expressions, independent of matrix solvers.
    max_ou_error=0.0; max_harm_error=0.0; max_lyap=0.0
    for q in [0.15,0.4,0.7], g in [0.03,0.2,1.5], tau in [0.02,1.,50.]
        a=from_q(q,g); alpha=q; beta=g*(1-q); b=q*(1-q)
        exact=b^2/(alpha*(alpha+1/tau+beta*tau))
        P=oucov(a,tau)
        max_ou_error=max(max_ou_error,abs(P[1,1]/exact-1))
        L=zeros(3,3);L[1:2,1:2]=a.J;L[1:2,3]=a.b;L[3,3]=-1/tau
        Q=zeros(3,3);Q[3,3]=2/tau
        max_lyap=max(max_lyap,norm(L*P+P*L'+Q)/norm(Q))
        for w in [0.02,0.3,3.]
            v=b^2*w^2/((beta-w^2)^2+alpha^2*w^2)
            max_harm_error=max(max_harm_error,abs(abs2(response(a,w)[1])/v-1))
        end
    end
    @assert max_ou_error<1e-10 && max_harm_error<1e-10 && max_lyap<1e-10
    # Full structural derivative versus a central difference of the response itself.
    derivative_error=0.0
    for w in [0.03,0.3,1.0]
        H=inv(im*w*I-p.J); jt,jd,je,db=derivatives(p)
        analytic=H*jt*H*p.b+H*db; step=p.a*1e-5
        numeric=(response(model(p.a+step,p.s),w)-response(model(p.a-step,p.s),w))/(2step)
        derivative_error=max(derivative_error,norm(analytic-numeric)/norm(numeric))
    end
    @assert derivative_error<1e-7

    # Figure 1: signed harmonic effects, same intervention throughout.
    absmap=zeros(length(FREQ),length(GAMMAS)); cvmap=similar(absmap)
    rows=Vector{Any}()
    for (j,g) in enumerate(GAMMAS), (i,w) in enumerate(FREQ)
        a=from_q(BASE_Q,g); b=modified(a)
        @assert a.stable && b.stable
        v=harmvar(a,w)[1]; vm=harmvar(b,w)[1]
        absmap[i,j]=percent(v,vm); cvmap[i,j]=percent(v/a.R^2,vm/b.R^2)
        push!(rows,(w,g,v,vm,absmap[i,j],cvmap[i,j]))
    end
    csv("harmonic_map.csv",["omega_over_r","consumer_speed_s_m_over_r","resource_variance","modified_variance","variance_change_percent","cv2_change_percent"],rows)
    v0=[harmvar(p,w)[1] for w in FREQ]; v1=[harmvar(pm,w)[1] for w in FREQ]
    dif=v1-v0
    eligible=findall(v0 .>= 0.10maximum(v0))
    ilo=eligible[argmax(dif[eligible])]; ihi=eligible[argmin(dif[eligible])]
    chosen=[FREQ[ilo],FREQ[ihi]]
    @assert dif[ilo]>0 && dif[ihi]<0
    f=Figure(size=(1400,980),figure_padding=26)
    Label(f[0,1:3],"Does weaker consumption buffer or amplify resource fluctuations?",fontsize=29,font=:bold,tellwidth=false)
    ax=Axis(f[1,1],title="A  Change in resource variance",xlabel="Environmental frequency  ω / r",ylabel="Consumer response speed  s m / r",xscale=log10,yscale=log10)
    hm=heatmap!(ax,FREQ,GAMMAS,absmap,colormap=:RdBu_11,colorrange=(-60,60),lowclip=:darkred,highclip=:darkblue)
    contour!(ax,FREQ,GAMMAS,absmap,levels=[0.],color=:black,linewidth=2)
    hlines!(ax,[BASE_GAMMA],color=:gray25,linestyle=:dash)
    scatter!(ax,chosen,fill(BASE_GAMMA,2),color=:white,strokecolor=:black,strokewidth=2,markersize=13)
    ax=Axis(f[1,2],title="B  Change in relative variance (CV²)",xlabel="Environmental frequency  ω / r",ylabel="Consumer response speed  s m / r",xscale=log10,yscale=log10)
    heatmap!(ax,FREQ,GAMMAS,cvmap,colormap=:RdBu_11,colorrange=(-60,60),lowclip=:darkred,highclip=:darkblue)
    contour!(ax,FREQ,GAMMAS,cvmap,levels=[0.],color=:black,linewidth=2)
    Colorbar(f[1,3],hm,label="Change after 10% lower attack rate (%)")
    ax=Axis(f[2,1],title="C  Baseline slice: absolute response",xlabel="Environmental frequency  ω / r",ylabel="Resource standard deviation / K",xscale=log10)
    lines!(ax,FREQ,sqrt.(v0),color=BLUE,linewidth=3,label="Original attack rate")
    lines!(ax,FREQ,sqrt.(v1),color=ORANGE,linewidth=3,label="10% lower attack rate")
    vlines!(ax,chosen,color=:gray40,linestyle=:dash)
    axislegend(ax,position=:rt)
    ax=Axis(f[2,2],title="D  Same slice: sign and measurement",xlabel="Environmental frequency  ω / r",ylabel="Change (%)",xscale=log10)
    lines!(ax,FREQ,percent.(v0,v1),color=INK,linewidth=3,label="Absolute variance")
    lines!(ax,FREQ,percent.(v0./p.R^2,v1./pm.R^2),color=:gray50,linewidth=3,linestyle=:dash,label="Relative variance (CV²)")
    hlines!(ax,[0.],color=:gray70);vlines!(ax,chosen,color=:gray60,linestyle=:dot)
    axislegend(ax,position=:rb)
    Label(f[3,1:3],"Type-I consumer–resource model • resource growth is forced • each intervention uses its own equilibrium\nBlack contours: zero effect. Red: less variable; blue: more variable. Colour scale saturates at ±60%.",fontsize=17,tellwidth=false)
    savepng("01_periodic_response_map",f)

    # Figure 2: independent nonlinear checks and a useful mechanistic diagnostic.
    sims=[simulate(a,w) for a in (p,pm),w in chosen]
    simrows=Vector{Any}(); validation=Vector{Any}()
    f=Figure(size=(1400,1100),figure_padding=26)
    Label(f[0,1:2],"The same attack-rate reduction can have opposite effects",fontsize=29,font=:bold,tellwidth=false)
    for (k,w) in enumerate(chosen)
        title=k==1 ? "A  Frequency with increased variance" : "B  Frequency with decreased variance"
        ax=Axis(f[1,k],title=title,xlabel="Environmental cycles after transients",ylabel="Resource displacement from its own mean / K")
        for (j,a) in enumerate((p,pm))
            sim=sims[j,k]; sel=sim.data[:,1].<=3sim.period
            lines!(ax,sim.data[sel,1]./sim.period,sim.data[sel,2].-sim.av[1],color=[BLUE,ORANGE][j],linewidth=3,label=["Original","10% lower attack rate"][j])
            y=response(a,w)[1]*ENV_AMPLITUDE
            lines!(ax,sim.data[sel,1]./sim.period,real.(y.*exp.(im*w.*sim.data[sel,1])),color=[BLUE,ORANGE][j],linestyle=:dash,linewidth=1.2)
            v=harmvar(a,w)
            push!(validation,(k,w,j,sim.av[1],a.R,sim.v[1],v[1],abs(sim.v[1]/v[1]-1),sim.v[2],v[2],abs(sim.v[2]/v[2]-1)))
            for z in eachindex(sim.data[:,1]);push!(simrows,(k,w,j,sim.data[z,1],sim.data[z,2],sim.data[z,3]));end
        end
        maxamp=maximum(sqrt.(2 .* vcat(v0,v1)))
        ylims!(ax,-1.1maxamp,1.35maxamp)
        text!(ax,0.03,0.97,text=@sprintf("ω/r = %.3f   |   Δ variance = %+.1f%%",w,percent(v0[[ilo,ihi][k]],v1[[ilo,ihi][k]])),space=:relative,align=(:left,:top),fontsize=17)
    end
    Legend(f[2,1:2],[LineElement(color=BLUE,linewidth=3),LineElement(color=ORANGE,linewidth=3)],["Original","10% lower attack rate"],orientation=:horizontal)
    ax=Axis(f[3,1],title="C  What changes with attack rate?",xlabel="Change (%)",yticks=(1:5,["Resource abundance","Consumer abundance","Resource damping","Feedback strength","Environmental coupling"]))
    changes=[percent(p.R,pm.R),percent(p.C,pm.C),percent(-p.J[1,1],-pm.J[1,1]),percent(-p.J[1,2]*p.J[2,1],-pm.J[1,2]*pm.J[2,1]),percent(p.b[1],pm.b[1])]
    barplot!(ax,1:5,changes,direction=:x,color=[x>=0 ? BLUE : ORANGE for x in changes])
    vlines!(ax,[0.],color=:gray65);xlims!(ax,-12,16)
    for (j,x) in enumerate(changes);text!(ax,x+(x>0 ? .6 : -.6),j,text=@sprintf("%+.1f%%",x),align=(x>0 ? :left : :right,:center),fontsize=16);end
    ax=Axis(f[3,2],title="D  Accounting for the whole biological change",xlabel="Environmental frequency  ω / r",ylabel="First-order variance change (%)",xscale=log10)
    terms=hcat([signed_terms(p,w) for w in FREQ]...)'
    for (j,col,label) in [(1,BLUE,"Direct Jacobian change"),(2,ORANGE,"Equilibrium-mediated Jacobian change"),(3,"#009E73","Change in environmental coupling")]
        lines!(ax,FREQ,terms[:,j],color=col,linewidth=2.3,label=label)
    end
    lines!(ax,FREQ,vec(sum(terms,dims=2)),color=INK,linewidth=3,label="Total first-order effect")
    lines!(ax,FREQ,percent.(v0,v1),color=INK,linestyle=:dash,linewidth=2,label="Exact finite-change local effect")
    hlines!(ax,[0.],color=:gray70)
    Legend(f[4,2],ax,labelsize=14,nbanks=2)
    Label(f[4,1],"Consumer lag is 90° at every frequency in this model.\nA changing delay alone cannot explain the reversal.\nBars describe the equilibrium and its local dynamics.",fontsize=16,tellwidth=false)
    Label(f[5,1:2],"Top: nonlinear simulations (solid), local harmonic predictions (thin dashed); identical vertical scales.\nBottom right: diagnostic decomposition, not separately implementable ecological interventions.",fontsize=17,tellwidth=false)
    savepng("02_mechanism_and_time_series",f)
    csv("nonlinear_validation.csv",["example","omega","intervention","nonlinear_resource_mean","equilibrium_resource","nonlinear_resource_variance","linear_resource_variance","resource_relative_error","nonlinear_consumer_variance","linear_consumer_variance","consumer_relative_error"],validation)
    csv("nonlinear_trajectories.csv",["example","omega","intervention","time","resource","consumer"],simrows)
    csv("derivative_decomposition.csv",["omega","direct_percent","equilibrium_percent","input_percent","total_first_order_percent","exact_percent"],[(w,terms[i,1],terms[i,2],terms[i,3],sum(terms[i,:]),percent(v0[i],v1[i])) for (i,w) in enumerate(FREQ)])

    # Figure 3: OU forcing can average away a narrow-band reversal.
    oumap=zeros(length(TAUS),length(QS)); oucv=similar(oumap); ourows=Vector{Any}()
    for (j,q) in enumerate(QS), (i,tau) in enumerate(TAUS)
        a=from_q(q,BASE_GAMMA); b=modified(a)
        @assert a.stable && b.stable
        v=ouvar(a,tau); vm=ouvar(b,tau)
        oumap[i,j]=percent(v[1],vm[1]);oucv[i,j]=percent(v[1]/a.R^2,vm[1]/b.R^2)
        push!(ourows,(tau,q,v[1],vm[1],oumap[i,j],oucv[i,j],percent(v[2],vm[2]),a.R,b.R,a.C,b.C))
    end
    @assert maximum(oucv)<0
    # A dimensionless sign relationship, tested independently against small interventions.
    sign_error=0.0
    for q in [.15,.35,.4,.6,.8], g in [.03,.2,1.], theta in [.01,.3,3.,100.]
        D=q+1/theta+g*(1-q)*theta
        N=-2q^2+(1-3q)/theta+g*theta*(1-q)*(1-2q)
        analytic=N/((1-q)*D) # derivative of log variance w.r.t. log q
        a=from_q(q,g)
        plus=from_q(q*exp(1e-5),g);minus=from_q(q*exp(-1e-5),g)
        numeric=(log(ouvar(plus,theta)[1])-log(ouvar(minus,theta)[1]))/2e-5
        sign_error=max(sign_error,abs(analytic-numeric))
    end
    @assert sign_error<1e-7
    csv("persistent_environment_map.csv",["persistence_tau_times_r","resource_equilibrium_over_K","resource_variance_per_unit_environment_variance","modified_variance","variance_change_percent","cv2_change_percent","consumer_variance_change_percent","resource_mean","modified_resource_mean","consumer_mean","modified_consumer_mean"],ourows)
    f=Figure(size=(1400,970),figure_padding=26)
    Label(f[0,1:3],"Does the reversal survive a fluctuating, persistent environment?",fontsize=28,font=:bold,tellwidth=false)
    ax=Axis(f[1,1],title="A  Absolute resource variance",xlabel="Environmental persistence  τ r",ylabel="Original resource abundance / carrying capacity",xscale=log10)
    hm=heatmap!(ax,TAUS,QS,oumap,colormap=:RdBu_11,colorrange=(-35,35))
    contour!(ax,TAUS,QS,oumap,levels=[0.],color=:black,linewidth=2)
    hlines!(ax,[.2,.4,.6],color=:gray30,linestyle=:dash)
    ax=Axis(f[1,2],title="B  Relative resource variance (CV²)",xlabel="Environmental persistence  τ r",ylabel="Original resource abundance / carrying capacity",xscale=log10)
    heatmap!(ax,TAUS,QS,oucv,colormap=:RdBu_11,colorrange=(-35,35))
    Colorbar(f[1,3],hm,label="Change after 10% lower attack rate (%)")
    ax=Axis(f[2,1],title="C  Three ecological starting states",xlabel="Environmental persistence  τ r",ylabel="Change in resource variance (%)",xscale=log10)
    for (q,col) in [(0.2,BLUE),(0.4,ORANGE),(0.6,"#009E73")]
        a=from_q(q,BASE_GAMMA);b=modified(a)
        lines!(ax,TAUS,[percent(ouvar(a,t)[1],ouvar(b,t)[1]) for t in TAUS],color=col,linewidth=3,label=@sprintf("Resource / K = %.1f",q))
    end
    hlines!(ax,[0.],color=:gray60);axislegend(ax,position=:rb)
    ax=Axis(f[2,2],title="D  Absolute size of fluctuations",xlabel="Environmental persistence  τ r",ylabel="Resource SD / K at environmental SD = 0.02",xscale=log10)
    lines!(ax,TAUS,[ENV_AMPLITUDE*sqrt(ouvar(p,t)[1]) for t in TAUS],color=BLUE,linewidth=3,label="Original")
    lines!(ax,TAUS,[ENV_AMPLITUDE*sqrt(ouvar(pm,t)[1]) for t in TAUS],color=ORANGE,linewidth=3,label="10% lower attack rate")
    axislegend(ax,position=:rt)
    Label(f[3,1:3],"OU environmental noise with fixed variance; only its persistence changes. Curves are exact stationary local covariances.\nAbsolute-variance conclusions can reverse while relative variability declines throughout this type-I model.",fontsize=17,tellwidth=false)
    savepng("03_environmental_persistence",f)

    # Figure 4: limited robustness, exposing rather than hiding model dependence.
    f=Figure(size=(1400,970),figure_padding=26)
    Label(f[0,1:2],"Which conclusions survive changes to the first model?",fontsize=29,font=:bold,tellwidth=false)
    robustrows=Vector{Any}()
    configs=[("A  Consumer response speed",[("s m / r = $g",BASE_Q,g,0.0,FRACTION) for g in [.05,.2,.8]]),
        ("B  Size of the attack-rate reduction",[("$(Int(round(100v)))% reduction",BASE_Q,BASE_GAMMA,0.0,v) for v in [.01,.05,.1,.2]]),
        ("C  Saturating consumption",[("Saturation χ = $c",BASE_Q,BASE_GAMMA,c,FRACTION) for c in [0.,.15,.3]]),
        ("D  Consumer variability",[("Resource / K = $q",q,BASE_GAMMA,0.0,FRACTION) for q in [.2,.4,.6]])]
    colors=[BLUE,ORANGE,"#009E73","#CC79A7"]
    for (k,(ttl,cases)) in enumerate(configs)
        ax=Axis(f[div(k-1,2)+1,mod(k-1,2)+1],title=ttl,xlabel="Environmental persistence  τ r",ylabel=k==4 ? "Change in consumer variance (%)" : "Change in resource variance (%)",xscale=log10)
        for (j,(label,q,g,chi,frac)) in enumerate(cases)
            a=from_q(q,g;chi);b=modified(a;fraction=frac);@assert a.stable && b.stable
            idx=k==4 ? 2 : 1
            vals=[percent(ouvar(a,t)[idx],ouvar(b,t)[idx]) for t in TAUS]
            lines!(ax,TAUS,vals,color=colors[j],linewidth=3,label=label)
            for (t,v) in zip(TAUS,vals);push!(robustrows,(k,q,g,chi,frac,t,idx,v));end
        end
        hlines!(ax,[0.],color=:gray65);axislegend(ax,position=:rb,labelsize=16)
    end
    Label(f[3,1:2],"All cases recompute coexistence equilibria and pass local stability checks. Saturation comparisons keep the initial R*/K fixed.\nThese are bounded robustness checks, not a survey of food webs or evidence of a universal rule.",fontsize=17,tellwidth=false)
    savepng("04_robustness_and_consumer",f)
    csv("robustness.csv",["panel","q","gamma","saturation_chi","attack_reduction_fraction","tau","output_species","variance_change_percent"],robustrows)

    # Independent spectral quadrature of the OU covariance, and dt refinement.
    ws=10 .^ range(-7,5,length=14001);quaderr=0.0
    for tau in [.05,1.,20.]
        integrand=[abs2(response(p,w)[1])*2tau/(1+(w*tau)^2)/pi for w in ws]
        integral=sum(diff(ws).*(integrand[1:end-1]+integrand[2:end])/2)
        quaderr=max(quaderr,abs(integral/ouvar(p,tau)[1]-1))
    end
    @assert quaderr<1e-5
    refined=simulate(pm,chosen[2];dtmax=.01)
    dterr=maximum(abs.(refined.v./sims[2,2].v.-1))
    nlerr=maximum(max(row[8],row[11]) for row in validation)
    @assert dterr<1e-5 && nlerr<0.02
    csv("equilibria.csv",["state","attack_rate","resource","consumer","resource_over_K","max_eigenvalue_real"],[(name,a.a,a.R,a.C,a.q,maximum(real.(eigvals(a.J)))) for (name,a) in [("original",p),("modified",pm)]])
    open(joinpath(OUT,"checks_and_findings.txt"),"w") do io
        println(io,"Completed checks (errors are fractions, not percentages)")
        for (name,x) in [("OU analytic vs Lyapunov",max_ou_error),("Harmonic analytic vs resolvent",max_harm_error),("Lyapunov residual",max_lyap),("Full derivative vs finite difference",derivative_error),("Dimensionless sign derivative",sign_error),("OU covariance vs spectral quadrature",quaderr),("Nonlinear RK4 step halving",dterr),("Nonlinear vs local variance, largest error",nlerr)]
            println(io,"$name: $x")
        end
        println(io,"\nBaseline equilibrium R=$(p.R), C=$(p.C); modified R=$(pm.R), C=$(pm.C)")
        for (k,w) in enumerate(chosen);println(io,"Harmonic example $k: omega=$w, resource variance change=$(percent(harmvar(p,w)[1],harmvar(pm,w)[1]))%, CV2 change=$(percent(harmvar(p,w)[1]/p.R^2,harmvar(pm,w)[1]/pm.R^2))%");end
        for q in [.2,.4,.6]
            a=from_q(q,BASE_GAMMA);b=modified(a)
            for tau in [.01,1.,100.];println(io,"OU q=$q tau=$tau: absolute change=$(percent(ouvar(a,tau)[1],ouvar(b,tau)[1]))%, CV2 change=$(percent(ouvar(a,tau)[1]/a.R^2,ouvar(b,tau)[1]/b.R^2))%");end
        end
        println(io,"OU CV2 map maximum effect=$(maximum(oucv))%; all tested CV2 changes negative.")
        println(io,"All 4 figures are PNG only. Exact finite-change local effects are used unless explicitly labelled first-order.")
    end
    println(read(joinpath(OUT,"checks_and_findings.txt"),String))
end
main()
