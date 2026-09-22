using LinearAlgebra, Statistics, DelimitedFiles, Test, Printf, CairoMakie
const ROOT=@__DIR__;const OUT=joinpath(ROOT,"outputs","oscillatory_test");mkpath(OUT)
const P=joinpath(ROOT,"processed")
const A=readdlm(joinpath(P,"interactions.csv"),',',Float64)
const R=vec(readdlm(joinpath(P,"growth.csv"),',',Float64))
const EPS=vec(readdlm(joinpath(P,"susceptibilities.csv"),',',Float64))
const IX=[4,5,6,7,8,9,11];const B=A[IX,IX];const EE=EPS[IX];const FOCAL=6
const LABELS=["Blautia → C. difficile","C. difficile → Akkermansia"]
const LINKS=[(6,2),(4,6)]
const U0=.001;const AMP=.0002;const FRACTION=.001
const PERIODS=[2.,30.];const GRID=10 .^ range(0,log10(180),length=181)
const COLORS=["#0072B2","#D55E00"]
function csv(name,header,rows)
    open(joinpath(OUT,name),"w") do io;println(io,join(header,","));for row in rows;println(io,join(row,","));end;end
end
function state(background,BB=B)
    rr=R[IX]+EE*background;n=-(BB\rr);AA=copy(A);AA[IX,IX]=BB
    full=zeros(11);full[IX]=n;Jfull=Diagonal(R+EPS*background+AA*full)+Diagonal(full)*AA
    margin=maximum(real.(eigvals(Jfull)))
    @assert minimum(n)>0 && margin<0 "Reference support is infeasible or unstable"
    (;rr,n,J=Diagonal(n)*BB,b=n.*EE,margin)
end
function components(s,l,period)
    i,j=LINKS[l];E=zeros(7,7);E[i,j]=1;dn=-(B\(E*s.n));H=(im*2pi/period*I-s.J)\Matrix{Float64}(I,7,7);y=H*s.b
    direct=H*(Diagonal(s.n)*E*y)
    equilibrium=H*(Diagonal(dn)*B*y)
    input=H*(dn.*EE)
    (;E,dn,y,direct,equilibrium,input,total=direct+equilibrium+input)
end
function simulate(s,BB,period,burn;dt=.025,amp=AMP)
    duration=(burn+3)*period;steps=ceil(Int,duration/dt);h=duration/steps
    keep=round(Int,burn*period/h);X=zeros(7,steps-keep+1);tt=collect(keep:steps)*h;x=copy(s.n)
    f(x,t)=x.*(s.rr+BB*x+EE*amp*cos(2pi*t/period))
    for z in 0:steps
        z>=keep && (X[:,z-keep+1]=x-s.n)
        z==steps && break
        t=z*h;a=f(x,t);b=f(x+h*a/2,t+h/2);c=f(x+h*b/2,t+h/2);d=f(x+h*c,t+h)
        x+=h*(a+2b+2c+d)/6
        @assert all(isfinite,x) && minimum(x)>0
    end
    tt,X
end
function main()
    s=state(U0);rows=[];validation=[];traces=[];robust=[]
    direct=zeros(2,length(GRID));total=similar(direct)
    @testset "Oscillatory equilibrium and decomposition" begin
        @test U0-AMP>0
        @test norm(s.rr+B*s.n)<1e-12
        for (l,(i,j)) in enumerate(LINKS),(q,period) in enumerate(GRID)
            d=components(s,l,period);delta=FRACTION*B[i,j]
            direct[l,q]=AMP*abs(delta*d.direct[FOCAL])/sqrt(2)
            total[l,q]=AMP*abs(delta*d.total[FOCAL])/sqrt(2)
            push!(rows,(LABELS[l],period,direct[l,q],AMP*abs(delta*d.equilibrium[FOCAL])/sqrt(2),AMP*abs(delta*d.input[FOCAL])/sqrt(2),total[l,q],real(d.direct[FOCAL]*conj(d.total[FOCAL]))/abs2(d.total[FOCAL]),real(d.equilibrium[FOCAL]*conj(d.total[FOCAL]))/abs2(d.total[FOCAL]),real(d.input[FOCAL]*conj(d.total[FOCAL]))/abs2(d.total[FOCAL])))
        end
        for l in 1:2,p in PERIODS
            d=components(s,l,p);h=1e-6
            y(a)=begin st=state(U0,B+a*d.E);(im*2pi/p*I-st.J)\st.b end
            @test norm((y(h)-y(-h))/(2h)-d.total)/norm(d.total)<1e-5
        end
    end
    for period in PERIODS
        altered=[state(U0,B+FRACTION*B[LINKS[l]...]*components(s,l,period).E) for l in 1:2]
        decay=minimum(-maximum(real.(eigvals(st.J))) for st in vcat([s],altered))
        burn=ceil(Int,20/decay/period)
        t,X=simulate(s,B,period,burn)
        for l in 1:2
            i,j=LINKS[l];d=components(s,l,period);delta=FRACTION*B[i,j];BB=B+delta*d.E;sp=altered[l]
            _,Xp=simulate(sp,BB,period,burn)
            exact=AMP*((im*2pi/period*I-sp.J)\sp.b-d.y)
            predicted=AMP*delta*d.total;structural=AMP*delta*d.direct
            phase=transpose(exp.(im*2pi/period*t));D=Xp-X;localtrace=real.(exact.*phase)
            push!(validation,(LABELS[l],period,U0,AMP,FRACTION,s.margin,sp.margin,minimum(sp.n),abs(exact[FOCAL])/sqrt(2),norm(exact)/sqrt(2),norm(predicted-exact)/norm(exact),norm(D-localtrace)/norm(localtrace),abs(predicted[FOCAL]-exact[FOCAL])/abs(exact[FOCAL]),abs(structural[FOCAL]-exact[FOCAL])/abs(exact[FOCAL]),norm(sp.n-s.n)))
            stride=max(1,round(Int,period/.025/250));z=1:stride:length(t)
            push!(traces,(;l,period,time=t[z].-t[1],difference=D[FOCAL,z],localpred=localtrace[FOCAL,z],first=real.(predicted[FOCAL]*exp.(im*2pi/period*t[z]))))
        end
        if period==2.
            tf,Xf=simulate(s,B,period,burn;dt=.0125)
            @test norm(X-Xf[:,1:2:end])/norm(Xf[:,1:2:end])<1e-6
        end
    end
    # Background and finite-change checks are predefined, not optimized for reversal.
    for background in [.0005,.001,.002],fraction in [.0005,.001,.0025],period in PERIODS
        st=state(background)
        for l in 1:2
            i,j=LINKS[l];d=components(st,l,period);sp=state(background,B+fraction*B[i,j]*d.E)
            exact=AMP*((im*2pi/period*I-sp.J)\sp.b-d.y)
            push!(robust,(background,fraction,period,LABELS[l],abs(exact[FOCAL])/sqrt(2),AMP*abs(fraction*B[i,j]*d.direct[FOCAL])/sqrt(2),sp.margin))
        end
    end
    @testset "Nonlinear check at prescribed small forcing" begin
        @test all(v[12]<.02 for v in validation)
    end
    csv("decomposition.csv",["link","period_days","direct_structural_rms","equilibrium_J_rms","input_coupling_rms","total_rms","direct_projection_fraction","equilibrium_projection_fraction","input_projection_fraction"],rows)
    csv("validation.csv",["link","period_days","background","oscillation_amplitude","coefficient_fraction","baseline_full_stability","modified_full_stability","minimum_modified_abundance","exact_pathogen_rms","exact_community_rms","derivative_community_error","nonlinear_local_community_error","derivative_pathogen_error","direct_only_pathogen_error","equilibrium_shift_norm"],validation)
    csv("robustness.csv",["background","coefficient_fraction","period_days","link","exact_pathogen_rms","direct_structural_rms","full_stability_margin"],robust)
    csv("reference_state.csv",["original_species_index","abundance","input_coupling"],[(IX[i],s.n[i],s.b[i]) for i in 1:7])
    for (z,tr) in enumerate(traces);csv("trace_$z.csv",["time_days","nonlinear_centred_difference","exact_local_difference","first_order_difference"],zip(tr.time,tr.difference,tr.localpred,tr.first));end
    set_theme!(Theme(fontsize=15,Axis=(xgridvisible=false,ygridvisible=false)))
    fig=Figure(size=(1450,940),figure_padding=28)
    Label(fig[0,1:2],"Stein: does the frequency contrast come from interaction structure?",fontsize=23,font=:bold)
    Label(fig[1,1:2],"Nonnegative exposure u(t) = 0.001 + 0.0002 cos(2πt/period) • coefficient changes 0.1% • focal group: C. difficile",fontsize=14)
    for (c,V,title) in [(1,total,"A  Full biological-parameter derivative"),(2,direct,"B  Direct interaction contribution at fixed reference")]
        ax=Axis(fig[2,c],title=title,xlabel="Period (days)",ylabel="Predicted RMS change in C. difficile response",xscale=log10,yscale=log10,xticks=[1,2,7,30,180])
        for l in 1:2;lines!(ax,GRID,V[l,:],color=COLORS[l],linewidth=3,label=LABELS[l]);end
        vlines!(ax,PERIODS,color=:gray60,linestyle=:dash);axislegend(ax,position=:lt,labelsize=12,framevisible=false)
    end
    for (c,period) in enumerate(PERIODS)
        ax=Axis(fig[3,c],title="$(period)-day oscillation: nonlinear response differences",xlabel="Time after transient (days)",ylabel="Modified minus original centred C. difficile")
        for tr in filter(tr->tr.period==period,traces)
            lines!(ax,tr.time,tr.difference,color=COLORS[tr.l],linewidth=3,label=LABELS[tr.l])
            lines!(ax,tr.time,tr.localpred,color=:black,linestyle=:dash,linewidth=1)
        end
        axislegend(ax,position=:rt,labelsize=11,framevisible=false)
    end
    Label(fig[4,1:2],"Panel B is a derivative decomposition, not a standalone biological intervention. Full changes also shift equilibrium and input coupling.\nDashed black time curves are exact local predictions. Periods 2 and 30 days were fixed before this test from the previous screening; no crossover optimization.",fontsize=14,color=:gray35)
    rowgap!(fig.layout,22);colgap!(fig.layout,40)
    save(joinpath(OUT,"oscillatory_decomposition.png"),fig,px_per_unit=1.4);save(joinpath(OUT,"oscillatory_decomposition.svg"),fig)
    println("VALIDATION");for row in validation;println(row);end
    println("DECOMPOSITION AT EXAMPLE PERIODS")
    for l in 1:2,period in PERIODS
        d=components(s,l,period)
        println(LABELS[l]," period=",period," component projections=",[real(z[FOCAL]*conj(d.total[FOCAL]))/abs2(d.total[FOCAL]) for z in [d.direct,d.equilibrium,d.input]])
    end
end
main()
