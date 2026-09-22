using LinearAlgebra, Statistics, DelimitedFiles, Test, Printf
using CairoMakie
const ROOT=@__DIR__;const OUT=joinpath(ROOT,"outputs");mkpath(OUT)
const P=joinpath(ROOT,"processed")
const NAMES=readlines(joinpath(P,"species.txt"))
const SHORT=["Barnesiella","Lachno. undefined","Lachno. unclassified","Other","Blautia","Mollicutes","Akkermansia","Coprobacillus","C. difficile","Enterococcus","Enterobacter."]
const PERIODS=10 .^ range(0,log10(180),length=161)
const DURATIONS=[.25,.5,1.,2.,4.,7.,14.,28.]
const HORIZON=365.;const DT=.05;const EXPOSURE=.001
const COL=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9"]
rank(v)=Float64.(invperm(sortperm(v;rev=true)))
rms(X)=sqrt(sum(abs2,X[:,2:end-1])+sum(abs2,X[:,[1,end]])/2)/sqrt(size(X,2)-1)
function writecsv(name,header,rows)
    open(joinpath(OUT,name),"w") do io
        println(io,join(header,","));for row in rows;println(io,join(row,","));end
    end
end
function audit(r,A)
    states=[]
    for mask in 1:2^length(r)-1
        ix=[i for i in eachindex(r) if (mask>>(i-1))&1==1]
        B=A[ix,ix];isfinite(cond(B)) || continue;n=-(B\r[ix]);minimum(n)>1e-9 || continue
        J=Diagonal(n)*B;maximum(real.(eigvals(J))) < -1e-9 || continue
        full=zeros(length(r));full[ix]=n;Jfull=Diagonal(r+A*full)+Diagonal(full)*A
        margin=maximum(real.(eigvals(Jfull)));margin < -1e-9 || continue
        push!(states,(;ix,n,J,margin))
    end
    states
end
function deriv(B,n,e,i,j)
    E=zeros(size(B));E[i,j]=1;dn=-(B\(E*n))
    (;E,dn,dJ=Diagonal(dn)*B+Diagonal(n)*E,db=dn.*e)
end
response(B,n,e,w)=(im*w*I-Diagonal(n)*B)\(n.*e)
# Exact step propagation of linear ODE with rectangular input: no time-integration error.
function pulse(J,b,duration;exposure=EXPOSURE,dt=DT,horizon=HORIZON)
    k=length(b);Z=zeros(k+1,k+1);Z[1:k,1:k]=J;Z[1:k,end]=b*exposure/duration
    on=exp(Z*dt);off=exp(J*dt);steps=round(Int,horizon/dt);switch=round(Int,duration/dt)
    @assert isapprox(switch*dt,duration;atol=1e-10)
    X=zeros(k,steps+1);x=zeros(k+1);x[end]=1
    for t in 1:steps
        if t<=switch;x=on*x;else;x[1:k]=off*x[1:k];end
        X[:,t+1]=x[1:k]
    end
    X
end
function nonlinear(r,B,n,e,duration;exposure=EXPOSURE,dt=DT,horizon=HORIZON)
    steps=round(Int,horizon/dt);switch=round(Int,duration/dt);X=zeros(length(n),steps+1);X[:,1]=n;x=copy(n)
    for k in 1:steps
        # Keep pulse constant throughout each step, including at the discontinuity.
        input=k<=switch ? exposure/duration : 0.
        f(x)=x.*(r+B*x+e*input)
        a=f(x);b=f(x+dt*a/2);c=f(x+dt*b/2);d=f(x+dt*c)
        x+=dt*(a+2b+2c+d)/6
        @assert all(isfinite,x) && minimum(x)>0
        X[:,k+1]=x
    end
    X.-n
end
function main()
    A=readdlm(joinpath(P,"interactions.csv"),',',Float64);r=vec(readdlm(joinpath(P,"growth.csv"),',',Float64));e=vec(readdlm(joinpath(P,"susceptibilities.csv"),',',Float64))
    states=audit(r,A);@assert length(states)==2
    cd=findfirst(==("Clostridium_difficile"),NAMES)
    s=only(filter(s->cd in s.ix,states));ix=s.ix;n=s.n;B=A[ix,ix];rr=r[ix];ee=e[ix];J=s.J;b=n.*ee;focal=findfirst(==(cd),ix);k=length(ix)
    links=[(i,j) for i in 1:k for j in 1:k if i!=j && B[i,j]!=0]
    labels=[SHORT[ix[j]]*" → "*SHORT[ix[i]] for (i,j) in links]
    ds=[deriv(B,n,ee,i,j) for (i,j) in links]
    # Frequency responses are derivatives per unit input amplitude, not worst-case norms.
    comm=zeros(length(links),length(PERIODS));path=similar(comm);absolute=similar(comm)
    for (p,period) in enumerate(PERIODS)
        H=(im*2pi/period*I-J)\Matrix{Float64}(I,k,k);y=H*b
        for (l,(i,j)) in enumerate(links)
            d=ds[l];dy=H*(d.dJ*y+d.db);delta=.01abs(B[i,j])
            comm[l,p]=delta*norm(dy)/sqrt(2);path[l,p]=delta*abs(dy[focal])/sqrt(2);absolute[l,p]=norm(dy)/sqrt(2)
        end
    end
    cr=hcat([rank(comm[:,p]) for p in eachindex(PERIODS)]...);pr=hcat([rank(path[:,p]) for p in eachindex(PERIODS)]...)
    finite_frequency=[]
    for fraction in [.001,.01],(p,period) in enumerate(PERIODS)
        baseline=response(B,n,ee,2pi/period);ec=zeros(length(links));ep=similar(ec)
        for (l,(i,j)) in enumerate(links)
            Bp=B+fraction*B[i,j]*ds[l].E;np=-(Bp\rr)
            delta=response(Bp,np,ee,2pi/period)-baseline
            ec[l]=norm(delta);ep[l]=abs(delta[focal])
        end
        push!(finite_frequency,(fraction,period,cor(rank(ec),cr[:,p]),cor(rank(ep),pr[:,p]),labels[argmax(ec)],labels[argmax(ep)]))
    end
    writecsv("finite_frequency_rank_summary.csv",["fraction","period_days","community_rank_correlation","pathogen_rank_correlation","top_community","top_pathogen"],finite_frequency)
    tests=0
    @testset "Source, equilibrium, derivatives and finite changes" begin
        @test norm(rr+B*n)<1e-12
        @test s.margin<0
        for (l,(i,j)) in enumerate(links)
            d=ds[l];h=1e-5*max(abs(B[i,j]),.01)
            for period in [1.,7.,30.,180.]
                y(a)=begin BB=B+a*d.E;nn=-(BB\rr);response(BB,nn,ee,2pi/period) end
                H=(im*2pi/period*I-J)\Matrix{Float64}(I,k,k);v=H*(d.dJ*(H*b)+d.db)
                @test norm((y(h)-y(-h))/(2h)-v)/norm(v)<1e-5
            end
            for frac in [-.01,.01]
                Ap=copy(A);Ap[ix[i],ix[j]]*=1+frac;np=-(Ap[ix,ix]\rr);full=zeros(11);full[ix]=np
                @test minimum(np)>0 && maximum(real.(eigvals(Diagonal(r+Ap*full)+Diagonal(full)*Ap)))<0
            end
        end
    end
    println("Frequency screen complete. Screening finite-duration pulses.")
    pc=zeros(length(links),length(DURATIONS));pp=similar(pc)
    for (l,(i,j)) in enumerate(links)
        d=ds[l];K=[J zeros(k,k);d.dJ J];bb=vcat(b,d.db)
        for (q,duration) in enumerate(DURATIONS)
            X=pulse(K,bb,duration);D=.01B[i,j]*X[k+1:end,:]
            pc[l,q]=rms(D);pp[l,q]=rms(D[focal:focal,:])
        end
    end
    pcr=hcat([rank(pc[:,q]) for q in eachindex(DURATIONS)]...);ppr=hcat([rank(pp[:,q]) for q in eachindex(DURATIONS)]...)
    finite_rows=[];finite_summary=[]
    for duration in [.5,14.],fraction in [.001,.01]
        base=pulse(J,b,duration);ec=zeros(length(links));ep=similar(ec);q=findfirst(==(duration),DURATIONS)
        for (l,(i,j)) in enumerate(links)
            Bp=B+fraction*B[i,j]*ds[l].E;np=-(Bp\rr)
            D=pulse(Diagonal(np)*Bp,np.*ee,duration)-base
            ec[l]=rms(D);ep[l]=rms(D[focal:focal,:])
        end
        rc=rank(ec);rp=rank(ep)
        for l in eachindex(links);push!(finite_rows,(labels[l],duration,fraction,ec[l],ep[l],rc[l],rp[l]));end
        push!(finite_summary,(duration,fraction,cor(rc,pcr[:,q]),cor(rp,ppr[:,q]),labels[argmax(ec)],labels[argmax(ep)]))
    end
    writecsv("finite_pulse_rank_check.csv",["link","duration_days","fraction","exact_local_community_rms","exact_local_pathogen_rms","community_rank","pathogen_rank"],finite_rows)
    writecsv("finite_pulse_rank_summary.csv",["duration_days","fraction","community_rank_correlation_with_derivative","pathogen_rank_correlation_with_derivative","top_community","top_pathogen"],finite_summary)
    # Fixed example durations; select top pathogen link for each, deduplicated. No reversal required.
    q1=findfirst(==(.5),DURATIONS);q2=findfirst(==(14.),DURATIONS)
    chosen=unique([argmax(pp[:,q1]),argmax(pp[:,q2])]);validation=[];traces=[]
    for duration in [.5,14.]
        base=pulse(J,b,duration);baseNL=nonlinear(rr,B,n,ee,duration)
        for l in chosen,frac in [.001,.005,.01,.05,.10]
            i,j=links[l];d=ds[l];delta=frac*B[i,j];Bp=B+delta*d.E;np=-(Bp\rr)
            Ap=copy(A);Ap[ix[i],ix[j]]+=delta;full=zeros(11);full[ix]=np
            margin=maximum(real.(eigvals(Diagonal(r+Ap*full)+Diagonal(full)*Ap)))
            if minimum(np)<=0 || margin>=0
                push!(validation,(labels[l],duration,frac,"infeasible_or_unstable",NaN,NaN,NaN,NaN,NaN,norm(np-n),margin));continue
            end
            exact=pulse(Diagonal(np)*Bp,np.*ee,duration)-base
            total=delta*pulse([J zeros(k,k);d.dJ J],vcat(b,d.db),duration)[k+1:end,:]
            NL=nonlinear(rr,Bp,np,ee,duration)-baseNL
            # Separately show the consequences of incorrectly freezing equilibrium/input.
            fixed=delta*pulse([J zeros(k,k);Diagonal(n)*d.E J],vcat(b,zeros(k)),duration)[k+1:end,:]
            push!(validation,(labels[l],duration,frac,"ok",rms(exact),rms(NL),norm(total-exact)/norm(exact),norm(NL-exact)/norm(exact),norm(fixed-exact)/norm(exact),norm(np-n),margin))
            if frac==.01 && l==argmax(pp[:,findfirst(==(duration),DURATIONS)])
                push!(traces,(;duration,link=labels[l],t=collect(0:DT:HORIZON),base=baseNL[focal,:],modified=(NL+baseNL)[focal,:],difference=NL[focal,:],pred=total[focal,:],exact=exact[focal,:]))
            end
        end
    end
    @testset "Pulse and nonlinear numerical checks" begin
        half=nonlinear(rr,B,n,ee,.5;dt=DT/2)
        coarse=nonlinear(rr,B,n,ee,.5)
        @test norm(coarse-half[:,1:2:end])/norm(half[:,1:2:end])<1e-6
        fine=pulse(J,b,.5;dt=DT/2)
        @test abs(rms(fine)/rms(pulse(J,b,.5))-1)<1e-3
        # Spectral zero frequency agrees with integral of the impulse response.
        @test norm(response(B,n,ee,0.)+J\b)<1e-10
        @test all(v[4]!="ok" || v[8]<.02 for v in validation)
    end
    writecsv("equilibria.csv",["state","species","abundance","full_max_real_eigenvalue_per_day"],[(a,SHORT[j],st.n[z],st.margin) for (a,st) in enumerate(states) for (z,j) in enumerate(st.ix)])
    writecsv("frequency_sensitivity.csv",["source","recipient","period_days","community_rms_per_unit_input_1pct","pathogen_rms_per_unit_input_1pct","community_absolute_coefficient_sensitivity","community_rank","pathogen_rank"],[(NAMES[ix[j]],NAMES[ix[i]],period,comm[l,p],path[l,p],absolute[l,p],cr[l,p],pr[l,p]) for (l,(i,j)) in enumerate(links) for (p,period) in enumerate(PERIODS)])
    writecsv("pulse_sensitivity.csv",["source","recipient","duration_days","integrated_scaled_exposure_days","horizon_days","community_rms_1pct","pathogen_rms_1pct","community_rank","pathogen_rank"],[(NAMES[ix[j]],NAMES[ix[i]],du,EXPOSURE,HORIZON,pc[l,q],pp[l,q],pcr[l,q],ppr[l,q]) for (l,(i,j)) in enumerate(links) for (q,du) in enumerate(DURATIONS)])
    writecsv("nonlinear_checks.csv",["link","duration_days","fraction","status","exact_local_community_rms","nonlinear_community_rms","first_order_relative_error","nonlinear_local_relative_error","frozen_equilibrium_input_relative_error","equilibrium_shift_norm","full_max_real_eigenvalue"],validation)
    for (z,tr) in enumerate(traces)
        writecsv("pulse_trace_$z.csv",["time_days","original_centered_pathogen","modified_centered_pathogen","nonlinear_difference","first_order_difference","exact_local_difference"],zip(tr.t,tr.base,tr.modified,tr.difference,tr.pred,tr.exact))
    end
    writedlm(joinpath(OUT,"resident_jacobian.csv"),J,',')
    boundaries=[]
    for l in chosen
        i,j=links[l]
        feasible(frac)=begin
            Ap=copy(A);Ap[ix[i],ix[j]]*=1+frac;np=-(Ap[ix,ix]\rr);full=zeros(11);full[ix]=np
            minimum(np)>0 && maximum(real.(eigvals(Diagonal(r+Ap*full)+Diagonal(full)*Ap)))<0
        end
        lo=0.;hi=.1
        if !feasible(hi)
            for z in 1:50;mid=(lo+hi)/2;if feasible(mid);lo=mid;else;hi=mid;end;end
        end
        push!(boundaries,(labels[l],lo,hi))
    end
    writecsv("selected_link_boundary.csv",["link","last_feasible_stable_fraction","first_rejected_fraction"],boundaries)
    plotfigures(s,ix,n,links,labels,comm,path,cr,pr,pc,pp,pcr,ppr,traces,validation)
    strengths=rank([abs(B[i,j]) for (i,j) in links]);good=filter(v->v[4]=="ok",validation)
    open(joinpath(OUT,"summary.txt"),"w") do io
        println(io,"Julia ",VERSION,"; CairoMakie ",Base.pkgversion(CairoMakie))
        println(io,"Pathogen-present residents: ",join(SHORT[ix],", "))
        println(io,"Full Jacobian margin (per day): ",s.margin)
        println(io,"Frequency endpoint rank correlation community: ",cor(cr[:,1],cr[:,end]))
        println(io,"Frequency endpoint rank correlation pathogen: ",cor(pr[:,1],pr[:,end]))
        println(io,"Max frequency rank range community: ",maximum(maximum(cr;dims=2)-minimum(cr;dims=2)))
        println(io,"Max frequency rank range pathogen: ",maximum(maximum(pr;dims=2)-minimum(pr;dims=2)))
        println(io,"Frequency winners community: ",join(unique([labels[argmax(comm[:,q])] for q in eachindex(PERIODS)]),"; "))
        println(io,"Frequency winners pathogen: ",join(unique([labels[argmax(path[:,q])] for q in eachindex(PERIODS)]),"; "))
        for (q,du) in enumerate(DURATIONS)
            println(io,"Pulse ",du," d; top community: ",labels[argmax(pc[:,q])],"; top pathogen: ",labels[argmax(pp[:,q])])
        end
        println(io,"Pulse endpoint rank correlation community: ",cor(pcr[:,1],pcr[:,end]))
        println(io,"Pulse endpoint rank correlation pathogen: ",cor(ppr[:,1],ppr[:,end]))
        println(io,"Link magnitude vs pathogen sensitivity rank correlation at 7 d: ",cor(strengths,ppr[:,findfirst(==(7.),DURATIONS)]))
        println(io,"Max nonlinear/local error: ",maximum(v[8] for v in good))
        for frac in [.001,.005,.01,.05,.10]
            subset=filter(v->v[3]==frac,good)
            println(io,"Max derivative error at fraction ",frac,": ",isempty(subset) ? "no feasible/stable tested case" : maximum(v[7] for v in subset))
        end
        println(io,"Max frozen equilibrium/input error: ",maximum(v[9] for v in good))
        println(io,"Validation rows not feasible/stable: ",length(validation)-length(good))
        for boundary in boundaries;println(io,"Selected link boundary: ",boundary);end
    end
    print(read(joinpath(OUT,"summary.txt"),String))
end

function plotfigures(s,ix,n,links,labels,comm,path,cr,pr,pc,pp,pcr,ppr,traces,validation)
    set_theme!(Theme(fontsize=15,Axis=(xgridvisible=false,ygridvisible=false)))
    fig=Figure(size=(1500,1000),figure_padding=28)
    Label(fig[0,1:2],"Stein screening: which interactions shape the antibiotic response?",fontsize=24,font=:bold)
    Label(fig[1,1:2],"Published fitted input direction • stable seven-group equilibrium • first-order sensitivities scaled to 1% coefficient changes",fontsize=15)
    selections=[unique(vcat(unique([argmax(S[:,p]) for p in eachindex(PERIODS)]),sortperm(vec(maximum(S;dims=2));rev=true)))[1:6] for S in [comm,path]]
    palette=vcat(COL,["#332288","#882255","#666666"])
    linkcolors=Dict(l=>palette[c] for (c,l) in enumerate(unique(vcat(selections...))))
    for (col,S,title) in [(1,comm,"A  Whole-community sensitivity"),(2,path,"B  C. difficile sensitivity")]
        ax=Axis(fig[2,col],title=title,xlabel="Forcing period (days)",ylabel="RMS response change / input amplitude",xscale=log10,yscale=log10,xticks=[1,7,30,180])
        top=selections[col]
        for l in top;lines!(ax,PERIODS,S[l,:],color=linkcolors[l],linewidth=2.5,label=labels[l]);end
        axislegend(ax,position=:lt,labelsize=11,framevisible=false)
    end
    select=sortperm(vec(maximum(pr;dims=2)-minimum(pr;dims=2));rev=true)[1:12]
    grid=GridLayout();fig[3,1]=grid
    ax=Axis(grid[1,1],title="C  Largest pathogen rank changes",xlabel="Forcing period (days)",xticks=(log10.([1,7,30,180]),string.([1,7,30,180])),yticks=(1:12,labels[select]),yticklabelsize=11)
    hm=heatmap!(ax,log10.(PERIODS),1:12,pr[select,:]',colorrange=(1,length(links)),colormap=:viridis)
    Colorbar(grid[1,2],hm,width=12,label="Rank (1 = highest)",labelsize=12)
    ax=Axis(fig[3,2],title="D  Pulse duration distinguishes the two outputs",xlabel="Pulse duration (days)",ylabel="Effect ratio (Akkermansia link / Other link)",xscale=log10,xticks=[.25,1,7,28])
    a=argmax(pp[:,1]);bb=argmax(pc[:,1])
    scatterlines!(ax,DURATIONS,pc[a,:]./pc[bb,:],label="Whole community",linewidth=2.5,color=COL[1])
    scatterlines!(ax,DURATIONS,pp[a,:]./pp[bb,:],label="C. difficile",linewidth=2.5,color=COL[2])
    hlines!(ax,[1.],color=:gray50,linestyle=:dash)
    axislegend(ax,position=(.05,.5),labelsize=12,framevisible=false)
    Label(fig[4,1:2],"Panel D compares C. difficile → Akkermansia with C. difficile → Other. Its near-tie reversal disappears in the finite-change check.\nPulses share exposure and a 365-day RMS window. Panel C selects largest rank changes. Frequency curves are model diagnostics, not measured oscillations.",fontsize=14,color=:gray35)
    colgap!(fig.layout,40);rowgap!(fig.layout,22)
    save(joinpath(OUT,"01_sensitivity_screen.png"),fig,px_per_unit=1.4);save(joinpath(OUT,"01_sensitivity_screen.svg"),fig)
    fig=Figure(size=(1450,930),figure_padding=28)
    Label(fig[0,1:2],"Small antibiotic pulses: predictions within the fitted model",fontsize=24,font=:bold)
    Label(fig[1,1:2],"Both time panels show C. difficile • nonlinear simulations centred on each model's own equilibrium • 1% coefficient change",fontsize=15)
    for (z,tr) in enumerate(traces)
        ax=Axis(fig[2,z],title="$(tr.duration)-day pulse; $(tr.link)",xlabel="Time (days)",ylabel="C. difficile departure from own equilibrium")
        lines!(ax,tr.t,tr.base,color=:gray45,label="Original structure",linewidth=2.5)
        lines!(ax,tr.t,tr.modified,color=COL[z],label="Modified structure",linewidth=2.5)
        xlims!(ax,0,180);axislegend(ax,position=:rb,framevisible=false,labelsize=12)
    end
    ax=Axis(fig[3,1],title="C  Whole-community departure, separately",xlabel="Interaction change (%)",ylabel="RMS difference over 365 days")
    good=filter(v->v[4]=="ok",validation)
    for du in [.5,14.],link in unique([v[1] for v in good])
        rows=filter(v->v[1]==link && v[2]==du,good)
        scatterlines!(ax,100 .* [v[3] for v in rows],[v[6] for v in rows],label="$(du) d: "*link)
    end
    axislegend(ax,position=:lt,labelsize=10,framevisible=false)
    ax=Axis(fig[3,2],title="D  Approximation errors",xlabel="Interaction change (%)",ylabel="Whole-community waveform error (%)",yscale=log10)
    for (c,du) in enumerate([.5,14.])
        rows=filter(v->v[2]==du && v[1]==first(tr for tr in traces if tr.duration==du).link,good)
        scatterlines!(ax,100 .* [v[3] for v in rows],100 .* [v[7] for v in rows],color=COL[c],label="$(du) d: first derivative")
        scatterlines!(ax,100 .* [v[3] for v in rows],100 .* [v[8] for v in rows],color=COL[c],linestyle=:dash,label="$(du) d: nonlinear vs exact local")
    end
    axislegend(ax,position=:rb,labelsize=11,framevisible=false)
    Label(fig[4,1:2],"Total scaled exposure = $(EXPOSURE) days for both pulses; these are hypothetical weak disturbances, not clinical doses.\nThe selected link fails feasibility/stability at 5% and 10%; those cases are recorded, not plotted as valid local responses.",fontsize=14,color=:gray35)
    colgap!(fig.layout,35);rowgap!(fig.layout,22)
    save(joinpath(OUT,"02_pulse_checks.png"),fig,px_per_unit=1.4);save(joinpath(OUT,"02_pulse_checks.svg"),fig)
end
main()
