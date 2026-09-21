using LinearAlgebra, Statistics, DelimitedFiles, Printf, Test
using CairoMakie

const ROOT=@__DIR__
const INPUT=joinpath(ROOT,"processed")
const OUT=joinpath(ROOT,"outputs")
const SPECIES=readlines(joinpath(INPUT,"species.txt"))
const TAGS=["T1","T2","T3","T4"]
const FORCE=0.001 # hypothetical common per-capita growth-rate oscillation, h^-1
const PERIODS=10 .^ range(log10(6),log10(336),length=121)
const COLORS=["#0072B2","#D55E00","#009E73","#CC79A7"]
loadmodel(tag)=(vec(readdlm(joinpath(INPUT,tag*"_growth.csv"),',',Float64)),readdlm(joinpath(INPUT,tag*"_interactions.csv"),',',Float64))
equilibrium(u,B)=-(B\u)
jacobian(u,B,n)=Diagonal(u+B*n)+Diagonal(n)*B
transfer(J,w)=(im*w*I-J)\Matrix{Float64}(I,size(J,1),size(J,1))
rankvalues(v)=Float64[findfirst(==(i),sortperm(v;rev=true)) for i in eachindex(v)]
label(names,link)=names[link[2]]*" → "*names[link[1]]

function resident(u,B)
    # Enumerate all supports; require feasibility, internal stability AND exclusion of invaders.
    candidates=[]; stablecounts=zeros(Int,12)
    for mask in 1:2^12-1
        ix=[i for i in 1:12 if (mask>>(i-1))&1==1]; BB=B[ix,ix]
        abs(det(BB))<1e-15 && continue
        n=equilibrium(u[ix],BB); minimum(n)>1e-9 || continue
        margin=maximum(real.(eigvals(Diagonal(n)*BB))); margin < -1e-8 || continue
        stablecounts[length(ix)]+=1
        absent=setdiff(1:12,ix);inv=u+B[:,ix]*n
        (isempty(absent) || maximum(inv[absent]) < -1e-8) || continue
        push!(candidates,(;ix,n,margin,invasion=isempty(absent) ? NaN : maximum(inv[absent])))
    end
    @assert length(candidates)==1 "Multiple or absent resident supports require manual interpretation."
    only(candidates),stablecounts
end

function rk4(u,B,x0;duration,dt=.05,period=Inf,q=zeros(length(u)),amp=0.,dilutions=false)
    steps=ceil(Int,duration/dt); dt=duration/steps;t=collect(range(0,duration,length=steps+1))
    X=zeros(length(x0),steps+1); X[:,1]=x0;x=copy(x0)
    f(x,s)=x.*(u+B*x+amp*q*cos(2pi*s/period))
    for k in 1:steps
        s=t[k];a=f(x,s);b=f(x+dt*a/2,s+dt/2);c=f(x+dt*b/2,s+dt/2);d=f(x+dt*c,s+dt)
        x+=dt*(a+2b+2c+d)/6
        if dilutions && (abs(t[k+1]-24)<1e-8 || abs(t[k+1]-48)<1e-8);x*=.05;end
        @assert all(isfinite,x) && minimum(x)>=-1e-10
        X[:,k+1]=x
    end
    t,X
end

function derivative(u,B,n,link,q)
    i,j=link;E=zeros(size(B));E[i,j]=1
    dn=-(B\(E*n))
    dJ=Diagonal(dn)*B+Diagonal(n)*E
    db=dn.*q
    (;E,dn,dJ,db)
end

function sensitivity(u,B,n,names;q=ones(length(n)))
    J=Diagonal(n)*B;b=n.*q
    links=[(i,j) for i in eachindex(n) for j in eachindex(n) if i!=j && abs(B[i,j])>1e-10]
    S=zeros(length(links),length(PERIODS)); ABS=similar(S);BIOM=similar(S)
    derivs=[derivative(u,B,n,l,q) for l in links]
    for (k,p) in enumerate(PERIODS)
        H=transfer(J,2pi/p);y=H*b
        for (l,(i,j)) in enumerate(links)
            d=derivs[l];dy=H*(d.dJ*y+d.db)
            ABS[l,k]=FORCE*norm(dy)/sqrt(2)
            S[l,k]=.01abs(B[i,j])*ABS[l,k]
            BIOM[l,k]=FORCE*.01abs(B[i,j])*abs(sum(dy))/sqrt(2)
        end
    end
    ranks=hcat([rankvalues(S[:,k]) for k in eachindex(PERIODS)]...)
    (;links,S,ABS,BIOM,ranks,derivs,J,q,b,names)
end

function forced_sim(u,B,n,period,q;dt=.05)
    decay=-maximum(real.(eigvals(Diagonal(n)*B)))
    burn=ceil(Int,25/decay/period);duration=(burn+4)*period
    t,X=rk4(u,B,n;duration,dt,period,q,amp=FORCE)
    keep=findall(t .>= burn*period-1e-8)
    t[keep],X[:,keep].-n
end
function rmsmatrix(D,t)
    v=vec(sum(abs2,D;dims=1));sqrt(sum(diff(t).*(v[1:end-1]+v[2:end])/2)/(t[end]-t[1]))
end

function validate(u,B,n,s)
    # Predefined periods; select top-ranked links at short/long ends, never by a desired reversal.
    chosen=unique([argmax(s.S[:,1]),argmax(s.S[:,end])]);rows=[];traces=[]
    for period in [24.,168.]
        H=transfer(s.J,2pi/period); y=H*s.b
        for l in chosen, fraction in [.01,.05,.10]
            i,j=s.links[l];delta=fraction*B[i,j];Bp=B+delta*s.derivs[l].E
            np=equilibrium(u,Bp)
            if minimum(np)<=0 || maximum(real.(eigvals(Diagonal(np)*Bp)))>=0
                push!(rows,(period=period,link=label(s.names,s.links[l]),fraction=fraction,status="infeasible_or_unstable",exact=NaN,predicted=NaN,nonlinear=NaN,approx_error=NaN,nonlinear_error=NaN,mean_shift=norm(np-n)))
                continue
            end
            yp=transfer(Diagonal(np)*Bp,2pi/period)*(np.*s.q)
            exact=FORCE*(yp-y)
            d=s.derivs[l];prediction=FORCE*delta*H*(d.dJ*y+d.db)
            # Use identical absolute time grid and sufficient burn-in for both models.
            decay=min(-maximum(real.(eigvals(s.J))),-maximum(real.(eigvals(Diagonal(np)*Bp))))
            burn=ceil(Int,25/decay/period);duration=(burn+4)*period
            t,X=rk4(u,B,n;duration,period,q=s.q,amp=FORCE)
            _,Xp=rk4(u,Bp,np;duration,period,q=s.q,amp=FORCE)
            keep=findall(t .>= burn*period-1e-8);tt=t[keep]
            D=(Xp[:,keep].-np)-(X[:,keep].-n)
            predtrace=real.(exact.*transpose(exp.(im*2pi/period*tt)))
            nonlinear_error=norm(D-predtrace)/norm(predtrace)
            push!(rows,(period=period,link=label(s.names,s.links[l]),fraction=fraction,status="ok",exact=norm(exact)/sqrt(2),predicted=norm(prediction)/sqrt(2),nonlinear=rmsmatrix(D,tt),approx_error=norm(prediction-exact)/norm(exact),nonlinear_error=nonlinear_error,mean_shift=norm(np-n)))
            if fraction==.05
                focal=argmax(abs.(exact))
                # Downsample only exported/displayed traces; all checks use full resolution.
                ix=1:10:length(tt)
                push!(traces,(period=period,link=label(s.names,s.links[l]),species=s.names[focal],time=tt[ix].-tt[1],observed=D[focal,ix],predicted=predtrace[focal,ix]))
            end
        end
    end
    rows,traces
end

function exportall(models,residents,counts,sens,validation,traces,protocol)
    mkpath(OUT)
    open(joinpath(OUT,"equilibrium_audit.csv"),"w") do io
        println(io,"model,full_coexistence_feasible,minimum_full_equilibrium,resident_species,internal_max_real_eigenvalue,max_absent_invasion_rate")
        for tag in TAGS
            u,B=models[tag];n=equilibrium(u,B);r=residents[tag]
            println(io,join((tag,minimum(n)>0,minimum(n),join(SPECIES[r.ix],";"),r.margin,r.invasion),","))
        end
    end
    for tag in ["T3","T4"]
        s=sens[tag];r=residents[tag]
        open(joinpath(OUT,tag*"_link_sensitivity.csv"),"w") do io
            println(io,"source,recipient,period_hours,predicted_rms_for_one_percent_change,equal_absolute_sensitivity,summed_OD_proxy_sensitivity,rank")
            for (l,(i,j)) in enumerate(s.links),(k,p) in enumerate(PERIODS)
                println(io,join((s.names[j],s.names[i],p,s.S[l,k],s.ABS[l,k],s.BIOM[l,k],s.ranks[l,k]),","))
            end
        end
        u,B=models[tag];J=Diagonal(r.n)*B[r.ix,r.ix];tau=-1 ./diag(J);A=Diagonal(tau)*J
        writedlm(joinpath(OUT,tag*"_resident_jacobian.csv"),J,',')
        writedlm(joinpath(OUT,tag*"_resident_normalized_A.csv"),A,',')
        writedlm(joinpath(OUT,tag*"_resident_timescales_hours.csv"),tau,',')
        open(joinpath(OUT,tag*"_resident_equilibrium.csv"),"w") do io
            println(io,"species,OD_proxy");for (name,x) in zip(s.names,r.n);println(io,name,",",x);end
        end
    end
    open(joinpath(OUT,"nonlinear_validation.csv"),"w") do io
        println(io,join(string.(keys(first(validation))),","))
        for row in validation;println(io,join(values(row),","));end
    end
    for (k,tr) in enumerate(traces)
        writedlm(joinpath(OUT,"validation_trace_$(k).csv"),hcat(tr.time,tr.observed,tr.predicted),',')
    end
    for (tag,(t,X)) in protocol
        writedlm(joinpath(OUT,tag*"_as_shipped_dilution.csv"),hcat(t,X'),',')
    end
end

function figures(models,residents,sens,validation,traces,protocol,observed)
    set_theme!(Theme(fontsize=15,Axis=(xgridvisible=false,ygridvisible=false)))
    fig=Figure(size=(1400,930),figure_padding=28)
    Label(fig[0,1:2],"Venturelli feasibility: the experimental protocol and equilibrium are different regimes",fontsize=23,font=:bold)
    ax=Axis(fig[1,1],title="A  Formal 12-species equilibria",xticks=(1:12,SPECIES),xticklabelrotation=pi/4,ylabel="Equilibrium OD proxy (negative is infeasible)")
    for (k,tag) in enumerate(["T3","T4"]);u,B=models[tag];scatterlines!(ax,1:12,equilibrium(u,B),color=COLORS[k],label=tag);end
    hlines!(ax,[0],color=:gray40,linestyle=:dash);axislegend(ax,position=:lb,framevisible=false)
    ax=Axis(fig[1,2],title="B  Same six residents in T3 and T4",xticks=(1:6,SPECIES[residents["T3"].ix]),ylabel="Positive equilibrium OD proxy")
    for (k,tag) in enumerate(["T3","T4"]);barplot!(ax,(1:6).+(k==1 ? -.18 : .18),residents[tag].n,width=.32,color=COLORS[k],label=tag);end
    axislegend(ax,position=:rt,framevisible=false)
    ax=Axis(fig[2,1],title="C  Supplied SBML protocol: 95% removals",xlabel="Time (hours)",ylabel="Total OD proxy")
    for (k,tag) in enumerate(["T3","T4"]);t,X=protocol[tag];lines!(ax,t,vec(sum(X;dims=1)),color=COLORS[k],label=tag);end
    vlines!(ax,[24,48],color=:gray50,linestyle=:dash);axislegend(ax,position=:rt,framevisible=false)
    ax=Axis(fig[2,2],title="D  Supplied full-community observations",xlabel="Sampling index (exact timestamps absent in workbook)",ylabel="Relative abundance")
    for (i,name) in enumerate(SPECIES)
        lines!(ax,0:6,observed[:,i],linewidth=name in ["BU","BO","EL"] ? 3 : 1.3,label=name)
    end
    axislegend(ax,position=:rt,nbanks=3,labelsize=10,framevisible=false)
    Label(fig[3,1:2],"Protocol simulation uses the SBML initial values (0.01 per species), not reconstructed experimental initial conditions. It is not a new goodness-of-fit test.\nThe resolvent analysis below concerns the no-dilution resident equilibrium, not the observed serial-transfer trajectory.",fontsize=14,color=:gray35)
    save(joinpath(OUT,"01_feasibility.png"),fig,px_per_unit=1.5);save(joinpath(OUT,"01_feasibility.svg"),fig)

    s=sens["T3"];s4=sens["T4"];nlinks=length(s.links)
    selected=sortperm(vec(maximum(s.ranks;dims=2)-minimum(s.ranks;dims=2));rev=true)[1:min(12,nlinks)]
    fig=Figure(size=(1450,960),figure_padding=28)
    Label(fig[0,1:2],"Which fitted interactions matter for a hypothetical growth-rate disturbance?",fontsize=24,font=:bold)
    Label(fig[1,1:2],"T3 six-resident equilibrium • common growth-rate forcing 0.001 h⁻¹ • equal 1% coefficient changes • equilibrium and input coupling allowed to change",fontsize=14)
    grid=GridLayout();fig[2,1]=grid
    ax=Axis(grid[1,1],title="A  Largest rank changes across periods",xlabel="Forcing period (hours)",yticks=(1:length(selected),[label(s.names,s.links[l]) for l in selected]),xticks=(log10.([6,24,72,168,336]),string.([6,24,72,168,336])),yticklabelsize=12)
    hm=heatmap!(ax,log10.(PERIODS),1:length(selected),s.ranks[selected,:]',colorrange=(1,nlinks),colormap=:viridis)
    Colorbar(grid[1,2],hm,label="Rank (1 = highest)",width=12,labelsize=12)
    colgap!(grid,10)
    ax=Axis(fig[2,2],title="B  Profiles of the largest overall effects",xlabel="Forcing period (hours)",ylabel="Predicted RMS change in fluctuations (OD proxy)",xscale=log10,yscale=log10)
    top=sortperm(vec(maximum(s.S;dims=2));rev=true)[1:min(6,nlinks)]
    for l in top;lines!(ax,PERIODS,s.S[l,:],linewidth=2.5,label=label(s.names,s.links[l]));end
    axislegend(ax,position=:lt,labelsize=11,framevisible=false)
    ax=Axis(fig[3,1],title="C  Sensitivity ranking is not simply link magnitude",xlabel="Rank by |interaction coefficient|",ylabel="Rank by response effect (1 = highest)")
    u,B=models["T3"];BB=B[residents["T3"].ix,residents["T3"].ix];strength=rankvalues([abs(BB[l...]) for l in s.links])
    scatter!(ax,strength,s.ranks[:,1],color=COLORS[1],label="6 h");scatter!(ax,strength,s.ranks[:,end],color=COLORS[2],marker=:diamond,label="336 h")
    lines!(ax,[1,nlinks],[1,nlinks],color=:gray65,linestyle=:dash);axislegend(ax,position=:rt,framevisible=false)
    ax=Axis(fig[3,2],title="D  Dependence on the fitted parameter set",xlabel="T3 sensitivity rank",ylabel="T4 sensitivity rank")
    common=intersect(s.links,s4.links)
    for (k,labeltext,col) in [(1,"6 h",COLORS[1]),(length(PERIODS),"336 h",COLORS[2])]
        x=[s.ranks[findfirst(==(l),s.links),k] for l in common];y=[s4.ranks[findfirst(==(l),s4.links),k] for l in common]
        scatter!(ax,x,y,color=col,label=labeltext)
    end
    axislegend(ax,position=:rt,framevisible=false)
    Label(fig[4,1:2],"Ranks compare nonzero resident-to-resident coefficients. T3 and T4 are different fits, not posterior samples.\nPanel A selects rank-changing links descriptively; all links and both absolute/fractional sensitivities are exported. This is not empirical validation.",fontsize=14,color=:gray35)
    rowgap!(fig.layout,20);colgap!(fig.layout,32)
    save(joinpath(OUT,"02_interaction_priorities.png"),fig,px_per_unit=1.5);save(joinpath(OUT,"02_interaction_priorities.svg"),fig)

    fig=Figure(size=(1400,900),figure_padding=28)
    Label(fig[0,1:2],"Checking the approximation against the fitted nonlinear model",fontsize=24,font=:bold)
    Label(fig[1,1:2],"Each model fluctuates around its own equilibrium. The equilibrium shift is recorded separately. These are simulations, not new observations.",fontsize=14)
    ax=Axis(fig[2,1],title="A  Exact local response versus nonlinear simulations",xlabel="Local linear prediction of RMS departure",ylabel="Nonlinear simulation RMS departure",xscale=log10,yscale=log10)
    good=filter(r->r.status=="ok",validation);xx=[r.exact for r in good];yy=[r.nonlinear for r in good]
    scatter!(ax,xx,yy,color=[r.period==24 ? COLORS[1] : COLORS[2] for r in good],markersize=11)
    lo=min(minimum(xx),minimum(yy))*.8;hi=max(maximum(xx),maximum(yy))*1.2;lines!(ax,[lo,hi],[lo,hi],color=:gray40,linestyle=:dash)
    ax=Axis(fig[2,2],title="B  Derivative error as coefficient changes increase",xlabel="Coefficient change (%)",ylabel="Relative complex-response error (%)")
    for link in unique([r.link for r in good]),p in [24.,168.]
        rows=filter(r->r.link==link && r.period==p,good)
        scatterlines!(ax,100 .* [r.fraction for r in rows],100 .* [r.approx_error for r in rows],label=link*" / "*string(Int(p))*" h")
    end
    axislegend(ax,position=:lt,labelsize=10,framevisible=false)
    for (q,tr) in enumerate(traces[1:min(2,length(traces))])
        ax=Axis(fig[3,q],title="$(tr.link), $(Int(tr.period)) h; displayed species $(tr.species)",xlabel="Hours after transient",ylabel="Change in centred fluctuations")
        lines!(ax,tr.time,tr.observed,color=COLORS[q],linewidth=3,label="Nonlinear gLV simulation")
        lines!(ax,tr.time,tr.predicted,color=:black,linestyle=:dash,linewidth=2,label="Exact local linear response")
        q==1 && axislegend(ax,position=:rt,labelsize=11,framevisible=false)
    end
    Label(fig[4,1:2],"Forcing is a stated scenario, not measured environmental input. Validation checks local approximation within the published model.\nCoefficient changes modify equilibrium abundances, the Jacobian, and forcing amplitudes; the analysis does not hold T artificially fixed.",fontsize=14,color=:gray35)
    save(joinpath(OUT,"03_nonlinear_check.png"),fig,px_per_unit=1.5);save(joinpath(OUT,"03_nonlinear_check.svg"),fig)
end

function main()
    mkpath(OUT);models=Dict(tag=>loadmodel(tag) for tag in TAGS);residents=Dict();counts=Dict()
    for tag in TAGS;residents[tag],counts[tag]=resident(models[tag]...);println(tag," residents ",SPECIES[residents[tag].ix]);end
    @assert residents["T3"].ix==residents["T4"].ix
    sens=Dict()
    for tag in ["T3","T4"]
        u,B=models[tag];r=residents[tag];sens[tag]=sensitivity(u[r.ix],B[r.ix,r.ix],r.n,SPECIES[r.ix])
    end
    r=residents["T3"];u,B=models["T3"];u=u[r.ix];B=B[r.ix,r.ix];s=sens["T3"]
    @testset "Model and total parameter derivative" begin
        @test norm(u+B*r.n)<1e-12
        for (l,(i,j)) in enumerate(s.links)
            h=1e-5*max(abs(B[i,j]),.01);E=s.derivs[l].E
            for period in [6.,24.,168.,336.]
                function y(e)
                    BB=B+e*E;nn=equilibrium(u,BB);transfer(Diagonal(nn)*BB,2pi/period)*(nn.*s.q)
                end
                H=transfer(s.J,2pi/period);expected=H*(s.derivs[l].dJ*(H*s.b)+s.derivs[l].db)
                @test norm((y(h)-y(-h))/(2h)-expected)/norm(expected)<1e-6
            end
        end
        # Jacobian including extinct species must be stable, not just resident block.
        for tag in TAGS
            uu,BB=models[tag];rr=residents[tag];nn=zeros(12);nn[rr.ix]=rr.n
            @test maximum(real.(eigvals(jacobian(uu,BB,nn))))<0
        end
    end
    validation,traces=validate(u,B,r.n,s)
    @testset "Nonlinear response and integration refinement" begin
        @test all(row.status=="ok" && row.nonlinear_error<0.01 for row in validation)
        t,x=forced_sim(u,B,r.n,24.,s.q;dt=.05)
        tf,xf=forced_sim(u,B,r.n,24.,s.q;dt=.025)
        @test length(tf)==2length(t)-1
        @test norm(x-xf[:,1:2:end])/norm(xf[:,1:2:end])<1e-6
    end
    protocol=Dict(tag=>rk4(models[tag]...,vec(readdlm(joinpath(INPUT,tag*"_initial.csv"),',',Float64));duration=72.,dilutions=true) for tag in ["T3","T4"])
    observed=readdlm(joinpath(INPUT,"observed_full.csv"),',',Float64)
    exportall(models,residents,counts,sens,validation,traces,protocol)
    figures(models,residents,sens,validation,traces,protocol,observed)
    open(joinpath(OUT,"numerical_summary.txt"),"w") do io
        println(io,"Julia ",VERSION,"; CairoMakie ",Base.pkgversion(CairoMakie))
        println(io,"Residents T3/T4: ",SPECIES[r.ix]);println(io,"T3 nonzero resident links: ",length(s.links))
        println(io,"T3 slow-fast rank correlation: ",cor(s.ranks[:,1],s.ranks[:,end]))
        println(io,"Maximum rank change: ",maximum(abs.(s.ranks[:,1]-s.ranks[:,end])))
        println(io,"Top at 6 h: ",label(s.names,s.links[argmax(s.S[:,1])]))
        println(io,"Top at 336 h: ",label(s.names,s.links[argmax(s.S[:,end])]))
        good=filter(x->x.status=="ok",validation)
        println(io,"Largest nonlinear-vs-local waveform error: ",maximum(x.nonlinear_error for x in good))
        println(io,"Largest derivative error: ",maximum(x.approx_error for x in good))
        for tag in ["T3","T4"];println(io,tag," resident stability ",residents[tag].margin," invasion margin ",residents[tag].invasion);end
    end
    print(read(joinpath(OUT,"numerical_summary.txt"),String))
end
main()
