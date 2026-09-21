using LinearAlgebra, Statistics, Printf, Test
using CairoMakie

# Run: julia --startup-file=no single_link_analysis.jl
# Convention: T dx/dt = A x + a*b*cos(omega*t), fixed T and fixed reference equilibrium.
# A[i,j] is the effect j -> i. All changes are equal ABSOLUTE changes in A[i,j].
const OUT = joinpath(@__DIR__, "outputs")
mkpath(OUT)

function illustrative_system()
    # Same three oscillatory modules and feed-forward couplings as Figure5.jl,
    # before its display permutation. These are illustrative, not fitted species.
    A = -Matrix{Float64}(I, 6, 6)
    for (k, w) in enumerate([0.35, 1.2, 4.0])
        i = 2k-1
        A[i,i+1] = -w; A[i+1,i] = w
    end
    for (i,j) in [(3,1),(5,3)]
        A[i,j] = 2.5; A[i+1,j+1] = 2.5
        A[i,j+1] = 1.0; A[i+1,j] = -1.0
    end
    return A, Diagonal([1.0,1.0,1.0,1.0,1.0,1.0])
end

resolvent(A,T,w) = (im*w*T-A) \ Matrix{Float64}(I,size(A,1),size(A,1))
linklabel(link) = "$(link[2]) → $(link[1])"
function tied_ranks(v)
    # Symmetric modules generate true ties; do not display roundoff as rank swaps.
    order=sortperm(v;rev=true); ranks=zeros(length(v)); first=1
    while first <= length(v)
        last=first
        while last < length(v) && isapprox(v[order[last+1]],v[order[first]];rtol=1e-10,atol=1e-14)
            last+=1
        end
        ranks[order[first:last]] .= (first+last)/2
        first=last+1
    end
    ranks
end
function perturbation(n, link)
    P = zeros(n,n); P[link...] = 1.0; P
end

function profiles(A,T,ws,links,b)
    n = size(A,1); m = length(links)
    source = zeros(m,length(ws)); propagation = similar(source)
    potential = similar(source); realized = similar(source)
    identity_error = 0.0; bound_excess = 0.0
    for (k,w) in enumerate(ws)
        R = resolvent(A,T,w)
        for (l,(i,j)) in enumerate(links)
            source[l,k] = norm(R[j,:])
            propagation[l,k] = norm(R[:,i])
            potential[l,k] = source[l,k]*propagation[l,k]
            realized[l,k] = propagation[l,k]*abs((R*b)[j])
            H = R*perturbation(n,(i,j))*R
            identity_error = max(identity_error,abs(opnorm(H)-potential[l,k])/potential[l,k])
            @assert isapprox(norm(H*b),realized[l,k]; rtol=1e-11,atol=1e-13)
            bound_excess = max(bound_excess,realized[l,k]-potential[l,k])
        end
    end
    (;source,propagation,potential,realized,identity_error,bound_excess)
end

function select_pair(S)
    # Transparent selection: maximize the weaker of the two opposing dominance
    # ratios. Exclude negligible tails: both links >= 10% of own peak at each end.
    best = -Inf; selected = nothing
    for a in axes(S,1), b in a+1:size(S,1)
        valid = findall((S[a,:] .>= 0.1maximum(S[a,:])) .& (S[b,:] .>= 0.1maximum(S[b,:])))
        isempty(valid) && continue
        ratio = log.(S[a,valid]./S[b,valid])
        lo = argmin(ratio); hi = argmax(ratio)
        score = min(-ratio[lo],ratio[hi])
        if score > best
            best = score; selected = (a,b,valid[hi],valid[lo])
        end
    end
    @assert best > 0 "No ranking reversal found; do not manufacture one."
    a,b,k1,k2 = selected
    k1 > k2 && ((a,b,k1,k2) = (b,a,k2,k1))
    return a,b,k1,k2,exp(best)
end

function integrate_response(A,T,b,w,t,a)
    # Independent time-domain RK4 integration, starting at zero displacement.
    J = T\A; q = T\b; x = zeros(length(b)); X = zeros(length(b),length(t))
    f(x,s) = J*x + a*q*cos(w*s)
    for k in 1:length(t)-1
        h = t[k+1]-t[k]; s = t[k]
        k1=f(x,s); k2=f(x+h*k1/2,s+h/2)
        k3=f(x+h*k2/2,s+h/2); k4=f(x+h*k3,s+h)
        x += h*(k1+2k2+2k3+k4)/6
        X[:,k+1] = x
    end
    X
end

function validation(A,T,b,ws,links,chosen,ks)
    rows = NamedTuple[]; traces = NamedTuple[]
    epsilons = [0.001,0.003,0.01,0.03,0.1]
    for k in ks, l in chosen
        w = ws[k]; P=perturbation(size(A,1),links[l]); R=resolvent(A,T,w)
        H = R*P*R
        for epsilon in epsilons
            Ap = A+epsilon*P
            @assert maximum(real.(eigvals(T\Ap))) < 0
            exact = (resolvent(Ap,T,w)-R)*b
            predicted = epsilon*H*b
            err = norm(exact-predicted)/norm(exact)
            push!(rows,(link=linklabel(links[l]),omega=w,epsilon=epsilon,error=err))
        end
        epsilon=0.01; amplitude=0.01; Ap=A+epsilon*P
        decay = min(-maximum(real.(eigvals(T\A))),-maximum(real.(eigvals(T\Ap))))
        period=2pi/w; burn=40/decay
        dt=min(0.01,period/400,0.05/opnorm(T\Ap))
        t=collect(range(0,burn+3period;length=ceil(Int,(burn+3period)/dt)+1))
        X=integrate_response(A,T,b,w,t,amplitude)
        Y=integrate_response(Ap,T,b,w,t,amplitude)
        exact_complex=amplitude*(resolvent(Ap,T,w)-R)*b
        predicted_complex=amplitude*epsilon*H*b
        keep=findall(t .>= burn)
        delta=Y[:,keep]-X[:,keep]
        exact=real.(exact_complex .* transpose(exp.(im*w*t[keep])))
        prediction=real.(predicted_complex .* transpose(exp.(im*w*t[keep])))
        ode_error=norm(delta-exact)/norm(exact)
        @assert ode_error < 1e-4 "Time integration does not match exact frequency response"
        species=argmax(abs.(exact_complex))
        push!(traces,(link=linklabel(links[l]),omega=w,t=(t[keep].-burn)./period,
            actual=delta[species,:],prediction=prediction[species,:],species=species,
            ode_error=ode_error,linear_error=norm(delta-prediction)/norm(delta)))
    end
    rows,traces
end

function make_figures(A,ws,links,p,chosen,ks,rows,traces)
    colors=["#0072B2","#D55E00"]
    set_theme!(Theme(fontsize=15,Axis=(spinewidth=0.8,xgridvisible=false,ygridvisible=false)))
    fig=Figure(size=(1550,1020),figure_padding=28)
    Label(fig[0,1:3],"Why a link matters depends on the forcing timescale",fontsize=27,font=:bold)
    Label(fig[1,1:3],"Illustrative six-species linear community • equal absolute link changes • fixed T = I",fontsize=16,color=:gray35)
    ax=Axis(fig[2,1],title="A  All existing links",xlabel="Angular frequency ω",ylabel="Source → recipient",
        yticks=(1:length(links),linklabel.(links)),xticks=([-2.,-1.,0.,1.],["0.01","0.1","1","10"]),yticklabelsize=11)
    ranks=zeros(size(p.potential))
    for k in eachindex(ws)
        ranks[:,k] = tied_ranks(p.potential[:,k])
    end
    hm=heatmap!(ax,log10.(ws),collect(1:length(links)),transpose(ranks),colormap=:viridis,colorrange=(1,length(links)))
    ylims!(ax,length(links)+0.5,0.5)
    Colorbar(fig[3,1],hm,vertical=false,label="Rank within frequency (ties averaged)",ticks=[1,4,7,10,14],height=12)
    axes=[Axis(fig[2,2],title="B  Source responsiveness",ylabel="Maximum source amplitude",xlabel="Angular frequency ω",xscale=log10,yscale=log10),
          Axis(fig[2,3],title="C  Recipient propagation",ylabel="Community response to recipient input",xlabel="Angular frequency ω",xscale=log10,yscale=log10),
          Axis(fig[4,1],title="D  Their product: potential sensitivity",ylabel="‖R P R‖₂",xlabel="Angular frequency ω",xscale=log10,yscale=log10),
          Axis(fig[4,2],title="E  One fixed disturbance",ylabel="Sensitivity",xlabel="Angular frequency ω",xscale=log10,yscale=log10),
          Axis(fig[4,3],title="F  First-order prediction improves",ylabel="Relative error of complex response",xlabel="Absolute interaction change ε",xscale=log10,yscale=log10)]
    for (c,l) in enumerate(chosen)
        label="Link "*linklabel(links[l]); color=colors[c]
        for (ax,values) in zip(axes[1:3],[p.source,p.propagation,p.potential])
            lines!(ax,ws,values[l,:],color=color,linewidth=3,label=label)
            scatter!(ax,ws[ks],values[l,ks],color=color,markersize=9)
        end
        lines!(axes[4],ws,p.potential[l,:],color=(color,0.45),linestyle=:dash,linewidth=2)
        lines!(axes[4],ws,p.realized[l,:],color=color,linewidth=3,label=label)
        for (kk,k) in enumerate(ks)
            rr=filter(r->r.link==linklabel(links[l]) && r.omega==ws[k],rows)
            lines!(axes[5],[r.epsilon for r in rr],[max(r.error,1e-15) for r in rr],color=color,linestyle=kk==1 ? :solid : :dash,linewidth=2)
            scatter!(axes[5],[r.epsilon for r in rr],[max(r.error,1e-15) for r in rr],color=color,markersize=6)
        end
    end
    axislegend(axes[3],position=:lb,framevisible=false,labelsize=13)
    Label(fig[5,1],"B × C = D  •  selected links illustrate a ranking reversal",fontsize=13)
    Label(fig[5,2],"Solid: forcing on species 1 only\nDashed: maximum over unit complex inputs",fontsize=13)
    Label(fig[5,3],"Solid: slower marked frequency\nDashed: faster marked frequency",fontsize=13)
    Label(fig[6,1:3],"Frequency and time are in model units. Sensitivity measures community-vector amplitude, not total biomass.\nThe highlighted pair is selected transparently from this example; a reversal is not a universal claim.",fontsize=14,color=:gray35)
    rowgap!(fig.layout,18); colgap!(fig.layout,30)
    save(joinpath(OUT,"single_link_mechanism.png"),fig,px_per_unit=1.6)
    save(joinpath(OUT,"single_link_mechanism.pdf"),fig)
    fig2=Figure(size=(1250,760),figure_padding=28)
    Label(fig2[0,1:2],"Do the predicted changes appear in time-domain simulations?",fontsize=24,font=:bold)
    Label(fig2[1,1:2],"Same forcing on species 1 in every comparison • forcing amplitude 0.01 • link change ε = 0.01",fontsize=14)
    for (q,tr) in enumerate(traces)
        row=div(q-1,2)+2; col=mod(q-1,2)+1
        ax=Axis(fig2[row,col],title=@sprintf("Link %s • ω = %.3g • species %d",tr.link,tr.omega,tr.species),xlabel="Cycles after transient discarded",ylabel="Change in species displacement")
        lines!(ax,tr.t,tr.actual,color=colors[col],linewidth=3,label="Direct ODE simulation")
        lines!(ax,tr.t,tr.prediction,color=:black,linestyle=:dash,linewidth=2,label="First-order prediction")
        q==1 && axislegend(ax,position=:rt,labelsize=12,framevisible=false)
    end
    Label(fig2[4,1:2],"Each panel shows the species with the largest exact response change for that comparison.\nValidation errors use all species, not only the displayed species. This tests linear dynamics, not nonlinear ecological validity.",fontsize=13,color=:gray35)
    rowgap!(fig2.layout,22); colgap!(fig2.layout,28)
    save(joinpath(OUT,"time_domain_validation.png"),fig2,px_per_unit=1.6)
    save(joinpath(OUT,"time_domain_validation.pdf"),fig2)
end

function main()
    A,T=illustrative_system(); n=size(A,1)
    ws=10.0 .^ range(-2,1.3;length=350)
    links=[(i,j) for i in 1:n for j in 1:n if i!=j && A[i,j]!=0]
    b=zeros(n); b[1]=1.0 # Same real unit forcing vector across links and frequencies.
    p=profiles(A,T,ws,links,b)
    a,c,k1,k2,dominance=select_pair(p.potential)
    chosen=[a,c]; ks=[k1,k2]
    rows,traces=validation(A,T,b,ws,links,chosen,ks)
    @testset "Scientific checks" begin
        @test tied_ranks([3.,3.0+1e-13,1.]) == [1.5,1.5,3.]
        @test maximum(real.(eigvals(T\A))) < 0
        @test p.identity_error < 1e-11
        @test p.bound_excess < 1e-11
        @test p.potential[a,k1] > p.potential[c,k1]
        @test p.potential[c,k2] > p.potential[a,k2]
        @test maximum(t.ode_error for t in traces) < 1e-4
        # Also test identity and derivatives with heterogeneous timescales.
        Th=Diagonal([0.4,0.8,1.0,1.7,2.0,3.0])
        for w in [0.0,0.1,1.0,4.0], l in links
            R=resolvent(A,Th,w); P=perturbation(n,l); i,j=l
            @test opnorm(R*P*R) ≈ norm(R[:,i])*norm(R[j,:])
            h=1e-5
            fd=(resolvent(A+h*P,Th,w)-resolvent(A-h*P,Th,w))/(2h)
            @test norm(fd-R*P*R)/norm(R*P*R) < 1e-7
        end
        for l in chosen, k in ks
            rr=filter(r->r.link==linklabel(links[l]) && r.omega==ws[k],rows)
            @test rr[1].error < rr[end].error
        end
    end
    open(joinpath(OUT,"profiles.csv"),"w") do io
        println(io,"source,recipient,omega,source_maximum,recipient_propagation,potential_sensitivity,fixed_forcing_sensitivity")
        for (l,(i,j)) in enumerate(links), (k,w) in enumerate(ws)
            println(io,join((j,i,w,p.source[l,k],p.propagation[l,k],p.potential[l,k],p.realized[l,k]),","))
        end
    end
    open(joinpath(OUT,"validation.csv"),"w") do io
        println(io,"link,omega,epsilon,relative_complex_response_error")
        for r in rows; println(io,join((r.link,r.omega,r.epsilon,r.error),",")); end
    end
    open(joinpath(OUT,"summary.txt"),"w") do io
        println(io,"Illustrative single-link analysis; Julia ",VERSION)
        println(io,"CairoMakie ",Base.pkgversion(CairoMakie))
        println(io,"Baseline maximum real eigenvalue: ",maximum(real.(eigvals(T\A))))
        println(io,"Highlighted links: ",linklabel.(links[chosen]))
        println(io,"Highlighted angular frequencies: ",ws[ks])
        println(io,"Minimum opposing dominance ratio: ",dominance)
        println(io,"Maximum relative factorisation error: ",p.identity_error)
        println(io,"Maximum ODE vs exact frequency response error: ",maximum(t.ode_error for t in traces))
        println(io,"Maximum ODE vs first-order error at epsilon=0.01: ",maximum(t.linear_error for t in traces))
        println(io,"Fixed forcing b = ",b)
        println(io,"A = "); show(io,"text/plain",A); println(io,"\nT = "); show(io,"text/plain",T)
        for k in ks, l in chosen
            println(io,"\nLink ",linklabel(links[l])," omega=",ws[k]," potential=",p.potential[l,k]," realized=",p.realized[l,k])
        end
    end
    open(joinpath(OUT,"time_traces.csv"),"w") do io
        println(io,"link,omega,displayed_species,cycles_after_burn,simulated_difference,first_order_prediction")
        for tr in traces, k in eachindex(tr.t)
            println(io,join((tr.link,tr.omega,tr.species,tr.t[k],tr.actual[k],tr.prediction[k]),","))
        end
    end
    make_figures(A,ws,links,p,chosen,ks,rows,traces)
    println(read(joinpath(OUT,"summary.txt"),String))
    println("Figures and numeric results: ",OUT)
end

main()
