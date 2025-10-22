
begin
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "u_cv",
        ylabel = "diff_res_min_u",
        title = "Scatter plot of diff_res_min_u vs u_cv (all IS)"
    )

    scatter!(ax, df_tr.u_cv, df_tr.diff_res_min_u;
        color = :dodgerblue,
        markersize = 8,
        strokewidth = 0.5
    )

    display(fig)
end

using LinearAlgebra

A = [ 0  1 -1;
     -1  0  1;
      1 -1  0 ]

for u in [[1,1,1], [1,0.6,0.4], [1,0.6,0.1]]
    J = Diagonal(u) * (A - I)
    λ = eigvals(J)
    println("u = ", u,
            "  →  λ = ", round.(λ, digits=3),
            "  →  Re_max = ", round(maximum(real.(λ)), digits=3))
end
