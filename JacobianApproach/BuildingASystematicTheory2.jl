# --- plug your existing builders here ---
#   build_random_trophic_ER(S; conn, mean_abs, mag_cv, rho_sym, rng)
#   build_random_trophic(S; conn, mean_abs, mag_cv, degree_family, deg_param, rho_sym, rng)
#   random_u(S; mean, cv, rng)
#   realized_IS(A)
#   jacobian(A, u)
#   median_return_rate(J, u; t, perturbation)
#   r2_to_identity(x::AbstractVector, y::AbstractVector)

# ---------------- small helpers ----------------
finite(v) = filter(isfinite, v)
absdiff(x,y) = @. abs(x - y)

function scale_IS!(A; mean_abs)
    is0 = realized_IS(A)
    if is0 != 0.0
        A .*= mean_abs / is0
    end
    return A
end

# --- R̃med series (safe) via real Schur and expm on quasi-triangular T ---
# We keep the standard median_return_rate(J,u; t, perturbation) you already have,
# and provide a loop that calls it; it’s slower than a custom Schur-kernel,
# but it’s robust and won’t crash the session.
function rmed_series(J::AbstractMatrix, u::AbstractVector, tvals; perturbation::Symbol)
    [median_return_rate(J, u; t=t, perturbation=perturbation) for t in tvals]
end

# ---------------- experiment 5a: scramble eigenvectors only ----------------
function schur_scramble(J::AbstractMatrix; rng::AbstractRNG)
    # Real Schur by default on real matrices in Julia 1.11
    F = schur(Matrix{Float64}(J))   # J = Qsch * Tsch * Qsch'
    # You can use either pair below; they are aliases.
    Qsch = F.Z        # == F.vectors
    Tsch = F.T        # == F.Schur

    # Random orthogonal matrix via QR
    Qrand = Matrix(qr!(randn(rng, size(J)...)).Q)

    # Similarity in the Schur subspace: preserves eigenvalues
    Tnew = Qrand' * Tsch * Qrand
    Jnew = Qsch * Tnew * Qsch'   # back to the original basis
    return Jnew
end

# ---------------- experiment 5b: change only diagonal (A fixed) -------------
function diagonal_only_change(A; u_mean, u_cv, rng::AbstractRNG)
    u2 = random_u(size(A,1); mean=u_mean, cv=u_cv, rng=rng)
    return jacobian(A, u2), u2
end

# ---------------- bulk reshape (alter spectrum “bulk” of A) -----------------
function bulk_reshape(; S, conn, mean_abs, mag_cv, rho_sym_new, degcv_new, rng::AbstractRNG)
    A2 = build_random_trophic(S; conn=conn, mean_abs=mean_abs, mag_cv=mag_cv,
                              degree_family=:lognormal, deg_param=degcv_new,
                              rho_sym=rho_sym_new, rng=rng)
    scale_IS!(A2; mean_abs=mean_abs)
    return A2
end

# ---------------- baseline ----------------
function make_baseline(; S, conn, mean_abs, mag_cv, rho_sym, u_mean, u_cv, seed)
    rng = Random.Xoshiro(seed)
    A0 = build_random_trophic_ER(S; conn=conn, mean_abs=mean_abs,
                                 mag_cv=mag_cv, rho_sym=rho_sym, rng=rng)
    # A0 = build_random_nontrophic(S; conn=conn, mean_abs=mean_abs,
    #                              mag_cv=mag_cv, rho_sym=rho_sym, rng=rng,
    #                              degree_family=:uniform, deg_param=0.0)                             
    scale_IS!(A0; mean_abs=mean_abs)
    u0 = random_u(S; mean=u_mean, cv=u_cv, rng=rng)
    J0 = jacobian(A0, u0)
    return (; A0, u0, J0)
end

# ---------------- run the three tests ----------------
function run_three_tests(; S=120, conn=0.10, mean_abs=0.50, mag_cv=0.60, rho_sym=0.50,
                         u_mean=1.0, u_cv=0.6, t_vals=10 .^ range(-2, 2; length=40),
                         perturbation::Symbol=:uniform, seed=20251111)

    A0, u0, J0 = make_baseline(; S, conn, mean_abs, mag_cv, rho_sym, u_mean, u_cv, seed)
    rng = Random.Xoshiro(seed ⊻ 0x9e3779b97f4a7c15)

    # baseline
    f0 = rmed_series(J0, u0, t_vals; perturbation)

    # (5a) eigenvectors scrambled, eigenvalues preserved
    Jscr = schur_scramble(J0; rng)
    fscr = rmed_series(Jscr, u0, t_vals; perturbation)

    # (5b) change only diagonal (new u), A fixed
    Jdiag, u2 = diagonal_only_change(A0; u_mean, u_cv, rng)
    fdiag = rmed_series(Jdiag, u2, t_vals; perturbation)

    # bulk reshape of A (e.g., ↑deg_cv and tweak ρ_sym), keep same u to isolate A
    Abulk = bulk_reshape(; S, conn, mean_abs, mag_cv,
                         rho_sym_new=min(0.95, rho_sym + 0.3),
                         degcv_new=1.0, rng)
    Jbulk = jacobian(Abulk, u0)
    fbulk = rmed_series(Jbulk, u0, t_vals; perturbation)

    return (; t_vals, perturbation, f0, fscr, fdiag, fbulk, J0, Jscr, Jdiag, Jbulk)
end

# ---------------- spectra helper ----------------
re_parts(J) = sort!(real.(eigvals(Matrix(J))))

# ---------------- plotting ----------------
function plot_three_tests(res)
    t_vals, perturbation = res.t_vals, res.perturbation
    f0, fscr, fdiag, fbulk = res.f0, res.fscr, res.fdiag, res.fbulk
    J0, Jscr, Jdiag, Jbulk = res.J0, res.Jscr, res.Jdiag, res.Jbulk

    fig = Figure(size=(1200, 750))
    Label(fig[0, 1:3], "Bulk vs edge diagnostics — proofs (perturbation=$(perturbation))";
          fontsize=22, font=:bold)

    labels = ["(5a) eigenvector shuffle (eigs fixed)",
              "(5b) reshuffle diagonal only (A fixed, new u)",
              "(5c) bulk reshape (↑deg_cv, Δρ_sym)"]
    others = [fscr, fdiag, fbulk]
    Js     = [Jscr, Jdiag, Jbulk]

    for j in 1:3
        # R̃med overlays
        ax1 = Axis(fig[1, j]; xscale=log10, title=labels[j],
                   ylabel=(j==1 ? "R̃med" : ""), xlabel="t")
        lines!(ax1, t_vals, f0; color=:steelblue, linewidth=2, label="baseline")
        lines!(ax1, t_vals, others[j]; color=:orange, linewidth=2, label="modified")
        if j == 1; axislegend(ax1; position=:lt, framevisible=false); end

        # |ΔR̃med|
        ax2 = Axis(fig[2, j]; xscale=log10, ylabel=(j==1 ? "|ΔR̃med|" : ""), xlabel="t")
        values = abs.(f0 .- others[j])
        values[values .< 1e-3] .= 0.0
        lines!(ax2, t_vals, values; color=:crimson, linewidth=2)

        # eigenvalue real parts
        ax3 = Axis(fig[3, j]; ylabel=(j==1 ? "Re(λ)" : ""), xlabel="mode index")
        rp0   = re_parts(J0)
        rpnew = re_parts(Js[j])
        scatter!(ax3, 1:length(rp0),  rp0;   color=:gray70, markersize=4, label="baseline eigs")
        scatter!(ax3, 1:length(rpnew), rpnew; color=:black,  markersize=3, label="modified eigs")
        if j == 1; axislegend(ax3; position=:lb, framevisible=false); end
    end

    display(fig)
end

# ---------------- run & show ----------------
res = run_three_tests(
    ; S=120,
    conn=0.10, mean_abs=0.50, mag_cv=0.60, rho_sym=0.50,
    u_mean=1.0, u_cv=0.6,
    t_vals=10 .^ range(-2, 2; length=40),
    perturbation=:biomass,
    seed=0xC001D00D
)

plot_three_tests(res)
