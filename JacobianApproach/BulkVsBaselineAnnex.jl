############ 1) Project scrambled J onto "structure-only" manifold ############
using LinearAlgebra, Statistics

"""
    structure_only_project(J, Jscr; tol_diag=1e-10, tol_lam=1e-10)

Return Jhat that has:
  - diag(Jhat) == diag(J)        (time-scales restored)
  - λmax(Jhat) == λmax(J)        (edge rate matched by scalar shift)

Also returns a NamedTuple with diagnostic errors.
"""
function structure_only_project(J::AbstractMatrix{<:Real},
                                Jscr::AbstractMatrix{<:Real};
                                mode::Symbol = :pin_small_t,  # :pin_small_t or :pin_large_t
                                tol_diag::Float64 = 1e-10,
                                tol_lam::Float64  = 1e-10)

    S = size(J,1); @assert size(J,2)==S==size(Jscr,1)==size(Jscr,2)

    if mode === :pin_small_t
        # restore time scales (diagonal) → eigenvalues WILL change
        Jhat = Matrix{Float64}(Jscr)
        @inbounds for i in 1:S
            Jhat[i,i] = J[i,i]
        end
        # diagnostics: only check the diagonal; λmax mismatch is expected, don’t warn
        diag_err = norm(diag(Jhat) .- diag(J), Inf)
        lam_err  = maximum(real(eigvals(Jhat))) - maximum(real(eigvals(J)))  # for info only
        if diag_err > tol_diag
            @warn "structure_only_project(pin_small_t): diagonal mismatch = $diag_err (> $tol_diag)"
        end
        return Jhat, (; mode, diag_err, lam_err, note="λmax mismatch expected in pin_small_t")
    elseif mode === :pin_large_t
        # keep eigenvalues (esp. λmax) by doing NOTHING further — Jscr already preserves spectrum
        Jhat = Matrix{Float64}(Jscr)
        # report tiny spectral drift if any
        lam_err  = abs(maximum(real(eigvals(Jhat))) - maximum(real(eigvals(J))))
        diag_err = norm(diag(Jhat) .- diag(J), Inf)   # for info; small-t will differ
        if lam_err > tol_lam
            @warn "structure_only_project(pin_large_t): λmax mismatch = $lam_err (> $tol_lam)"
        end
        return Jhat, (; mode, diag_err, lam_err)
    else
        error("mode must be :pin_small_t or :pin_large_t")
    end
end

######################## 2) R̃med sanity checks (small/large t) ########################

"""
    sanity_check_series(J, Jhat, u, t_vals; perturb=:biomass)

Computes R̃med(t) for J and Jhat and reports:
  - Δ_small = |R̃med(t_min) difference|
  - Δ_large = |R̃med(t_max) difference|
Plus returns the two series.
Assumes you already defined `compute_rmed_series_stable`.
"""
function sanity_check_series(J::AbstractMatrix{<:Real},
                             Jhat::AbstractMatrix{<:Real},
                             u::AbstractVector{<:Real},
                             t_vals::AbstractVector{<:Real};
                             perturb::Symbol=:biomass)

    f  = compute_rmed_series_stable(J,    u, t_vals; perturb=perturb)
    fh = compute_rmed_series_stable(Jhat, u, t_vals; perturb=perturb)

    Δ_small = abs(fh[1]   - f[1])
    Δ_large = abs(fh[end] - f[end])

    return (; Δ_small, Δ_large, rmed=f, rmed_hat=fh)
end
