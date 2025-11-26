#############################################################
# NORMAL vs NON-NORMAL vs MISALIGNED BIOMASS
#############################################################
# ----------------------------
# 1. DEFINE MATRICES
# ----------------------------
# Normal matrix
J_normal = [-1.0  0.3;
             0.3 -1.2]

# Non-normal – strong shear → transient amplification possible
J_nonnormal = [-1.0  5.0;
                0.0 -1.2]

# Non-normal but **designed** so transient direction is e₁,
# while biomass points in e₂ → NO amplification for our biomass.
J_nonnormal2 = [-1.0  8.0;
                 0.0 -2.5]

# Biomass orthogonal to amplifying direction:
u_bad = [0.0, 1.0]   # normalized below

# Check normality
println("\nNormality check (0 = normal):")
println("  normal:     ", norm(J_normal*J_normal'   - J_normal'*J_normal))
println("  non-normal: ", norm(J_nonnormal*J_nonnormal' - J_nonnormal'*J_nonnormal))
println("  non-normal2:", norm(J_nonnormal2*J_nonnormal2' - J_nonnormal2'*J_nonnormal2))

# ----------------------------
# 2. TRANSIENT AMPLIFICATION fn
# ----------------------------

function transient_amplification(J; tmax=5, nt=200)
    ts = range(0, tmax; length=nt)
    ampl = zeros(nt)
    for (i,t) in enumerate(ts)
        ampl[i] = opnorm(exp(J*t))   # max singular value
    end
    return ts, ampl
end

# Biomass-weighted version:
function rmed_weighted(J, u; tmax=5, nt=200)
    n = size(J,1)
    u = u / norm(u) # normalize biomass

    ts = range(0, tmax; length=nt)
    r = zeros(nt)

    for (i,t) in enumerate(ts)
        M = exp(J*t)
        x = M * u
        r[i] = norm(x)   # biomass–weighted response
    end

    return ts, r ./ maximum(r)  # normalized for comparison
end

# ----------------------------
# 3. RUN ALL CASES
# ----------------------------
ts1, amp_norm    = transient_amplification(J_normal)
ts2, amp_non     = transient_amplification(J_nonnormal)
ts3, amp_misal   = transient_amplification(J_nonnormal2)

_, r_nonbio      = rmed_weighted(J_nonnormal,  [1,1])
_, r_misaligned  = rmed_weighted(J_nonnormal2, u_bad)

# ----------------------------
# 4. MAKE THE PLOTS
# ----------------------------
begin
    f = Figure(; size=(1100,500))

    ax1 = Axis(f[1,1], title="Max Singular Value (transient amplification)",
            xlabel="t", ylabel="‖exp(Jt)‖₂")
    lines!(ax1, ts1, amp_norm,    label="Normal")
    lines!(ax1, ts2, amp_non,     label="Non-normal (amplifies)", linestyle=:dash)
    lines!(ax1, ts3, amp_misal,   label="Non-normal + bad direction", linestyle=:dot)
    axislegend(ax1)

    ax2 = Axis(f[1,2], title="Biomass-weighted r(t) response",
            xlabel="t", ylabel="scaled response")
    lines!(ax2, ts2, r_nonbio,    label="Non-normal, good u")
    lines!(ax2, ts3, r_misaligned, label="Non-normal, bad u (hidden)", linestyle=:dash)
    axislegend(ax2)

    display(f)

end