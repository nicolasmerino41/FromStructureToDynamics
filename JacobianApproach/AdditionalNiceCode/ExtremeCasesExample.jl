S = 50    # matrix size
a = 1.0   # choose your magnitude
H = zeros(Float64, S, S)
for i in 1:S
    for j in i+1:S
        H[i,j] = a
        H[j,i] = -a
    end
end
H
# Compute normality measure:
# If the matrix is normal, this value should be 0 (or extremely close to 0 due to floating-point precision).
normality_error_A = norm(H * H' - H' * H)

# Non-hierarchical matrix
NH = zeros(Float64, S, S)
for i in 1:S
    for j in i+1:S
        s = rand(Bool) ? a : -a
        NH[i,j] = s
        NH[j,i] = -s
    end
end
NH
# Normality measure for NH:
normality_error_NH = norm(NH * NH' - NH' * NH)

# TRIANGULAR MATRIX
S = 50
a = 1.0  # set magnitude
T = zeros(Float64, S, S)
for i in 1:S
    for j in i+1:S
        T[i,j] = a
    end
end
T
# Normality measure for T:
normality_error_T = norm(T * T' - T' * T)

# almost TRIANGULAR MATRIX
S = 50
a = 1.0  # set magnitude
aT = zeros(Float64, S, S)
for i in 1:S
    for j in i+1:S
        aT[i,j] = -a
        aT[j,i] = 0.5
    end
end
aT
# Normality measure for aT:
normality_error_aT = norm(aT * aT' - aT' * aT)

D = Diagonal(u)            # diagonal matrix

J_H = H - D
J_NH = NH - D
J_T = T - D
J_aT = aT - D

# stability
λH = maximum(real, eigvals(J_H))
λNH = maximum(real, eigvals(J_NH))
λT = maximum(real, eigvals(J_T))
λaT = maximum(real, eigvals(J_aT))

# ---- TIME RANGE ----
ts = 10 .^ range(log10(0.01), log10(100.0); length=30)

# ---- METRIC OVER TIME ----
r_H = [median_return_rate(J_H, u; t=t) for t in ts]
r_NH = [median_return_rate(J_NH, u; t=t) for t in ts]
r_T  = [median_return_rate(J_T,  u; t=t) for t in ts]
r_aT  = [median_return_rate(J_aT,  u; t=t) for t in ts]

# ---- PLOT ----
begin
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "t", ylabel = "rₘₑd(t)", title = "Median Return Rate vs Time")

    lines!(ax, ts, r_H, label = "Hierarchical Antisymmetry")
    lines!(ax, ts, r_NH, label = "Random Antisymmetry")
    lines!(ax, ts, r_T,  label = "Triangular Community")
    lines!(ax, ts, r_aT,  label = "Almost Triangular Community")

    axislegend(ax, position = :rb)
    display(f)
end

noise_strength = 0.1
sym_noise = noise_strength .* randn(S, S)
sym_noise = 0.5 .* (sym_noise + sym_noise')    # symmetric only
A_H_noisy = H + sym_noise
A_NH_noisy = NH + sym_noise
J_H_noisy = A_H_noisy - D
J_NH_noisy = A_NH_noisy - D
for i in 2:S
    for j in 1:i-1
        T[i,j] = noise_strength * rand()
    end
end

J_T_noisy = T - D
r_H = [median_return_rate(J_H_noisy, u; t=t) for t in ts]
r_NH = [median_return_rate(J_NH_noisy, u; t=t) for t in ts]
r_T  = [median_return_rate(J_T_noisy,  u; t=t) for t in ts]

begin
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "t", ylabel = "rₘₑd(t)", title = "Median Return Rate vs Time")

    lines!(ax, ts, r_H, label = "Hierarchical Antisymmetry")
    lines!(ax, ts, r_NH, label = "Random Antisymmetry")
    lines!(ax, ts, r_T,  label = "Triangular Community")

    axislegend(ax, position = :rt)
    display(f)
end