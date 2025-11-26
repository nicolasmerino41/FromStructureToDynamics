function make_antisym_matrix(S, gp)
    A = zeros(S, S)
    for i in 1:S-1, j in (i+1):S
        p_ij = rand() < gp ? 1.0 : 0.0
        A[i,j] = p_ij * (2*rand() - 1)
        A[j,i] = -p_ij * (2*rand() - 1)
    end
    return A
end   

A = make_antisym_matrix(50, 1.0)
println(A)
B = abs.(A)

non_normality_A = norm(A - A')
non_normality_B = norm(B - B')

norm(B)
rho_A = maximum(real, eigvals(A))
rho_A = maximum(abs.(eigvals(A)))   # take absolute value, not real part!
println("Spectral radius of A: ", rho_A)

function make_upper_triangular_matrix(S, gp)
    A = zeros(S, S)
    for i in 1:S-1, j in (i+1):S
        p_ij = rand() < gp ? 1.0 : 0.0
        A[i,j] = p_ij * (2*rand(UInt8) - 1)
    end
    return A
end

A_upper = make_upper_triangular_matrix(50, 0.05)
println(A_upper)

rho_A_upper = maximum(real, eigvals(A_upper))
non_normality_A_upper = norm(A_upper - A_upper')

println("Spectral radius of upper triangular A: ", rho_A_upper)
rho_A = maximum(abs.(eigvals(A_upper)))

