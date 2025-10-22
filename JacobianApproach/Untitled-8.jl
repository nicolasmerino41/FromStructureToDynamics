
using LinearAlgebra, Random, Statistics

# 4x4 antisymmetric matrix: two independent rotations
A = [ 0  1  0  0;
     -1  0  0  0;
      0  0  0  1;
      0  0 -1  0 ]

function regime(u)
    J = Diagonal(u) * (A - I)
    λmax = maximum(real.(eigvals(J)))
    diff = λmax + minimum(u)
    return λmax, diff
end

rng = MersenneTwister(42)
N = 20_000
countt = Dict(:neutral=>0, :stab=>0, :destab=>0)
ε = 0.1
Aε = A + ε * randn(size(A))  # break perfect antisymmetry slightly
Aε .-= Aε' ./ 2              # keep it "mostly" antisymmetric

for _ in 1:20_000
    u = rand(LogNormal(0.0, 1.0), 4)
    J = Diagonal(u) * (Aε - I)
    λmax = maximum(real.(eigvals(J)))
    diff = λmax + minimum(u)
    if abs(diff) < 1e-3
        countt[:neutral] += 1
    elseif diff < 0
        countt[:stab] += 1
    else
        countt[:destab] += 1
    end
end

countt
