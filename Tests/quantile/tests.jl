using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
using .QR
using Plots, PlotThemes #, CSV, DataFrames, StatFiles
# using KernelDensity
theme(:juno)

# generate data
n = 100;
β, α, θ, σ = [2.1, 0.8], 0.5, 2., 1.;
X = [repeat([1], n) rand(Uniform(-3, 3), n)]
y = X * β .+ rand(Normal(0, σ), n);
# √(var(y-X*inv(X'*X)*X'*y));

nMCMC = 50000
σ = zeros(nMCMC)
σ[1] = 1.
β = zeros(nMCMC, 2)
β[1, :] = [2.1, 0.8]
θ = zeros(nMCMC)
θ[1] = 2.

for i in 2:nMCMC
    u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i-1])
    (i % 10000 === 0) && println(round.(θinterval(X, y, u1, u2, β[i-1,:], α, σ[i-1]), digits = 3))
    σ[i] = sampleσ(X, y, u1, u2, β[i-1, :], α, θ[i-1], 1)
    θ[i] = sampleθ(θ[i-1], 2., X, y, u1, u2, β[i-1, :], α, σ[i])
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i], σ[i], 10.)
end



plot(β[:, 1])
plot(σ)
plot(θ)
plot!(σ)

autocor(θ, [1,3,10,40]) |> println

thin = ((1:nMCMC) .% 20) .=== 0
autocor(θ[thin], [1,3,10,40]) |> println
plot(θ[thin])
plot(σ[thin])

1-((θ[2:nMCMC] .=== θ[1:(nMCMC - 1)]) |> mean)

plot(cumsum(σ) ./ (1:nMCMC))
plot(cumsum(β[:, 2]) ./ (1:nMCMC))
plot(cumsum(θ) ./ (1:nMCMC))
median(θ[10000:nMCMC])


"""
CSV.write("beta.csv", DataFrame(β), header = false)
CSV.write("theta.csv", DataFrame(reshape(θ, nMCMC, 1)), header = false)
CSV.write("sigma.csv", DataFrame(reshape(σ, nMCMC, 1)), header = false)
"""
