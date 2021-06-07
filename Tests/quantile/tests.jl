using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
using .QR
using Plots, PlotThemes, Formatting #, CSV, DataFrames, StatFiles
# using KernelDensity
theme(:juno)

## Sampling of σ

rand(Pareto(n-1, 0.99), 1000) |> mean

σ[1:1000] |> plot
plot!(θ[1:1000])

sampleσ(X, y, U1[1000,:], U2[1000,:], β[1000-1, :], α, θ[1000-1], 1)

iter = 1000
lower = zeros(n)
for i in 1:n
    μ = X[i,:] ⋅ β[iter,:]
    if (U1[iter,i] > 0) && (y[i] < μ)
        lower[i] = (μ - y[i]) / (α * U1[iter,i]^(1/θ[iter-1]))
    elseif (U2[iter,i] > 0) && (y[i] >= μ)
        lower[i] = (y[i] - μ) / ((1-α) * U2[iter,i]^(1/θ[iter-1]))
    end
end
maximum(lower)



##

# generate data
n = 300;
β, α, θ, σ = [2.1, 0.8], 0.5, 2., 1.;
X = [repeat([1], n) rand(Uniform(-3, 3), n)]
y = X * β .+ rand(Normal(0, σ), n);
√(var(y-X*inv(X'*X)*X'*y))

nMCMC = 500000
σ = zeros(nMCMC)
σ[1] = 1.
β = zeros(nMCMC, 2)
β[1, :] = [2.1, 0.8]
θ = zeros(nMCMC)
θ[1] = 2.5
U1, U2 = zeros(nMCMC, n), zeros(nMCMC, n)

for i in 2:nMCMC
    u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i-1])
    U1[i,:] = u1
    U2[i,:] = u2
    interval = round.(θinterval(X, y, u1, u2, β[i-1,:], α, σ[i-1]), digits = 3)
    (i % 10000 === 0) && printfmt("iter: {1}, θ ∈ [{2:.2f}, {3:.2f}] \n", i, interval[1], interval[2])
    σ[i] = sampleσ(X, y, u1, u2, β[i-1, :], α, θ[i-1], 1)
    θ[i] = sampleθ(θ[i-1], 100., X, y, u1, u2, β[i-1, :], α, σ[i])
    # θ[i] = 2.
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i], σ[i], 10.)
end

plot(β[:, 2])
plot(σ)
plot(θ)
plot!(σ)

autocor(θ, [1,3,10,40]) |> println

thin = ((1:nMCMC) .% 100) .=== 0
autocor(θ[thin], [1,3,10,40]) |> println
plot(θ[thin])
plot(σ[thin])

1-((θ[2:nMCMC] .=== θ[1:(nMCMC - 1)]) |> mean)

plot(cumsum(σ) ./ (1:nMCMC))
plot(cumsum(β[:, 2]) ./ (1:nMCMC))
plot(cumsum(θ) ./ (1:nMCMC))
median(θ[10000:nMCMC])
median(θ[thin])
##
plot!(cumsum(σ) ./ (1:nMCMC))
plot!(cumsum(θ) ./ (1:nMCMC))
"""
CSV.write("beta.csv", DataFrame(β), header = false)
CSV.write("theta.csv", DataFrame(reshape(θ, nMCMC, 1)), header = false)
CSV.write("sigma.csv", DataFrame(reshape(σ, nMCMC, 1)), header = false)
"""
