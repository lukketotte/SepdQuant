include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
using Plots, PlotThemes, CSV, DataFrames, StatFiles
# using KernelDensity

## TODO: why is the variance sampled so differently for linear model?
# β
n = 500;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ = 1.
X = [repeat([1], n) rand(Uniform(-3, 3), n)]
y = X * β .+ rand(aepd(0., σ, θ, α), n);

nMCMC = 5000
β = zeros(nMCMC, 2)
β[1,:] = [2.1, 0.8]
σ = zeros(nMCMC)
σ[1] = 2.
θ = zeros(nMCMC)
θ[1] = 1.

for i ∈ 2:nMCMC
    u₁, u₂ = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i-1])
    β[i,:] = sampleβ(X, y, u₁, u₂, β[i-1,:], α, θ[i-1], σ[i-1], 100.)
    σ[i], _ = sampleσ(X, y, u₂, u₂, β[i,:], α, θ[i-1], 1, 1.)
    θ[i] = sampleθ(θ[i-1], .1, X, y, u₁, u₂, β[i, :], α, σ[i])
end

plot(σ)
plot(cumsum(σ)./(1:nMCMC))
median(σ)
plot(β[:, 1])
plot(θ)
