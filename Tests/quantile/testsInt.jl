using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles
# using KernelDensity
theme(:juno)

n = 1500;
μ, α, σ = 2.2, 0.5, 2;
θ = 1
y = rand(aepd(μ, σ, θ, α), n);

μ,σ,l,θ = mcmc(y, 0.5, 200000)

plot(σ, label="σ")
# plot!(l)
plot!(μ, label="μ")
plot!(θ, label="θ")

##

u1, u2 = sampleLatent(y, μ, α, θ, σ)
μ = sampleμ(y, u1, u2, α, θ, σ, 100.)
σ = sampleσ(y, u1, u2, μ, α, θ, 1, 1.)
mean(u1)
mean(u2)


nMCMC = 100000
μ = zeros(nMCMC)
μ[1] = mean(y)
σ = zeros(nMCMC)
σ[1] = 5.3# √var(y)
θ = zeros(nMCMC)
θ[1] = 1.3
U1, U2 = zeros(nMCMC, n), zeros(nMCMC, n)

for i in 2:nMCMC
    u1, u2 = sampleLatent(y, μ[i-1], α, θ[i-1], σ[i-1])
    U1[i,:], U2[i,:] = u1, u2
    μ[i] = sampleμ(y, u1, u2, α, θ[i-1], σ[i-1], 100.)
    # β[i,:] = [2.1, 0.8]
    σ[i] = sampleσ(y, u1, u2, μ[i], α, θ[i-1], 1, 1.)
    # σ[i] = 8.
    θ[i] = sampleθ(θ[i-1], .1, y, u1, u2, μ[i], α, σ[i])
    # θ[i] = 1.
    if i % 5000 === 0
        interval = θinterval(y, u1, u2, μ[i], α, σ[i])
        printfmt("iter: {1}, θ ∈ [{2:.2f}, {3:.2f}], σ = {4:.2f} \n", i, interval[1], interval[2], σ[i])
    end
end

plot(μ)
plot!(cumsum(μ) ./ (1:nMCMC))

plot(σ)
plot!(cumsum(σ) ./ (1:nMCMC))

plot(θ)
plot!(cumsum(θ) ./ (1:nMCMC))
