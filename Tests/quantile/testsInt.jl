using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles
# using KernelDensity
theme(:juno)

n = 2000;
μ, α, σ = 2.2, 0.5, 2.;
θ = 1.
y = rand(aepd(μ, σ, θ, α), n);

μ,σ,l,θ = mcmc(y, 0.5, 50000, θinit = 1.)

plot(σ, label="σ")
# plot!(l)
plot(μ, label="μ")
plot(θ, label="θ")

plot(μ)
plot!(cumsum(μ) ./ (1:length(μ)))

plot(σ)
plot!(cumsum(σ) ./ (1:nMCMC))

plot(θ)
plot!(cumsum(θ) ./ (1:nMCMC))
