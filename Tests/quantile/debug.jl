using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff
using Plots, PlotThemes
theme(:juno)

include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg


n = 200;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(Normal(0., 1.), n);

α = 0.5
par = Sampler(y, X, .5, 10000, 1, 1);
β, θ, σ = mcmc(par, 0.5, 0.5, 2, 2);

plot(θ)
plot(σ)
plot(β[:,1])
