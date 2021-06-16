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

u1, u2 = sampleLatent(X, y, β, α, θ, σ)
βsim = [rand(Normal(2.1, 0.05), 1)[1],rand(Normal(0.8, 0.05), 1)[1]]
println(βsim)
sampleσ(X, y, u1, u2, βsim, α, θ, 1, 1.)

n = 500;
μ, α, σ = 2.2, 0.5, 2.;
θ = 1.
y = rand(aepd(μ, σ, θ, α), n);

u1, u2 = sampleLatent(y, μ, α, θ, σ)
sampleσ(y, u1, u2, rand(Normal(2.2, 0.05), 1)[1], α, θ, 1, 1.)
