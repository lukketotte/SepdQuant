using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, QuantileRegressions
include("../../QuantileReg/QuantileReg.jl")
using .QuantileReg
using Random
using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)
# using Formatting

## All covariates
dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

Random.seed!(12)

ids = sample(1:length(y), length(y)-100; replace = false)
trainX, trainy = X[ids,:], y[ids]
testX, testy = X[Not(ids),:], y[Not(ids)]

par = Sampler(trainy, trainX, 0.8, 21000, 5, 1000);
#βinit = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]
#βinit = [-1.5, -.1, -3.5, 3.7, 0., 0.15, 1.88, 0., 0.36]
#β, θ, σ = mcmc(par, 10000., 1., 1.4, βinit, 1.6, 1.1);
βinit = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, par.α) |> coef
β, θ, σ = mcmc(par, 0.4, 0.4, 4., 1.4, βinit); # α = 0.5
#β, θ, σ = mcmc(par, 0.4, 0.3, 1., 1., βinit); # α = 0.1
#β, θ, σ = mcmc(par, 0.4, 0.1, 1., 1., βinit); # α = 0.9
[mean(β[:,i]) for i in 1:9] |> println
plot(β[:,4])
1-((β[2:size(β, 1), 1] .=== β[1:(size(β, 1) - 1), 1]) |> mean)

βest = [mean(β[:,i]) for i in 1:9]

Q = zeros(100)
Q2 = zeros(100)
for i in 1:100
    Q[i] = testy[i] <= ceil(exp(testX[i,:] ⋅ βest) + par.α - 1)
    Q2[i] = testy[i] <= ceil(exp(testX[i,:] ⋅ βinit) + par.α - 1)
end
mean(Q)
mean(Q2)
