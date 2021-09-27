using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, Turing
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

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

ids = sample(1:length(y), length(y)-100; replace = false)
trainX, trainy = X[ids,:], y[ids]
testX, testy = X[Not(ids),:], y[Not(ids)]

par = Sampler(trainy, trainX, 0.5, 90000, 30, 10000);
βinit = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]
βinit = [-1.5, -.1, -3.5, 3.7, 0., 0.15, 1.88, 0., 0.36]
β, θ, σ = mcmc(par, 10000., 1., 1.4, βinit, 1.6, 1.1);
βest = [median(β[:,i]) for i in 1:9]

βest = coef(ResultQR)

Q = zeros(100)
for i in 1:100
    Q[i] = testy[i] <= ceil(exp(testX[i,:] ⋅ βest) + 0.5 - 1)
end
mean(Q)
