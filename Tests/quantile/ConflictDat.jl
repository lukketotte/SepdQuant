using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]

## All covariates
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[findall(y.>0),:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

names(dat) |> println

par = MCMCparams(y, X, 2*500000, 4, 100000);
ε = [0.09, 0.02, 0.02, 0.02, 0.00065, 0.02, 0.02, 0.00065, 0.006]
β1, θ1, σ1 = mcmc(par, 0.5, 100., 0.05, ε, inv(X'*X)*X'*log.(y), 2., 1., true);


1-((β1[2:length(θ1), 1] .=== β1[1:(length(θ1) - 1), 1]) |> mean)
plot(β1[:,4])
plot(θ1)

p = 9
plot(1:length(θ1), cumsum(β1[:,p])./(1:length(θ1)))

## Excluding some covariates
X = dat[:, Not(["osvAll", "policeLag", "militaryobserversLag"])] |> Matrix
X = hcat([1 for i in 1:length(y)], X);
y = y[y.>0];

par = MCMCparams(y, X, 500000, 4, 100000);
ε = [0.09, 0.015, 0.00065, 0.015, 0.015, 0.00065, 0.006]
β1, θ1, σ1 = mcmc(par, 0.5, 100., 0.05, ε, inv(X'*X)*X'*log.(y), 2., 1., true);
