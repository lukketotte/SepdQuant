@everywhere using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
@everywhere include(joinpath(@__DIR__, "../aepd.jl"))
@everywhere include(joinpath(@__DIR__, "../../QuantileReg/QuantileReg.jl"))
@everywhere using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)

α = range(0.1, 0.9, length = 17)

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

par = MCMCparams(y, X, 1000, 5, 1);
βinit = [-1.85, -0.046, -2.477, 2.628, 0.0, 0.412, 1.522, -0.003, 0.206]
res = pmap(a -> mcmc(par, a, 100., .8, .25, βinit, 3, 1.5, true), α)
