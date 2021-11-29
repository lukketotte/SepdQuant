using Distributions, QuantileRegressions, LinearAlgebra, Random, SpecialFunctions, QuadGK
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles, KernelDensity
theme(:juno)

using RDatasets

## QuantileReg data
RDatasets.datasets("datasets")
dat = dataset("datasets", "mtcars")

y = log.(dat[:, :MPG])
X = log.(dat[:, Not(["FoodExp"])] |> Matrix)
X = hcat([1. for i in 1:length(y)], dat[:, :WT]);

par = Sampler(y, X, 0.5, 100000, 5, 1000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3), x, 0.5) |> coef;
println(b)
median(β, dims = 1) |> println
β, θ, σ, α = mcmc(par, 0.01, 0.1, 0.1, 2, 1, 0.5, b);
acceptance(β)
plot(β[:,2])
plot(σ)
plot(θ)
plot(α)
acceptance(θ)



b = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3), x, 0.1) |> coef;
q = X * b;
μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));
τ = [quantconvert(q[j], median(θ), median(α), μ[j],
    median(σ)) for j in 1:length(par.y)] |> mean

par = Sampler(y, X, τ, 10000, 2, 5000);
βres, _ = mcmc(par, 1.3, median(θ), median(σ), b);
plot(βres[:,1])

[par.y[i] <= X[i,:] ⋅ b for i in 1:length(y)] |> mean
[par.y[i] <= X[i,:] ⋅ median(βres, dims = 1)  for i in 1:length(par.y)] |> mean

## Boston
#dat = load(string(pwd(), "/Tests/data/BostonHousing2.csv")) |> DataFrame;
dat = dataset("MASS", "Boston")
y = log.(dat[:, :MedV])
X = dat[:, Not(["MedV"])] |> Matrix
X = hcat([1 for i in 1:length(y)], X);

par = Sampler(y, X, 0.5, 3000, 1, 1);
b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 +
        x11 + x12 + x13 + x14 + x15), x, 0.5) |> coef;

β, θ, σ, α = mcmc(par, 0.25, .25, 0.05, 1, 2, 0.5, b);

acceptance(β)
acceptance(θ)
acceptance(α)
plot(α)
plot(θ)
plot(β[:,6])
