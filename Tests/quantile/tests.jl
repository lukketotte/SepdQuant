using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ProgressMeter, ForwardDiff
using CSV, DataFrames, StatFiles, CSVFiles, Plots
include("../../QuantileReg/QuantileReg.jl")
using .QuantileReg

dat = load(string(pwd(), "/Tests/data/BostonHousing2.csv")) |> DataFrame;
names(dat)


y = log.(dat[:, :MEDV])
X = dat[:, Not(["MEDV"])] |> Matrix
X = hcat([1 for i in 1:length(y)], X);



par = Sampler(y, X, 0.5, 20000, 5, 1);
βinit = [-1.3851, -0.0509, 0.0237, -0.0109, 0.0008, 0.0024, 0.0820, -0.4540, 0.1663, -0.0007, -0.0403,  0.0106, -0.0005, -0.0326,  0.0006, -0.0229]
β, θ, σ = mcmc(par, 100., 0.5, 0.01, βinit, 1., 1.);
1-((θ[2:length(σ)] .=== θ[1:(length(σ) - 1)]) |> mean)
1-((β[2:length(σ), 1] .=== β[1:(length(σ) - 1), 1]) |> mean)

plot(θ)
median(θ)
plot(β[:, 1])
[mean(β[:,i]) for i in 1:size(X)[2]]

DataFrame(val = [median(β[:,i]) for i in 1:size(X)[2]],
    param = append!(["idx"], names(dat[:, Not(["MEDV"])]))) |> println

DataFrame(val = [median(β[:,i]) for i in 1:size(X)[2]], param = names(dat[:, Not(["MEDV"])])) |> println


names(dat[:, Not(["medv"])]) |> println;
append!(["idx"], names(dat[:, Not(["MEDV"])]))
