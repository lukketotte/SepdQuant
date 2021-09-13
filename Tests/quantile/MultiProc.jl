using Distributed, SharedArrays
@everywhere using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
@everywhere include(joinpath(@__DIR__, "../aepd.jl"))
@everywhere include(joinpath(@__DIR__, "../../QuantileReg/QuantileReg.jl"))
@everywhere using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)

using Formatting

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

par = Sampler(y, X, 0.5, 100, 1, 1);
βinit = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]

M = 5
par = Sampler(y, X, 0.5, 200000, 5, 100000);

N = Integer((par.nMCMC - par.burnIn)/par.thin) + 1

β = SharedArray{Float64}((N, size(X)[2], M))
σ = SharedArray{Float64}((N,M))
θ = SharedArray{Float64}((N,M))

α = range(0.1, 0.9, length = 9)

@distributed for i = 1:20
    βt,θt,σt = mcmc(par, 100., .8, .25, βinit, 0.5, 1.)
    β[:,:,i] = βt
    θ[:,i] = θt
    σ[:,i] = σt
end

mean([mean(θ[:, i]) for i in 1:M])
[mean([mean(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]]
[mean([√var(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]]

colnames =["intercept" ; names(dat[:,2:9])]


"sig_" .* colnames

append!(colnames, "σ_" .* colnames)

[[mean([mean(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]] ; [mean([√var(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]]]

a = DataFrame(param = colnames,
    value =[mean([mean(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]],
    sd = [mean([√var(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]], α = 0.5)

b = DataFrame(param = ["sigma", "theta"], value = [mean([mean(σ[:, i]) for i in 1:M]), mean([mean(θ[:, i]) for i in 1:M])],
    sd = [mean([√var(σ[:, i]) for i in 1:M]), mean([√var(θ[:, i]) for i in 1:M])], α = 0.5)

print(b)
vcat(a, b)
