using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, KernelDensity, CSVFiles
theme(:juno)

## test
n = 1000;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(aepd(0., σ^(1/θ), θ, α), n);

par = MCMCparams(y, X, 10000, 1, 1000)
β, θ, σ = mcmc(par, 0.5, 100., 0.05, [2.1, 0.8], 2., 1.)

plot(β[:,2])
plot(θ, label="θ")
plot(σ, label="σ")

1-((β[2:nMCMC, 1] .=== β[1:(nMCMC - 1), 1]) |> mean)
1-((b[2:length(o), 1] .=== b[1:(length(o) - 1), 1]) |> mean)

## Conflict
dat = load(string(pwd(), "/Tests/data/nsa_ff.dta")) |> DataFrame;
cols = [:best_fatality, :foreign_f, :intensity, :pop_dens_ln, :fatality_lag_ln, :ethnic, :rebstrength, :loot, :length,
    :territorial, :govtbestfatal_ln];
dat = dat[:, cols];
dat = dat[:, Not(filter(c -> count(ismissing, dat[:,c])/size(dat,1) > 0., names(dat)))];

dat[findall(y.>0), "foreign_f"] |> mean

y = Int64.(dat."best_fatality");
X = Float64.(dat[:, Not("best_fatality")] |> Matrix);
X = X[findall(y.>0),:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);
# θ stepsizes : (0.5, 0.05), (0.9, .5), (0.1, 0.7)


inv(X'*X)*X'*log.(y)
par = MCMCparams(log.(y), X, 100000, 1, 20000);
β, θ, σ = mcmc(par, 0.5, 100., 0.05, 0.0165, inv(X'*X)*X'*log.(y), 2., 1.);

"""
α = 0.1, ε = 0.7
β₁ = [180.09, 0.494, 177.71, 177.58, -177.06, 0.06, 0.005, 0.46, 0.159]
θ₁ = 0.102
σ₁ = 0.122

α = 0.9, ε = 0.75, thin=5
β₁ = [-99.741, 51.35, -51.145, -1.379, 52.487, -0.167, 0.033, 51.003, 0.091]
θ₁ = 0.154
σ₁ = 0.19
"""
par = MCMCparams(y, X, 200000, 5, 1);
β, θ, σ = mcmc(par, 0.9, 100., 0.75, βest, 0.154, 0.19); # scale mix rep

p = 2
plot(1:length(θ), β[:,p])
plot!(1:length(θ), cumsum(β[:,p])./(1:length(θ)), lw=3)

plot(σ)
plot(cumsum(σ)./(1:length(θ)))

plot(θ)
plot(cumsum(θ)./(1:length(θ)))
median(σ)
1-((β[2:length(θ), 1] .=== β[1:(length(θ) - 1), 1]) |> mean)
1-((θ[2:length(θ)] .=== θ[1:(length(θ) - 1)]) |> mean)

βest = Float64[]
for b in eachcol(β)
    append!(βest, median(b))
end

println(round.(βest, digits = 3))

## BostonHousing
dat = load(string(pwd(), "/Tests/data/BostonHousing.csv")) |> DataFrame
y = dat[!, "medv"]
X = dat[!, Not(:medv)] |> Matrix
# X = hcat([1 for i in 1:length(y)], X)
# works much better over all α
# θ stepsizes : (0.5, 0.05), (0.8, 0.05), (0.2, ?)

par = MCMCparams(y, X, 500000, 10, 200000)
β, θ, σ = mcmc(par, 0.5, 100., 0.05, 0.00071, nothing, 3., .8) # MH step

par = MCMCparams(y, X, 1000000, 10, 200000)
β, θ, σ = mcmc(par, 0.9, 100., 0.05, nothing, 3., .8) # using scale mixture

plot(β[:,8])
plot(σ)
plot(θ)
plot(cumsum(θ)./(1:length(θ)))
plot(cumsum(β[:,11])./(1:length(θ)))

1-((β[2:length(θ), 1] .=== β[1:(length(θ) - 1), 1]) |> mean)
