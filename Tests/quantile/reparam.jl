using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, StaticArrays
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
dat = load(string(pwd(), "/Tests/data/nsa_ff.dta")) |> DataFrame
dat = dat[:, Not(filter(c -> count(ismissing, dat[:,c])/size(dat,1) > 0.05, names(dat)))]
dropmissing!(dat)

y = Float64.(dat."fatality_lag_ln")
colSub = [:intensity, :pop_dens_ln, :foreign_f, :ethnic, :rebstrength, :loot,
    :territorial,  :length, :govtbestfatal_ln]
X = Float64.(dat[:, colSub] |> Matrix)
y = y[y.>0]
X = X[findall(y.>0),:]
X = hcat([1 for i in 1:260], X)
α, n = 0.5, length(y)
y = trunc.(Int, exp.(y))

# θ stepsizes : (0.5, 0.05), (0.8, ?), (0.2, ?)

par = MCMCparams(log.(y), X, 500000, 20, 100000);
β, θ, σ = mcmc(par, 0.5, 100., 0.05, 0.0165, nothing, 2., 1.);

par = MCMCparams(log.(y), X, 100000, 1, 1);
β, θ, σ = mcmc(par, 0.1, 100., 0.1, nothing, .5, .3); # scale mix rep

plot(β[:,10])
plot(σ)
plot(θ)
plot(cumsum(θ)./(1:length(θ)))
plot(cumsum(σ)./(1:length(θ)))
1-((β[2:length(θ), 1] .=== β[1:(length(θ) - 1), 1]) |> mean)
1-((θ[2:length(θ)] .=== θ[1:(length(θ) - 1)]) |> mean)

βest8 = []
for b in eachcol(β)
    append!(βest8, median(b))
end

println(βest8)

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
