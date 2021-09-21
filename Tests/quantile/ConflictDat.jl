using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
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

par = Sampler(y, X, 0.5, 100000, 20, 20000);
βinit = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]
β, θ, σ = mcmc(par, 100., 1., 1.7, βinit, 1.5, 1.1);
inits = [median(β[:,i]) for i in 1:9]
##
using Turing, StatsPlots
chain = Chains(β, ["intercept";names(dat[:, Not(["osvAll"])])]);
chain2 = Chains([θ σ], ["θ", "σ"]);

summaries = summarystats(chain)
summaries2 = summarystats(chain2)
mean(summarystats(chain)[:, :ess]) / length(θ)

(length(θ) / (1+2*sum(autocor(β[:,1])))) / length(θ)
plot(chain)
plot(chain2)


##
p = 2
plot(β[:, 8])
1-((β[2:length(θ), 1] .=== β[1:(length(θ) - 1), 1]) |> mean)
plot(θ)
## Estimate over multiple quantiles
colnames = names(dat)
colnames[1] = "intercept"
feMCMC = FormatExpr("mcmc_{}.csv")
α = range(0.1, 0.9, length = 17)
βinit = [-1.85, -0.046, -2.477, 2.628, 0.0, 0.412, 1.522, -0.003, 0.206]
par = MCMCparams(y, X, 200000, 5, 100000);
for a in α
    println(a)
    β, θ, σ = mcmc(par, 0.9, 100., .8, .25, βinit, 3, 1.5, true)
    CSV.write(format(feMCMC, 0.9), hcat(DataFrame(β, colnames), DataFrame([σ, θ], ["σ", "θ"])))
end
