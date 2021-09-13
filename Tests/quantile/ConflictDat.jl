using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)
# using Formatting

# TODO: marginal effects plot
# TODO: θᵢ = (2, 5, 6, 7, 9)'ξ ...ev. log(brv_AllLag + 1)


## All covariates
dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

par = Sampler(y, X, 0.9, 20000, 5, 10000);
βinit = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]
β, θ, σ = mcmc(par, 100., .8, .25, βinit, 0.6, 1.1);

#α = 0.9, ε =  0.15
#α = 0.1, ε =  0.25

p = 2
plot(β[:, p])
plot!(1:length(θ), cumsum(β[:,p])./(1:length(θ)))
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
