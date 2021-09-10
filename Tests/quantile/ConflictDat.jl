using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)

# TODO: marginal effects plot
# TODO: θᵢ = (2, 5, 6, 7, 9)'ξ ...ev. log(brv_AllLag + 1)

using Formatting
fe = FormatExpr("β_{}.csv")
format(fe, 0.05)


## All covariates
dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

colnames = names(dat)
colnames[1] = "intercept"


par = MCMCparams(y, X, 1000, 5, 1);
βinit = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]
β, θ, σ = mcmc(par, 0.5, 100., .8, .25, βinit, 3, 1.5, true);



hcat(DataFrame(β, colnames), DataFrame([σ, θ], ["σ", "θ"]))


convert(DataFrame, β)
DataFrame(β, colnames)
DataFrame(β, :auto)
p = 2
plot(β[:,p])
plot!(1:length(θ), cumsum(β[:,p])./(1:length(θ)))
plot!(β1[:,p])
plot!(1:length(θ), cumsum(β1[:,p])./(1:length(θ)))

p = 9
plot(1:length(θ), cumsum(β2[:,p])./(1:length(θ)), label = "α = 0.1")
plot!(1:length(θ), cumsum(β[:,p])./(1:length(θ)), label = "α = 0.5")
plot!(1:length(θ), cumsum(β1[:,p])./(1:length(θ)), label = "α = 0.9")


1-((β[2:length(θ), 1] .=== β[1:(length(θ) - 1), 1]) |> mean)
round.([median(β[:,i]) for i in 1:9], digits = 3) |> println
1-((θ[2:length(θ)] .=== θ[1:(length(θ) - 1), 1]) |> mean)
plot(θ, label = "α = 0.5")
plot!(θ1, label = "α = 0.9")
plot!(σ, label = "σ")


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
# format(fe, 0.05)
