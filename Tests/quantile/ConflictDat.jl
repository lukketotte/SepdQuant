using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:default)

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]

## All covariates
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[findall(y.>0),:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

names(dat) |> println

par = MCMCparams(y, X, 500000, 4, 100000);
ε = [0.09, 0.02, 0.02, 0.02, 0.00065, 0.02, 0.02, 0.00065, 0.006]

β1, θ1, σ1 = mcmc(par, 0.1, 100., 0.05, ε.*0.4, inv(X'*X)*X'*log.(y), 2., 1., true);

β_01, θ_01, σ_01 = β1, θ1, σ1

1-((β1[2:length(θ1), 1] .=== β1[1:(length(θ1) - 1), 1]) |> mean)
plot(β1[:,6])
plot(θ1)
plot(1:length(θ1), cumsum(θ1)./(1:length(θ1)))

p = 2
plot(1:length(θ1), cumsum(β1[:,p])./(1:length(θ1)))

median(β_01[:, 1])
median(β_02[:, 1])
median(β_03[:, 1])

β = [β_01, β_02, β_03, β_04, β_05, β_06 ,β_07, β_08, β_09]
θ = [θ_01, θ_02, θ_03, θ_04, θ_05, θ_06 ,θ_07, θ_08, θ_09]
α = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

p = 7
plot([median(b[:,p]) for b in β])

# shape
θpl = plot(α, [median(a) for a in θ], color = "blue",  legend = false)
θpl = plot!(α, [median(a) + √(var(a)) for a in θ], ls=:dash, color="red")
θpl = plot!(α, [median(a) - √(var(a)) for a in θ], ls=:dash, color="red")
θpl = ylabel!("θ")


p = 2
βpl2 = plot(α, [median(a[:,p]) for a in β], color = "blue",  legend = false)
βpl2 = plot!(α, [median(a[:,p]) + √(var(a[:,p])) for a in β], ls=:dash, color="red")
βpl2 = plot!(α, [median(a[:,p]) - √(var(a[:,p])) for a in β], ls=:dash, color="red")
βpl2 = ylabel!("troopLag")

p = 7
βpl7 = plot(α, [median(a[:,p]) for a in β], color = "blue",  legend = false)
βpl7 = plot!(α, [median(a[:,p]) + √(var(a[:,p])) for a in β], ls=:dash, color="red")
βpl7 = plot!(α, [median(a[:,p]) - √(var(a[:,p])) for a in β], ls=:dash, color="red")
βpl7 = ylabel!("incomp")
βpl7 = xlabel!("α")

p = 9
βpl9 = plot(α, [median(a[:,p]) for a in β], color = "blue",  legend = false)
βpl9 = plot!(α, [median(a[:,p]) + √(var(a[:,p])) for a in β], ls=:dash, color="red")
βpl9 = plot!(α, [median(a[:,p]) - √(var(a[:,p])) for a in β], ls=:dash, color="red")
βpl9 = ylabel!("lntpop")
βpl9 = xlabel!("α")


plot(θpl, βpl2, βpl7, βpl9, layout=(2,2))
plot!(size=(500,450))
savefig("test")
names(dat) |> println
## Excluding some covariates
X = dat[:, Not(["osvAll", "policeLag", "militaryobserversLag"])] |> Matrix
X = hcat([1 for i in 1:length(y)], X);
y = y[y.>0];

par = MCMCparams(y, X, 500000, 4, 100000);
ε = [0.09, 0.015, 0.00065, 0.015, 0.015, 0.00065, 0.006]
β1, θ1, σ1 = mcmc(par, 0.5, 100., 0.05, ε, inv(X'*X)*X'*log.(y), 2., 1., true);
