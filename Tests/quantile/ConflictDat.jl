using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, Turing, QuantileRegressions
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

# par = Sampler(y, X, 0.5, 150000, 20, 10000);

# α = 0.1: thin = 30, ϵ = .9, ϵθ = 1.
# α = 0.2: thin = 30, ϵ = 1.2
# α = 0.3: thin = 30, ϵ = 1.3
# α = 0.4: thin = 30, ϵ = 1.5
# α = 0.5: thin = 30, ϵ = 1.5
# α = 0.6: thin = 30, ϵ = 1.6
# α = 0.7: thin = 30, ϵ = 1.6
# α = 0.8: thin = 30, ϵ = 1.4
# α = 0.9: thin = 10, ϵ = .6

α = 0.9
par = Sampler(y, X, α, 1000, 5, 1);

βinit = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, α) |> coef
#par = Sampler(y, X, 0.7, 200000, 5, 100000);
β, θ, σ = mcmc(par, 0.4, 0.2, 1., 1., βinit);
#β, θ, σ = mcmc(par, 1000., .8, 0.4, 0.9, 1.1, βinit);
#βa, σa = mcmc(par, 1000., 0.5, βinit, 1);

1-((β[2:size(β, 1), 1] .=== β[1:(size(β, 1) - 1), 1]) |> mean)

acceptance(θ::AbstractMatrix{<:Real}) = size(θ, 1) |> n -> 1-((θ[2:n, 1] .=== θ[1:(n - 1), 1]) |> mean)
acceptance(θ::AbstractVector{<:Real}) = length(θ) |> n -> 1-((θ[2:n] .=== θ[1:(n - 1), 1]) |> mean)

acceptance(β)
acceptance(θ)

mean(β, dims = 1)
βinit
p = 1
plot(β[:, 7])
#plot(βa[:, 2])
plot(θ)
plot(cumsum(β[:,9])./(1:length(θ)))

b = [mean(β[:,i]) for i in 1:9]
println(b)

Q = zeros(length(y))
for i ∈ 1:length(y)
    Q[i] = y[i] <= ceil(exp(X[i,:] ⋅ b) + 0.5)
end
mean(Q)
mean(y <= ceil.(exp.(X * b) .+ 0.5))
##
chain = Chains(β, ["intercept";names(dat[:, Not(["osvAll"])])]);
mean(summarystats(chain)[:, :ess]) / length(θ)

##
p = 3
plot(β[:, 1])
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
