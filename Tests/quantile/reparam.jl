using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, StaticArrays
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles, KernelDensity
theme(:juno)

## test
n = 1000;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(aepd(0., σ^(1/θ), θ, α), n);

par = MCMCparams(y, X, 50000, 5, 1000)
β, θ, σ = mcmc(par, 0.5, 100., 0.05, [0.05, 0.001], [2.1, 0.8], 2., 1.)

plot(β[:,1])
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
inv(X'*X)*X'*log.(y)
# y = SArray{n}(y)
# X = SMatrix{260,10}(X)

typeof(par.y[1]) <: Integer

par = MCMCparams(log.(y), X, 500000, 20, 100000)
β, θ, σ = mcmc(par, 0.5, 100., 0.05, 0.0165, nothing, 2., 1.)

plot(β[:,8])
plot(σ)
plot(θ)
1-((β[2:length(θ), 1] .=== β[1:(length(θ) - 1), 1]) |> mean)

βest = []
for b in eachcol(β)
    append!(βest, median(b))
end

println(βest)

nMCMC = 1000000
β = zeros(nMCMC, 10)
β[1,:] = inv(X'*X)*X'*y₁
σ, θ = zeros(nMCMC), zeros(nMCMC)
σ[1] = 1.
θ[1] = 1.

for i ∈ 2:nMCMC
    if i % 10000 === 0
        println("iter: ", i)
    end
    global y = log.((exp.(y₁) + rand(Uniform(), length(y₁)) .- α))
    θ[i] = sampleθBlock(θ[i-1], X, y, β[i-1,:], α, 0.05)
    σ[i] = sampleσBlock(X, y, β[i-1,:], α, θ[i])
    global u1, u2 = sampleLatentBlock(X, y, β[i-1,:], α, θ[i], σ[i])
    β[i,:] = sampleβBlock(X, y, u1, u2, β[i-1,:], α, θ[i], σ[i], 100.)
end

thin = ((1:nMCMC) .% 20) .=== 0
plot(θ[thin], label="θ")
plot(σ[thin], label="σ")
plot(β[thin,10], label="β")

plot(cumsum(β[:,10]) ./ (1:length(σ)))
plot(cumsum(θ) ./ (1:length(σ)))
median(β[thin, 7])


p = 10
b1 = kde(β[thin, p])
x = range(median(β[thin, p])-1, median(β[thin, p])+1, length = 1000)
plot(x, pdf(b1, x))
