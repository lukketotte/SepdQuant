using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles
using KernelDensity
theme(:juno)

##
function θBlockCond(θ::T, X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T) where {T <: Real}
    n = length(y)
    z  = y-X*β
    pos = findall(z .> 0)
    a = δ(α, θ)*(sum(abs.(z[Not(pos)]).^θ)/α^θ + sum(z[pos].^θ)/(1-α)^θ)
    n/θ * log(δ(α, θ))  - n*log(gamma(1+1/θ)) - n*log(a)/θ + loggamma(n/θ)
end

function sampleθBlock(θ::T, X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1},
    α::T, ε::T) where {T <: Real}
    prop = rand(Uniform(maximum([0., θ-ε]), θ + ε), 1)[1]
    θBlockCond(prop, X, y, β, α) - θBlockCond(θ, X, y, β, α) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

## generate data
n = 2000;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ = 0.7
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(aepd(0., σ, θ, α), n);
# β, σ, θ = mcmc(y, X, 0.5, 10, θinit = 1.0, σinit = 1.5)

Θ = range(0.1, 3., length = 2000)
pplot = [θBlockCond(p, X, y, β, α) for p in Θ]
plot(Θ, pplot)

##

nMCMC = 100000
β = zeros(nMCMC, 2)
β[1,:] = [2.1, 0.8]
σ, θ = zeros(nMCMC), zeros(nMCMC)
σ[1] = 2.
θ[1] = 1.
# seems like θ goes towards 1 with this sampling order
for i ∈ 2:nMCMC
    if i % 5000 === 0
        println("iter: ", i)
    end
    θ[i] = sampleθBlock(θ[i-1], X, y, β[i-1,:], α, 0.01)
    z = y - X*β[i-1,:]
    pos = findall(z .> 0)
    # using jacobian u = σ^p https://barumpark.com/blog/2019/Jacobian-Adjustments/
    b = (δ(α, θ[i]) * sum(abs.(z[Not(pos)]).^θ[i]) / α^θ[i]) + (δ(α, θ[i]) * sum(abs.(z[pos]).^θ[i]) / (1-α)^θ[i])
    σ[i] = (rand(InverseGamma(n/θ[i], b), 1)[1])^(1/θ[i])
    # σ[i] = 1
    global u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i], σ[i])
    """
    try
        θ[i] = sampleθ(θ[i-1], 1., X, y, u1, u2, β[i-1, :], α, σ[i])
    catch e
        if isa(e, ArgumentError)
            θ[i] = θ[i-1]
            break
        else
            println(e)
        end
    end
    """
    # θ[i] = 2.
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i], σ[i], 100.)
end

plot(θ, label="θ")
plot!(σ, label="σ")

# thin = ((1:nMCMC) .% 30) .=== 0
plot(β[:, 1], label="trace")
plot!(cumsum(β[:, 1])./(1:length(β[:, 1])), label="running")
plot(σ, label="running")
plot!(cumsum(σ)./(1:length(σ)), label="σ")

median(β[50000:nMCMC, 2])
mean(σ[10000:nMCMC])
mean(θ[10000:nMCMC])


u1, u2 = sampleLatent(X, y, β, α, θ, σ)
β = sampleβ(X, y, u1, u2, β, α, θ, σ, 100.)
σ,_ = sampleσ(X, y, u1, u2, β, α, θ, 1, 1.)

##

autocor(θ, [1,3,10,40]) |> println
thin = ((1:nMCMC) .% 30) .=== 0
autocor(θ[thin], [1,3,10,40]) |> println
# Ess
1-((θ[2:nMCMC] .=== θ[1:(nMCMC - 1)]) |> mean)

plot(cumsum(σ) ./ (1:length(σ)))
median(θ[10000:nMCMC])
median(θ[thin])


"""
CSV.write("beta.csv", DataFrame(β), header = false)
CSV.write("theta.csv", DataFrame(reshape(θ, nMCMC, 1)), header = false)
CSV.write("sigma.csv", DataFrame(reshape(σ, nMCMC, 1)), header = false)
"""
