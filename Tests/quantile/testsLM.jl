using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles
using KernelDensity
theme(:juno)

## Sampling from σ with u1 and u2 integrated out
function δ(α::T, θ::T)::T where {T <: Real}
    2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

z = y - X*[2.08, 0.85]
pos = findall(z .> 0)
b = (δ(α, θ) * sum(abs.(z[Not(pos)]).^θ) / α^θ) + (δ(α, θ) * sum(abs.(z[pos]).^θ) / (1-α)^θ)
rand(InverseGamma(n, b), 1000) |> mean

## generate data
n = 500;
β, α, σ = [2.1, 0.8], 0.5, 1.5;
θ = 1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(aepd(0., σ, θ, α), n);

β, σ, θ = mcmc(y, X, 0.5, 10, θinit = 1.0, σinit = 1.5)

nMCMC = 100
β = zeros(nMCMC, 2)
β[1,:] = [2.1, 0.8]
σ, θ = zeros(nMCMC), zeros(nMCMC)
σ[1] = 2.
θ[1] = 1.
# σ gets sampled higher for θ = 2 than θ = 3
for i ∈ 2:nMCMC
    if i % 1000 === 0
        println("iter: ", i)
    end
    z = y - X*β[i-1,:]
    pos = findall(z .> 0)
    b = (δ(α, θ[i-1]) * sum(abs.(z[Not(pos)]).^θ[i-1]) / α^θ[i-1]) + (δ(α, θ[i-1]) * sum(abs.(z[pos]).^θ[i-1]) / (1-α)^θ[i-1])
    σ[i] = rand(InverseGamma(n, b), 1)[1]
    u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i])
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i-1], σ[i], 100.)
    # σ[i], _ = sampleσ(X, y, u1, u2, β[i, :], α, θ, 1, 1.)
    θ[i] = sampleθ(θ[i-1], .1, X, y, u1, u2, β[i, :], α, σ[i])
    # θ[i] = 1.
end

## possible last piece of puzzle, θinterval
i = 7
u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i])
β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i-1], σ[i-1], 100.)
# sampleθ(θ[i-1], .1, X, y, u1, u2, β[i, :], α, σ[i])
θinterval(X, y, u1, u2, β[i,:], α, σ[i]) |> println

u₁, u₂ = u1, u2

findall((y.-X*β[i,:]) .> 0)

id_pos = findall(((X*β[i,:] .- y) .> 0) .& (u₁ .> 0))
id_neg = findall(((y.-X*β[i,:]) .> 0) .& (u₂ .> 0))
ids1 = id_pos[findall(log.((X[id_pos,:]*β[i,:] - y[id_pos])./(σ[i]*α)) .< 0)]
ids2 = id_pos[findall(log.((X[id_pos,:]*β[i,:] - y[id_pos])./(σ[i]*α)) .> 0)]
ids3 = id_neg[findall(log.((y[id_neg]-X[id_neg,:]*β[i,:])./(σ[i]*(1-α))) .< 0)]
ids4 = id_neg[findall(log.((y[id_neg]-X[id_neg,:]*β[i,:])./(σ[i]*(1-α))) .> 0)]


l1 = length(ids1) > 0 ? maximum(log.(u₁[ids1])./log.((X[ids1,:]*β[i,:] - y[ids1])./(α*σ[i]))) : 0
l2 = length(ids3) > 0 ? maximum(log.(u₂[ids3])./log.(( y[ids3]-X[ids3,:]*β[i,:])./((1-α)*σ[i]))) : 0

findall(log.(u₂[ids3])./log.(( y[ids3]-X[ids3,:]*β[i,:])./((1-α)*σ[i])) .> 1.2)
u₂[27]

(y[27]-X[27,:]⋅β[i,:])./((1-α)*σ[i])

up1 = length(ids2) > 0 ? minimum(log.(u₁[ids2]) ./ log.((X[ids2,:]*β[i,:] - y[ids2])./(α*σ[i]))) : Inf
up2 = length(ids4) > 0 ? minimum(log.(u₂[ids4]) ./log.((y[ids4]-X[ids4,:]*β[i,:])./((1-α)*σ[i]))) : Inf

[maximum([0 l1 l2]) minimum([up1 up2])]


# thin = ((1:nMCMC) .% 30) .=== 0
plot(β[:, 1], label="trace")
plot!(cumsum(β[:, 2])./(1:length(β[:, 1])), label="running")
plot(σ, label="running")
plot!(cumsum(σ)./(1:length(σ)), label="σ")
plot(θ, label="θ")

median(β[50000:nMCMC, 2])
mean(σ)

plot(σ[2:300])
plot!(l[2:300])

u1, u2 = sampleLatent(X, y, β, α, θ, σ)
β = sampleβ(X, y, u1, u2, β, α, θ, σ, 100.)
σ,_ = sampleσ(X, y, u1, u2, β, α, θ, 1, 1.)

##

autocor(θ, [1,3,10,40]) |> println
thin = ((1:nMCMC) .% 10) .=== 0
autocor(θ[thin], [1,3,10,40]) |> println
# Ess
1-((θ[2:nMCMC] .=== θ[1:(nMCMC - 1)]) |> mean)

plot(cumsum(σ) ./ (1:nMCMC))
median(θ[10000:nMCMC])
median(θ[thin])


"""
CSV.write("beta.csv", DataFrame(β), header = false)
CSV.write("theta.csv", DataFrame(reshape(θ, nMCMC, 1)), header = false)
CSV.write("sigma.csv", DataFrame(reshape(σ, nMCMC, 1)), header = false)
"""
