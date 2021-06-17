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

## generate data
n = 5000;
β, α, σ = [2.1, 0.8], 0.5, 2.2;
θ = 1.3
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(aepd(0., σ, θ, α), n);
# β, σ, θ = mcmc(y, X, 0.5, 10, θinit = 1.0, σinit = 1.5)

##
θ = 2.
z = y - X*β
pos = findall(z .> 0)
b = (δ(α, θ) * sum(abs.(z[Not(pos)]).^θ) / α^θ) + (δ(α, θ) * sum(abs.(z[pos]).^θ) / (1-α)^θ)
(rand(InverseGamma(n/θ, b), 1)[1])^(1/θ)


##

nMCMC = 30000
β = zeros(nMCMC, 2)
β[1,:] = [2.1, 0.8]
σ, θ = zeros(nMCMC), zeros(nMCMC)
σ[1] = 2.2
θ[1] = 1.3
# seems like θ goes towards 1 with this sampling order
for i ∈ 2:nMCMC
    if i % 5000 === 0
        println("iter: ", i)
    end
    z = y - X*β[i-1,:]
    pos = findall(z .> 0)
    # using jacobian u = σ^p https://barumpark.com/blog/2019/Jacobian-Adjustments/
    b = (δ(α, θ[i-1]) * sum(abs.(z[Not(pos)]).^θ[i-1]) / α^θ[i-1]) + (δ(α, θ[i-1]) * sum(abs.(z[pos]).^θ[i-1]) / (1-α)^θ[i-1])
    σ[i] = (rand(InverseGamma(n/θ[i-1], b), 1)[1])^(1/θ[i-1])
    # σ[i] = 1
    global u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i])
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
    # θ[i] = 2.
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i], σ[i], 100.)
end

plot(θ, label="θ")
plot(σ, label="σ")

# thin = ((1:nMCMC) .% 30) .=== 0
plot(β[:, 1], label="trace")
plot!(cumsum(β[:, 1])./(1:length(β[:, 1])), label="running")
plot(σ, label="running")
plot!(cumsum(σ)./(1:length(σ)), label="σ")

median(β[50000:nMCMC, 2])
mean(σ)
mean(θ)

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
