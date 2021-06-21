using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
using ForwardDiff
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles

theme(:juno)

function sampleLatentBlock(X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T, θ::T, σ::T) where {T <: Real}
    u₁, u₂ = zeros(n), zeros(n)
    μ = X*β
    for i ∈ 1:n
        if y[i] <= μ[i]
            l = ((μ[i] - y[i]) / (σ^(1/θ) * α))^θ
            u₁[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        else
            l = ((y[i] - μ[i]) / (σ^(1/θ) * (1-α)))^θ
            u₂[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        end
    end
    u₁, u₂
end

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

function sampleσBlock(X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T, θ::T) where {T, N <: Real}
    z = y - X*β
    pos = findall(z .> 0)
    b = (δ(α, θ) * sum(abs.(z[Not(pos)]).^θ) / α^θ) + (δ(α, θ) * sum(abs.(z[pos]).^θ) / (1-α)^θ)
    rand(InverseGamma(length(y)/θ, b), 1)[1]
end

function logβCond(β::Array{T, 1}, X::Array{T, 2}, y::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    z = y - X*β
    pos = findall(z .> 0)
    b = δ(α, θ)/σ * (sum(abs.(z[Not(pos)]).^θ) / α^θ + sum(abs.(z[pos]).^θ) / (1-α)^θ)
    -b -1/(2*τ) * β'⋅β
end


function βMh(β::Array{T, 1}, ε::T,  X::Array{T, 2}, y::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    # prop = vec(rand(MvNormal(β, ε), 1))
    # try MALA sampling
    z = y - X*β
    ∇ = δ(α, θ)/σ .* β .* (sum(z .< 0)/α^θ - sum(z .> 0)/(1-α)^θ - 1/τ)
    μ = β + ε^2/2 .* ∇
    prop = vec(rand(MvNormal(μ, ε), 1))
    βCond(prop, X, y, α, θ, σ, 100.) - βCond(β, X, y, α, θ, σ, 100.) > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

function sampleβBlock(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    n, p = size(X)
    βsim = zeros(p)
    for k in 1:p
        l, u = [-Inf], [Inf]
        for i in 1:n
            a = (y[i] - X[i, 1:end .!= k] ⋅  β[1:end .!= k]) / X[i, k]
            b₁ = α*σ^(1/θ)*(u₁[i]^(1/θ)) / X[i, k]
            b₂ = (1-α)*σ^(1/θ)*(u₂[i]^(1/θ)) / X[i, k]
            if (u₁[i] > 0) && (X[i, k] < 0)
                append!(l, a + b₁)
            elseif (u₂[i] > 0) && (X[i, k] > 0)
                append!(l, a - b₂)
            elseif (u₁[i] > 0) && (X[i, k] > 0)
                append!(u, a + b₁)
            elseif (u₂[i] > 0) && (X[i, k] < 0)
                append!(u, a - b₂)
            end
        end
        βsim[k] =  maximum(l) < minimum(u) ? rand(truncated(Normal(0, τ), maximum(l), minimum(u)), 1)[1] : β[k]
    end
    βsim
end

## test
n = 200;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(aepd(0., σ^(1/θ), θ, α), n);

nMCMC = 10000
β = zeros(nMCMC, 2)
β[1,:] = [2.1, 0.8]
σ, θ = zeros(nMCMC), zeros(nMCMC)
σ[1] = 2^(1/1.)
θ[1] = 1.
# seems like θ goes towards 1 with this sampling order
for i ∈ 2:nMCMC
    if i % 5000 === 0
        println("iter: ", i)
    end
    θ[i] = sampleθBlock(θ[i-1], X, y, β[i-1,:], α, 0.05)
    σ[i] = sampleσBlock(X, y, β[i-1,:], α, θ[i])
    global u1, u2 = sampleLatentBlock(X, y, β[i-1,:], α, θ[i], σ[i])
    # β[i,:] = sampleβBlock(X, y, u1, u2, β[i-1,:], α, θ[i], σ[i], 100.)
    β[i,:] = [2.1, 0.8]
    # β[i,:] = βMh(β[i-1,:], 0.01, X, y, α, θ[i], σ[i], 100.)
end

plot(θ, label="θ")
plot(σ, label="σ")
plot(β[:, 2], label="β")

median(β[1000:nMCMC, 1])
median(σ[1000:nMCMC])
median(θ[1000:nMCMC])
