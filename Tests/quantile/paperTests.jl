using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("../aepd.jl")
using .AEPD
using Plots, PlotThemes
theme(:juno)

## Made to test how the variance in the paper works
function paperLatent(X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T, θ::T, σ::T) where {T <: Real}
    n,_ = size(X)
    u₁, u₂ = zeros(n), zeros(n)
    μ = X*β
    for i ∈ 1:n
        if y[i] <= μ[i]
            l = ((μ[i] - y[i]) * gamma(1+1/θ) / (σ * α))^θ
            u₁[i] = rand(truncated(Exponential(1), l, Inf), 1)[1]
        else
            l = ((y[i] - μ[i]) * gamma(1+1/θ) / (σ * (1-α)))^θ
            u₂[i] = rand(truncated(Exponential(1), l, Inf), 1)[1]
        end
    end
    u₁, u₂
end

function paperσ(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, a::N = 1, b::T = 1.) where {T, N <: Real}
    n = length(y)
    lower = zeros(n)
    μ = X * β
    for i ∈ 1:n
        if u₁[i] > 0
            lower[i] = (μ[i] - y[i])* gamma(1+1/θ) / (α * u₁[i]^(1/θ))
        elseif u₂[i] > 0
            lower[i] = (y[i] - μ[i])* gamma(1+1/θ) / ((1-α) * u₂[i]^(1/θ))
        end
    end
    rand(Pareto(a + n - 1, maximum(lower)), 1)[1]
end

function paperβ(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    n, p = size(X)
    βₛ = zeros(p)
    for k ∈ 1:p
        l, u = [], []
        for i ∈ 1:n
            μ = X[i, 1:end .!= k] ⋅  β[1:end .!= k]
            if X[i,k] > 0
                u₂[i] > 0 && append!(l, (y[i] - μ - ((1-α)*σ/ gamma(1+1/θ))*u₂[i]^(1/θ))/X[i,k])
                u₁[i] > 0 && append!(u, (y[i] - μ + (α*σ/ gamma(1+1/θ))*u₁[i]^(1/θ))/X[i,k])
            elseif X[i, k] < 0
                u₁[i] > 0 && append!(l, (μ - y[i] - (α*σ/ gamma(1+1/θ))*u₁[i]^(1/θ))/(-X[i,k]))
                u₂[i] > 0 && append!(u, (μ - y[i] + ((1-α)*σ/ gamma(1+1/θ))*u₂[i]^(1/θ))/(-X[i,k]))
            end
        end
        length(l) == 0. && append!(l, -Inf)
        length(u) == 0. && append!(u, Inf)
        βₛ[k] = rand(truncated(Normal(0, τ), maximum(l), minimum(u[findall(u .>= maximum(l))])), 1)[1]# maximum(l) < minimum(u) ? rand(truncated(Normal(0, τ), maximum(l), minimum(u)), 1)[1] : β[k]
    end
    βₛ
end

function paperβ(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    n, p = size(X)
    βsim = zeros(p)
    for k ∈ 1:p
        l, u = [], []
        for i in 1:n
            a = (y[i] - X[i, 1:end .!= k] ⋅  β[1:end .!= k]) / X[i, k]
            b₁ = ((α*σ/gamma(1+1/θ))*(u₁[i]^(1/θ))) / X[i, k]
            b₂ = (((1-α)*σ/gamma(1+1/θ))*(u₂[i]^(1/θ))) / X[i, k]
            if (u₁[i] > 0) && (X[i, k] < 0)
                append!(l, a + b₁)
            elseif (u₂[i] > 0) && (X[i, k] > 0)
                append!(l, a - b₂)
            elseif (u₁[i] > 0) && (X[i, k] > 0)
                append!(u, a + b₁)
            elseif (u₂[i] > 0) && (X[i, k] < 0)
                append!(u, a - b₂)
            end
            append!(l, -Inf)
            append!(u, Inf)
        end
        βsim[k] = rand(truncated(Normal(0, τ), maximum(l), minimum(u)), 1)[1]
    end
    βsim
end


n = 500;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ = 1.
X = [repeat([1], n) rand(Uniform(-3, 3), n)]
y = X * β .+ rand(aepd(0., σ, θ, α), n);


rand(aepd(0., 3., 1., 0.5), 100000) |> var
rand(Laplace(0, 3), 100000) |> var

u1, u2 = paperLatent(X, y, β, α, θ, σ)
mean(u1)

nMCMC = 20000
σ = zeros(nMCMC)
σ[1] = 2.
β = zeros(nMCMC, 2)
β[1,:] = inv(X'*X)*X'*y

# paper
for i ∈ 2:nMCMC
    u1, u2 = paperLatent(X, y, β[i-1,:], α, θ, σ[i-1])
    σ[i] = paperσ(X, y, u1, u2, β[i-1,:], α, θ, 1, 1.)
    β[i,:] = paperβ(X, y, u1, u2, β[i-1,:], α, θ, σ[i], 100.)
    #β[i,:] = [2.1, 0.8]
end

plot(σ)
plot(β[:,1])
plot(cumsum(β[:,1]) ./ (1:nMCMC))
