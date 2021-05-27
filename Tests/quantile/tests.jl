using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
using Plots, PlotThemes
theme(:juno)

## generate data
n = 100
β = [2.1, 0.8]
α, θ = 0.5, 1.5
x₁ = rand(Normal(0, 5), n)
repeat([1], 100)
x₂ = rand(Normal(2, 5), n)
X = [x₁ x₂]
y = X * β .+ rand(Normal(0, 2), n)

function δ(α::T, θ::T)::T where {T <: Real}
    2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function sampleLatent(X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T, θ::T, σ::T) where {T <: Real}
    n,_ = size(X)
    u₁ = zeros(n)
    u₂ = zeros(n)
    for i in 1:n
        μ = X[i,:] ⋅ β
        if y[i] <= μ
            l = ((μ - y[i]) * gamma(1+1/θ) / (σ * α))^θ
            u₁[i] = rand(truncated(Gamma(1, 1/δ(α, θ)), l, Inf), 1)[1]
        else
            l = ((y[i] - μ) * gamma(1+1/θ) / (σ * (1- α)))^θ
            u₂[i] = rand(truncated(Gamma(1, 1/δ(α, θ)), l, Inf), 1)[1]
        end
    end
    u₁, u₂
end

function sampleSigma(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, ν::N = 1) where {T, N <: Real}
    u1Pos = u₁ .> 0
    u2Pos = u₂ .> 0
    l₁ = maximum((X[u1Pos,:] * β .- y[u1Pos]) .* gamma(1+1/θ) ./ (α .* u₁[u1Pos].^(1/θ)))
    l₂ = maximum((y[u2Pos] .- X[u2Pos,:] * β) .* gamma(1+1/θ) ./ ((1-α) .* u₂[u2Pos].^(1/θ)))
    # σnew = rand(truncated(InverseGamma(ν + size(y)[1] - 1, 1), maximum([l₁ l₂]), Inf), 1)[1]
    # σnew === Inf ? maximum([l₁ l₂]) : σnew
    rand(Pareto(ν + length(y) - 1, maximum([l₁ l₂])), 1)[1]
end

function θinterval(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, σ::T) where {T <: Real}
    n = length(y)
    θ = range(0.01, 3, length = 2000)
    θₗ, θᵤ = zeros(n), zeros(n)
    for i in 1:n
        if u₁[i] > 0
            θside = (u₁[i] .^ (1 ./ θ)) ./ gamma(1 + 1 ./ θ)
            c = (X[i,:] ⋅ β - y[i]) / (α * σ)
            θᵤ[i] = θ[maximum(findall(θside .> c))]
            θₗ[i] = θ[minimum(findall(θside .> c))]
        else
            θside = (u₂[i] .^ (1 ./ θ)) ./ gamma(1 + 1 ./ θ)
            c = (y[i] - X[i,:] ⋅ β) / (α * σ)
            θᵤ[i] = θ[maximum(findall(θside .> c))]
            θₗ[i] = θ[minimum(findall(θside .> c))]
        end
    end
    [maximum(θₗ) minimum(θᵤ)]
end


function sampleβ(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    n, p = size(X)
    βsim = zeros(p)

    for k in 1:p
        u1Pos = (X[:, k] .< 0) .& (u₁ .> 0)
        u2Pos = (X[:, k] .> 0) .& (u₂ .> 0)
        l = []
        if sum(u1Pos) > 0
            a1 = (y[u1Pos] .- X[u1Pos, 1:end .!= k] * β[1:end .!= k]) ./ X[u1Pos, k]
            b1 = σ*α ./ X[u1Pos, k] .* u₁[u1Pos].^(1/θ) ./ gamma(1 + 1/θ)
            l = append!(l, a1.+b1)
        end
        if sum(u2Pos) > 0
            a2 = (y[u2Pos] .- X[u2Pos, 1:end .!= k] * β[1:end .!= k]) ./ X[u2Pos, k]
            b2 = σ*(1-α) ./ X[u2Pos, k] .* u₂[u2Pos].^(1/θ) ./ gamma(1 + 1/θ)
            l = append!(l, a2.-b2)
        end
        u1Pos = (X[:, k] .> 0) .& (u₁ .> 0)
        u2Pos = (X[:, k] .< 0) .& (u₂ .> 0)
        u = []
        if sum(u1Pos) > 0
            a1 = (y[u1Pos] .- X[u1Pos, 1:end .!= k] * β[1:end .!= k]) ./ X[u1Pos, k]
            b1 = σ*α ./ X[u1Pos, k] .* u₁[u1Pos].^(1/θ) ./ gamma(1 + 1/θ)
            u = append!(u, a1.+b1)
        end
        if sum(u2Pos) > 0
            a2 = (y[u2Pos] .- X[u2Pos, 1:end .!= k] * β[1:end .!= k]) ./ X[u2Pos, k]
            b2 = σ*(1-α) ./ X[u2Pos, k] .* u₂[u2Pos].^(1/θ) ./ gamma(1 + 1/θ)
            u = append!(u, a2.-b2)
        end

        βsim[k] = rand(truncated(Normal(0, τ), maximum(l), minimum(u)), 1)[1]
    end
    βsim
end



nMCMC = 20000
σ = zeros(nMCMC)
σ[1] = 1
β = zeros(nMCMC, 2)
β[1, :] = [2.1, 0.8]


for i in 2:nMCMC
    (i % 500 === 0) && println(i)
    u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ, σ[i-1])
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ, σ[i-1], 10.)
    σ[i] = sampleSigma(X, y, u1, u2, β[i-1, :], α, θ, 1)
end

plot(β[:, 1])
plot(σ)

plot(cumsum(σ) ./ (1:nMCMC))
plot(cumsum(β[:, 2]) ./ (1:nMCMC))

mean(σ) * √(gamma(1+1/θ))
