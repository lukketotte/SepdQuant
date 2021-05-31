using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
using Plots, PlotThemes
theme(:juno)

## generate data
n = 100
β = [2.1, 0.8]
α, θ, σ = 0.5, 2., 1.
x₂ = rand(Uniform(-3, 3), n)
X = [repeat([1], n) x₂]
y = X * β .+ rand(Normal(0, σ), n)

## tests

u1, u2 = sampleLatent(X, y, β, α, θ, σ)

ids = u1 .> 0

findall((y[ids] - X[ids,:] * β) .> 0)
# θinterval(X, y, u1, u2, β, α, σ)
sampleθ(2., X, y, u1, u2, β, α, σ)
sampleSigma(X, y, u1, u2, β, α, σ)

p = range(0.01, 4., length = 200)
pl = [θcond(a, u1, u2, 0.5)  for a in p]
plot(p, pl, legend = false)


##

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
            l = ((μ - y[i]) / (σ * α))^θ
            u₁[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        else
            l = ((y[i] - μ) / (σ * (1-α)))^θ
            u₂[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        end
    end
    u₁, u₂
end

function sampleSigma(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, ν::N = 1) where {T, N <: Real}
    n = length(y)
    lower = zeros(n)
    for i in 1:n
        μ = X[i,:] ⋅ β
        if u₁[i] > 0
            lower[i] = (μ - y[i]) / (α * u₁[i]^(1/θ))
        else
            lower[i] = (y[i] - μ) / ((1-α) * u₂[i]^(1/θ))
        end
    end
    rand(Pareto(ν + length(y) - 1, maximum(lower)), 1)[1]
end

function θinterval(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, σ::T) where {T <: Real}
    n = length(y)
    θ = range(0.01, 5, length = 1000)
    θₗ, θᵤ = zeros(n), zeros(n)
    for i in 1:n
        if u₁[i] > 0
            θside = (u₁[i] .^ (1 ./ θ))
            c = (X[i,:] ⋅ β - y[i]) / (α * σ)
            ids = findall(θside .> c)
            θᵤ[i] = length(ids) > 0 ? θ[maximum(ids)] : Inf
            θₗ[i] = length(ids) > 0 ? θ[minimum(ids)] : 0
        else
            θside = (u₂[i] .^ (1 ./ θ))
            c = (y[i] - X[i,:] ⋅ β) / ((1-α) * σ)
            ids = findall(θside .> c)
            θᵤ[i] = length(ids) > 0 ? θ[maximum(ids)] : Inf
            θₗ[i] = length(ids) > 0 ? θ[minimum(ids)] : 0
        end
    end
    [maximum(θₗ) minimum(θᵤ)]
end


function θcond(θ::T, u₁::Array{T, 1}, u₂::Array{T, 1}, α::T) where {T <: Real}
    n = length(u₁)
    n*(1+1/θ) * log(δ(α, θ))  - n*log(gamma(1+1/θ)) - δ(α, θ) * sum(u₁ .+ u₂)
end

function sampleθ(θ::T, X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, σ::T) where {T <: Real}
    interval = θinterval(X, y, u₁, u₂, β, α, σ)
    if minimum(interval) > maximum(interval)
        prop = rand(Uniform(minimum(interval), maximum(interval)), 1)[1]
        θcond(prop, u₁, u₂, α) - θcond(θ, u₁, u₂, α) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
    else
        minimum(interval)
    end
end

function sampleβ(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    n, p = size(X)
    βsim = zeros(p)

    for k in 1:p
        l, u = [], []
        for i in 1:n
            a = (y[i] - X[i, 1:end .!= k] ⋅  β[1:end .!= k]) / X[i, k]
            b₁ = α*σ*(u₁[i]^(1/θ)) / X[i, k]
            b₂ = (1-α)*σ*(u₂[i]^(1/θ)) / X[i, k]
            if (u₁[i] > 0) && (X[i, k] < 0)
                l = append!(l, a + b₁)
            elseif (u₂[i] > 0) && (X[i, k] > 0)
                l = append!(l, a - b₂)
            elseif (u₁[i] > 0) && (X[i, k] > 0)
                u = append!(u, a + b₁)
            elseif (u₂[i] > 0) && (X[i, k] < 0)
                u = append!(u, a - b₂)
            end
        end
        βsim[k] = maximum(l) < minimum(u) ? rand(truncated(Normal(0, τ), maximum(l), minimum(u)), 1)[1] : maximum(l)
    end
    βsim
end


nMCMC = 10
σ = zeros(nMCMC)
σ[1] = 1.
β = zeros(nMCMC, 2)
β[1, :] = [2.1, 0.8]
θ = zeros(nMCMC)
θ[1] = 2.

simU1 = zeros(nMCMC, n)
simU2 = zeros(nMCMC, n)
for i in 2:nMCMC
    (i % 250 === 0) && println(i)
    println(i)
    u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i-1])
    simU1[i,:] = u1
    simU2[i,:] = u2
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i-1], σ[i-1], 10.)
    σ[i] = sampleSigma(X, y, u1, u2, β[i, :], α, θ[i-1], 1)
    θ[i] = sampleθ(θ[i-1], X, y, u1, u2, β[i, :], α, σ[i])
    # θ[i] = 2.
end

plot(β[:, 2])
plot(σ)
plot(θ)

plot(cumsum(σ) ./ (1:nMCMC))
plot(cumsum(β[:, 1]) ./ (1:nMCMC))

## whats going on...
i = 3
θinterval(X, y, simU1[i,:], simU2[i,:], β[i,:], α, σ[i])


ids = simU1[i,:] .> 0

simU2[i,:] |> sum
