include("QR.jl")
using .QR
using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
using Plots, PlotThemes, CSV, DataFrames, StatFiles


function δ(α::T, θ::T)::T where {T <: Real}
    2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function θcond(θ::T, u₁::Array{T, 1}, u₂::Array{T, 1}, α::T) where {T <: Real}
    n = length(u₁)
    n*(1+1/θ) * log(δ(α, θ))  - n*log(gamma(1+1/θ)) - δ(α, θ) * sum(u₁ .+ u₂)
end

##
n = 100
β = [2.1, 0.8]
α, θ, σ = 0.55, 1., 1.
x₂ = rand(Uniform(-3, 3), n)
X = [repeat([1], n) x₂]
y = X * β .+ rand(Normal(0, σ), n)

u1, u2 = sampleLatent(X, y, β, α, θ, σ)
p = range(0.1, 3, length = 2000)
plot(p, [θcond(a, u1, u2, α) for a in p])
# plot!(p, [θcond(a, u1, u2, α) for a in p])

n*(1+1/θ) * log(δ(α, θ))
n*log(gamma(1+1/0.5))
δ(α, θ) * sum(u1 .+ u1)

## σ
lower = zeros(n)
for i in 1:n
    μ = X[i,:] ⋅ β
    if (u1[i] > 0) && (y[i] < μ)
        lower[i] = (μ - y[i]) / (α * u1[i]^(1/θ))
    elseif (u2[i] > 0) && (y[i] >= μ)
        lower[i] = (y[i] - μ) / ((1-α) * u2[i]^(1/θ))
    end
end

maximum(lower)

rand(Pareto(n - 1, maximum(lower)), 10000) |> mean
