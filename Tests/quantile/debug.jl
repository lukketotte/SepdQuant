include("QR.jl")
using .QR
using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
using Plots, PlotThemes, CSV, DataFrames, StatFiles
# using KernelDensity

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


## Truncated gamma following Philippe
a, t, b = 100, 0.99, 1;
v, w = zeros(a), zeros(a);
v[1], w[1] = 1,1;
for k in 2:a
    v[k] = v[k-1] * (a-k+1)/(t*b)
    w[k] = w[k-1] + v[k]
end

wt = v./w[a]

any(wt .>= u)

n = 1000
x = zeros(n)
for i in 1:n
    u = rand(Uniform(), 1)[1]
    k = any(wt .>= u) ? minimum(findall(wt .>= u)) : a
    x[i] = t * (rand(InverseGamma(k, 1/(t*b)), 1)[1] + 1)
end

mean(x)
x

d = kde(x);
a = range(t, 1.2, length = 1000)
plot(pdf(d, a));

function rtruncGamma(n::N, a::N, b::T, t::T) where {N, T <: Real}
    a, t, b = 100, 0.99, 1;
    v, w = zeros(a), zeros(a);
    v[1], w[1] = 1,1;
    for k in 2:a
        v[k] = v[k-1] * (a-k+1)/(t*b)
        w[k] = w[k-1] + v[k]
    end
    wt = v./w[a]
    x = zeros(n)
    for i in 1:n
        u = rand(Uniform(), 1)[1]
        k = any(wt .>= u) ? minimum(findall(wt .>= u)) : a
        x[i] = t * (rand(InverseGamma(k, 1/(t*b)), 1)[1] + 1)
    end
    x
end

rtruncGamma(1, 100, 1., 0.98)[1]
