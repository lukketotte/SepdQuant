using Distributions, LinearAlgebra, StatsBase
using Plots, PlotThemes
theme(:juno)

## Testing data
y = rand(Normal(0, 2), 50)

## helpers
function ESS(W::Vector{T})::T where {T <: Real}
    weightSum = sum(W)
    1/sum((W ./ weightSum).^2)
end

function α₁(Tₖ::N, t::N; p::T = 2.) where {N <: Integer, T <: Real}
    (t/Tₖ)^p
end

# πₜ^{2, k}(θₜ)|_{k=1} invariant kernel
function μ_update(y::Vector{T}, σ::T, μ_0::T, σ_0::T, t::N, Tₖ::N) where {T<: Real, N <: Integer}
    αₜ = α₁(Tₖ, t)
    n = length(y)
    μ_post = (σ_0 * mean(y) + σ/(n*αₜ) * μ_0) / (σ/(n*αₜ) + σ_0)
    σ_post = 1/(n*αₜ / σ + 1/σ_0)
    rand(Normal(μ_post, √σ_post), 1)[1]
end

μ_update(y, σ[1], μ₀, σ₀, 1, T)

function σ_update(y::Vector{T}, μ::T, α₀::T, β₀::T, t::N, Tₖ::N) where {T<: Real, N <: Integer}
    αₜ = α₁(Tₖ, t)
    n = length(y)
    α_post = α₀ + n*αₜ/2
    β_post = β₀ + αₜ/2 * sum((y.- μ).^2)
    rand(InverseGamma(α_post, β_post))
end


σ_update(y, μ[1], α₀, β₀, 1, T)

## initialization step

N = 10
θ = zeros(N, 2)
T = 100
# hyperparameters
μ₀, σ₀ = 0., 10.
α₀, β₀ = 1., 1.

# initialize using ν ~ π(θ)
σ = 1 ./ rand(Gamma(α₀,β₀), N)
μ = zeros(N)
for i in 1:N
    μ[i] = rand(Normal(μ₀, √(σ₀ / σ[i])), 1)[1]
end

θ = hcat(μ, σ)

# weights
W = zeros(N, T)
W[:, 1] .= 1/N

for i in 1:N
    W[i, 2] = W[i, 1] * prod(pdf.(Normal(μ[i], √σ[i]), y)) ^ α₁(T, 1)
end

ESS(W[:, 2])

W[:, 2] = W[:, 2] ./ sum(W[:, 2])

1/sum(W[:,2].^2)

## Attempt at full run
N = 20000
T = 500
y = rand(Normal(0, 2), 50)
μ₀, σ₀ = 0., 10.
α₀, β₀ = 1., 1.
μ = zeros(N, T)
σ = zeros(N, T)
W = zeros(N, T)
essRun = zeros(T)
z = ones(T)
essRun[1] = N
μ[:, 1] = rand(Normal(μ₀, √σ₀), N)
σ[:, 1] = rand(InverseGamma(α₀, β₀), N)
W[:, 1] .= 1/N
resampFactor = 0.5

for t in 2:T
    # Weights
    for i in 1:N
        W[i, t] = W[i, t-1] * prod(pdf.(Normal(μ[i], √σ[i]), y)) ^ (α₁(T, t) - α₁(T, t-1))
    end
    z[t] = sum(W[:, t])
    # normalize weights and calculate ESS
    W[:, t] = W[:, t] ./ sum(W[:, t])
    essRun[t] = 1/sum(W[:,t].^2)
    # check ESS if resampling is necessary
    if essRun[t] < resampFactor * N
        ids = StatsBase.sample(1:N, StatsBase.weights(W[:, t]), N, replace = true)
        W[:,t] .= 1/N
        μ[:, t-1] = μ[ids, t-1]
        σ[:, t-1] = σ[ids, t-1]
    end
    # update parameters
    for i in 1:N
        μ[i, t] = μ_update(y, σ[i, t-1], μ₀, σ₀, t, T)
        σ[i, t] = σ_update(y, μ[i, t-1], α₀, β₀, t, T)
    end
end

println(log(prod(z)/N))

plot(essRun)
plot(essRun[400:500])

W

α₁(T, 95)
