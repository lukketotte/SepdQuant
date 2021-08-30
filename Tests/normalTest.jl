using Distributions, LinearAlgebra, StatsBase, ForwardDiff, EpdTest, SpecialFunctions
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

## For the AEPD
function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function θBlockCond(θ::Real, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, β::AbstractVector{<:Real}, α::Real,
    t::Integer, T::Integer)
    z  = y-X*β
    n = length(y)
    a = δ(α, θ)*(sum((.-z[z.<0]).^θ)/α^θ + sum(z[z.>=0].^θ)/(1-α)^θ)
    return α₁(T,t)*n/θ * log(δ(α, θ))  - α₁(T,t)*n*log(gamma(1+1/θ)) - n*log(α₁(T,t)*a)/θ + loggamma(α₁(T,t)*n/θ)
end

function sampleθ(θ::Real, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, β::AbstractVector{<:Real},
    α::Real, ε::Real, t::Integer, T::Integer)
    prop = rand(Uniform(maximum([0., θ-ε]), minimum([3., θ + ε])), 1)[1]
    return θBlockCond(prop, X, y, β, α, t, T) - θBlockCond(θ, X, y, β, α, t, T) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

function sampleσ(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, β::AbstractVector{<:Real},
    α::Real, θ::Real, t::Integer, T::Integer)
    n = length(y)
    z = y - X*β
    b = (α₁(T,t)*δ(α, θ) * sum((.-z[z.<0]).^θ) / α^θ) + (δ(α, θ) * sum(z[z.>=0].^θ) / (1-α)^θ)
    return rand(InverseGamma(α₁(T,t)*n/θ, b), 1)[1]
end

function logβCond(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real,
        σ::Real, τ::Real, λ::AbstractVector{<:Real}, t::Integer, T::Integer)
    z = y - X*β
    pos = findall(z .> 0)
    b = α₁(T,t)*δ(α, θ)/σ * (sum((.-z[z.< 0]).^θ) / α^θ + sum(z[z.>=0].^θ) / (1-α)^θ)
    return -b -1/(2*τ) * β'*diagm(λ.^(-2))*β
end

∂β(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}, t::Integer, T::Integer) = ForwardDiff.gradient(β -> logβCond(β, X, y, α, θ, σ, τ, λ, t, T), β)

function sampleβ(β::AbstractVector{<:Real}, ε::Union{Real, AbstractVector{<:Real}},  X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real, t::Integer, T::Integer)
    λ = abs.(rand(Cauchy(0,1), length(β)))
    ∇ = ∂β(β, X, y, α, θ, σ, τ, λ, t, T)
    prop = β + ε .^2 / 2 .* ∇ + ε .* vec(rand(MvNormal(zeros(length(β)), 1), 1))
    ∇ₚ = ∂β(prop, X, y, α, θ, σ, τ, λ, t, T)
    αᵦ = logβCond(prop, X, y, α, θ, σ, τ, λ, t, T) - logβCond(β, X, y, α, θ, σ, τ, λ, t, T)
    αᵦ += - logpdf(MvNormal(β + ε .^2 / 2 .* ∇, ε), prop) + logpdf(MvNormal(prop + ε .^2/2 .* ∇ₚ, ε), β)
    αᵦ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

n = 100;
β, α, σ = [2.1, 0.8, -1.], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n) rand(Uniform(-5, 0), n)]
y = X*β .+ rand(Laplace(0, 1), n)

y = log.(y)

ε = [0.09, 0.02, 0.02, 0.02, 0.00065, 0.02, 0.02, 0.00065, 0.006]
plot(βs[6,:,T])
plot(θ[:, T])
plot(σ[:, T])
# Testing it out
N = 2000
T = 30
k = 1

σ, θ = zeros(N, T), zeros(N, T);
βs = zeros(size(X[:, 1:end .!= k])[2], N, T)
βs[:,:,1] = reshape(repeat(inv(X[:, 1:end .!= k]'*X[:, 1:end .!= k])*X[:, 1:end .!= k]'*y, outer = N), (size(X[:, 1:end .!= k])[2], N))
σ[:,1] = ones(N);
θ[:,1] = ones(N);

W = zeros(N, T)
essRun = zeros(T-1)
z = ones(T)
W[:, 1] .= 1/N
resampFactor = 0.5

for t in 2:T
    for i in 1:N
        pr = 0
        for j in 1:length(y)
            pr += (α₁(T, t) - α₁(T, t-1)) * logpdf.(aepd(X[j, 1:end .!= k] ⋅ βs[:,i,t-1], σ[i,t-1]^(1/θ[i,t-1]), θ[i,t-1], 0.5), y[j])
        end
        # W[i, t] = W[i, t-1] * pr
        W[i, t] = exp(log(W[i, t-1]) + pr)
    end
    # normalize weights and calculate ESS
    z[t] = sum(W[:, t])
    W[:, t] = W[:, t] ./ sum(W[:, t])
    essRun[t-1] = 1/sum(W[:,t].^2)
    # check ESS if resampling is necessary
    if essRun[t-1] < resampFactor * N
        ids = StatsBase.sample(1:N, StatsBase.weights(W[:, t]), N, replace = true)
        W[:,t] .= 1/N
        βs[:,:,t-1] = βs[:,ids,t-1]
        σ[:, t-1] = σ[ids, t-1]
        θ[:, t-1] = θ[ids, t-1]
    end
    # update parameters
    for i in 1:N
        βs[:,i,t] = sampleβ(βs[:,i,t-1],ε[1:end .!= k], X[:, 1:end .!= k], y, 0.5, θ[i, t-1], σ[i, t-1], 100., t, T)
        σ[i, t] = sampleσ(X[:, 1:end .!= k], y, βs[:,i,t-1], 0.5, θ[i, t-1], t, T)
        θ[i, t] = sampleθ(θ[i, t-1], X[:, 1:end .!= k], y,  βs[:,i,t-1], 0.5, 0.05, t, T)
    end
end

sum(log.(z)) |> println

println(z)
println(log(prod(z)/N))
plot(essRun)

(α₁(T, T) - α₁(T, T-1))

## need a sampler for


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
N = 1000
T = 100
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
