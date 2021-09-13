using Distributions, LinearAlgebra, StatsBase, ForwardDiff, EpdTest, SpecialFunctions
include("aepd.jl")
include("../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg
using Plots, PlotThemes
theme(:juno)

## helpers
function α₁(Tₖ::N, t::N; p::T = 5.) where {N <: Integer, T <: Real}
    (t/Tₖ)^p
end

function resample(W::AbstractVector{<:Real})
    W = cumsum(W)
    ids = zeros(Int64, length(W))

    for i in 1:length(W)
        ids[i] = findall(rand(Uniform(), 1)[1] .<= W)[1]
    end
    ids
end

## For the AEPD
function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function θBlockCond(θ::Real, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
    β::AbstractVector{<:Real}, α::Real, ϕ::Real)
    z  = y-X*β
    n = length(y)
    a = δ(α, θ)*(sum((.-z[z.<0]).^θ)/α^θ + sum(z[z.>=0].^θ)/(1-α)^θ)
    return ϕ*n/θ * log(δ(α, θ))  - ϕ*n*log(gamma(1+1/θ)) - n*log(ϕ*a)/θ + loggamma(ϕ*n/θ)
end

function sampleθ(θ::Real, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, β::AbstractVector{<:Real},
    α::Real, ε::Real, ϕ::Real)
    prop = rand(Uniform(maximum([0., θ-ε]), minimum([3., θ + ε])), 1)[1]
    return θBlockCond(prop, X, y, β, α, ϕ) - θBlockCond(θ, X, y, β, α, ϕ) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

function sampleσ(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, β::AbstractVector{<:Real},
    α::Real, θ::Real, ϕ::Real)
    n = length(y)
    z = y - X*β
    b = (ϕ*δ(α, θ) * sum((.-z[z.<0]).^θ) / α^θ) + (ϕ*δ(α, θ) * sum(z[z.>=0].^θ) / (1-α)^θ)
    return rand(InverseGamma(ϕ*n/θ, b), 1)[1]
end

function logβCond(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real,
        σ::Real, τ::Real, λ::AbstractVector{<:Real}, ϕ::Real)
    z = y - X*β
    pos = findall(z .> 0)
    b = ϕ*δ(α, θ)/σ * (sum((.-z[z.< 0]).^θ) / α^θ + sum(z[z.>=0].^θ) / (1-α)^θ)
    return -b -1/(2*τ) * β'*diagm(λ.^(-2))*β
end

∂β(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}, ϕ::Real) = ForwardDiff.gradient(β -> logβCond(β, X, y, α, θ, σ, τ, λ, ϕ), β)
∂β2(β::AbstractVector{<:Real},  X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}, ϕ::Real) = ForwardDiff.jacobian(β -> -∂β(β, X, y, α, θ, σ, τ, λ, ϕ), β)

function sampleβ(β::AbstractVector{<:Real}, ε::Real,  X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real, ϕ::Real)
    λ = abs.(rand(Cauchy(0,1), length(β)))
    ∇ = ∂β(β, X, y, α, θ, σ, τ, λ, ϕ)
    H = (∂β2(β, X, y, α, maximum([θ, 1.0001]), σ, τ, λ, ϕ))^(-1) |> Symmetric
    prop = β + ε^2 * H / 2 * ∇ + ε * √H * vec(rand(MvNormal(zeros(length(β)), 1), 1))
    ∇ₚ = ∂β(prop, X, y, α, θ, σ, τ, λ, ϕ)
    Hₚ = (∂β2(prop, X, y, α, maximum([θ, 1.0001]), σ, τ, λ, ϕ))^(-1) |> Symmetric
    αᵦ = logβCond(prop, X, y, α, θ, σ, τ, λ, ϕ) - logβCond(β, X, y, α, θ, σ, τ, λ, ϕ)
    αᵦ += - logpdf(MvNormal(β + ε .^2 / 2 * ∇, ε^2 * H), prop) + logpdf(MvNormal(prop + ε^2/2 * ∇ₚ, ε^2 * Hₚ), β)
    αᵦ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

n = 200;
β, α, σ = [2.1, 0.8, 0.01], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n) rand(Uniform(-5, 0), n)]
y = X*β .+ rand(Laplace(0., 1.), n)

# Testing it out
N = 500;
T = 500;
k = 30;

ϕ = range(0, 1, length = T)
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

# Y = y
# y = log.(Y + rand(Uniform(), length(Y)) .- 0.5)

for t in 2:T
    for i in 1:N
        pr = 0
        for j in 1:length(y)
            pr += (ϕ[t] -ϕ[t-1]) * logpdf(aepd(X[j, 1:end .!= k] ⋅ βs[:,i,t-1], σ[i,t-1], θ[i,t-1], 0.5), y[j])
        end
        W[i, t] = exp(log(W[i, t-1]) + pr)
    end
    # normalize weights and calculate ESS
    z[t] = sum(W[:, t])

    cess = N*(sum(W[:, t] .* W[:, t-1]))^2 / sum(W[:, t].^2 .* W[:, t-1])

    W[:, t] = W[:, t] ./ sum(W[:, t])
    essRun[t-1] = 1/sum(W[:,t].^2) #cess
    # check ESS if resampling is necessary
    if essRun[t-1] < resampFactor * N
        # ids = StatsBase.sample(1:N, StatsBase.weights(W[:, t]), N, replace = true)
        ids = resample(W[:,t])
        W[:,t] .= 1/N
        βs[:,:,t-1] = βs[:,ids,t-1]
        σ[:, t-1] = σ[ids, t-1]
        θ[:, t-1] = θ[ids, t-1]
    end
    # update parameters
    for i in 1:N
        # y = log.(Y + rand(Uniform(), length(Y)) .- 0.5)
        θ[i, t] = sampleθ(θ[i, t-1], X[:, 1:end .!= k], y,  βs[:,i,t-1], 0.5, 0.1, ϕ[t])
        σ[i, t] = sampleσ(X[:, 1:end .!= k], y, βs[:,i,t-1], 0.5, θ[i, t], ϕ[t])
        βs[:,i,t] = sampleβ(βs[:,i,t-1], 0.5, X[:, 1:end .!= k], y, 0.5, θ[i, t], σ[i, t], 100., ϕ[t])
    end
end

sum(log.(z)) |> println
plot(essRun)

[median(βs[i,:,T]) for i in 1:9] |> println
[median(β2[:,i]) for i in 1:9] |> println

[median(βs[3, :, t]) for t in 1:T]

plot(βs[2,:,T])
plot(θ[:, T])
plot(σ[:, T])
median(σ[:,T])
median(θ[:,T])
median(βs[1,:,T])

plot(1:N, cumsum(θ[:,T])./(1:N))
plot!(1:N, cumsum(θ2[1:N])./(1:N))

plot(1:N, cumsum(σ[:,T])./(1:N))
plot!(1:N, cumsum(σ2[1:N])./(1:N))

p = 3
plot(1:N, cumsum(βs[p,:,T])./(1:N))
plot!(1:N, cumsum(β2[1:N, p])./(1:N))

1-((θ[2:N, T] .=== θ[1:(N - 1), T]) |> mean)

1-((βs[1, 2:N, T] .=== βs[1, 1:(N - 1), T]) |> mean)

## TODO: Alg 4 och SMC paper for model comparison
