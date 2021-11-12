using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff

include("../../QuantileReg/QuantileReg.jl")
using.QuantileReg

kernel(s::Sampler, β::AbstractVector{<:Real}, θ::Real) = s.y-s.X*β |> z -> (sum((.-z[z.<0]).^θ)/s.α^θ + sum(z[z.>0].^θ)/(1-s.α)^θ)

function logβCond(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real)
    return - gamma(1+1/θ)^θ/σ^θ * kernel(s, β, θ)
end

∂β(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real) = ForwardDiff.gradient(β -> logβCond(β, s, θ, σ), β)
∂β2(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real) = ForwardDiff.jacobian(β -> -∂β(β, s, θ, σ), β)

function sampleβ(β::AbstractVector{<:Real}, ε::Real,  s::Sampler, θ::Real, σ::Real)
    ∇ = ∂β(β, s, θ, σ)
    H = (∂β2(β, s, maximum([θ, 1.01]), σ))^(-1) |> Symmetric
    #H = diagm([1 for i in 1:length(β)])
    prop = β + ε^2 * H / 2 * ∇ + ε * √H * vec(rand(MvNormal(zeros(length(β)), 1), 1))
    ∇ₚ = ∂β(prop, s, θ, σ)
    Hₚ = (∂β2(prop, s, maximum([θ, 1.01]), σ))^(-1) |> Symmetric
    #Hₚ = diagm([1 for i in 1:length(β)])
    αᵦ = logβCond(prop, s, θ, σ) - logβCond(β, s, θ, σ)
    αᵦ += - logpdf(MvNormal(β + ε^2 / 2 * H * ∇, ε^2 * H), prop) + logpdf(MvNormal(prop + ε^2/2 * Hₚ * ∇ₚ, ε^2 * Hₚ), β)
    return αᵦ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

function mhβ(β::AbstractVector{<:Real}, ε::Real,  s::Sampler, θ::Real, σ::Real)
    prop = β + ε * rand(MvNormal(zeros(length(β)), 1))
    αᵦ = logβCond(prop, s, θ, σ) - logβCond(β, s, θ, σ)
    return αᵦ > log(rand(Uniform(0,1))) ? prop : β
end

N = 40000
betas = zeros(N, size(X, 2))
betas[1,:] = b
for i in 2:N
    betas[i,:] = sampleβ(betas[i-1,:], .8, par,  2.3, 4.35)
    #betas[i,:] = mhβ(betas[i-1,:], 0.001, par,  2.3, 4.35)
end

plot(betas[:,1])
acceptance(betas)

idx = 4
plot(cumsum(betas[:,idx]) ./ (1:N));plot!([b[idx] for i in 1:N])

mean(betas, dims = 1) |> println
println(b)

[par.y[i] <= X[i,:] ⋅ b for i in 1:length(y)] |> mean
[par.y[i] <= X[i,:] ⋅ mean(betas, dims = 1)  for i in 1:length(par.y)] |> mean

## using scale mixture
function sampleLatent(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, β::AbstractVector{<:Real},
    α::Real, θ::Real, σ::Real)
    n, p = size(X)
    u₁, u₂ = zeros(n), zeros(n)
    μ = X*β
    for i ∈ 1:n
        if y[i] <= μ[i]
            l = gamma(1+1/θ)^θ * ((μ[i] - y[i]) / (σ * α))^θ
            u₁[i] = rand(truncated(Exponential(1), l, Inf), 1)[1]
        else
            l =  gamma(1+1/θ)^θ * ((y[i] - μ[i]) / (σ * (1-α)))^θ
            u₂[i] = rand(truncated(Exponential(1), l, Inf), 1)[1]
        end
    end
    return u₁, u₂
end

function sampleβ(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, u₁::AbstractVector{<:Real}, u₂::AbstractVector{<:Real},
    β::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real)
    n, p = size(X)
    for k in 1:p
        l, u = Float64[-Inf], Float64[Inf]
        for i in 1:n
            a = (y[i] - X[i, 1:end .!= k] ⋅  β[1:end .!= k]) / X[i, k]
            b₁ = α*σ/gamma(1+1/θ)*(u₁[i]^(1/θ)) / X[i, k]
            b₂ = (1-α)*σ/gamma(1+1/θ)*(u₂[i]^(1/θ)) / X[i, k]
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
        β[k] = rand(Uniform(maximum(l), minimum(u)))
    end
    return β
end

N = 20000
betas = zeros(N, size(X, 2))
betas[1,:] = b
for i in 2:N
    u1, u2 = sampleLatent(par.X, par.y, betas[i-1,:], 0.1, 1., 1.)
    betas[i,:] = sampleβ(par.X, par.y, u1, u2, betas[i-1,:], 0.1, 1., 1., 1000)
end

thin = ((5000:N) .% 5) .=== 0
betas = (betas[5000:N,:])[thin,:]

plot(betas[:,3])
plot(cumsum(betas[:,4]) ./ (1:size(betas, 1)))


mean(betas, dims = 1) |> println
println(b)
