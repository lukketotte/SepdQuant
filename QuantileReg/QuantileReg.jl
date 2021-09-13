module QuantileReg

export mcmc, Sampler

using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ProgressMeter, ForwardDiff

struct Sampler{T <: Real, M <: Real, Response <: AbstractVector, ModelMat <: AbstractMatrix}
    y::Response
    X::ModelMat
    α::Real
    nMCMC::Int
    thin::Int
    burnIn::Int
end

function Sampler(y::AbstractVector{T}, X::AbstractMatrix{M}, α::Real, nMCMC::Int, thin::Int, burnIn::Int) where {T,M <: Real}
    nMCMC > 0 || thin > 0 || burnIn > 0 || throw(DomainError("Integers can't be negative"))
    α > 0 || α < 1 || throw(DomainError("α ∉ (0,1)"))
    y = T <: Int ?  log.(y + rand(Uniform(), length(y)) .- α) : y
    length(y) === size(X)[1] || throw(DomainError("Size of y and X not matching"))
    Sampler{T, M, typeof(y), typeof(X)}(y, X, α, nMCMC, thin, burnIn)
end

Sampler(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, α::Real, nMCMC::Int) = Sampler(y, X, α, nMCMC, 1, 1)
Sampler(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, nMCMC::Int) = Sampler(y, X, 0.5, nMCMC, 1, 1)
data(s::Sampler) = (s.y, s.X)
param(s::Sampler) = (s.y, s.X, s.α)

function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function θcond(s::Sampler, θ::Real, β::AbstractVector{<:Real})
    z  = s.y-s.X*β
    n = length(z)
    a = δ(s.α, θ)*(sum((.-z[z.<0]).^θ)/s.α^θ + sum(z[z.>=0].^θ)/(1-s.α)^θ)
    return n/θ * log(δ(s.α, θ))  - n*log(gamma(1+1/θ)) - n*log(a)/θ + loggamma(n/θ)
end

function sampleθ(s::Sampler, θ::Real, β::AbstractVector{<:Real}, ε::Real)
    prop = rand(Uniform(maximum([0., θ-ε]), minimum([3., θ + ε])), 1)[1]
    return θcond(s, prop, β) - θcond(s, θ, β) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

function sampleσ(s::Sampler, θ::Real, β::AbstractVector{<:Real})
    z = s.y - s.X*β
    n = length(z)
    b = (δ(s.α, θ) * sum((.-z[z.<0]).^θ) / s.α^θ) + (δ(s.α, θ) * sum(z[z.>=0].^θ) / (1-s.α)^θ)
    return rand(InverseGamma(n/θ + 1, b + 1), 1)[1]
end


function logβCond(β::AbstractVector{<:Real},s::Sampler, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real})
    z = s.y - s.X*β
    b = δ(s.α, θ)/σ * (sum((.-z[z.< 0]).^θ) / s.α^θ + sum(z[z.>=0].^θ) / (1-s.α)^θ)
    return -b -1/(2*τ) * β'*diagm(λ.^(-2))*β
end

# first derivative
∂β(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}) = ForwardDiff.gradient(β -> logβCond(β, s, θ, σ, τ, λ), β)
# -Hessian
∂β2(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}) = ForwardDiff.jacobian(β -> -∂β(β, s, θ, σ, τ, λ), β)

function sampleβ(β::AbstractVector{<:Real}, ε::Real,  s::Sampler, θ::Real, σ::Real, τ::Real) where {T <: Real}
    λ = abs.(rand(Cauchy(0,1), length(β)))
    ∇ = ∂β(β, s, θ, σ, τ, λ)
    H = (∂β2(β, s, maximum([θ, 1.0001]), σ, τ, λ))^(-1) |> Symmetric
    prop = β + ε^2 * H / 2 * ∇ + ε * √H * vec(rand(MvNormal(zeros(length(β)), 1), 1))
    ∇ₚ = ∂β(prop, s, θ, σ, τ, λ)
    Hₚ = (∂β2(prop, s, maximum([θ, 1.0001]), σ, τ, λ))^(-1) |> Symmetric
    αᵦ = logβCond(prop, s, θ, σ, τ, λ) - logβCond(β, s, θ, σ, τ, λ)
    αᵦ += - logpdf(MvNormal(β + ε .^2 / 2 * ∇, ε^2 * H), prop) + logpdf(MvNormal(prop + ε^2/2 * ∇ₚ, ε^2 * Hₚ), β)
    return αᵦ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

function sampleβ(β::AbstractVector{<:Real}, ε::AbstractVector{<:Real},  s::Sampler, θ::Real, σ::Real, τ::Real)
    λ = abs.(rand(Cauchy(0,1), length(β)))
    prop = vec(rand(MvNormal(β, diagm(ε)), 1))
    return logβCond(prop, X, y, α, θ, σ, τ, λ) - logβCond(β, X, y, α, θ, σ, τ, λ) > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

function mcmc(s::Sampler, τ::Real, ε::Real, εᵦ::Union{Real, AbstractVector{<:Real}},
    β₁::Union{AbstractVector{<:Real}, Nothing} = nothing, σ₁::Real = 1, θ₁::Real = 1; verbose = true)
    n, p = size(s.X)
    σ₁ > 0 || θ₁ > 0 || throw(DomainError("Shape ands scale must be positive"))
    β = zeros(s.nMCMC, p)
    σ, θ = zeros(s.nMCMC), zeros(s.nMCMC)
    β[1,:] = typeof(β₁) <: Nothing ? inv(s.X'*s.X)*s.X'*s.y : β₁
    σ[1], θ[1] = σ₁, θ₁

    p = verbose && Progress(s.nMCMC-1, dt=0.5,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen=50, color=:green)

    for i ∈ 2:s.nMCMC
        next!(p; showvalues=[(:iter,i) (:θ, round(θ[i-1], digits = 3)) (:σ, round(σ[i-1], digits = 3))])
        mcmcInner!(s, θ, σ, β, i, ε, εᵦ, τ)
    end
    return mcmcThin(θ, σ, β, s)
end

function mcmcInner!(s::Sampler, θ::AbstractVector{<:Real}, σ::AbstractVector{<:Real},
    β::AbstractMatrix{<:Real}, i::Int, ε::Real, εᵦ::Real, τ::Real)
        θ[i] = sampleθ(s, θ[i-1], β[i-1,:], ε)
        σ[i] = sampleσ(s, θ[i-1], β[i-1,:])
        β[i,:] = sampleβ(β[i-1,:], εᵦ, s, θ[i], σ[i], τ)
        nothing
end

function mcmcThin(θ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}, β::Array{<:Real, 2}, s::Sampler)
    thin = ((s.burnIn:s.nMCMC) .% s.thin) .=== 0
    β = (β[s.burnIn:s.nMCMC,:])[thin,:]
    θ = (θ[s.burnIn:s.nMCMC])[thin]
    σ = (σ[s.burnIn:s.nMCMC])[thin]
    return β, θ, σ
end

end
