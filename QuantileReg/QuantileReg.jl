module QuantileReg

export mcmc, Sampler, acceptance

using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ProgressMeter, ForwardDiff

mutable struct Sampler{T <: Real, M <: Real, Response <: AbstractVector, ModelMat <: AbstractMatrix}
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
    y = T <: Int ?  log.(y + rand(Uniform(), length(y))) : y
    length(y) === size(X)[1] || throw(DomainError("Size of y and X not matching"))
    Sampler{T, M, typeof(y), typeof(X)}(y, X, α, nMCMC, thin, burnIn)
end

Sampler(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, α::Real, nMCMC::Int) = Sampler(y, X, α, nMCMC, 1, 1)
Sampler(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, nMCMC::Int) = Sampler(y, X, 0.5, nMCMC, 1, 1)
Sampler(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}) = Sampler(y, X, 0.5, 5000, 1, 1)

data(s::Sampler) = (s.y, s.X)
param(s::Sampler) = (s.y, s.X, s.α)

kernel(s::Sampler, β::AbstractVector{<:Real}, θ::Real) = s.y-s.X*β |> z -> (sum((.-z[z.<0]).^θ)/s.α^θ + sum(z[z.>0].^θ)/(1-s.α)^θ)
kernel(s::Sampler, β::AbstractVector{<:Real}, θ::Real, α::Real) = s.y-s.X*β |> z -> (sum((.-z[z.<0]).^θ)/α^θ + sum(z[z.>0].^θ)/(1-α)^θ)

function πθ(θ::Real)
    θ^(-3/2) * √((1+1/θ) * trigamma(1+1/θ))
end

function θcond(s::Sampler, θ::Real, β::AbstractVector{<:Real})
    n = length(s.y)
    a = gamma(1+1/θ)^θ * kernel(s, β, θ)
    return -log(θ) + loggamma(n/θ) - (n/θ) * log(a) + log(πθ(θ))
end

function sampleθ(s::Sampler, θ::Real, β::AbstractVector{<:Real}, ε::Real; trunc = 1.)
    prop = rand(Truncated(Normal(θ, ε^2), trunc, Inf))
    a = logpdf(Truncated(Normal(prop, ε^2), trunc, Inf), θ) - logpdf(Truncated(Normal(θ, ε^2), trunc, Inf), prop)
    return θcond(s, prop, β) - θcond(s, θ, β) + a >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

function αcond(α::Real, s::Sampler, θ::Real, σ::Real, β::AbstractVector{<:Real})
    return - (gamma(1+1/θ)/σ)^θ * kernel(s, β, θ, α)
end

function sampleα(s::Sampler, ε::Real, θ::Real, σ::Real, β::AbstractVector{<:Real})
    prop = rand(Truncated(Normal(s.α, ε^2), 0, 1))
    a = logpdf(Truncated(Normal(prop, ε^2), 0, 1), s.α) - logpdf(Truncated(Normal(s.α, ε^2), 0, 1), prop) +
        αcond(prop, s, θ, σ, β) - αcond(s.α, s, θ, σ, β)
    s.α = a >= log(rand(Uniform(0,1), 1)[1]) ? prop : s.α
    nothing
end

function sampleσ(s::Sampler, θ::Real, β::AbstractVector{<:Real})
    b = gamma(1+1/θ)^θ * kernel(s, β, θ)
    return (rand(InverseGamma(length(s.y)/θ, b), 1)[1])^(1/θ)
end

function logβCond(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real)
    return - gamma(1+1/θ)^θ/σ^θ * kernel(s, β, θ)
end

function logβCond(β::AbstractVector{<:Real},s::Sampler, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real})
    return - gamma(1+1/θ)^θ/σ^θ * kernel(s, β, θ) -1/(2*τ) * β'*diagm(λ.^(-2))*β
end


∂β(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real) = ForwardDiff.gradient(β -> logβCond(β, s, θ, σ), β)
∂β(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}) = ForwardDiff.gradient(β -> logβCond(β, s, θ, σ, τ, λ), β)
∂β2(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real) = ForwardDiff.jacobian(β -> -∂β(β, s, θ, σ), β)
∂β2(β::AbstractVector{<:Real}, s::Sampler, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}) = ForwardDiff.jacobian(β -> -∂β(β, s, θ, σ, τ, λ), β)

function sampleβ(β::AbstractVector{<:Real}, ε::Real,  s::Sampler, θ::Real, σ::Real)
    ∇ = ∂β(β, s, θ, σ)
    H = (∂β2(β, s, maximum([θ, 1.01]), σ))^(-1) |> Symmetric
    prop = β + ε^2 * H / 2 * ∇ + ε * √H * vec(rand(MvNormal(zeros(length(β)), 1), 1))
    ∇ₚ = ∂β(prop, s, θ, σ)
    Hₚ = (∂β2(prop, s, maximum([θ, 1.01]), σ))^(-1) |> Symmetric
    αᵦ = logβCond(prop, s, θ, σ) - logβCond(β, s, θ, σ)
    αᵦ += - logpdf(MvNormal(β + ε^2 / 2 * H * ∇, ε^2 * H), prop)
    αᵦ += logpdf(MvNormal(prop + ε^2/2 * Hₚ * ∇ₚ, ε^2 * Hₚ), β)
    return αᵦ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

function sampleβ(β::AbstractVector{<:Real}, ε::Real,  s::Sampler, θ::Real, σ::Real, τ::Real)
    λ = abs.(rand(Cauchy(0,1), length(β)))
    ∇ = ∂β(β, s, θ, σ, τ, λ)
    H = (∂β2(β, s, maximum([θ, 1.0001]), σ, τ, λ))^(-1) |> Symmetric
    prop = β + ε^2 * H / 2 * ∇ + ε * √H * vec(rand(MvNormal(zeros(length(β)), 1), 1))
    ∇ₚ = ∂β(prop, s, θ, σ, τ, λ)
    Hₚ = (∂β2(prop, s, maximum([θ, 1.0001]), σ, τ, λ))^(-1) |> Symmetric
    αᵦ = logβCond(prop, s, θ, σ, τ, λ) - logβCond(β, s, θ, σ, τ, λ)
    αᵦ += - logpdf(MvNormal(β + ε .^2 / 2 * H * ∇, ε^2 * H), prop) + logpdf(MvNormal(prop + ε^2/2 * Hₚ * ∇ₚ, ε^2 * Hₚ), β)
    return αᵦ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

# AEPD jeffrey's prior, α known
function mcmc(s::Sampler, ε::Real, εᵦ::Union{Real, AbstractVector{<:Real}}, σ₁::Real, θ₁::Real,
    β₁::Union{AbstractVector{<:Real}, Nothing} = nothing; verbose = true)
    n, p = size(s.X)
    σ₁ > 0 || θ₁ > 0 || throw(DomainError("Shape ands scale must be positive"))
    β = zeros(s.nMCMC, p)
    σ, θ = [σ₁ ; zeros(s.nMCMC-1)], [θ₁ ; zeros(s.nMCMC-1)]
    β[1,:] = typeof(β₁) <: Nothing ? inv(s.X'*s.X)*s.X'*s.y : β₁

    p = verbose && Progress(s.nMCMC-1, dt=0.5,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen=50, color=:green)

    for i ∈ 2:s.nMCMC
        verbose && next!(p; showvalues=[(:iter,i) (:θ, round(θ[i-1], digits = 3)) (:σ, round(σ[i-1], digits = 3))])
        mcmcInner!(s, θ, σ, β, i, ε, εᵦ)
    end
    return mcmcThin(θ, σ, β, s)
end

function mcmcInner!(s::Sampler, θ::AbstractVector{<:Real}, σ::AbstractVector{<:Real},
    β::AbstractMatrix{<:Real}, i::Int, ε::Real, εᵦ::Real)
        θ[i] = sampleθ(s, θ[i-1], β[i-1,:], ε)
        σ[i] = sampleσ(s, θ[i], β[i-1,:])
        β[i,:] = sampleβ(β[i-1,:], εᵦ, s, θ[i], σ[i])
        nothing
end

# AEPD jeffrey's prior, α unknown
function mcmc(s::Sampler, ε::Real, εₐ::Real, εᵦ::Union{Real, AbstractVector{<:Real}}, σ₁::Real, θ₁::Real, α₁::Real,
    β₁::Union{AbstractVector{<:Real}, Nothing} = nothing; verbose = true)
    n, p = size(s.X)
    σ₁ > 0 || θ₁ > 0 || α > 0 || α < 1 || throw(DomainError("Parameter(s) not in domain"))
    β = zeros(s.nMCMC, p)
    σ, θ, α = [σ₁ ; zeros(s.nMCMC-1)], [θ₁ ; zeros(s.nMCMC-1)], [α₁ ; zeros(s.nMCMC-1)]
    s.α = α₁
    β[1,:] = typeof(β₁) <: Nothing ? inv(s.X'*s.X)*s.X'*s.y : β₁

    p = verbose && Progress(s.nMCMC-1, dt=0.5,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen=50, color=:green)

    for i ∈ 2:s.nMCMC
        verbose && next!(p; showvalues=[(:iter,i) (:θ, round(θ[i-1], digits = 2)) (:σ, round(σ[i-1], digits = 2)) (:α, round(α[i-1], digits = 2))])
        mcmcInner!(s, θ, σ, α, β, i, ε, εᵦ, εₐ)
    end
    return mcmcThin(θ, σ, α, β, s)
end

function mcmcInner!(s::Sampler, θ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}, α::AbstractVector{<:Real},
    β::AbstractMatrix{<:Real}, i::Int, ε::Real, εᵦ::Real, εₐ::Real)
        θ[i] = sampleθ(s, θ[i-1], β[i-1,:], ε)
        σ[i] = sampleσ(s, θ[i], β[i-1,:])
        sampleα(s, εₐ, θ[i], σ[i], β[i-1,:])
        α[i] = s.α
        β[i,:] = sampleβ(β[i-1,:], εᵦ, s, θ[i], σ[i])
        nothing
end

# AEPD horse-shoe prior
"""function mcmc(s::Sampler, τ::Real, ε::Real, εᵦ::Union{Real, AbstractVector{<:Real}}, σ₁::Real, θ₁::Real,
    β₁::Union{AbstractVector{<:Real}, Nothing} = nothing; verbose = true)
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
        verbose && next!(p; showvalues=[(:iter,i) (:θ, round(θ[i-1], digits = 3)) (:σ, round(σ[i-1], digits = 3))])
        mcmcInner!(s, θ, σ, β, i, ε, εᵦ, τ)
    end
    return mcmcThin(θ, σ, β, s)
end

function mcmcInner!(s::Sampler, θ::AbstractVector{<:Real}, σ::AbstractVector{<:Real},
    β::AbstractMatrix{<:Real}, i::Int, ε::Real, εᵦ::Real, τ::Real)
        θ[i] = sampleθ(s, θ[i-1], β[i-1,:], ε, trunc = 0.01)
        σ[i] = sampleσ(s, θ[i-1], β[i-1,:])
        β[i,:] = sampleβ(β[i-1,:], εᵦ, s, θ[i], σ[i], τ)
        nothing
end"""


# convert to quantile estimates
function mcmc(s::Sampler, εᵦ::Union{Real, AbstractVector{<:Real}}, θ::Real, σ₁::Real,
    β₁::Union{AbstractVector{<:Real}, Nothing} = nothing; verbose = true)
    n, p = size(s.X)
    σ₁ > 0 || throw(DomainError("Shape ands scale must be positive"))
    β = zeros(s.nMCMC, p)
    σ = [σ₁; zeros(s.nMCMC-1)]
    β[1,:] = typeof(β₁) <: Nothing ? inv(s.X'*s.X)*s.X'*s.y : β₁
    σ[1] = σ₁
    β = zeros(s.nMCMC, p)
    β[1,:] = typeof(β₁) <: Nothing ? inv(s.X'*s.X)*s.X'*s.y : β₁

    p = verbose && Progress(s.nMCMC-1, dt=0.5,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen=50, color=:green)

    for i ∈ 2:s.nMCMC
        verbose && next!(p; showvalues=[(:iter,i) (:σ, round(σ[i-1], digits = 3))])
        mcmcInner!(s, σ, θ, β, i, εᵦ)
    end
    return mcmcThin(σ, β, s)
end

function mcmcInner!(s::Sampler, σ::AbstractVector{<:Real}, θ::Real,
    β::AbstractMatrix{<:Real}, i::Int, εᵦ::Real)
        σ[i] = σ[i-1]# sampleσ(s, θ, β[i-1,:])
        β[i,:] = sampleβ(β[i-1,:], εᵦ, s, θ, σ[i])
        nothing
end

function mcmcThin(θ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}, β::Array{<:Real, 2}, s::Sampler)
    thin = ((s.burnIn:s.nMCMC) .% s.thin) .=== 0
    β = (β[s.burnIn:s.nMCMC,:])[thin,:]
    θ = (θ[s.burnIn:s.nMCMC])[thin]
    σ = (σ[s.burnIn:s.nMCMC])[thin]
    return β, θ, σ
end

function mcmcThin(θ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}, α::AbstractVector{<:Real}, β::Array{<:Real, 2}, s::Sampler)
    thin = ((s.burnIn:s.nMCMC) .% s.thin) .=== 0
    β = (β[s.burnIn:s.nMCMC,:])[thin,:]
    θ = (θ[s.burnIn:s.nMCMC])[thin]
    σ = (σ[s.burnIn:s.nMCMC])[thin]
    α = (α[s.burnIn:s.nMCMC])[thin]
    return β, θ, σ, α
end

function mcmcThin(σ::AbstractVector{<:Real}, β::Array{<:Real, 2}, s::Sampler)
    thin = ((s.burnIn:s.nMCMC) .% s.thin) .=== 0
    β = (β[s.burnIn:s.nMCMC,:])[thin,:]
    σ = (σ[s.burnIn:s.nMCMC])[thin]
    return β, σ
end

acceptance(θ::AbstractMatrix{<:Real}) = size(θ, 1) |> n -> 1-((θ[2:n, 1] .=== θ[1:(n - 1), 1]) |> mean)
acceptance(θ::AbstractVector{<:Real}) = length(θ) |> n -> 1-((θ[2:n] .=== θ[1:(n - 1), 1]) |> mean)

end
