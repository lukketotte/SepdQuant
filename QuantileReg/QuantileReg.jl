module QuantileReg

export MCMCparams, MCMC

using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, Formatting, DataFrames, ProgressMeter
include("Validation.jl")
using .Validation

abstract type MCMCAbstractType end
# TODO: documentaiton
mutable struct MCMCparams <: MCMCAbstractType
    y::Array{<:Real, 1}
    X::Array{<:Real, 2}
    nMCMC::Int
    thin::Int
    burnIn::Int
end

"""
    δ(α, θ)

δ-function of the AEPD pdf
```math
\\delta_{\\alpha, \\theta} = \\frac{2\\alpha^\\theta (1-\\alpha)^\\theta}{\\alpha^\\theta + (1-\\alpha)^\\theta}
```

# Arguments
- `α::Real`: assymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
"""
function δ(α::Real, θ::Real)
    validateParams(α, θ)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

"""
    sampleLatent(X, y, β, α, θ, σ)

Samples latent u₁ and u₂ based on the uniform mixture

# Arguments
- `X::Array{<:Real, 2}`: model matrix
- `y::Array{<:Real, 1}`: dependent variable
- `β::Array{<:Real, 1}`: coefficient vector
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
- `σ::Real`: scale parameter, σ ≥ 0
"""
function sampleLatent(X::Array{<:Real, 2}, y::Array{<:Real, 1}, β::Array{<:Real, 1}, α::Real, θ::Real, σ::Real)
    n, p = validateParams(X, y, β, α, θ, σ)
    u₁, u₂ = zeros(n), zeros(n)
    μ = X*β
    for i ∈ 1:n
        if y[i] <= μ[i]
            l = ((μ[i] - y[i]) / (σ^(1/θ) * α))^θ
            u₁[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        else
            l = ((y[i] - μ[i]) / (σ^(1/θ) * (1-α)))^θ
            u₂[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        end
    end
    return u₁, u₂
end

"""
    θBlockCond(θ, X, y, β, α)

Computes the conditional distribution of θ with σ marginalized as
```math
\\int_0^\\infty \\pi(\\theta, \\sigma | \\ldots)\\ d\\sigma
```

# Arguments
- `θ::Real`: shape parameter, θ ≥ 0
- `X::Array{<:Real, 2}`: model matrix
- `y::Array{<:Real, 1}`: dependent variable
- `β::Array{<:Real, 1}`: coefficient vector
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
"""
function θBlockCond(θ::T, X::Array{<:Real, 2}, y::Array{<:Real, 1}, β::Array{<:Real, 1}, α::T) where {T <: Real}
    n, _ = validateParams(X, y, β, α, θ)
    z  = y-X*β
    pos = findall(z .> 0)
    a = δ(α, θ)*(sum(abs.(z[Not(pos)]).^θ)/α^θ + sum(z[pos].^θ)/(1-α)^θ)
    return n/θ * log(δ(α, θ))  - n*log(gamma(1+1/θ)) - n*log(a)/θ + loggamma(n/θ)
end

"""
    sampleθ(θ, X, y, β, α, ε)

Samples from the marginalized conditional distribution of θ via MH using the proposal
```math
q(\\theta^*|\\theta) = U(\\max(0, \\theta - \\varepsilon), \\theta + \\varepsilon)
```

# Arguments
- `θ::Real`: shape parameter, θ ≥ 0
- `X::Array{<:Real, 2}`: model matrix
- `y::Array{<:Real, 1}`: dependent variable
- `β::Array{<:Real, 1}`: coefficient vector
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `ε::Real`: Controls width of propsal interval, ε > 0
"""
function sampleθ(θ::Real, X::Array{<:Real, 2}, y::Array{<:Real, 1}, β::Array{<:Real, 1}, α::Real, ε::Real)
    prop = rand(Uniform(maximum([0., θ-ε]), θ + ε), 1)[1]
    return θBlockCond(prop, X, y, β, α) - θBlockCond(θ, X, y, β, α) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

"""
    sampleσ(X, y, β, α, θ)

Samples from the marginalized conditional distribution of σ as a Gibbs step

# Arguments
- `θ::Real`: shape parameter, θ ≥ 0
- `X::Array{<:Real, 2}`: model matrix
- `y::Array{<:Real, 1}`: dependent variable
- `β::Array{<:Real, 1}`: coefficient vector
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
"""
function sampleσ(X::Array{<:Real, 2}, y::Array{<:Real, 1}, β::Array{<:Real, 1}, α::Real, θ::Real)
    n, _ = validateParams(X, y, β, α, θ)
    z = y - X*β
    pos = findall(z .> 0)
    b = (δ(α, θ) * sum(abs.(z[Not(pos)]).^θ) / α^θ) + (δ(α, θ) * sum(abs.(z[pos]).^θ) / (1-α)^θ)
    return rand(InverseGamma(n/θ, b), 1)[1]
end

"""
    logβCond(X, y, β, α, θ, σ, τ, λ)

Computes log of the conditional distribution of β with X being a n × p matrix

# Arguments
- `β::Array{<:Real, 1}`: coefficient vector
- `X::Array{<:Real, 2}`: model matrix
- `y::Array{<:Real, 1}`: dependent variable
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
- `σ::Real`: scale parameter, σ ≥ 0
- `τ::Real`: scale of π(β), τ ≥ 0
- `λ::Array{Real, 1}`: Horse-shoe hyper-parameter
"""
function logβCond(β::Array{<:Real, 1}, X::Array{<:Real, 2}, y::Array{<:Real, 1}, α::Real, θ::Real,
        σ::Real, τ::Real, λ::Array{<:Real, 1})
    z = y - X*β
    pos = findall(z .> 0)
    b = δ(α, θ)/σ * (sum(abs.(z[Not(pos)]).^θ) / α^θ + sum(abs.(z[pos]).^θ) / (1-α)^θ)
    return -b -1/(2*τ) * β'*diagm(λ.^(-2))*β
end

"""
    logβCond(β, X, y, α, θ, σ, τ, λ)

Computes log of the conditional distribution of β with X being a n × 1 vector

# Arguments
- `β::Real`: coefficient, β ∈ ℜ
- `X::Array{<:Real, 1}`: vector of independent variable
- `y::Array{<:Real, 1}`: dependent variable
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
- `σ::Real`: scale parameter, σ ≥ 0
- `τ::Real`: scale of π(β), τ ≥ 0
- `λ::Real`: Horse-shoe hyper-parameter
"""
function logβCond(β::Real, X::Array{<:Real, 1}, y::Array{<:Real, 1}, α::Real, θ::Real,
        σ::Real, τ::Real, λ::Real)
    z = y - X*β
    pos = findall(z .> 0)
    b = δ(α, θ)/σ * (sum(abs.(z[Not(pos)]).^θ) / α^θ + sum(abs.(z[pos]).^θ) / (1-α)^θ)
    return -b -1/(2*(τ*λ)^2) * β^2
end

"""
    ∇ᵦ(β, X, y, α, θ, σ, τ, λ)

Computes
```math
\\nabla_\\beta \\log \\pi(\\beta | \\ldots)
```

# Arguments
- `β::Array{<:Real, 1}`: coefficient vector
- `X::Array{<:Real, 2}`: model matrix
- `y::Array{<:Real, 1}`: dependent variable
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
- `σ::Real`: scale parameter, σ ≥ 0
- `τ::Real`: scale of π(β), τ ≥ 0
- `λ::Array{<:Real, 1}`: Horse-shoe hyper-parameter
"""
function ∇ᵦ(β::Array{<:Real, 1}, X::Array{<:Real, 2}, y::Array{<:Real, 1}, α::Real, θ::Real, σ::Real,
        τ::Real, λ::Array{<:Real, 1})
    z = y - X*β
    posId = findall(z.>0)
    p=length(β)
    ∇ = zeros(p)
    for k in 1:p
        ℓ₁ = θ/α^θ * sum(abs.(z[Not(posId)]).^(θ-1) .* X[Not(posId), k])
        ℓ₂ = θ/(1-α)^θ * sum(z[posId].^(θ-1) .* X[posId, k])
        ∇[k] = -δ(α,θ)/σ * (ℓ₁ - ℓ₂) - β[k]/(τ^2 * λ[k]^2)
    end
    return ∇
end

"""
    sampleβ(X, y, u₁, u₂, β, α, θ, σ)

Samples β using latent u₁ and u₂ via Gibbs

# Arguments
- `X::Array{<:Real, 2}`: model matrix
- `y::Array{<:Real, 1}`: dependent variable
- `u₁::Array{<:Real, 1}`: latent variable
- `u₂::Array{<:Real, 1}`: latent variable
- `β::Array{<:Real, 1}`: coefficient vector
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
- `σ::Real`: scale parameter, σ ≥ 0
- `τ::Real`: scale of π(β), τ ≥ 0
"""
function sampleβ(X::Array{<:Real, 2}, y::Array{<:Real, 1}, u₁::Array{<:Real, 1}, u₂::Array{<:Real, 1},
    β::Array{<:Real, 1}, α::Real, θ::Real, σ::Real, τ::Real)
    n, p = validateParams(X, y, β, α, θ, σ)
    βsim = zeros(p)
    for k in 1:p
        l, u = [-Inf], [Inf]
        for i in 1:n
            a = (y[i] - X[i, 1:end .!= k] ⋅  β[1:end .!= k]) / X[i, k]
            b₁ = α*σ^(1/θ)*(u₁[i]^(1/θ)) / X[i, k]
            b₂ = (1-α)*σ^(1/θ)*(u₂[i]^(1/θ)) / X[i, k]
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
        λ = abs(rand(Cauchy(0 , 1), 1)[1])
        βsim[k] =  maximum(l) < minimum(u) ? rand(truncated(Normal(0, λ*τ), maximum(l), minimum(u)), 1)[1] : β[k]
    end
    return βsim
end

"""
    sampleβ(X, y, β, α, θ, σ, MALA)

Samples β using via MALA-MH

# Arguments
- `β::Array{<:Real, 1}`: coefficient vector
- `ε::Union{Real, Array{<:Real, 1}}`: vector or scalar of propsal variance(s)
- `X::Array{<:Real, 2}`: model matrix
- `y::Array{<:Real, 1}`: dependent variable
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
- `σ::Real`: scale parameter, σ ≥ 0
- `τ::Real`: scale of π(β), τ ≥ 0
- `MALA::Bool`: Set to true for MALA-MH step, false otherwise
"""
function sampleβ(β::Array{<:Real, 1}, ε::Union{Real, Array{<:Real, 1}},  X::Array{<:Real, 2},
        y::Array{<:Real, 1}, α::Real, θ::Real, σ::Real, τ::Real, MALA::Bool = true)
    _, p = validateParams(X, y, β, ε, α, θ, σ)
    if MALA
        λ = abs.(rand(Cauchy(0,1), p))
        ∇ = ∇ᵦ(β, X, y, α, θ, σ, τ, λ)
        prop = rand(MvNormal(β + ε.^2 ./ 2 .* ∇, typeof(ε) <: Real ? ε : diagm(ε)), 1) |> vec
    else
        prop = vec(rand(MvNormal(β, typeof(ε) <: Real ? ε : diagm(ε))), 1)
    end
    λ = abs.(rand(Cauchy(0,1), p))
    ∇ = ∇ᵦ(β, X, y, α, θ, σ, τ, λ)

    α₁ = logβCond(prop, X, y, α, θ, σ, 100., λ) - logβCond(β, X, y, α, θ, σ, 100., λ)
    α₁ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

InitParam = Union{Real, Array{<:Real, 1}, Nothing}

"""
    MCMC(Params)

MCMC algorithm for the the AEPD with known α

# Arguments
- `params <: MCMCAbstractType`: struct with all settings for sampler
- `α::Real`: Asymmetry parameter which determines quantile, α ∈ (0,1)
- `τ::Real`: Hyperparameter for π(β) scale, τ > 0
- `ε::Real`: Width of proposal interval for MH step for θ, ε > 0
- `εᵦ::Union{Real, Array{<:Real, 1}}`: Variance of proposal for MH step for β
- `β₁::Union{Real, Array{<:Real, 1}, Nothing}`: Initial value for β
- `σ₁::Real`: Initial value for σ
- `θ₁::Real`: Initial value for θ
- `MALA::Bool`: Set to true for MALA-MH step, false otherwise
"""
function MCMC(params::MCMCparams, α::Real, τ::Real, ε::Real = 0.05, εᵦ::Union{Real, Array{<:Real, 1}} = 0.01,
        β₁::InitParam = nothing, σ₁::Real = 1, θ₁::Real = 1, MALA = true)
    # TODO: validation
    n, p = size(params.X)
    β, σ, θ = zeros(params.nMCMC, p), zeros(params.nMCMC), zeros(params.nMCMC)
    β[1,:] = typeof(β₁) <: Nothing ? inv(X'*X)*X'*y : β₁
    σ[1], θ[1] = σ₁, θ₁

    @showprogress 1 "Sampling..." for i in 2:params.nMCMC
        θ[i] = sampleθ(θ[i-1], params.X, params.y, β[i-1,:], α, ε)
        σ[i] = sampleσ(params.X, params.y, β[i-1,:], α, θ[i])
        β[i,:] = sampleβ(β[i-1,:], εᵦ, params.X, params.y, α, θ[i], σ[i], τ, MALA)
    end
    thin = ((params.burnIn:params.nMCMC) .% params.thin) .=== 0

    β = (β[params.burnIn:params.nMCMC,:])[thin,:]
    θ = (θ[params.burnIn:params.nMCMC])[thin]
    σ = (σ[params.burnIn:params.nMCMC])[thin]
    return β, θ, σ
end
end
