module QuantileReg
using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, Formatting, DataFrames

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
    (α < 0 || α > 1) && throw(DomainError(α, "argument α must be on (0,1) interval"))
    θ < 0 && throw(DomainError(θ, "argument θ must be nonnegative"))
    2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

δ

"""
    sampleLatent(X, y, β, α, θ, σ)

Samples latent u₁ and u₂ based on the uniform mixture

# Arguments
- `X::Array{Real, 2}`: model matrix
- `y::Array{Real, 1}`: dependent variable
- `β::Array{Real, 1}`: coefficient vector
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
- `σ::Real`: scale parameter, σ ≥ 0
"""
function sampleLatent(X::Array{Real, 2}, y::Array{Real, 1}, β::Array{Real, 1}, α::Real, θ::Real, σ::Real)
    n, p = size(X)
    n == length(y) || throw(DomainError("nrow of X not equal to length of y"))
    p == length(β) || throw(DomainError("ncol of X not equal to length of β"))
    (α < 0) || (α > 1) && throw(DomainError(α, "argument must be on (0,1) interval"))
    (α < 0 || α > 1) && throw(DomainError(α, "argument α must be on (0,1) interval"))
    θ < 0 && throw(DomainError(θ, "argument θ must be nonnegative"))
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
    u₁, u₂
end

"""
    sampleLatent(X, y, β, α, θ, σ)

Samples β using latent u₁ and u₂

# Arguments
- `X::Array{Real, 2}`: model matrix
- `y::Array{Real, 1}`: dependent variable
- `β::Array{Real, 1}`: coefficient vector
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
- `σ::Real`: scale parameter, σ ≥ 0
- `τ::Real`: scale of π(β), τ ≥ 0
"""
function sampleβBlock(X::Array{Real, 2}, y::Array{Real, 1}, u₁::Array{Real, 1}, u₂::Array{Real, 1},
    β::Array{Real, 1}, α::Real, θ::Real, σ::Real, τ::Real)
    n, p = size(X)
    n == length(y) || throw(DomainError("nrow of X not equal to length of y"))
    p == length(β) || throw(DomainError("ncol of X not equal to length of β"))
    (α < 0) || (α > 1) && throw(DomainError(α, "argument must be on (0,1) interval"))
    (α < 0 || α > 1) && throw(DomainError(α, "argument α must be on (0,1) interval"))
    θ < 0 && throw(DomainError(θ, "argument θ must be nonnegative"))
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
    βsim
end

"""
    θBlockCond(θ, X, y, β, α)

Computes the conditional distribution of θ with σ marginalized as
```math
\\int_0^\\infty \\pi(\\theta, \\sigma | \\ldots)\\ d\\sigma
```

# Arguments
- `θ::Real`: shape parameter, θ ≥ 0
- `X::Array{Real, 2}`: model matrix
- `y::Array{Real, 1}`: dependent variable
- `β::Array{Real, 1}`: coefficient vector
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
"""
function θBlockCond(θ::T, X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T) where {T <: Real}
    n, p = size(X)
    n == length(y) || throw(DomainError("nrow of X not equal to length of y"))
    p == length(β) || throw(DomainError("ncol of X not equal to length of β"))
    (α < 0) || (α > 1) && throw(DomainError(α, "argument must be on (0,1) interval"))
    (α < 0 || α > 1) && throw(DomainError(α, "argument α must be on (0,1) interval"))
    θ < 0 && throw(DomainError(θ, "argument θ must be nonnegative"))

    z  = y-X*β
    pos = findall(z .> 0)
    a = δ(α, θ)*(sum(abs.(z[Not(pos)]).^θ)/α^θ + sum(z[pos].^θ)/(1-α)^θ)
    n/θ * log(δ(α, θ))  - n*log(gamma(1+1/θ)) - n*log(a)/θ + loggamma(n/θ)
end


"""
    sampleθ(θ, X, y, β, α, ε)

Samples from the marginalized conditional distribution of θ via MH using the proposal
```math
q(\\theta^*|\\theta) = U(\\max(0, \\theta - \\varepsilon), \\theta + \\varepsilon)
```

# Arguments
- `θ::Real`: shape parameter, θ ≥ 0
- `X::Array{Real, 2}`: model matrix
- `y::Array{Real, 1}`: dependent variable
- `β::Array{Real, 1}`: coefficient vector
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
- `ε::Real`: Controls width of propsal interval, ε > 0
"""
function sampleθ(θ::Real, X::Array{Real, 2}, y::Array{Real, 1}, β::Array{Real, 1}, α::Real, ε::Real)
    prop = rand(Uniform(maximum([0., θ-ε]), θ + ε), 1)[1]
    θBlockCond(prop, X, y, β, α) - θBlockCond(θ, X, y, β, α) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

"""
    sampleσ(X, y, β, α, θ)

Samples from the marginalized conditional distribution of θ via MH using the proposal
```math
q(\\theta^*|\\theta) = U(\\max(0, \\theta - \\varepsilon), \\theta + \\varepsilon)
```

# Arguments
- `θ::Real`: shape parameter, θ ≥ 0
- `X::Array{Real, 2}`: model matrix
- `y::Array{Real, 1}`: dependent variable
- `β::Array{Real, 1}`: coefficient vector
- `α::Real`: Asymmetry parameter, α ∈ (0,1)
"""
function sampleσ(X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T, θ::T) where {T, N <: Real}
    z = y - X*β
    pos = findall(z .> 0)
    b = (δ(α, θ) * sum(abs.(z[Not(pos)]).^θ) / α^θ) + (δ(α, θ) * sum(abs.(z[pos]).^θ) / (1-α)^θ)
    rand(InverseGamma(length(y)/θ, b), 1)[1]
end

end
