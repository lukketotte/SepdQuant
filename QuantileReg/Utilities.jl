module Utilities

export validateParams, MCMCparams, MixedVec, MixedMat

using StaticArrays

MixedVec = Union{SVector, Array{<:Real, 1}}
MixedMat = Union{SArray{<:Tuple, <:Real}, Array{<:Real, 2}}
ParamReal = Union{MVector, Real}
ParamVec = Union{MVector, Array{<:Real, 1}}

"""
    validateParams(α, θ)

Checks that the domains are valid

# Arguments
- `α::Real`: assymmetry parameter, α ∈ (0,1)
- `θ::Real`: shape parameter, θ ≥ 0
"""
function validateParams(α::Real, θ::Real)
    (α < 0 || α > 1) && throw(DomainError(α, "argument α must be on (0,1) interval"))
    θ < 0 && throw(DomainError(θ, "argument θ must be nonnegative"))
    return nothing
end

function validateParams(X::MixedMat, y::MixedVec, β::MixedVec, α::Real, θ::Real)
    n, p = size(X)
    n == length(y) || throw(DomainError("nrow of X not equal to length of y"))
    p == length(β) || throw(DomainError("ncol of X not equal to length of β"))
    (α < 0) || (α > 1) && throw(DomainError(α, "argument must be on (0,1) interval"))
    (α < 0 || α > 1) && throw(DomainError(α, "argument α must be on (0,1) interval"))
    θ < 0 && throw(DomainError(θ, "argument θ must be nonnegative"))
    return n, p
end

function validateParams(X::MixedMat, y::MixedVec, β::MixedVec, α::Real, θ::Real, σ::Real)
    n, p = size(X)
    n == length(y) || throw(DomainError("nrow of X not equal to length of y"))
    p == length(β) || throw(DomainError("ncol of X not equal to length of β"))
    (α < 0) || (α > 1) && throw(DomainError(α, "argument must be on (0,1) interval"))
    (α < 0 || α > 1) && throw(DomainError(α, "argument α must be on (0,1) interval"))
    θ < 0 && throw(DomainError(θ, "argument θ must be nonnegative"))
    σ < 0 && throw(DomainError(σ, "argument σ must be nonnegative"))
    return n, p
end

function validateParams(X::MixedMat, y::MixedVec, β::MixedVec,
        εᵦ::Union{Real, Array{<:Real, 1}}, α::Real, θ::Real, σ::Real)
    n, p = size(X)
    n == length(y) || throw(DomainError("nrow of X not equal to length of y"))
    p == length(β) || throw(DomainError("ncol of X not equal to length of β"))
    if typeof(εᵦ) <: Array{<:Real, 1}
        p == length(εᵦ) || throw(DomainError("length of εᵦ not equal to length of β"))
    elseif typeof(εᵦ) <: Real
        εᵦ < 0 && throw(DomainError(εᵦ, "argument εᵦ must be nonnegative"))
    end
    (α < 0) || (α > 1) && throw(DomainError(α, "argument must be on (0,1) interval"))
    (α < 0 || α > 1) && throw(DomainError(α, "argument α must be on (0,1) interval"))
    θ < 0 && throw(DomainError(θ, "argument θ must be nonnegative"))
    σ < 0 && throw(DomainError(σ, "argument σ must be nonnegative"))
    return n, p
end

function validateParams(X::MixedMat, y::MixedVec, β::MixedVec, ε::Real,
        εᵦ::Union{Real, Array{<:Real, 1}}, α::Real, θ::Real, σ::Real)
    n, p = size(X)
    n == length(y) || throw(DomainError("nrow of X not equal to length of y"))
    p == length(β) || throw(DomainError("ncol of X not equal to length of β"))
    if typeof(εᵦ) <: Array{<:Real, 1}
        p == length(εᵦ) || throw(DomainError("length of εᵦ not equal to length of β"))
    elseif typeof(εᵦ) <: Real
        εᵦ < 0 && throw(DomainError(εᵦ, "argument εᵦ must be nonnegative"))
    end
    (α < 0) || (α > 1) && throw(DomainError(α, "argument must be on (0,1) interval"))
    (α < 0 || α > 1) && throw(DomainError(α, "argument α must be on (0,1) interval"))
    θ < 0 && throw(DomainError(θ, "argument θ must be nonnegative"))
    σ < 0 && throw(DomainError(σ, "argument σ must be nonnegative"))
    ε < 0 && throw(DomainError(ε, "argument ε must be nonnegative"))
    return n, p
end

end
