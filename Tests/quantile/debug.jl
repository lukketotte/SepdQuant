using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, CSV, DataFrames, StatFiles
using StaticArrays, ForwardDiff

function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function logβCond(β::Array{<:Real, 1}, X::Array{<:Real, 2}, y::Array{<:Real, 1}, α::Real, θ::Real,
        σ::Real, τ::Real, λ::Array{<:Real, 1})
    z = y - X*β
    b = δ(α, θ)/σ * (sum((.-z[z.< 0]).^θ) / α^θ + sum(z[z.>=0].^θ) / (1-α)^θ)
    return -b -1/(2*τ) * β'*diagm(λ.^(-2))*β
end

∂β(β::Array{<:Real, 1}, X::Array{<:Real, 2}, y::Array{<:Real, 1}, α::Real, θ::Real, σ::Real, τ::Real, λ::Array{<:Real, 1}) = ForwardDiff.gradient(β -> logβCond(β, X, y, α, θ, σ, τ, λ), β)
∂β2(β::Array{<:Real, 1}, X::Array{<:Real, 2}, y::Array{<:Real, 1}, α::Real, θ::Real, σ::Real, τ::Real, λ::Array{<:Real, 1}) = ForwardDiff.jacobian(β -> ∇ᵦ(β, X, y, α, θ, σ, τ, λ), β)

function ∇ᵦ(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
    α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real})
    z = y - X*β # z will be SArray, not MArray
    p = length(β)
    # ∇ = AbstractVector{<:Real}{p}(zeros(p))
    ∇ = zeros(length(β))
    for i in 1:length(z)
        if z[i] < 0
            ∇ -= ((δ(α,θ)/σ) * (θ/α^θ) * (-z[i])^(θ-1)) .* X[i, :]
        else
            ∇ += ((δ(α,θ)/σ) * (θ/(1-α)^θ) * z[i]^(θ-1)).* X[i, :]
        end
    end
    ∇ - 1/τ^2 * (β' * diagm(1 ./ λ.^2))'
end
zeros(2,2)

function Hessian(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
    α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real})
    z = y - X*β # z will be SArray, not MArray
    p = length(β)
    ∇ = zeros(p,p)
    for i in 1:length(z)
        if z[i] < 0
            ∇ += ((δ(α,θ)/σ) * (θ*(θ-1)/α^θ) * (-z[i])^(θ-2)) * (X[i,:] * X[i,:]')
        else
            ∇ -= ((δ(α,θ)/σ) * (θ*(θ-1)/(1-α)^θ) * z[i]^(θ-2)) * (X[i,:] * X[i,:]')
        end
    end
     - 1/τ^2 * diagm(1 ./ λ.^2)
end


n = 1000;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X*β .+ rand(Laplace(0, 1), n)

∂β(β, X, y, α, θ, σ, 100., [1., 1.]) |> println
∇ᵦ(β, X, y, α, θ, σ, 100., [1., 1.]) |> println
print(Hessian(β, X, y, α, .9, σ, 100., [1., 1.]) |> inv)
print(∂β2(β, X, y, α, .9, σ, 100., [1., 1.]) |> inv)
