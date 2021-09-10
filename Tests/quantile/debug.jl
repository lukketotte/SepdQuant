using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff
using CSV, DataFrames, CSVFiles
#include("../aepd.jl")
#include("../../QuantileReg/QuantileReg.jl")
#using .AEPD, .QuantileReg

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function logβCond(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real,
        σ::Real, τ::Real, λ::AbstractVector{<:Real})
    z = y - X*β
    b = δ(α, θ)/σ * (sum((.-z[z.< 0]).^θ) / α^θ + sum(z[z.>=0].^θ) / (1-α)^θ)
    return -b -1/(2*τ^2) * β'*diagm(λ.^(-2))*β
end

∂β(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}) = ForwardDiff.gradient(β -> logβCond(β, X, y, α, θ, σ, τ, λ), β)
∂β2(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}) = ForwardDiff.jacobian(β -> -∂β(β, X, y, α, θ, σ, τ, λ), β)

function Hessian(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real,
        σ::Real, τ::Real, λ::AbstractVector{<:Real})
        a = δ(α, θ)/σ * θ * (θ - 1)
        A = zeros(length(β), length(β))
        for i in 1:length(y)
                μ = X[i, :] ⋅ β
                if(y[i] <= μ)
                        A += 1/α^θ * (μ - y[i])^(θ-2) * X[i,:] * X[i,:]'
                else
                        A += 1/(1-α)^θ * (y[i] - μ)^(θ-2) * X[i,:] * X[i,:]'
                end
        end
        -(-a * A - diagm(λ.^(-2)) / τ^2)
end




β = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]
α, θ, σ, τ = 0.99, 1.5, 12.5, 100
ε = 0.1


λ = abs.(rand(Cauchy(0,1), length(β)))
H = (∂β2(β, X, log.(y), α, .7, σ, τ, λ))^(-1) |> Symmetric
Hessian(β, X, log.(y), α, .7, σ, τ, λ)^(-1)
