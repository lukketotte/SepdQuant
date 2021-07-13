using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, CSV, DataFrames, StatFiles
using StaticArrays

MixedVec = Union{SVector, Array{<:Real, 1}}
MixedMat = Union{SArray{<:Tuple, <:Real}, Array{<:Real, 2}}
ParamReal = Union{MVector, Real}
ParamVec = Union{MVector, Array{<:Real, 1}}

n = 500;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X*β .+ rand(Laplace(0, 1), n)

function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function ∇ᵦ(β::MixedVec, X::MixedMat, y::MixedVec, α::Real, θ::Real, σ::Real, τ::Real, λ::MixedVec)
    z = y - X*β
    p = length(β)
    ∇ = zeros(length(β))
    for i in 1:length(z)
        if z[i] < 0
            ∇ -= ((δ(α,θ)/σ) * (θ/α^θ) * (-z[i])^(θ-1)) .* X[i, :]
        else
            ∇ += ((δ(α,θ)/σ) * (θ/(1-α)^θ) * z[i]^(θ-1)).* X[i, :]
        end
    end
    ∇ - 2/τ^2 * (β' * diagm(1 ./ λ.^2))'
end

function inner∇ᵦ!(∇::MVector, β::MixedVec, k::Int, z::MixedVec,
        X::MixedMat, α::Real, θ::Real, σ::Real, τ::Real, λ::MixedVec)
    ℓ₁ = θ/α^θ * sum((.-z[z.<0]).^(θ-1) .* X[z.<0, k])
    ℓ₂ = θ/(1-α)^θ * sum(z[z.>=0].^(θ-1) .* X[z.>=0, k])
    ∇[k] = -δ(α,θ)/σ * (ℓ₁ - ℓ₂) - 2*β[k]/((τ*λ[k])^2)
    nothing
end

function logβCond(β::MixedVec, X::MixedMat, y::MixedVec, α::Real, θ::Real,
        σ::Real, τ::Real, λ::MixedVec)
    z = y - X*β
    pos = findall(z .> 0)
    b = δ(α, θ)/σ * (sum((.-z[z.< 0]).^θ) / α^θ + sum(z[z.>=0].^θ) / (1-α)^θ)
    return -b -1/(2*τ) * β'*diagm(λ.^(-2))*β
end

∇ᵦ(β, X, y, α, θ, σ, 100, [100., 1.]) |> println

### MALA
ε = 0.001
∇ = ∇ᵦ(β, X, y, α, θ, σ,  100, [100., 1.])
prop = rand(MvNormal(β + ε .* ∇, 2*ε), 1) |> vec
# println(∇ᵦ(prop, X, y, α, θ, σ, 100, [100., 1.]))

## HOW IN THE WORLD DOES THE GRADIENT SKYROCKET?
# println(∇ᵦ([2.1001, 0.8001], X, y, α, θ, σ, 100, [1., 1.]))



∇ₚ = ∇ᵦ(prop, X, y, α, θ, σ, 100, [100., 1.])
αᵦ = logβCond(prop, X, y, α, θ, σ, 100, [100., 1.]) - logβCond(β, X, y, α, θ, σ, 100, [100., 1.])
αᵦ += -sum((β - prop - ε * ∇ₚ).^2)/ (4*ε) + sum((prop - β - ε * ∇).^2)/ (4*ε)
# αᵦ -= sum((β - prop - ε * ∇ₚ).^2)/ (4*ε) + sum((prop - β - ε * ∇).^2)/ (4*ε)
# - 1.5181594
αᵦ > log(rand(Uniform(0,1), 1)[1]) # ? prop : β
