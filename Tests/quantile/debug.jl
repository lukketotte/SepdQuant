using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, CSV, DataFrames, StatFiles
using StaticArrays, ForwardDiff

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

function logβCond(β::MixedVec, X::MixedMat, y::MixedVec, α::Real, θ::Real,
        σ::Real, τ::Real, λ::MixedVec)
    z = y - X*β
    pos = findall(z .> 0)
    b = δ(α, θ)/σ * (sum((.-z[z.< 0]).^θ) / α^θ + sum(z[z.>=0].^θ) / (1-α)^θ)
    return -b -1/(2*τ) * β'*diagm(λ.^(-2))*β
end

∂β(β, X, y, α, θ, σ, τ, λ) = ForwardDiff.gradient(β -> logβCond(β, X, y, α, θ, σ, 100, [100., 1.]), β)

ε = 0.001
∇ = ∂β(β, X, y, α, θ, σ, 100, [100., 1.])
prop = β + ε^2/2 .* ∇ + ε .* vec(rand(MvNormal(zeros(length(β))), 1))
∇n = ∂β(prop, X, y, α, θ, σ, 100, [100., 1.])
- logpdf(MvNormal(β + ε^2/2 .* ∇, ε), prop) + logpdf(MvNormal(prop + ε^2/2 .* ∇n, ε), β) +
    logβCond(prop, X, y, α, θ, σ, 100, [100., 1.]) - logβCond(β, X, y, α, θ, σ, 100, [100., 1.]) > log(rand(Uniform(0,1), 1)[1])

logβCond(prop, X, y, α, θ, σ, 100, [100., 1.]) - logβCond(β, X, y, α, θ, σ, 100, [100., 1.])

### MALA
n = 500;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X*β .+ rand(Laplace(0, 1), n)
αᵦ = logβCond(prop, X, y, α, θ, σ, 100, [100., 1.]) - logβCond(β, X, y, α, θ, σ, 100, [100., 1.])



ε = 0.00001
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
