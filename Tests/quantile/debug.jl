using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, CSV, DataFrames, StatFiles


## Struct test
struct MCMCparams
    y::Array{<:Real, 1}
    X::Array{<:Real, 2}
    nMCMC::Int
    thin::Int
    burnIn::Int
    ε::Union{Real, Array{<:Real, 1}}
    Θ::Union{Array{Any, 2}}

    function MCMCparams(y::Array{<:Real, 1}, X::Array{<:Real, 2}, nMCMC::Int,
            thin::Int, burnIn::Int, ε::Union{Real, Array{<:Real, 1}})
        new(y, X, nMCMC, thin, burnIn, ε, [[inv(X'*X)*X'*y] [1.] [1.]])
    end

    function MCMCparams(y::Array{<:Real, 1}, X::Array{<:Real, 2}, nMCMC::Int,
            thin::Int, burnIn::Int)
        new(y, X, nMCMC, thin, burnIn, 0.01, [[inv(X'*X)*X'*y] [1.] [1.]])
    end
end

@showprogress 1 "Computing..." for i in 1:500000000
    log(i)
end


n = 500;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X*β .+ rand(Laplace(0, 1), n)

function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function ∇ᵦ(β::AbstractVector, X::AbstractMatrix, y::AbstractVector,
    α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector)
    z = y - X*β # z will be SArray, not MArray
    p = length(β)
    ∇ = zeros(p)
    for k in 1:p
        inner∇ᵦ!(∇, β, k, z, X, α, θ, σ, τ, λ)
    end
    return ∇
end

function inner∇ᵦ!(∇::AbstractVector, β::AbstractVector, k::Int, z::AbstractVector,
        X::AbstractMatrix, α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector)
    ℓ₁ = θ/α^θ * sum((.-z[z.<0]).^(θ-1) .* X[z.<0, k])
    ℓ₂ = θ/(1-α)^θ * sum(z[z.>=0].^(θ-1) .* X[z.>=0, k])
    ∇[k] = -δ(α,θ)/σ * (ℓ₁ - ℓ₂) - β[k]/((τ*λ[k])^2)
    nothing
end

∇ᵦ(β, X, y, α, θ, σ, 100, [100., 1.]) |> println

z = y - X*β
∂L = -(δ(α,θ)/σ) * (θ/α^θ) * sum((.-z[z.<0]).^(θ-1) * X[z.<0, :])

τ = 100
sumterm = [0, 0]
for i in 1:length(z)
    if z[i] < 0
        global sumterm -= ((δ(α,θ)/σ) * (θ/α^θ) * (-z[i])^(θ-1)) .* X[i, :]
    else
        global sumterm += ((δ(α,θ)/σ) * (θ/(1-α)^θ) * z[i]^(θ-1)).* X[i, :]
    end
end

sumterm -= 2/τ^2 * (β' * diagm(1 ./ λ.^2))'
println(sumterm)

λ = [100., 1.]

(β' * diagm(1 ./ λ))'

((δ(α,θ)/σ) * (θ/α^θ) * z[3]^(θ-1)) .* X[3, :]

z

sum((.-z[z.<0]).^(θ-1) * X[z.<0, :])

sumterm + z[1] .* X[1, :]
z[1] .* X[1, :]
