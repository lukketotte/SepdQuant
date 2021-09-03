using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:default)

function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end
function θBlockCond(θ::Real, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, β::AbstractVector{<:Real}, α::Real)
    z  = y-X*β
    n = length(y)
    a = δ(α, θ)*(sum((.-z[z.<0]).^θ)/α^θ + sum(z[z.>=0].^θ)/(1-α)^θ)
    return n/θ * log(δ(α, θ))  - n*log(gamma(1+1/θ)) - n*log(a)/θ + loggamma(n/θ)
end
function sampleθ(θ::Real, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, β::AbstractVector{<:Real}, α::Real, ε::Real)
    prop = rand(Uniform(maximum([0., θ-ε]), minimum([3., θ + ε])), 1)[1]
    return θBlockCond(prop, X, y, β, α) - θBlockCond(θ, X, y, β, α) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end
function sampleσ(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, β::AbstractVector{<:Real}, α::Real, θ::Real)
    n = length(y)
    z = y - X*β
    b = (δ(α, θ) * sum((.-z[z.<0]).^θ) / α^θ) + (δ(α, θ) * sum(z[z.>=0].^θ) / (1-α)^θ)
    return rand(InverseGamma(n/θ, b), 1)[1]
end
function logβCond(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real,
        σ::Real, τ::Real, λ::AbstractVector{<:Real})
    z = y - X*β
    b = δ(α, θ)/σ * (sum((.-z[z.< 0]).^θ) / α^θ + sum(z[z.>=0].^θ) / (1-α)^θ)
    return -b -1/(2*τ^2) * β'*diagm(λ.^(-2))*β
end
∂β(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}) = ForwardDiff.gradient(β -> logβCond(β, X, y, α, θ, σ, τ, λ), β)
∂β2(β::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real, λ::AbstractVector{<:Real}) = ForwardDiff.jacobian(β -> -∂β(β, X, y, α, θ, σ, τ, λ), β)

function sampleβ(β::AbstractVector{<:Real}, ε::Real,  X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real}, α::Real, θ::Real, σ::Real, τ::Real, MALA::Bool = true) where {T <: Real}
    λ = abs.(rand(Cauchy(0,1), length(β)))
    if MALA
        ∇ = ∂β(β, X, y, α, θ, σ, τ, λ)
        H = (∂β2(β, X, y, α, maximum([θ, 1.01]), σ, τ, λ))^(-1) |> Symmetric
        prop = β + ε^2 * H / 2 * ∇ + ε * √H * vec(rand(MvNormal(zeros(length(β)), 1), 1))
        global test, H, ∇ = prop, H, ∇
        ∇ₚ = ∂β(prop, X, y, α, θ, σ, τ, λ)
        Hₚ = (∂β2(prop, X, y, α, maximum([θ, 1.01]), σ, τ, λ))^(-1) |> Symmetric
        αᵦ = logβCond(prop, X, y, α, θ, σ, τ, λ) - logβCond(β, X, y, α, θ, σ, τ, λ)
        αᵦ += - logpdf(MvNormal(β + ε .^2 / 2 * ∇, ε^2 * H), prop) + logpdf(MvNormal(prop + ε^2/2 * ∇ₚ, ε^2 * Hₚ), β)
        αᵦ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
    else
        prop = vec(rand(MvNormal(β, typeof(ε) <: AbstractArray ? diagm(ε) : ε), 1))
        logβCond(prop, X, y, α, θ, σ, τ, λ) - logβCond(β, X, y, α, θ, σ, τ, λ) >
            log(rand(Uniform(0,1), 1)[1]) ? prop : β
    end
end

n = 500;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(Laplace(0.,1.), n);

N = 1000;
θs, σs = zeros(N), zeros(N);
βs = zeros(N, size(X)[2]);
θs[1] = 1.;
σs[1] = 1.;
βs[1,:] = β;

for i in 2:N
    βs[i,:] = sampleβ(βs[i-1,:], 0.01, X, y, α, θs[i-1], σs[i-1], 1., true)
    θs[i] = sampleθ(θs[i-1], X, y, βs[i,:], α, 0.05)
    σs[i] = sampleσ(X, y, βs[i,:], α, θs[i])
end

plot(βs[:, 1])
plot(θs)
plot(σs)
1-((βs[2:length(θs), 1] .=== βs[1:(length(θs) - 1), 1]) |> mean)


## All covariates
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);
β = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]
α, θ, σ, τ = 0.99, 1.5, 12.5, 100
ε = 0.1
#y = log.(y)

N = 10000
βs = zeros(N, size(X)[2])
βs[1,:] = βinit

for i in 2:N
    βs[i,:] = sampleβ(βs[i-1,:], 0.01, X, log.(y), 0.8, 1., 2., 100.)
end

diagm(0.01, 10)

plot(βs[:,1])

λ = abs.(rand(Cauchy(0,1), length(β)))
∇ = ∂β(β, X, y, α, θ, σ, τ, λ)
H = (∂β2(β, X, y, α, θ, σ, τ, λ))^(-1) |> Symmetric
prop = β + ε^2 * H / 2 * ∇ + ε * √H * vec(rand(MvNormal(zeros(length(β)), 1), 1))
∇ₚ = ∂β(prop, X, y, α, θ, σ, τ, λ)
Hₚ = (∂β2(prop, X, y, α, θ, σ, τ, λ))^(-1) |> Symmetric
logpdf(MvNormal(β + ε .^2 / 2 * ∇, ε^2 * H), prop) + logpdf(MvNormal(prop + ε^2/2 * ∇ₚ, ε^2 * Hₚ), β)
