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

function leapfrog(β::Array{<:Real, 1}, p::Array{<:Real, 1}, M::Array{<:Real, 2}, ϵ::Real,
    X::Array{<:Real, 2}, y::Array{<:Real, 1},
    α::Real, θ::Real, σ::Real, τ::Real, λ::Array{<:Real, 1})
    p += ϵ/2 * ∂β(β, X, y, α, θ, σ, τ, λ)
    β = β .+ ϵ * inv(M) * p
    p += ϵ/2 * ∂β(β, X, y, α, θ, σ, τ, λ)
    return (β, p)
end


function HMC(β::Array{<:Real, 1}, L::Integer, ϵ::Real, X::Array{<:Real, 2}, y::Array{<:Real, 1},
        α::Real, θ::Real, σ::Real, τ::Real, λ::Array{<:Real, 1})
        M = diagm([1 for i in 1:length(β)])
        p = rand(MvNormal(zeros(length(β))), 1) |> vec
        prop, pn = leapfrog(β, p, M, ϵ, X, y, α, θ, σ, 100., λ)

        for i in 2:L
            prop, pn = leapfrog(prop, pn, M, ϵ, X, y, α, θ, σ, 100., λ)
        end

        numer = exp(logβCond(prop, X, y, α, θ, σ, 100., λ) - 0.5*pn'*M*pn)
        denom =  exp(logβCond(β, X, y, α, θ, σ, 100., λ) - 0.5*p'*M*p)

        rand(Uniform(), 1)[1] < minimum([1,numer/denom]) ? prop : β
end

∇ = ∂β(β, X, y, α, θ, σ, 100., [1., 1.])
ε = [0.01, 0.05]

a = vec(rand(MvNormal(zeros(length(β)), 1), 1))

a[1] * ε[1]
a[2] * ε[2]

ε .* a

β + ε.^2/2 .* ∇ + ε .* a

n = 1000;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X*β .+ rand(Laplace(0, 1), n)

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;

y = dat[:, :osvAll]
X = dat[:, Not("osvAll")] |> Matrix
X = X[findall(y.>0),:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);
b = β[N,:]
N = 500000
β = zeros(N, 9)
β[1,:] = b#[2.1, 0.8]

β[1,:] = inv(X'*X)*X'*log.(y)
β[1,:] = β[N,:]

for i in 2:N
    global y = typeof(y[1]) <: Integer ?
        log.(y + rand(Uniform(), length(y)) .- α) : y
    λ = abs.(rand(Cauchy(0,1), size(β)[2]))
    β[i,:] = HMC(β[i-1,:], 15, 0.0003, X, y, α, θ, σ, 100., λ)
end

p = 3
plot(β[1:N, p])
plot(1:N, cumsum(β[:,p])./(1:N))
1-((β[2:N, 1] .=== β[1:(N - 1), 1]) |> mean)

median(β[5000:N, 2])
