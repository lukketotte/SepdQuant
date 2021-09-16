using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff
using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)
include("../../QuantileReg/QuantileReg.jl")
using .QuantileReg

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);
Z = X[:, [1, 4, 7, 8, 9]]

names(dat)
## try with θ as a linear model
βinit = [-1.85, -0.046, -2.477, 2.628, 0.0, 0.412, 1.522, -0.003, 0.206]
par = Sampler(y, X,0.5, 10000, 1, 1);

function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function δ(α::Real, θ::AbstractVector{<:Real})
    return 2*(α*(1-α)) .^ θ ./ (α .^ θ + (1-α) .^ θ)
end

z  = par.y-par.X*βinit
η = Z * [1, 0.1, 0.001, 0.1]

a = (sum(δ(par.α, η[z.<0]).*(.-z[z.<0]).^η[z.<0]/par.α.^η[z.<0]) + sum(δ(par.α, η[z.>=0]).*z[z.>=0].^η[z.>=0] /(1-par.α).^η[z.>=0]))
n = length(z)
sum(n./η .* log.(δ(par.α, η))  .- n*log.(gamma.(1 .+1 ./η)) .- n*log(a)./η .+ loggamma.(n./η))


function θcond(θ::AbstractVector{<:Real}, s::Sampler, Z::AbstractMatrix{<:Real}, β::AbstractVector{<:Real})
    z  = s.y-s.X*β
    η = exp.(Z * θ)
    n = length(z)
    neg = z .< 0
    pos = z .>= 0
    a = (sum(δ(par.α, η[neg]).*(.-z[neg]).^η[neg]/par.α.^η[neg]) + sum(δ(par.α, η[pos]).*z[pos].^η[pos] /(1-par.α).^η[pos]))
    return sum(n./η .* log.(δ(par.α, η))  .- n*log.(gamma.(1 .+1 ./η)) .- n*log(a)./η .+ loggamma.(n./η))
end

function sampleθ(s::Sampler, θ::AbstractVector{<:Real}, Z::AbstractMatrix{<:Real}, β::AbstractVector{<:Real}, ε::Real)
    ∇ = ∂θ(θ, s, β, Z)
    #H = ∂θ2(θ, s, β, Z)^(-1) |> Symmetric
    prop = θ + (ε^2 / 2)  * ∇ + ε * vec(rand(MvNormal(zeros(length(θ)), 1), 1))
    ∇ₚ = ∂θ(prop, s, β, Z)
    #Hₚ = ∂θ2(prop, s, β, Z)^(-1) |> Symmetric
    α = θcond(prop, s, Z, β) - θcond(θ, s, Z, β)
    α += -logpdf(MvNormal(θ + (ε^2 / 2) * ∇, ε^2), prop) + logpdf(MvNormal(prop + (ε^2 / 2)* ∇ₚ, ε^2), θ)
    return log(rand(Uniform(0,1), 1)[1]) <= α ? prop : θ
end

function sampleθ(s::Sampler, θ::AbstractVector{<:Real}, Z::AbstractMatrix{<:Real},
    β::AbstractVector{<:Real}, ε::AbstractVector{<:Real})
    ∇ = ∂θ(θ, s, β, Z)
    prop = θ + vec(rand(MvNormal(zeros(length(θ)), diagm(ε)), 1))
    α = θcond(prop, s, Z, β) - θcond(θ, s, Z, β)
    return log(rand(Uniform(0,1), 1)[1]) <= α ? prop : θ
end

function sampleσ(s::Sampler, θ::AbstractVector{<:Real}, Z::AbstractMatrix{<:Real}, β::AbstractVector{<:Real})
    z = s.y - s.X*β
    η = exp.(Z * θ)
    b = sum(δ(s.α, η[z.<0]) .* (.-z[z.<0]).^ η[z.<0] ./ s.α.^η[z.<0])
    b += sum(δ(s.α, η[z.>=0]) .* (z[z.>=0].^η[z.>=0] ./ (1-s.α).^η[z.>=0]))
    return rand(InverseGamma(sum(1/η) + 1, b + 1), 1)[1]
end

ε^2 .* vec(rand(MvNormal(zeros(4), 1), 1))

∂θ(θ::AbstractVector{<:Real}, s::Sampler, β::AbstractVector{<:Real}, Z::AbstractMatrix{<:Real}) = ForwardDiff.gradient(θ -> θcond(θ, s, Z, β), θ)
∂θ2(θ::AbstractVector{<:Real}, s::Sampler, β::AbstractVector{<:Real}, Z::AbstractMatrix{<:Real}) = ForwardDiff.jacobian(θ -> -∂θ(θ, s, β, Z), θ)

∂θ2(θ[N,:], par, βinit, Z)^(-1)

N = 20000
σ = zeros(N)
θ = zeros(N, size(Z)[2])
σ[1] = 2.
θinit = θ[N,:]
θ[1,:] = [.01 for i in 1:size(Z)[2]]
θ[1,:] = θinit
ε = [0.00000001, 0.001, 0.0001, 0.00000001, 0.0001]
for i in 2:N
    θ[i,:] = sampleθ(par, θ[i-1,:], Z, βinit, ε)
    # σ[i] = sampleσ(par, θ[i,:], Z, βinit)
end
1-((θ[2:length(σ), 1] .=== θ[1:(length(σ) - 1), 1]) |> mean)
plot(θ[:,3])

median(σ)
plot(σ)
plot(θ[:,5])

[median(θ[:,i]) for i in 1:size(Z)[2]]
exp.(Z*[median(θ[:,i]) for i in 1:size(Z)[2]]) |> maximum
