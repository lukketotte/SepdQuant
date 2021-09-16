using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, Optim
using DataFrames, StatFiles, CSVFiles
include("../aepd.jl")
using .AEPD

struct ConvergenceError <:Exception
    msg::String
end

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

α = 0.5
ys = log.(y + rand(Uniform(), length(y)) .- α)

function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function σhat(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, β::AbstractVector{<:Real}, θ::Real, α::Real)
    z  = y-X*β
    θ / length(z) * δ(α, θ)*(sum((.-z[z.<0]).^θ)/α^θ + sum(z[z.>=0].^θ)/(1-α)^θ)
end

function Q(θ::Real, β::AbstractVector{<:Real}, y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, α::Real)
    θ = exp(θ)
    z  = y-X*β
    a = sum((.-z[z.<0]).^θ)/α^θ + sum(z[z.>=0].^θ)/(1-α)^θ
    d = δ(α,θ)
    log(d)/θ - 1/θ*log((θ*d*a)/length(z)) - loggamma(1+1/θ) - 1/θ
end

β = [-1.85, -0.046, -2.477, 2.628, 0.0, 0.412, 1.522, -0.003, 0.206]
ϑ = [-1.85, -0.046, -2.477, 2.628, 0.0, 0.412, 1.522, -0.003, 0.206, log(1)]

n = 1000;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
ys = X * β .+ rand(Laplace(), n);

# gives simiar results
α = .1
ys = log.(y + rand(Uniform(), length(y)) .- α)
optimFunc = TwiceDifferentiable(vars -> -Q(vars[1], vars[2:(size(X)[2]+1)], ys, X, α), ones(size(X)[2] + 1), autodiff =:forward)
optimum = optimize(optimFunc, ones(size(X)[2] + 1))
vals = Optim.minimizer(optimum)
exp(vals[1])
