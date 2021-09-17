using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, Optim, QuantileRegressions
using DataFrames, StatFiles, CSVFiles
include("../aepd.jl")
using .AEPD
using ProgressMeter

struct ConvergenceError <:Exception
    msg::String
end

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

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

# gives simiar results
α = .1
ys = log.(y + rand(Uniform(), length(y)) .- α)
optimFunc = TwiceDifferentiable(vars -> -Q(vars[1], vars[2:(size(X)[2]+1)], ys, X, α), zeros(size(X)[2] + 1), autodiff =:forward)
optimum = optimize(optimFunc, zeros(size(X)[2] + 1))


optimum = optimize(optimFunc, [ log(1), -0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28])
vals = Optim.minimizer(optimum)
exp(vals[1])
sample(1:length(y), length(y))

function bootstrap(ϑ::Real, y::AbstractVector{<:Integer},  X::AbstractMatrix{<:Real}, α::Real, N::Integer)
    θ = zeros(N)
    β = zeros(N, size(X)[2])
    σ = zeros(N)
    p = Progress(N, dt=0.5,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen=50, color=:green)
    ϑ = repeat([ϑ], size(X)[2] + 1)
    for i in 1:N
        next!(p)
        ys = log.(y[sample(1:length(y), length(y))] + rand(Uniform(), length(y)) .- α)
        # initial values
        ϑ[2:(size(X)[2] + 1)] = coef(qreg(@formula(y ~ x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9), hcat(DataFrame(y = ys), DataFrame(X, :auto)), α))
        optimFunc = TwiceDifferentiable(vars -> -Q(vars[1], vars[2:(size(X)[2]+1)], ys, X, α), ϑ, autodiff =:forward)
        try
            optimum = optimize(optimFunc, ϑ)
            vals = Optim.minimizer(optimum)
            θ[i] = exp(vals[1])
            β[i,:] = vals[2:length(vals)]
            σ[i] = σhat(ys, X, β[i,:], θ[i], α)
        catch e
            continue
        end
    end
    β[θ .> 0.3,:], θ[θ .> 0.3], σ[θ .> 0.3]
end

β, θ, σ = bootstrap(log(1), y, X, 0.9, 10000)

median(θ)
median(σ)
sort(θ[θ .> 0.5])[Integer(round((0.05/2) * 500))]

using QuantileRegressions

mod = qreg(@formula(y ~ x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9), hcat(DataFrame(y = log.(y)), DataFrame(X, :auto)), .5)

append!(ϑ, coef(mod))
