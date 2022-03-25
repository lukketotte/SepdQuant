using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff, QuantileRegressions, QuadGK
include("../../QuantileReg/QuantileReg.jl")
include("../aepd.jl")
include("../../QuantileReg/FreqQuantileReg.jl")
using .AEPD, .QuantileReg, .FreqQuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles, HTTP

## Testing package
using SepdQuantile, Random, Distributions

n = 200;
X = hcat(ones(n), rand(Normal(), n), rand(Normal(), n))
y = X*ones(3) + rand(TDist(3), n)#(1 .+ X[:,2]).*rand(Normal(), n)
y = X*ones(3) + [rand(Uniform()) < 0.8 ? rand(Normal()) : rand(Normal(0, 3)) for i in 1:n]
y = X*ones(3) + rand(Chisq(3), n)

par = Sampler(y, X, 0.5, 10000, 5, 2000);
β, θ, σ, α = mcmc(par, 0.5, 0.5, 1., 1, 2, 0.5, [0., 0., 0.]);
par.α = mcτ(0.5, mean(α), mean(θ), mean(σ))

bqr = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3 + x4), x, 0.5) |> coef
βlp = mcmc(par, .25, mean(θ), mean(σ), [0., 0., 0.])
blp = mean(βlp, dims = 1)'

mean((blp-ones(3)).^2)
mean((bqr-ones(3)).^2)

plot(βlp[:,2])
plot(α)
plot(θ)
plot(β[:,1])

acceptance(βlp)
acceptance(α)
acceptance(β)
acceptance(θ)

##
f(x, b, p, α, μ, σ) = abs(x-b)^(p-1) * pdf(Aepd(μ, σ, p, α), x)

function quantconvert(q, p, α, μ, σ)
    a₁ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, Inf)[1]
    a₂ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, q)[1]
    1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
end

function mcτ(τ, α, p, σ, n = 1000, N = 1000)
    res = zeros(N)
    for i in 1:N
        dat = rand(Aepd(0, σ, p, α), n)
        q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, τ) |> coef;
        res[i] = quantconvert(q[1], p, α, 0, σ)
    end
    mean(res)
end

function mcτ(τ, α, p, σ, n = 1000, N = 1000)
    res = zeros(N)
    for i in 1:N
        dat = rand(Aepd(0, σ, p, α), n)
        q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, τ) |> coef;
        a₁ = mean(abs.(dat .- q).^(p-1))
        a₂ = mean(abs.(dat .- q).^(p-1) .* (dat .< q))
        res[i] = 1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
    end
    mean(res)
end


dat = HTTP.get("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-max-temperatures.csv") |> x -> CSV.File(x.body) |> DataFrame
scatter(dat[1:(size(dat,1)-1), :Temperature], dat[2:size(dat,1), :Temperature])

y = log.(dat[2:size(dat, 1),:Temperature])
X = hcat(ones(length(y)), log.(dat[1:(size(dat,1)-1),:Temperature]))

par = Sampler(y, X, 0.5, 10000, 1, 1000);
β, θ, σ, α = mcmc(par, .3, 0.11, 1.1, 2, 1, 0.5, rand(size(par.X, 2)));

mcτ(0.7, mean(α), mean(θ), mean(σ))
mcτ2(0.7, mean(α), mean(θ), mean(σ))

par.α = mcτ2(0.7, mean(α), mean(θ), mean(σ), 5000)
#par.nMCMC, par.burnIn = 6000, 1000
βres = mcmc(par, 0.5, mean(θ), mean(σ), rand(size(par.X, 2)))
mean(par.y .<= par.X *  median(βres, dims = 1)')
