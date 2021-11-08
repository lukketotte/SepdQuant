using Distributions, QuantileRegressions, LinearAlgebra, Random, SpecialFunctions, QuadGK
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)

##
f(x, b, p, α, μ, σ) = abs(x-b)^(p-1) * pdf(Aepd(μ, σ, p, α), x)

function quantconvert(q, p, α, μ, σ)
    a₁ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, Inf)[1]
    a₂ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, q)[1]
    1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
end


α = range(0.01, 0.99, length = 101);
# normal
p = [0.75, 1.5, 2, 3];
ε = [0.05, 0.2, 0.1, 0.05];

# Laplace
p = [0.75, 1.25, 1.5];
ε = [0.025, 0.1, 0.1];
n = 1000;
x = rand(Normal(), n)
y = 0.5 .+ 1.2 .* x + raepd(n, 2, 2, 0.7);
X = hcat(ones(n), x)

par = Sampler(y,X, 0.5, 21000, 5, 6000);
β, θ, σ, α = mcmc(par, 0.4, 0.4, 0.5, 1, 2, 0.5)
acceptance(α)

plot(α)
plot(θ)
plot(σ)
plot(β[:,1])

β1 = (DataFrame(hcat(y, ones(n), x), :auto) |> x -> qreg(@formula(x1 ~  x3), x, 0.9) |> coef)
β2 = [mean(β[:,i]) for i in 1:size(β, 2)]

τ = [quantconvert(X[i,:] ⋅ β1, mean(θ), mean(α), X[i,:] ⋅ β2, mean(σ)) for i in 1:n] |> mean

par = Sampler(y,X, τ, 21000, 5, 6000);
β, θ, σ= mcmc(par, 0.4, 0.5, 1, 2)
β2 = [mean(β[:,i]) for i in 1:size(β, 2)]

[y[i] <= X[i,:] ⋅ β2 for i in 1:length(y)] |> mean
[y[i] <= X[i,:] ⋅ β1 for i in 1:length(y)] |> mean

println(β2)
