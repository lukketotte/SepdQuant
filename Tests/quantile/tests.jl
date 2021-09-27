using Distributions, QuantileRegressions
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)

n = 500;
β = [2.1, 0.8]
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(Aepd(0., 3., 0.2, 0.5), n);
scatter(X[:,2], y)


freq = qreg(@formula(y ~ x), DataFrame(y = y, x = X[:,2]), 0.9) |> coef

par = Sampler(y, X, 0.9, 10000, 3, 2000);
β, θ, σ = mcmc(par, 1000000., .5, 0.1, [2.1, 0.8], 2., 1.);

plot(β[:,1])
plot(θ)
plot(σ)

plot(cumsum(β[:,1])./(1:length(θ)))
1-((β[2:length(θ), 1] .=== β[1:(length(θ) - 1), 1]) |> mean)
bayes = [median(β[:,i]) for i in 1:2]

Q = zeros(length(y))
for i ∈ 1:length(y)
    Q[i] = y[i] <= X[i,:] ⋅ bayes
end
mean(Q)



α = range(0.1, 0.9, length = 9)
freqDiff = zeros(9)
for j ∈ 1:9
    β = qreg(@formula(y ~ x), DataFrame(y = y, x = X[:,2]), α[j]) |> coef
    Q = zeros(length(y))
    for i ∈ 1:length(y)
        Q[i] = y[i] <= X[i,:] ⋅ β
    end
    freqDiff[j] = mean(Q)
end

freqDiff
