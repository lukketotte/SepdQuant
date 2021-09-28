using Distributions, QuantileRegressions, LinearAlgebra
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

## Simulation study
μ₁, μ₂, μ₃ = [1., 0.], [4.,0.], [-2.,0.]
Σ = [1 0.6 ; 0.6 1]
η = (0.85, 0.0725, 0.0725)

n = 100
y = zeros(n, 2)
for i ∈ 1:n
    a = rand(Uniform(), 1)[1]
    if a <= 0.85
        y[i,:] = rand(MvNormal(μ₁, Σ), 1)
    elseif a <= (0.85 + 0.0725)
        y[i,:] = rand(MvNormal(μ₂, Σ), 1)
    else
        y[i,:] = rand(MvNormal(μ₃, Σ), 1)
    end
end

scatter(y[:,2], y[:,1])

X = hcat([1 for i in 1:n], y[:,2])
y = y[:,1]

par = Sampler(y, X, 0.5, 40000, 10, 10000);
β, θ, σ = mcmc(par, 100000., 1., 0.01, [1., 0.8]);

plot(β[:,2])
[median(β[:,i]) for i in 1:2]
plot(θ)
plot(σ)

√var(β[:,1])
1-((β[2:length(θ), 1] .=== β[1:(length(θ) - 1), 1]) |> mean)
1-((θ[2:length(θ)] .=== θ[1:(length(θ) - 1)]) |> mean)

logpdf(Beta(2, 2), 1/2)

## Simulation
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
