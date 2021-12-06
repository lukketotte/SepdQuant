using Distributions, QuantileRegressions, LinearAlgebra, Random, SpecialFunctions, QuadGK
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles, KernelDensity
theme(:juno)

using RDatasets

## QuantileReg data
dat = load(string(pwd(), "/Tests/data/Immunog.csv")) |> DataFrame;
names(dat)
y = dat[:, :IgG];
X = hcat(ones(size(dat,1)), dat[:,:Age], dat[:,:Age].^2)

par = Sampler(y, X, 0.5, 6000, 5, 2000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4), x, 0.5) |> coef;
β, θ, σ, α = mcmc(par, 1.3, 0.5, 1.5, 2, 1, 0.5, b);

plot(β[:,2])
plot(σ)
plot(θ)
plot(α)
acceptance(β)
acceptance(θ)
acceptance(α)

b = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3 + x4), x, 0.9) |> coef;
q = X * b;
μ = X * mean(β, dims = 1)' |> x -> reshape(x, size(x, 1));
τ = [quantconvert(q[j], mean(θ), mean(α), μ[j], mean(σ)) for j in 1:length(par.y)] |> mean

par.α = mcτ(0.1, median(α), median(θ), median(σ))

reps = 50
res = zeros(reps)
for i in 1:reps
    βres, _ = mcmc(par, 2, median(θ), median(σ), b);
    res[i] = ([par.y[i] <= X[i,:] ⋅ median(βres, dims = 1)  for i in 1:length(par.y)] |> mean)
end

mean(res)

##
RDatasets.datasets("datasets")
dat = dataset("datasets", "mtcars")

y = log.(dat[:, :MPG])
X = log.(dat[:, Not(["FoodExp"])] |> Matrix)
X = hcat([1. for i in 1:length(y)], dat[:, :WT]);

par = Sampler(y, X, 0.5, 100000, 5, 1000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3), x, 0.5) |> coef;
println(b)
median(β, dims = 1) |> println
β, θ, σ, α = mcmc(par, 0.01, 0.1, 0.1, 2, 1, 0.5, b);
acceptance(β)
plot(β[:,2])
plot(σ)
plot(θ)
plot(α)
acceptance(θ)



b = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3), x, 0.1) |> coef;
q = X * b;
μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));
τ = [quantconvert(q[j], median(θ), median(α), μ[j],
    median(σ)) for j in 1:length(par.y)] |> mean

par = Sampler(y, X, τ, 10000, 2, 5000);
βres, _ = mcmc(par, 1.3, median(θ), median(σ), b);
plot(βres[:,1])

[par.y[i] <= X[i,:] ⋅ b for i in 1:length(y)] |> mean
[par.y[i] <= X[i,:] ⋅ median(βres, dims = 1)  for i in 1:length(par.y)] |> mean

## Boston
#dat = load(string(pwd(), "/Tests/data/BostonHousing2.csv")) |> DataFrame;
dat = dataset("MASS", "Boston")
y = log.(dat[:, :MedV])
X = dat[:, Not(["MedV"])] |> Matrix
X = hcat([1 for i in 1:length(y)], X);

par = Sampler(y, X, 0.5, 10000, 5, 2000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 +
        x11 + x12 + x13 + x14 + x15), x, 0.5) |> coef;

β, θ, σ, α = mcmc(par, 0.25, .25, 0.05, 1, 2, 0.5, b);

acceptance(β)
acceptance(θ)
acceptance(α)
plot(α)
plot(θ)
plot(β[:,2])

b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 +
        x11 + x12 + x13 + x14 + x15), x, 0.1) |> coef;
q = X * b;
μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));
τ = [quantconvert(q[j], median(θ), median(α), μ[j],
    median(σ)) for j in 1:length(par.y)] |> mean

par.α = τ;
par.nMCMC = 4000
βres, _ = mcmc(par, 0.01, median(θ), median(σ), b);
plot(βres[:,1])
[par.y[i] <= X[i,:] ⋅ median(βres, dims = 1)  for i in 1:length(par.y)] |> mean

## Fishery data
using HTTP
dat = HTTP.get("https://people.brandeis.edu/~kgraddy/datasets/fish.out") |> x -> CSV.File(x.body) |> DataFrame
names(dat)

y = dat[:,:qty]
X = hcat(ones(length(y)), dat[:,:price])

par = Sampler(y, X, 0.5, 10000, 1, 5000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3), x, 0.5) |> coef;

β, θ, σ, α = mcmc(par, 1, 0.25, 0.7, 1, 2, 0.5, b);

q = par.X * (DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3), x, 0.4) |> coef);
μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));
τ = [quantconvert(q[j], median(θ), median(α), μ[j], median(σ)) for j in 1:length(par.y)] |> mean

par = Sampler(y, X, τ, 10000, 1, 5000);

n = 2000
s,p,a = median(σ), median(θ), median(α)
res = zeros(n)
for i in 1:n
    dat = rand(Aepd(0, s, p, a), n)
    q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.4) |> coef;
    res[i] = quantconvert(q[1], p, a, 0, s)
end
par.α = mean(res)

βres, _ = mcmc(par, 0.4, median(θ), median(σ), b);
plot(βres[:,2])

[par.y[i] <= q[i] for i in 1:length(y)] |> mean
[par.y[i] <= X[i,:] ⋅ median(βres, dims = 1)  for i in 1:length(par.y)] |> mean

## Prostate data
dat = load(string(pwd(), "/Tests/data/prostate.csv")) |> DataFrame;
names(dat)
y = dat[:, :lpsa]
X = hcat(ones(length(y)), Matrix(dat[:,2:9]))

par = Sampler(y, X, 0.5, 10000, 1, 5000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, 0.5) |> coef;

β, θ, σ, α = mcmc(par, 1, 0.25, 0.7, 1, 2, 0.5, b);
acceptance(β)
acceptance(α)
plot(σ)
plot(β[:,2])
plot(θ)

b = DataFrame(hcat(par.y, par.X), :auto) |> x -> qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, 0.1) |> coef;
q = par.X * b

n = 2000
s,p,a = median(σ), median(θ), median(α)
res = zeros(n)
for i in 1:n
    dat = rand(Aepd(0, s, p, a), n)
    q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.1) |> coef;
    res[i] = quantconvert(q[1], p, a, 0, s)
end
par.α = mean(res)

βres, _ = mcmc(par, 0.00005, median(θ), median(σ), b);
plot(βres[:,4])
acceptance(βres)

[par.y[i] <= q[i] for i in 1:length(y)] |> mean
[par.y[i] <= X[i,:] ⋅ median(βres, dims = 1)  for i in 1:length(par.y)] |> mean
