using Distributions, QuantileRegressions, LinearAlgebra, Random, SpecialFunctions, QuadGK
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
include("../../QuantileReg/FreqQuantileReg.jl")
using .AEPD, .QuantileReg, .FreqQuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles, KernelDensity
theme(:juno)

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

##

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

## Testing johans code
control =  Dict(:tol => 1e-3, :max_iter => 1000, :max_upd => 0.3,
  :is_se => false, :est_beta => true, :est_sigma => true,
  :est_p => true, :est_tau => true)

quantfreq(y, X, control)


##
par = Sampler(y, X, 0.5, 11000, 1, 1000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, 0.5) |> coef;
βt, _, _ = mcmc(par, .6, 1., 1.2, 4, b);

[par.y[i] <= X[i,:] ⋅ median(βt, dims = 1) for i in 1:length(par.y)] |> mean

β, θ, σ, α = mcmc(par, 0.8, .2, 1., 1, 2, 0.5, zeros(size(X,2)));
median(θ)#1.9659771517710287
median(σ)#3.925316251127944
median(α)#0.37734587781936146
median(β, dims = 1) |> println
#hcat(θ, σ, α, β[:, 4]) |> x -> DataFrame(x, [''])
#CSV.write("mcmc.csv", DataFrame(shape = θ, scale = σ, skewness = α, beta = β[:,4]))
plot(β[:,1])
plot(θ)


b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, 0.1) |> coef;
q = par.X * b;
μ = par.X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));
τ = [quantconvert(q[j], median(θ), median(α), μ[j],
    median(σ)) for j in 1:length(par.y)] |> mean

par.α = mcτ(0.1, median(α), median(θ), median(σ), 5000)
par.α = mcτ(0.5, a, p, s, 5000)
par.burnIn = 5000
par.nMCMC = 15000
βres = mcmc(par, .7, p, s, zeros(size(par.X, 2)));
#βres, _ = mcmc(par, .7, median(θ), median(σ), zeros(size(X, 2)));
[par.y[i] <= par.X[i,:] ⋅ median(βres, dims = 1)  for i in 1:length(par.y)] |> mean

plot(βres[:,3])
acceptance(βres)

#[par.y[i] <= par.X[i,:] ⋅ b for i in 1:length(par.y)] |> mean

α = 0.9
N = 2*5000
bs = zeros(N)
B = zeros(N, size(X, 2))
for i ∈ 1:N
    ids = sample(1:length(par.y), length(par.y))
    B[i,:] =  DataFrame(hcat(par.y[ids], par.X[ids,:]), :auto) |> x ->
    qreg(@formula(x1 ~ x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, α) |> coef
end

p = 2
res = kde(B[:,p])
x = range(minimum(B[:,p]), maximum(B[:,p]), length = 500)
plot!(x, pdf(res, x))

vcat(DataFrame(param = colnames,
        value =[median(B[:, j]) for j in 1:size(X)[2]],
        l = [sort(B, dims = 1)[Integer(round((0.05/2) * size(B, 1))), j] for j in 1:size(X)[2]],
        u =  [sort(B, dims = 1)[Integer(round(((1-0.05/2)) * size(B, 1))), j] for j in 1:size(X)[2]], α = α, est = "Ald"),
    DataFrame(param = colnames,
        value =[median(βres[:, j]) for j in 1:size(X)[2]],
        l = [sort(βres, dims = 1)[Integer(round((0.05/2) * size(βres, 1))), j] for j in 1:size(X)[2]],
        u =  [sort(βres, dims = 1)[Integer(round(((1-0.05/2)) * size(βres, 1))), j] for j in 1:size(X)[2]], α = α, est = "Aepd")
) |> x -> CSV.write("ci_05.csv", x)


##
α = range(0.01, 0.99, length = 101);
τ = zeros(length(α))
μ = X * mean(β, dims = 1)' |> x -> reshape(x, size(x, 1))

for i ∈ 1:length(α)
    b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
        qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, α[i]) |> coef
    q = X * b
    temp = 0
    for j ∈ 1:length(par.y)
        temp += quantconvert(q[j], p, a, μ[j], s) / length(par.y)
    end
    τ[i] = temp
end

plot(α, τ)
plot!(α, α)

##
