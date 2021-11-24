using Distributions, QuantileRegressions, LinearAlgebra, Random, SpecialFunctions, QuadGK
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles, KernelDensity
theme(:juno)

f(x, b, p, α, μ, σ) = abs(x-b)^(p-1) * pdf(Aepd(μ, σ, p, α), x)

function quantconvert(q, p, α, μ, σ)
    a₁ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, Inf)[1]
    a₂ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, q)[1]
    1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
end

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

##
α = 0.1;
par = Sampler(y, X, α, 20000, 5, 5000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, α) |> coef;
β, θ, σ = mcmc(par, .6, 1., 1.2, 4, b);



β, θ, σ, α = mcmc(par, 0.8, .25, 1.5, 1, 2, 0.5, b);
plot(α)
acceptance(θ)
acceptance(β)
plot(β[:,3])
plot(θ)
plot(cumsum(β[:,2]) ./ (1:size(β,1)))
plot(cumsum(θ) ./ (1:size(β,1)))
plot(cumsum(α) ./ (1:size(β,1)))

median(θ)#1.9
median(σ)#4.07
median(α)#0.445

b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, 0.5) |> coef;
q = X * b;
μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));
τ = [quantconvert(q[j], 1.96, 0.397, μ[j],
    3.98) for j in 1:length(par.y)] |> mean

par = Sampler(y, X, τ, 10000, 2, 5000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, 0.5) |> coef;
βres, _ = mcmc(par, 1.3, 1.899, 4.07, b);
plot(βres[:,6])
acceptance(βres)
plot(cumsum(βres[:,7]) ./ (1:size(βres,1)))

[par.y[i] <= X[i,:] ⋅ b for i in 1:length(y)] |> mean
[par.y[i] <= X[i,:] ⋅ median(β, dims = 1)  for i in 1:length(par.y)] |> mean
[par.y[i] <= X[i,:] ⋅ median(βres, dims = 1)  for i in 1:length(par.y)] |> mean

X*median(βres, dims = 1)'
X*b

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

par = Sampler(y, X, 0.5, 21000, 5, 6000);
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
