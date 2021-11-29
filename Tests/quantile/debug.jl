using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff

include("../../QuantileReg/QuantileReg.jl")
include("../aepd.jl")
using .QuantileReg, .AEPD,


n = 5000
p, a, s = 1.21, 0.1, 1

p, a, s = 1.9, 0.444, 4.07
p, a, s = median(θ), median(α), median(σ)
dat = rand(Aepd(0, s, p, a), n)

par2 = Sampler(dat, hcat(ones(n)), a, 5000, 4, 1000);
init = DataFrame(hcat(par2.y), :auto) |> x -> qreg(@formula(x1 ~  1), x, par2.α) |> coef;
bet, _ =  mcmc(par2, 1., p, s, init);
acceptance(bet)
plot(bet[:,1])

n = 500
N = 10000
res = zeros(N)
for i in 1:N
    dat = rand(Aepd(0, s, p, a), n)
    q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.1) |> coef;
    res[i] = quantconvert(q[1], p, a, 0, s)
end
τ = mean(res)


y = rand(Aepd(0, s, p, a), 100000)
q = DataFrame(hcat(y), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.9) |> coef;

1/((mean(abs.(y - ones(length(y)) .* q).^(p-1)) / mean(abs.(y - ones(length(y)) .* q).^(p-1) .* (y .<= q)) - 1)^(1/p) + 1)

##
n = 5000;
x = rand(Normal(), n);
y = 2.1 .+ 0.5 .* x + rand(Aepd(0, 1., 1.5, 0.2), n);
X = hcat(ones(n), x)

par = Sampler(y, X, 0.2, 10000, 5, 1000);
β, θ, σ, α = mcmc(par, 0.8, .25, 1.5, 1, 2, 0.5, [2.1, 0.5]);

b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~  x3), x, 0.9) |> coef;
q = X * b;
μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));
τ = [quantconvert(q[j], median(θ), median(α), μ[j],
    median(σ)) for j in 1:length(par.y)] |> mean

par = Sampler(y, X, τ, 10000, 2, 5000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
        qreg(@formula(x1 ~  x3), x, 0.9) |> coef;
βres, _ = mcmc(par, 1.3, median(θ), median(σ), b);

[par.y[i] <= X[i,:] ⋅ b for i in 1:length(y)] |> mean
[par.y[i] <= X[i,:] ⋅ median(βres, dims = 1)  for i in 1:length(par.y)] |> mean

n = 2000
res = zeros(n)
for i in 1:n
    dat = rand(Aepd(0, 1, 2, 0.7), n)
    q = DataFrame(hcat(dat), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.1) |> coef;
    res[i] = quantconvert(q[1], p, a, 0, s)
end
τ = mean(res)

## double check (this might be the real way to make a point actually)
n = 1000;
x = rand(Normal(), n);
# 2.5, 1.5
y = 2.1 .+ 0.5 .* x + rand(Aepd(0, 1, 2., 0.7), n);
X = hcat(ones(n), x)

reps = 10
α = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
res3 = zeros(length(α))
for i ∈ 1:length(α)
    temp = 0
    for j in 1:reps
        par = Sampler(y, X, α[i], 5000, 4, 1000)
        b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
            qreg(@formula(x1 ~ x3), x, α[i]) |> coef
        β, _, _ = mcmc(par, 1., 1., 1.2, 4, b)
        temp += abs(α[i] - ([par.y[i] <= X[i,:] ⋅ median(β, dims = 1)  for i in 1:length(par.y)] |> mean))/reps
    end
    res3[i] = temp
end

plot(α, res3, label = "p = 2.")
plot!(α, res2, label = "p = 1.5")
plot!(α, res3, label = "p = 0.7")

par = Sampler(y, X, α, 5000, 4, 1000);
b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
    qreg(@formula(x1 ~ x3), x, α) |> coef;
β, θ, σ = mcmc(par, 1., 1., 1.2, 4, b);
[par.y[i] <= X[i,:] ⋅ median(β, dims = 1)  for i in 1:length(par.y)] |> mean
acceptance(β)
plot(σ)
plot(θ)
plot(β[:,2])

β, θ, σ, α = mcmc(par, 0.8, .25, 1.5, 1, 2, 0.5, b);
q = X * b;
μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));
τ = [quantconvert(q[j], median(θ), median(α), μ[j],
    median(σ)) for j in 1:length(par.y)] |> mean

par = Sampler(y, X, τ, 6000, 5, 1000);
βres, _ = mcmc(par, 1.3, median(θ), median(σ), b);

acceptance(βres)

[par.y[i] <= X[i,:] ⋅ b for i in 1:length(y)] |> mean
[par.y[i] <= X[i,:] ⋅ median(β, dims = 1)  for i in 1:length(par.y)] |> mean
[par.y[i] <= X[i,:] ⋅ median(βres, dims = 1)  for i in 1:length(par.y)] |> mean
