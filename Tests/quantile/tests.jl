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
y = 0.5 .+ rand(Normal(), n);

par = Sampler(y, hcat(ones(n)), 0.5, 21000, 5, 6000);
β, θ, σ, α = mcmc(par, 0.4, 0.25, 0.2, 1, 2, 0.5)
1-((α[2:length(α)] .=== α[1:(length(α) - 1)]) |> mean)

plot(α)
plot(θ)
plot(σ)
plot(β[:,1])

βs = zeros(length(p));
σs = zeros(length(p));
for j in 1:length(p)
    β, σ = mcmc(par, ε[j], p[j], DataFrame(hcat(y, ones(n)), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.5) |> coef, 1.)
    βs[j] = median(β[:,1])
    σs[j] = median(σ)
end

β, σ = mcmc(par, 0.1, 1.5, DataFrame(hcat(y, ones(n)), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.5) |> coef, 1.);
1-((β[2:size(β, 1), 1] .=== β[1:(size(β, 1) - 1), 1]) |> mean)
plot(β[:,1])

βs
#βs = zeros(length(p))
βs[6] = median(β[:,1])

τl = zeros(length(α), length(p))
for i ∈ 1:length(α)
    for j ∈ 1:length(p)
        β1 = (DataFrame(hcat(y, ones(n)), :auto) |> x -> qreg(@formula(x1 ~  1), x, α[i]) |> coef)[1]
        τl[i, j] = quantconvert(β1, p[j], 0.5, βs[j], σs[j])
    end
end

τ

DataFrame(tau = reshape(τ, (404,)),
    p = repeat(p, inner = 101),
    a = repeat(α, outer = 4)) |> x -> save("res.csv", x)

α[51]
τ[51,:] |> println



plot(α, α)
plot!(α, τl[:,3])

τ[100,:]


for j ∈ 2:length(p)
    plot!(α, τ[:, 2])
end

α = 0.9
n = 1000
y = 0.5 .+ rand(Normal(), n)
#y = 0.5 .+ raepd(n, 2, 2, 0.1)
β1 = DataFrame(hcat(y, ones(n)), :auto) |> x -> qreg(@formula(x1 ~  1), x, α) |> coef
par = Sampler(y, hcat(ones(n)), 0.8360853, 11000, 5, 1000);
β, θ, σ = mcmc(par, 1000, 0.6, 0.1, 1., 1., β1);
β, σ = mcmc(par, 0.1, 2, β1, 1.);
1-((β[2:size(β, 1), 1] .=== β[1:(size(β, 1) - 1), 1]) |> mean)
plot(β[:,1])

mean(β[:,1])
mean(σ)
mean(θ)
β1[1]

mean(y .< β1)
mean(y .< mean(β[:,1]))

Q1 = zeros(length(y))
Q2 = zeros(length(y))
for i ∈ 1:length(y)
    Q2[i] = y[i] <= X[i,:] ⋅ β2
    Q1[i] = y[i] <= X[i,:] ⋅ β1
end
mean(Q1)
mean(Q2)

##
function raepd(n::Int, p1::Real, p2::Real, α::Real)
    u = rand(Uniform(), n)
    y1 = α .* rand(Gamma(1/p1, 1), n).^(1/p1) .* (sign.(u .- α) .- 1) ./ (2*gamma(1+1/p1))
    y2 = (1-α) .* rand(Gamma(1/p2, 1), n).^(1/p2) .* (sign.(u .- α) .+ 1) ./ (2*gamma(1+1/p2))
    y1 + y2
end

function simdata(x::AbstractVector{<:Real}, α::Real, f::Real, prop::Real)
    n = size(x, 1)
    y = zeros(n)
    for i ∈ 1:n
        u = rand(Uniform(), 1)[1]
        if u <= prop
            y[i] = 1.2 - 0.2 * x[i] + raepd(1, 1, 1, α)[1]
        else
            y[i] = 1.2 - 0.2 * x[i] + 10*raepd(1, 1, 1, α)[1]
        end
    end
    y
end

N = 5
MSEald = zeros(N)
MSEaepd = zeros(N)
α = 0.9
n = 500
x = rand(Normal(0, 1), n);

for i ∈ 1:N
    i % 10 == 0 && println(i)
    y = simdata(x, α, 10, 0.8)
    β1 = DataFrame(hcat(y, x), :auto) |> x -> qreg(@formula(x1 ~  x2), x, α) |> coef
    par = Sampler(y, hcat(ones(n), x), α, 30000, 5, 10000);
    β, θ, _ = mcmc(par, 1000, 0.6, 0.02, 1., 1., β1, verbose = false);
    β2 = median.([β[:,i] for i ∈ 1:2])

    MSEald[i] = sum((β1 - [1.2, -0.2]).^2)
    MSEaepd[i] = sum((β2 - [1.2, -0.2]).^2)
end

mean(MSEald)
mean(MSEaepd)
√var(MSEaepd)
√var(MSEald)

α = 0.1
n = 500;
x = rand(Normal(0, 1), n);
y = zeros(n)
for i ∈ 1:n
    u = rand(Uniform(), 1)[1]
    if u <= 1
        y[i] = 1.2 - 0.2 * x[i] + raepd(1, 2, 2, α)[1]
    else
        y[i] = 1.2 - 0.2 * x[i] + 5*raepd(1, 2, 2, α)[1]
    end
end

scatter(x, y)

β1 = DataFrame(hcat(y, x), :auto) |> x -> qreg(@formula(x1 ~  x2), x, α) |> coef
X =  hcat(ones(n), x)
par = Sampler(y, X, α, 40000, 5, 10000);
β, θ, _ = mcmc(par, 1000, 0.6, 0.01, 1., 2., β1);
1-((β[2:size(β, 1), 1] .=== β[1:(size(β, 1) - 1), 1]) |> mean)
plot(β[:,1])
plot(θ)

β2 = mean.([β[:,i] for i ∈ 1:2])# |> x -> reshape(x, 2)

sum((β1 - [1.2, -0.2]).^2)
sum((β2 - [1.2, -0.2]).^2)

Q1 = zeros(length(y))
Q2 = zeros(length(y))
for i ∈ 1:length(y)
    Q2[i] = y[i] <= X[i,:] ⋅ β2
    Q1[i] = y[i] <= X[i,:] ⋅ β1
end
mean(Q1)
mean(Q2)


Xt = hcat(ones(100), testX)
[testy[i] <= Xt[i,:] ⋅ β2 for i in 1:100] |> mean
[testy[i] <= Xt[i,:] ⋅ β1 for i in 1:100] |> mean

scatter(testX, testy, label = "")
plot!(testX, β1[1] .+ testX.*β1[2], label = "freq")
plot!(testX, β2[1] .+ testX.*β2[2], label = "aepd")
##
μ₁, μ₂, μ₃ = [1., 0.], [4.,0.], [-2.,0.]
Σ = [1 0.6 ; 0.6 1]
η = (0.85, 0.075, 0.075)

n = 200
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

X = hcat([1 for i in 1:n], y[:,2])
y = y[:,1]

par = Sampler(y, X, 0.9, 20000, 5, 5000);
β, _, _ = mcmc(par, 100000., .75, 0.031, 0.5, 0.5, [1., 0.8]);
βa, _ = mcmc(par, 100000., 0.03, [1., 0.8], 1);

plot(β[:,2])
plot!(βa[:,2])

√var(β[:,1])
1-((β[2:size(β,1), 1] .=== β[1:(size(β,1) - 1), 1]) |> mean)
1-((θ[2:length(θ)] .=== θ[1:(length(θ) - 1)]) |> mean)

logpdf(Beta(2, 2), 1/2)

freq = qreg(@formula(y ~ x), DataFrame(y = y, x = X[:,2]), 0.9) |> coef
bayes = [median(β[:,i]) for i in 1:2]
bayesqr = [median(βa[:,i]) for i in 1:2]

Q1 = zeros(length(y))
Q2 = zeros(length(y))
for i ∈ 1:length(y)
    Q1[i] = y[i] <= X[i,:] ⋅ bayes
    Q2[i] = y[i] <= X[i,:] ⋅ freq
end
mean(Q1)
mean(Q2)

scatter(y, X[:,2])
x = range(-3, 3, length = 1000)
ybayes = 1 .+ x .* bayes[2]
yfreq = 1 .+ x .* freq[2]
plot!(x, ybayes)
plot!(x, yfreq)

# bootstrap estimate
N = 1500
βboot = zeros(N, 2)
for i in 1:N
    ids = sample(1:length(y), length(y))
    βboot[i,:] = qreg(@formula(y ~ x), DataFrame(y = y[ids], x = X[ids,2]), 0.9) |> coef
end

b1 = sort(β, dims = 1)[[Integer(round((0.05/2) * size(β,1))),Integer(round((1-0.05/2) * size(β,1)))], 1]
b2 = sort(β, dims = 1)[[Integer(round((0.05/2) * size(β,1))),Integer(round((1-0.05/2) * size(β,1)))], 2]
f1 = sort(βboot, dims = 1)[[Integer(round((0.05/2) * size(βboot,1))),Integer(round((1-0.05/2) * size(βboot,1)))], 1]
f2 = sort(βboot, dims = 1)[[Integer(round((0.05/2) * size(βboot,1))),Integer(round((1-0.05/2) * size(βboot,1)))], 2]
a1 = sort(βa, dims = 1)[[Integer(round((0.05/2) * size(βa,1))),Integer(round((1-0.05/2) * size(βa,1)))], 1]
a2 = sort(βa, dims = 1)[[Integer(round((0.05/2) * size(βa,1))),Integer(round((1-0.05/2) * size(βa,1)))], 2]

errorsBayes = collect(zip(abs.(b1 .- bayes[1]), abs.(b2 .- bayes[2])))
errorsFreq = collect(zip(abs.(f1 .- freq[1]), abs.(f2 .- freq[2])))
errorsAld= collect(zip(abs.(a1 .- bayesqr[1]), abs.(a2 .- bayesqr[2])))

scatter([1,2], [bayes[1], bayes[2]], label="Bayesian", yerror=errorsBayes)
scatter!([1+0.05,2+0.05], [freq[1], freq[2]], label="Frequentist", yerror=errorsFreq)
scatter!([1-0.05,2-0.05], [freq[1], freq[2]], label="ALD", yerror=errorsAld)

histogram(rand(Pareto(.9, 100.), 1000))

## Generate data as in Johan and Randhal

# Actual data
dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
yt = dat[:, :osvAll]
yt = y[y.>0];


Random.seed!(731)
n = 500
p = 0.72
βtrue = (.5, 1.)
x1 = rand(Uniform(0, 1), n)
y = zeros(Int64, n)

Random.seed!(735)
Σ = [1. 0.4 0.3 ; 0.4 1.0 0.2 ; 0.3 0.2 1.]
n = 1000
X = hcat([1 for i in 1:n], rand(MvNormal([0, 0, 0], Σ), n)')# |> x -> reshape(x, n, 3)
b = [3., 0.2, 0.2, 0.2]
X * b
y = zeros(Int64, n)
for i ∈ 1:n
    if rand(Uniform(), 1)[1] < p
        #y[i] = rand(Poisson(exp(βtrue[1] + x1[i]*βtrue[2])), 1)[1]
        y[i] = rand(Poisson(exp(X[i,:] ⋅ b)), 1)[1]
    else
        # y[i] = Integer(round(rand(Pareto(0.5, 500), 1)[1]))
        y[i] = Integer(round(rand(Pareto(X[i,:] ⋅ b, 150), 1)[1]))
    end
end

# simulations more in line with Johans paper
α = 0.5
inits5 = qreg(@formula(x1 ~  x3 + x4 + x5), DataFrame(hcat(log.(y .- α), X), :auto), α) |> coef
inits9 = qreg(@formula(x1 ~  x3 + x4 + x5), DataFrame(hcat(log.(y .- α), X), :auto), α) |> coef
par = Sampler(y, X, α, 15000, 5, 5000);
β, θ, σ = mcmc(par, 1000., 0.1, 0.1, 0.5, 0.5, inits5);
β, θ, σ = mcmc(par, 1000., 0.4, 0.4, 0.5, 0.5, inits9);
βa, σa = mcmc(par, 1000., 0.006, inits5, 1); # α = 0.5
βa, σa = mcmc(par, 1000., 0.0025, inits9, 1);
1-((β[2:size(β,1), 1] .=== β[1:(size(β,1) - 1), 1]) |> mean)
1-((βa[2:size(βa,1), 1] .=== βa[1:(size(βa,1) - 1), 1]) |> mean)
p = 2
plot(β[:,1], label = "aepd")
plot!(βa[:,p], label = "ald")
plot(θ)


mean(β[:,1])

bayes = mean(β, dims = 1) |> x -> reshape(x, 4)
ald = mean(βa, dims = 1) |> x -> reshape(x, 4)
Q1, Q2 = zeros(length(y)), zeros(length(y))
for i ∈ 1:length(y)
    Q1[i] = y[i] <= ceil(exp(X[i,:] ⋅ bayes) + α - 1)
    Q2[i] = y[i] <= ceil(exp(X[i,:] ⋅ ald) + α - 1)
end
mean(Q1)
mean(Q2)

# α = 0.5, ϵ = 0.02, ϵθ = 0.01
# α = 0.9, ϵ = 0.0011, ϵθ = 0.5
# α = 0.1, ϵ = 0.0011, ϵθ = 0.5
par = Sampler(y[y .> 0], X[y .> 0, :], 0.5, 20000, 5, 5000);
inits1 = qreg(@formula(y ~ x), DataFrame(y = log.(y + rand(Uniform(), length(y))), x = x1), 0.1) |> coef
inits5 = qreg(@formula(y ~ x), DataFrame(y = log.(y + rand(Uniform(), length(y))), x = x1), 0.5) |> coef
inits9 = qreg(@formula(y ~ x), DataFrame(y = log.(y + rand(Uniform(), length(y))), x = x1), 0.9) |> coef
β, θ, σ = mcmc(par, 100000.,.5, 0.01, 0.5, 0.5, inits1); # α = 0.1
β, θ, σ = mcmc(par, 100000.,.5, 0.01, 0.5, 0.5, inits5); # α = 0.5
β, θ, σ = mcmc(par, 100000.,1., 0.3, 0.5, 0.5, inits9); # α = 0.9
βa, σa = mcmc(par, 100000., 0.03, inits1, 1); # α = 0.1
βa, σa = mcmc(par, 100000., 0.03, inits5, 1); # α = 0.5
βa, σa = mcmc(par, 100000., 0.01, inits9, 1); # α = 0.9
1-((βa[2:size(βa,1), 1] .=== βa[1:(size(βa,1) - 1), 1]) |> mean)

par = Sampler(y[y .> 0], X[y .> 0, :], 0.9, 20000, 5, 5000);
β, θ, σ = mcmc(par, 100000.,10., .6, 1., 1., inits9);
1-((θ[2:size(β,1)] .=== θ[1:(size(β,1) - 1)]) |> mean)
1-((β[2:size(β,1), 1] .=== β[1:(size(β,1) - 1), 1]) |> mean)
plot(β[:,1])
plot(θ)

p = 2
plot(βa[:,p], label = "ALD")
plot!(β[:,p], label = "AEPD")
qreg(@formula(y ~ x), DataFrame(y = log.(y + rand(Uniform(), length(y))), x = x1), 0.1) |> coef
[mean(βa[:,i]) for i in 1:2]
[mean(β[:,i]) for i in 1:2]
plot(θ)
# bootstrap
N = 2000
α = 0.9
βboot = zeros(N, 2)
for i in 1:N
    ids = sample(1:length(y), length(y))
    βboot[i,:] = qreg(@formula(y ~ x), DataFrame(y = log.(y[ids] + rand(Uniform(), length(y))), x = x1[ids]), α) |> coef
end

freq = mean(βboot, dims = 1) |> x -> reshape(x, 2)
bayes = mean(β, dims = 1) |> x -> reshape(x, 2)
ald = mean(βa, dims = 1) |> x -> reshape(x, 2)

b1 = sort(β, dims = 1)[[Integer(round((0.05/2) * size(β,1))),Integer(round((1-0.05/2) * size(β,1)))], 1]
b2 = sort(β, dims = 1)[[Integer(round((0.05/2) * size(β,1))),Integer(round((1-0.05/2) * size(β,1)))], 2]
f1 = sort(βboot, dims = 1)[[Integer(round((0.05/2) * size(βboot,1))),Integer(round((1-0.05/2) * size(βboot,1)))], 1]
f2 = sort(βboot, dims = 1)[[Integer(round((0.05/2) * size(βboot,1))),Integer(round((1-0.05/2) * size(βboot,1)))], 2]
a1 = sort(βa, dims = 1)[[Integer(round((0.05/2) * size(βa,1))),Integer(round((1-0.05/2) * size(βa,1)))], 1]
a2 = sort(βa, dims = 1)[[Integer(round((0.05/2) * size(βa,1))),Integer(round((1-0.05/2) * size(βa,1)))], 2]

errorsBayes = collect(zip(abs.(b1 .- bayes[1]), abs.(b2 .- bayes[2])))
errorsFreq = collect(zip(abs.(f1 .- freq[1]), abs.(f2 .- freq[2])))
errorsAld= collect(zip(abs.(a1 .- ald[1]), abs.(a2 .- ald[2])))

scatter([1,2], [bayes[1], bayes[2]], label="Bayesian", yerror=errorsBayes, legend=:bottomleft)
scatter!([1+0.05,2+0.05], [freq[1], freq[2]], label="Frequentist", yerror=errorsFreq)
scatter!([1-0.05,2-0.05], [ald[1], ald[2]], label="ALD", yerror=errorsAld)
xlims!((0.8, 2.2))


Q1, Q2, Q3 = zeros(length(y)), zeros(length(y)), zeros(length(y))
for i ∈ 1:length(y)
    Q1[i] = y[i] <= ceil(exp(X[i,:] ⋅ bayes) + α - 1)
    Q2[i] = y[i] <= ceil(exp(X[i,:] ⋅ freq) + α - 1)
    Q3[i] = y[i] <= ceil(exp(X[i,:] ⋅ ald) + α - 1)
end
mean(Q1)
mean(Q2)
mean(Q3)
