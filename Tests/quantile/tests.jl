using Distributions, QuantileRegressions, LinearAlgebra
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)

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

[median(β[:,i]) for i in 1:2]
plot(θ)
plot(σ)

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
n = 100
p = 0.8
βtrue = (.5, 2.)
x1 = rand(Uniform(0, 2), n)
exp.(βtrue[1] .+ x1.*βtrue[2])
sims = Poisson.(exp.(βtrue[1] .+ x1.*βtrue[2]))
#sims = truncated.(sims, 1, Inf)

y = [rand(sims[i], 1)[1] for i in 1:n]
sum(y.>0)
histogram(y)


y = zeros(Int64, n)
for i ∈ 1:n
    if rand(Uniform(), 1)[1] < p
        y[i] = rand(Poisson(exp(βtrue[1] + x1[i]*βtrue[2])), 1)[1]
    else
        y[i] = Integer(round(rand(Pareto(0.8, 500), 1)[1]))
    end
end
X = hcat(ones(sum(y.>0)), x1[y.>0])
y = y[y.>0]
histogram(y)
# α = 0.5, ϵ = 0.02, ϵθ = 0.01
# α = 0.9, ϵ = 0.0011, ϵθ = 0.5
# α = 0.1, ϵ = 0.0011, ϵθ = 0.5
par = Sampler(y, X, 0.5, 20000, 5, 5000);
β, θ, σ = mcmc(par, 100000.,.5, 0.02, 0.5, 0.5, [1., 0.8]);
βa, σa = mcmc(par, 100000., 0.01, [1., 0.8], 1);

1-((θ[2:size(β,1)] .=== θ[1:(size(β,1) - 1)]) |> mean)
1-((β[2:size(β,1), 1] .=== β[1:(size(β,1) - 1), 1]) |> mean)
plot(θ)
plot(β[:,1])

[mean(β[:,i]) for i in 1:2]

plot(β[:,1])
plot!(βa[:,1])

# bootstrap
N = 1500
βboot = zeros(N, 2)
for i in 1:N
    ids = sample(1:length(y), length(y))
    βboot[i,:] = qreg(@formula(y ~ x), DataFrame(y = log.(y[ids] + rand(Uniform(), length(y))), x = x1[ids]), 0.5) |> coef
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

scatter([1,2], [bayes[1], bayes[2]], label="Bayesian", yerror=errorsBayes, legend=:bottomright)
scatter!([1+0.05,2+0.05], [freq[1], freq[2]], label="Frequentist", yerror=errorsFreq)
scatter!([1-0.05,2-0.05], [ald[1], ald[2]], label="ALD", yerror=errorsAld)
xlims!((0.8, 2.2))


Q1 = zeros(length(y))
Q2 = zeros(length(y))
Q3 = zeros(length(y))
for i ∈ 1:length(y)
    Q1[i] = log(y[i]) <= ceil(exp(X[i,:] ⋅ bayes) + 0.5 - 1)
    Q2[i] = log(y[i]) <= ceil(exp(X[i,:] ⋅ freq) + 0.5 - 1)
    Q3[i] = log(y[i]) <= ceil(exp(X[i,:] ⋅ ald) + 0.5 - 1)
end
mean(Q1)
mean(Q2)
mean(Q3)
