using Distributed, SharedArrays
@everywhere using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, QuantileRegressions, DataFrames
@everywhere include(joinpath(@__DIR__, "../aepd.jl"))
@everywhere include(joinpath(@__DIR__, "../../QuantileReg/QuantileReg.jl"))
@everywhere using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, StatFiles, CSVFiles
theme(:juno)
using Formatting

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

colnames = ["intercept" ; names(dat[:, Not(["osvAll"])])]
βinit = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]
M = 40
p = 0.05
feMCMC = FormatExpr("mcmc_{}.csv")
α = range(0.1, 0.9, length = 9)
ϵ = [0.9, 1.2, 1.3, 1.5, 1.5, 1.6, 1.6, 1.4, 0.6]

for i ∈ 1:length(α)
    par = Sampler(y, X, α[i], 150000, 30, 30000);
    N = Integer((par.nMCMC - par.burnIn)/par.thin) + 1
    β = SharedArray{Float64}((N, size(X)[2], M))
    σ = SharedArray{Float64}((N,M))
    θ = SharedArray{Float64}((N,M))

    @sync @distributed for j ∈ 1:M
        βt,θt,σt = mcmc(par, 10000., 1., ϵ[i], βinit, 1., 1.)
        β[:,:,j] = βt
        θ[:,j] = θt
        σ[:,j] = σt
    end

    a = DataFrame(param = colnames,
        value =[mean([median(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]],
        l = [mean([sort(β, dims = 1)[Integer(round((p/2) * N)), j, i] for i in 1:M]) for j in 1:size(X)[2]],
        u =  [mean([sort(β, dims = 1)[Integer(round((1-p/2) * N)), j, i] for i in 1:M]) for j in 1:size(X)[2]], α = α[i])

    b = DataFrame(param = ["sigma", "theta"], value = [mean([median(σ[:, i]) for i in 1:M]), mean([median(θ[:, i]) for i in 1:M])],
            l = [mean(sort(σ, dims = 1)[Integer(round((p/2) * N)), :]) ; mean(sort(θ, dims = 1)[Integer(round((p/2) * N)), :])],
            u = [mean(sort(σ, dims = 1)[Integer(round((1-p/2) * N)), :]); mean(sort(θ, dims = 1)[Integer(round((1-p/2) * N)), :])], α = α[i])

    CSV.write(format(feMCMC, α[i]), vcat(a, b))
end

par = Sampler(y, X, 0.5, 150000, 30, 30000);
N = Integer((par.nMCMC - par.burnIn)/par.thin) + 1
M = 10
β = SharedArray{Float64}((N, size(X)[2], M))
σ = SharedArray{Float64}((N,M))
θ = SharedArray{Float64}((N,M))
@sync @distributed for i = 1:M
    βt,θt,σt =  mcmc(par, 10000., 1., 1.2, βinit, 1., 1.)
    β[:,:,i] = βt
    θ[:,i] = θt
    σ[:,i] = σt
end

a = DataFrame(param = colnames,
    value =[mean([median(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]],
    l = [mean([sort(β, dims = 1)[Integer(round((p/2) * N)), j, i] for i in 1:M]) for j in 1:size(X)[2]],
    u =  [mean([sort(β, dims = 1)[Integer(round((1-p/2) * N)), j, i] for i in 1:M]) for j in 1:size(X)[2]], α = 0.5)

b = DataFrame(param = ["sigma", "theta"], value = [mean([median(σ[:, i]) for i in  1:M]), mean([median(θ[:, i]) for i in 1:M])],
        l = [mean(sort(σ, dims = 1)[Integer(round((p/2) * N)), :]) ; mean(sort(θ, dims = 1)[Integer(round((p/2) * N)), :])],
        u = [mean(sort(σ, dims = 1)[Integer(round((1-p/2) * N)), :]); mean(sort(θ, dims = 1)[Integer(round((1-p/2) * N)), :])], α = 0.5)

CSV.write(format(feMCMC, 0.5), vcat(a, b))
mean(σ)

(1:M)[Not(5)]

[median(σ[:, i]) for i in 1:M]
[median(θ[:, i]) for i in 1:M]

print(a)
print(b)


## Prediction
α = 0.8
reps = 100
aepd = SharedVector{Float64}(reps)
freq = SharedVector{Float64}(reps)

@sync @distributed for j ∈ 1:reps
    ids = sample(1:length(y), length(y)-100; replace = false)
    trainX, trainy = X[ids,:], y[ids]
    testX, testy = X[Not(ids),:], y[Not(ids)]
    par = Sampler(trainy, trainX, α, 21000, 5, 1000)

    βinit = DataFrame(hcat(par.y, par.X), :auto) |> x ->
        qreg(@formula(x1 ~  x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), x, par.α) |> coef
    β, θ, σ = mcmc(par, 0.4, 0.4, 4., 1.4, βinit);
    βest = [mean(β[:,i]) for i in 1:9]

    Q = zeros(100)
    Q2 = zeros(100)
    for i in 1:100
        Q[i] = testy[i] <= ceil(exp(testX[i,:] ⋅ βest) + par.α - 1)
        Q2[i] = testy[i] <= ceil(exp(testX[i,:] ⋅ βinit) + par.α - 1)
    end
    aepd[j] = mean(Q)
    freq[j] = mean(Q2)
end

mean(aepd[aepd.>0])
mean(freq[aepd.>0])

print(aepd)

√var(aepd[aepd.>0])
√var(freq[aepd.>0])
print(freq)
