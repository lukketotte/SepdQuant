using Distributed, SharedArrays
@everywhere using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, QuantileRegressions, DataFrames, QuadGK
@everywhere include(joinpath(@__DIR__, "../aepd.jl"))
@everywhere include(joinpath(@__DIR__, "../../QuantileReg/QuantileReg.jl"))
@everywhere using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, StatFiles, CSVFiles
theme(:juno)
using Formatting

## α to τ
@everywhere f(x, b, p, α, μ, σ) = abs(x-b)^(p-1) * pdf(Aepd(μ, σ, p, α), x)

@everywhere function quantconvert(q, p, α, μ, σ)
    a₁ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, Inf)[1]
    a₂ = quadgk(x -> f(x, q, p, α, μ, σ), -Inf, q)[1]
    1/((maximum([a₁/a₂, 1.0001]) - 1)^(1/p) + 1)
end

## Simulation study with AEPD error term
n = 2000;
x = rand(Normal(), n);
X = hcat(ones(n), x)


p = [1.5, 2., 2.5]
skew = [0.2, 0.8, 0.5]
quant = range(0.1, 0.9, length = 3)

settings = DataFrame(p = repeat(p, inner = length(skew)) |> x -> repeat(x, inner = length(quant)),
    skew = repeat(skew, length(p)) |> x -> repeat(x, inner = length(quant)),
    tau = repeat(quant, length(skew) * length(p)),
    convertTau = 0, old = 0,  sdOld = 0, bayes = 0, sdBayes = 0)
cols = names(settings)
settings = SharedArray(Matrix(settings))
reps = 50

@sync @distributed for i ∈ 1:size(settings, 1)
    p, skew, τ = settings[i, 1:3]
    old, bayes, convertTau = [zeros(reps) for i in 1:3]
    for j ∈ 1:reps
        y = 2.1 .+ 0.5 .* x + rand(Aepd(0, 1, p, skew), n);
        par = Sampler(y, X, skew, 6000, 5, 1000);
        β, θ, σ, α = mcmc(par, 0.8, .25, 1.5, 1, 2, 0.5, [2.1, 0.5]);
        μ = X * median(β, dims = 1)' |> x -> reshape(x, size(x, 1));

        b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
            qreg(@formula(x1 ~  x3), x, τ) |> coef;
        q = X * b;
        convertTau[j] = [quantconvert(q[k], median(θ), median(α), μ[k], median(σ)) for k in 1:length(par.y)] |> mean

        par.α = convertTau[j]
        βres, _ = mcmc(par, 1.3, median(θ), median(σ), b);

        par.α = τ
        βt, _, _ = mcmc(par, .6, 1., 1.2, 4, b);

        bayes[j] = [par.y[k] <= X[k,:] ⋅ median(βres, dims = 1)  for k in 1:length(par.y)] |> mean
        old[j] = [par.y[k] <= X[k,:] ⋅ median(βt, dims = 1)  for k in 1:length(par.y)] |> mean
    end
    settings[i, 4] = mean(convertTau)
    settings[i, 5] = mean(old)
    settings[i, 6] = √var(old)
    settings[i, 7] = mean(bayes)
    settings[i, 8] = √var(bayes)
end

plt_dat = DataFrame(Tables.table(settings)) |> x -> rename!(x, cols)
CSV.write("sims.csv", plt_dat)

## Quantile with misspecified τ
n = 1000;
x = rand(Normal(), n);
X = hcat(ones(n), x)

α = range(0.1, 0.9, length = 17) |> Vector
res3 = SharedArray{Float64}(length(α))
reps = 50
@sync @distributed for i ∈ 1:length(α)
    temp = 0
    for j in 1:reps
        y = 2.1 .+ 0.5 .* x + rand(Aepd(0, 1, 0.7, 0.7), n);
        par = Sampler(y, X, α[i], 5000, 4, 1000)
        b = DataFrame(hcat(par.y, par.X), :auto) |> x ->
            qreg(@formula(x1 ~ x3), x, α[i]) |> coef
        β, _, _ = mcmc(par, 1., 1., 1.2, 4, b)
        temp += abs(α[i] - ([par.y[i] <= X[i,:] ⋅ median(β, dims = 1)  for i in 1:length(par.y)] |> mean))/reps
    end
    res3[i] = temp
end

vcat(hcat(res, α, [2. for i in 1:17]), hcat(res2, α, [1.5 for i in 1:17]),
    hcat(res3, α, [1. for i in 1:17])) |> x ->
    DataFrame(x, ["diff", "quant", "p"]) #|> x -> CSV.write("quantest.csv", x)


plot!(α, res3)

##
α = range(0.01, 0.99, length = 101);
p = [0.75, 1.5, 2, 3];
ε = [0.05, 0.2, 0.1, 0.05];
N = 20
n = 2000
τ = SharedArray{Float64}(length(α), length(p), N)

@sync @distributed for j ∈ 1:length(p)
    for iter ∈ 1:N
        y = 0.5 .+ rand(Normal(), n);
        inits = DataFrame(hcat(y, ones(n)), :auto) |> x -> qreg(@formula(x1 ~  1), x, 0.5) |> coef
        par = Sampler(y, hcat(ones(n)), 0.5, 16000, 5, 6000);
        β, σ = mcmc(par, ε[j], p[j], inits, 1., verbose = false)
        β = median(β[:,1])
        σ = median(σ)

        for i ∈ 1:length(α)
            q = (DataFrame(hcat(y, ones(n)), :auto) |> x -> qreg(@formula(x1 ~  1), x, α[i]) |> coef)[1]
            τ[i, j, iter] = quantconvert(q, p[j], 0.5, β, σ)
        end
    end
end


tau = mean(τ, dims = 3)[:,:,1]

plot(α, α)
plot!(α, tau[:,4])

DataFrame(tau = reshape(tau, (404,)),
    p = repeat(p, inner = 101),
    a = repeat(α, outer = 4)) |> x -> save("res.csv", x)

##
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
