using Distributed, SharedArrays
@everywhere using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
@everywhere include(joinpath(@__DIR__, "../aepd.jl"))
@everywhere include(joinpath(@__DIR__, "../../QuantileReg/QuantileReg.jl"))
@everywhere using .AEPD, .QuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)
using Formatting

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = y[y.>0];
X = hcat([1 for i in 1:length(y)], X);

β, θ, σ = bootstrap(log(1), y, X, 0.9, 100)

mean(θ)
[mean(β[:,i]) for i in 1:9]


## FREQUENTIST APPROACH
α = range(0.1, 0.8, length = 8)
θinit = range(2.7, 1.4, length = 8)

N = 1000
β = SharedArray{Float64}((N, size(X)[2], length(α)))
σ = SharedArray{Float64}((N, length(α)))
θ = SharedArray{Float64}((N, length(α)))

@sync @distributed for i = 1:(length(α))
    βt,θt,σt =  bootstrap(log(θinit[i]), y, X, α[i], N)
    β[:,:, i] = βt
    σ[:,i] = σt
    θ[:,i] = θt
end

mean(θ[:, ])
[mean(θ[:,i]) for i in 1:length(α)] |> println

[mean(β[:,i,1]) for i in 1:9]

βt,θt,σt =  bootstrap(log(θinit[5]), ry, X, α[5], N)
[mean(βt[:,i]) for i in 1:9] |> println
## BAYESIAN APPROACH
βinit = [-0.48, -0.14, -2.6, 3.7, 0., 0.1, 1.75, -0.05, 0.28]
M = 20
p = 0.05
feMCMC = FormatExpr("mcmc_{}.csv")
α = range(0.1, 0.9, length = 9)

for i in 1:length(α)
    par = Sampler(y, X, α[i], 110000, 10, 10000);
    N = Integer((par.nMCMC - par.burnIn)/par.thin) + 1
    β = SharedArray{Float64}((N, size(X)[2], M))
    σ = SharedArray{Float64}((N,M))
    θ = SharedArray{Float64}((N,M))

    @sync @distributed for i = 1:M
        βt,θt,σt = mcmc(par, 100., 1., ϵ[i], βinit, 1., 1.)
        β[:,:,i] = βt
        θ[:,i] = θt
        σ[:,i] = σt
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

M = 10
β = SharedArray{Float64}((N, size(X)[2], M))
σ = SharedArray{Float64}((N,M))
θ = SharedArray{Float64}((N,M))
@sync @distributed for i = 1:M
    βt,θt,σt = mcmc(Sampler(y, X, 0.9, 200000, 5, 100000), 100., .8, .25, βinit, 1., 1.)
    β[:,:,i] = βt
    θ[:,i] = θt
    σ[:,i] = σt
end

a = DataFrame(param = colnames,
    value =[mean([median(β[:, j, i]) for i in 1:M]) for j in 1:size(X)[2]],
    l = [mean([sort(β, dims = 1)[Integer(round((p/2) * N)), j, i] for i in 1:M]) for j in 1:size(X)[2]],
    u =  [mean([sort(β, dims = 1)[Integer(round((1-p/2) * N)), j, i] for i in 1:M]) for j in 1:size(X)[2]], α = 0.9)

b = DataFrame(param = ["sigma", "theta"], value = [mean([median(σ[:, i]) for i in  1:M]), mean([median(θ[:, i]) for i in 1:M])],
        l = [mean(sort(σ, dims = 1)[Integer(round((p/2) * N)), :]) ; mean(sort(θ, dims = 1)[Integer(round((p/2) * N)), :])],
        u = [mean(sort(σ, dims = 1)[Integer(round((1-p/2) * N)), :]); mean(sort(θ, dims = 1)[Integer(round((1-p/2) * N)), :])], α = 0.9)

CSV.write(format(feMCMC, 0.9), vcat(a, b))
mean(σ)

(1:M)[Not(5)]

[median(σ[:, i]) for i in 1:M]
[median(θ[:, i]) for i in 1:M]

print(a)
print(b)
