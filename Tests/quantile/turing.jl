using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, StatsModels
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, StatsPlots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles
theme(:juno)

using Turing

dat = load(string(pwd(), "/Tests/data/hks_jvdr.csv")) |> DataFrame;
y = dat[:, :osvAll]
X = dat[:, Not(["osvAll"])] |> Matrix
X = X[y.>0,:];
y = log.(y[y.>0] + rand(Uniform(0,1), size(X,1)));
X = hcat([1 for i in 1:length(y)], X);

## they all behave similarly with this data at least.
n = 200;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(Aepd(0., σ, θ, 0.5), n);

@model apdreg(X, y; predictors=size(X,2)) = begin
    # priors
    #θ =1.84#~ Uniform(0.2,3)
    σ = 4.5#~ InverseGamma(1, 1)
    β ~ filldist(Normal(0., 10000.), predictors)
    for i in 1:length(y)
        #y[i] ~ Aepd(X[i,:] ⋅ β, σ, θ, 0.5)
        y[i] ~ Laplace(X[i,:] ⋅ β, σ)
    end
end

model = apdreg(par.X, par.y);
chain = sample(model, MH(), 3001);
summaries = summarystats(chain)

plot(chain[:"β[1]"])
plot!(β[1:length(chain[:"β[5]"]),3])
println(inits)

reshape(chain[:"β[3]"].data, 7001)

plot(1:7001, cumsum(reshape(chain[:"β[9]"].data, 7001))./(1:7001))
plot!(1:7001, cumsum(β[:,9])./(1:7001))


plot(chain[:"θ"])
plot!(θ[1:length(chain[:"θ"])])

plot(chain[:"σ"])
plot!(σ[1:length(chain[:"σ"])])

using QuantileRegressions
coef(qreg(@formula(y ~ x2), hcat(DataFrame(y = y), DataFrame(X, :auto)), 0.5))


par = Sampler(y, X, 0.5, 50000, 10, 20000);
β, θ, σ = mcmc(par, 100., .05, .07, [2.5, 0.75], 1., 2.);
[median(β[:,i]) for i in 1:2]

1-((β[2:length(θ), 1] .=== β[1:(length(θ) - 1), 1]) |> mean)
1-((θ[2:length(θ)] .=== θ[1:(length(θ) - 1)]) |> mean)
plot(β[:,1])
plot(θ)
plot(σ)

##
using QuantileRegressions
qrdat = dat[dat.osvAll .> 0,:]
qrdat[!,:osvAll] = log.(qrdat[:,"osvAll"])
names(qrdat)[2:9]
ResultQR = qreg((Term(:osvAll)~sum(Term.(Symbol.(names(qrdat[:, Not(:osvAll)]))))), qrdat, .5)

Q = zeros(100)
for i in 1:100
    Q[i] = testy[i] <= ceil(exp(testX[i,:] ⋅ βest) + 0.5 - 1)
end

α = range(0.1, 0.9, length = 9)
freqDiff = zeros(9)
for j ∈ 1:9
    β = qreg((Term(:osvAll)~sum(Term.(Symbol.(names(qrdat[:, Not(:osvAll)]))))), qrdat, α[j]) |> coef
    Q = zeros(881)
    for i ∈ 1:881
        Q[i] = y[i] <= ceil(exp(X[i,:] ⋅ β) + α[j] - 1)
    end
    freqDiff[j] = mean(Q)
end

freqDiff

βinit = qreg((Term(:osvAll)~sum(Term.(Symbol.(names(qrdat[:, Not(:osvAll)]))))), qrdat, 0.6) |> coef
Q = zeros(881)
for i ∈ 1:881
    Q[i] = y[i] <= ceil(exp(X[i,:] ⋅ β) + 0.3 - 1)
end

mean(Q)
