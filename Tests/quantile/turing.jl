using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
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
    θ ~ Uniform(0.5,3)
    σ ~ InverseGamma(1, 1)
    β ~ filldist(Normal(0., 100.), predictors)
    for i in 1:length(y)
        y[i] ~ Aepd(X[i,:] ⋅ β, σ, θ, 0.5)
    end
end

model = apdreg(X, y);
chain = sample(model, NUTS(), 3000);
# chain = sample(model, Gibbs(PG(100, :σ), HMC(.03, 10, :β)), 1000);
summaries = summarystats(chain)
plot(chain[:"β[1]"])
plot!(β[1:length(chain[:"β[1]"]),1])
println(inits)

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
