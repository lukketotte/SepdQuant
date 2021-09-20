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

@model apdreg(X, y; predictors=size(X,2)) = begin
    # priors
    θ = 1.8#~ Uniform(0.01,3)
    # σ ~ InverseGamma(1, 1)
    σ = 4
    β ~ filldist(Normal(0., 100.), predictors)
    for i in 1:length(y)
        y[i] ~ Aepd(-2.1 + X[i,:] ⋅ β, σ, θ, 0.5)
    end
end

model = apdreg(X, log.(y));
chain = sample(model, NUTS(1000, 0.65), 5000, init_theta = inits[2:length(inits)]);
# chain = sample(model, Gibbs(PG(100, :σ), HMC(.03, 10, :β)), 1000);
summaries = summarystats(chain)
plot(chain[:"β[7]"])

println(inits)
