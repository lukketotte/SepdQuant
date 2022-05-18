using Distributions, QuantileRegressions, LinearAlgebra, Random, SpecialFunctions, QuadGK, RCall
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
include("../../QuantileReg/FreqQuantileReg.jl")
using .AEPD, .QuantileReg, .FreqQuantileReg

using Plots, PlotThemes, CSV, DataFrames, StatFiles, CSVFiles, HTTP
theme(:juno)

dat = load(string(pwd(), "/Tests/data/fishery.csv")) |> DataFrame
names(dat)
y = dat[:,:qty]

X = hcat(ones(length(y)), Matrix(dat[:,[:stormy, :mixed, :price]]))

dat = (load(string(pwd(), "/Tests/data/prostate.csv")) |> DataFrame)[:, 2:10];
y = dat[:, :lpsa]
X = hcat(ones(length(y)), Matrix(dat[:,Not(:lpsa)]))

# baseline estimates
par = Sampler(y, X, 0.5, 20000, 4, 5000);
β, θ, σ, α = mcmc(par, 1., 0.5, 1.2, 1., 1., 0.5, zeros(size(par.X, 2)));

plot(θ)
plot(β[:,4])
plot(α)
plot(σ)
acceptance(α)
acceptance(θ)
acceptance(β)

βb, θb, σb, αb = vec(median(β,dims=1)), median(θ), median(σ), median(α);

control =  Dict(:tol => 1e-4, :max_iter => 1000, :max_upd => 0.3,
  :is_se => false, :est_beta => true, :est_sigma => true,
  :est_p => true, :est_tau => true, :log => false, :verbose => false);

res = quantfreq(y, X, control)
res[:beta] |> println
println(βb)
res[:p]
res[:sigma]

par.α = mcτ(0.1, αb, θb, σb, 5000, 5000)
par.nMCMC = 20000
βres = mcmc(par, 1.2, θb, σb, zeros(size(par.X,2)));
acceptance(βres)
mean(par.y .<= vec(par.X*median(βres, dims = 1)'))

plot(βres[:,4])

b = rcopy(R"""
suppressWarnings(suppressMessages(library(bayesQR, lib.loc = "C:/Users/lukar818/Documents/R/win-library/4.0")))
dat = $dat
bayesQR(qty ~ stormy + mixed + price, dat, quantile = 0.7, ndraw = 10000, keep=4)[[1]]$betadraw
""")

mean(par.y .<= vec(par.X*median(b, dims = 1)'))
