include("QR.jl")
using .QR
using Plots, PlotThemes, CSV, DataFrames, StatFiles
using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
theme(:juno)

# dat = load("C:\\Users\\lukar818\\Documents\\PhD\\SMC\\Tests\\data\\nsa_ff.dta") |> DataFrame
dat = load(string(pwd(), "/Tests/data/nsa_ff.dta")) |> DataFrame
dat = dat[:, Not(filter(c -> count(ismissing, dat[:,c])/size(dat,1) > 0.05, names(dat)))]
dropmissing!(dat)

y = Float64.(dat."fatality_lag_ln")
X = Float64.(dat[:, [:intensity, :pop_dens, :foreign_f, :loot, :ethnic]] |> Matrix)
ids = y.>0
y = y[ids]
X = X[ids,:]

## mcmc test
n, p = size(X)
α = 0.5
nMCMC =  30000
σ = zeros(nMCMC)
σ[1] = 1.
β = zeros(nMCMC, p)
β[1, :] = inv(X'*X) * X' * y
θ = zeros(nMCMC)
θ[1] = 1.5
simU1 = zeros(nMCMC, n)
simU2 = zeros(nMCMC, n)

for i in 2:nMCMC
    (i % 100 === 0) && println(i)
    u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i-1])
    simU1[i,:] = u1
    simU2[i,:] = u2
    σ[i] = sampleσ(X, y, u1, u2, β[i-1, :], α, θ[i-1], 1)
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i-1], σ[i], 10.)
    interval = θinterval(X, y, u1, u2, β[i,:], α, σ[i])
    if minimum(interval) === maximum(interval)
        θ[i] = interval[1]
    else
        d = truncated(Normal(θ[i-1], 0.01), minimum(interval), maximum(interval))
        θ[i] = sampleθ(θ[i-1], d, interval, X, y, u1, u2, β[i, :], α, σ[i])
    end
end


plot(θ)
plot(β[:, 3])
plot(σ)

plot(cumsum(σ) ./ (1:nMCMC))
plot(cumsum(β[:, 2]) ./ (1:nMCMC))
plot(cumsum(θ) ./ (1:nMCMC))
