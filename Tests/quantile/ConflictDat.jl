using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
using .QR
using Plots, PlotThemes, CSV, DataFrames, StatFiles
theme(:juno)

# dat = load("C:\\Users\\lukar818\\Documents\\PhD\\SMC\\Tests\\data\\nsa_ff.dta") |> DataFrame
dat = load(string(pwd(), "/Tests/data/nsa_ff.dta")) |> DataFrame
dat = dat[:, Not(filter(c -> count(ismissing, dat[:,c])/size(dat,1) > 0.05, names(dat)))]
dropmissing!(dat)

y₁ = Float64.(dat."fatality_lag_ln")
X = Float64.(dat[:, [:intensity, :pop_dens, :foreign_f, :loot, :ethnic]] |> Matrix)
y₁ = y₁[y₁.>0]
X = X[findall(y₁.>0),:]
α, n = 0.5, length(y)

nMCMC = 50000
β = zeros(nMCMC, 5)
β[1,:] = inv(X'*X)*X'*y₁
σ, θ = zeros(nMCMC), zeros(nMCMC)
σ[1] = 1.
θ[1] = 1.

# seems like θ goes towards 1 with this sampling order
for i ∈ 2:nMCMC
    if i % 5000 === 0
        println("iter: ", i, " θ = ", round(θ[i-1], digits = 2))
    end
    # jittering to make quantiles continuous
    global y = log.((exp.(y₁) + rand(Uniform(), length(y₁)) .- α))
    z = y - X*β[i-1,:]
    pos = findall(z .> 0)
    # using jacobian u = σ^p https://barumpark.com/blog/2019/Jacobian-Adjustments/
    b = (δ(α, θ[i-1]) * sum(abs.(z[Not(pos)]).^θ[i-1]) / α^θ[i-1]) + (δ(α, θ[i-1]) * sum(abs.(z[pos]).^θ[i-1]) / (1-α)^θ[i-1])
    σ[i] = (rand(InverseGamma(n/θ[i-1], b), 1)[1])^(1/θ[i-1])
    # σ[i] = 1
    global u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i])
    try
        θ[i] = sampleθ(θ[i-1], 1., X, y, u1, u2, β[i-1, :], α, σ[i])
    catch e
        if isa(e, ArgumentError)
            θ[i] = θ[i-1]
            break
        else
            println(e)
        end
    end
    # θ[i] = 2.
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i], σ[i], 100.)
end

plot(θ)
plot(β[:, 5])
plot(σ)

plot(cumsum(σ) ./ (1:nMCMC))
plot(cumsum(β[:, 5]) ./ (1:nMCMC))
plot(cumsum(θ) ./ (1:nMCMC))
