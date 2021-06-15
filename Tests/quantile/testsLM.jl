using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles
using KernelDensity
theme(:juno)

# generate data
n = 500;
β, α, σ = [2.1, 0.8], 0.5, 3.;
θ = 1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(aepd(0., σ, θ, α), n);

β,σ,l,θ = mcmc(y, X, 0.5, 5000, θinit = 1.0, σinit = 3.)

plot(β[:, 1], label="β")
plot(σ[1:100], label="σ")
plot(θ, label="θ")

u1, u2 = sampleLatent(X, y, β, α, θ, √var(y))
β = sampleβ(X, y, u1, u2, β, α, θ, √var(y), 100.)
σ,_ = sampleσ(X, y, u1, u2, β, α, θ, 1, 1.)

##

nMCMC = 5000
σ = zeros(nMCMC)
σ[1] = 3.# √(sum((y-X*inv(X'*X)*X'*y).^2) / (n-2))
β = zeros(nMCMC, 2)
β[1, :] = inv(X'*X)*X'*y
θ = zeros(nMCMC)
θ[1] = 1.
U1, U2 = zeros(nMCMC, n), zeros(nMCMC, n)

# TODO: why is σ so high even when treating θ as known?
for i in 2:nMCMC
    u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i-1])
    U1[i,:], U2[i,:] = u1, u2
    σ[i],_ = sampleσ(X, y, u1, u2, β[i-1, :], α, θ[i-1], 1, 1.)
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i-1], σ[i], 100.)
    # σ[i] = 3.
    θ[i] = sampleθ(θ[i-1], .1, X, y, u1, u2, β[i, :], α, σ[i])
    # θ[i] = 1.
    if i % 5000 === 0
        interval = θinterval(X, y, u1, u2, β[i,:], α, σ[i])
        printfmt("iter: {1}, θ ∈ [{2:.2f}, {3:.2f}], σ = {4:.2f} \n", i, interval[1], interval[2], σ[i])
    end
end

plot(β[:, 1])
plot(σ)
plot(θ)
plot!(σ)

autocor(θ, [1,3,10,40]) |> println
thin = ((1:nMCMC) .% 10) .=== 0
autocor(θ[thin], [1,3,10,40]) |> println
# Ess
1-((θ[2:nMCMC] .=== θ[1:(nMCMC - 1)]) |> mean)

plot(cumsum(σ) ./ (1:nMCMC))
median(θ[10000:nMCMC])
median(θ[thin])


"""
CSV.write("beta.csv", DataFrame(β), header = false)
CSV.write("theta.csv", DataFrame(reshape(θ, nMCMC, 1)), header = false)
CSV.write("sigma.csv", DataFrame(reshape(σ, nMCMC, 1)), header = false)
"""
