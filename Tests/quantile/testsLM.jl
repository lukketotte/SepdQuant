using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles
using KernelDensity
theme(:juno)

##
function mcmc2(y::Array{T, 1}, X::Array{T, 2}, α::T, nMCMC::N;
    θinit::T = 1., σinit::T = 1., printIter::N = 5000) where {T <: Real, N <: Integer}
    n = length(y)
    σ, σₗ, θ = zeros(nMCMC), zeros(nMCMC), zeros(nMCMC)
    σ[1] = σinit
    β = zeros(nMCMC, 2)
    β[1, :] = inv(X'*X)*X'*y
    θ[1] = θinit

    for i ∈ 2:nMCMC
        u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i-1])
        β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i-1], σ[i-1], 100.)
        # σ[i], σₗ[i] = sampleσ(X, y, u1, u2, β[i, :], α, θ[i-1], 1, 1.)
        σ[i] = σinit
        # θ[i] = sampleθ(θ[i-1], .1, X, y, u1, u2, β[i, :], α, σ[i])
        θ[i] = θinit
        if i % printIter === 0
            interval = θinterval(X, y, u1, u2, β[i,:], α, σ[i])
            printfmt("iter: {1}, θ ∈ [{2:.2f}, {3:.2f}], σ = {4:.2f} \n", i, interval[1], interval[2], σ[i])
        end
    end
    β, σ, θ
end


## generate data
n = 2000;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ = 1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(aepd(0., σ, θ, α), n);

β,σ,l,θ = mcmc(y, X, 0.5, 30000, θinit = 1.0, σinit = 1.)

β,σ,θ = mcmc2(y, X, 0.5, 10000, θinit = 1.0, σinit = 2.)

plot(β[:, 1], label="β")
plot(σ, label="σ")
plot(θ, label="θ")

plot(σ[2:300])
plot!(l[2:300])

mean(σ[2:50000] - l[2:50000])

u1, u2 = sampleLatent(X, y, β, α, θ, σ)
β = sampleβ(X, y, u1, u2, β, α, θ, σ, 100.)
σ,_ = sampleσ(X, y, u1, u2, β, α, θ, 1, 1.)

##

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
