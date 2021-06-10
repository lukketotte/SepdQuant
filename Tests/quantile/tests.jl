using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles
# using KernelDensity
theme(:juno)

## Sampling of θ
u1, u2 = sampleLatent(X, y, β, α, θ, σ)
θinterval(X, y, u1, u2, β, α, σ) |> println


##

# generate data
n = 300;
β, α, σ = [2.1, 0.8], 0.5, 1.;
θ = 1.
X = [repeat([1], n) rand(Uniform(-3, 3), n)]
d = aepd(0., σ, θ, α);
# d = Laplace(0., 1.)
y = X * β .+ rand(d, n);

nMCMC = 200000
σ = zeros(nMCMC)
σ[1] = 1 # √(sum((y-X*inv(X'*X)*X'*y).^2) / (n-2))
β = zeros(nMCMC, 2)
β[1, :] = inv(X'*X)*X'*y
θ = zeros(nMCMC)
θ[1] = 1.
U1, U2 = zeros(nMCMC, n), zeros(nMCMC, n)

for i in 2:nMCMC
    u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i-1])
    U1[i,:], U2[i,:] = u1, u2
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i-1], σ[i-1], 10.)
    # β[i,:] = [2.1, 0.8]
    # σ[i] = sampleσ(X, y, u1, u2, β[i, :], α, θ[i-1], 1, 1.)
    # θ[i] = sampleθ(θ[i-1], .1, X, y, u1, u2, β[i, :], α, σ[i])
    if i % 5000 === 0
        interval = round.(θinterval(X, y, u1, u2, β[i,:], α, σ[i]), digits = 3)
        printfmt("iter: {1}, θ ∈ [{2:.2f}, {3:.2f}], σ = {4:.2f} \n", i, interval[1], interval[2], σ[i])
    end
    σ[i] = 1.
    θ[i] = 1.
end


plot(β[:, 1])
plot(σ[1:nMCMC])
plot(θ[1:nMCMC])
plot!(σ)

##

autocor(θ, [1,3,10,40]) |> println

thin = ((1:nMCMC) .% 10) .=== 0
autocor(θ[thin], [1,3,10,40]) |> println
plot(θ[thin])
plot(σ[thin])

1-((θ[2:nMCMC] .=== θ[1:(nMCMC - 1)]) |> mean)

plot(cumsum(σ) ./ (1:nMCMC))
plot(cumsum(β[:, 2]) ./ (1:nMCMC))
plot(cumsum(θ) ./ (1:nMCMC))
median(θ[10000:nMCMC])
median(θ[thin])
##
plot!(cumsum(σ) ./ (1:nMCMC))
plot!(cumsum(θ) ./ (1:nMCMC))

"""
CSV.write("beta.csv", DataFrame(β), header = false)
CSV.write("theta.csv", DataFrame(reshape(θ, nMCMC, 1)), header = false)
CSV.write("sigma.csv", DataFrame(reshape(σ, nMCMC, 1)), header = false)
"""
