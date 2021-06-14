using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("QR.jl")
include("../aepd.jl")
using .AEPD, .QR
using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles
# using KernelDensity
theme(:juno)

## Sampling of θ
u1, u2 = sampleLatent(X, y, β, α, θ, σ)
# θinterval(X, y, u1, u2, β, α, σ) |> println
β = sampleβ(X, y, u1, u2, β, α, θ, σ, 10.)
σ = sampleσ(X, y, u1, u2, β, α, θ, 1, 1.)
println(β,", ", σ)
# θ = sampleθ(θ, .1, X, y, u1, u2, [2.1, 0.8], α, σ)

## Testing different way of writing lm coefficient interval
u1, u2 = sampleLatent(X, y, β, α, θ, √(sum((y-X*inv(X'*X)*X'*y).^2) / (n-2)))
k = 1
l, u = [], []

for i ∈ 1:n
    μ = X[i, 1:end .!= k] ⋅  β[1:end .!= k]
    if X[i,k] > 0
        u2[i] > 0 && append!(l, (y[i]-μ-(1-α)*σ*u2[i]^(1/θ))/X[i,k])
        u1[i] > 0 && append!(u, (y[i]-μ+ α*σ*u1[i]^(1/θ))/X[i,k])
    elseif X[i, k] < 0
        u1[i] > 0 && append!(l, (μ - y[i] - α*σ*u1[i]^(1/θ))/(-X[i,k]))
        u2[i] > 0 && append!(u, (μ - y[i] + (1-α)*u2[i]^(1/θ))/(-X[i,k]))
    end
end

maximum(l) < minimum(u[findall(u .>= maximum(l))])

# other way
l, u = [], []
for i in 1:n
    a = (y[i] - X[i, 1:end .!= k] ⋅  β[1:end .!= k]) / X[i, k]
    b₁ = α*σ*(u1[i]^(1/θ)) / X[i, k]
    b₂ = (1-α)*σ*(u2[i]^(1/θ)) / X[i, k]
    if (u1[i] > 0) && (X[i, k] < 0)
        append!(l, a + b₁)
    elseif (u2[i] > 0) && (X[i, k] > 0)
        append!(l, a - b₂)
    elseif (u1[i] > 0) && (X[i, k] > 0)
        append!(u, a + b₁)
    elseif (u2[i] > 0) && (X[i, k] < 0)
        append!(u, a - b₂)
    end
end

maximum(l)
minimum(u)

##

# generate data
n = 500;
β, α, σ = [2.1, 0.8], 0.5, 0.5;
θ = 1.
X = [repeat([1], n) rand(Uniform(-3, 3), n)]
y = X * β .+ rand(aepd(0., σ, θ, α), n);

nMCMC = 20000
σ = zeros(nMCMC)
σ[1] = 3# √(sum((y-X*inv(X'*X)*X'*y).^2) / (n-2))
β = zeros(nMCMC, 2)
β[1, :] = inv(X'*X)*X'*y
θ = zeros(nMCMC)
θ[1] = 1.
U1, U2 = zeros(nMCMC, n), zeros(nMCMC, n)

# TODO: why is σ so high even when treating θ as known?
for i in 2:nMCMC
    u1, u2 = sampleLatent(X, y, β[i-1,:], α, θ[i-1], σ[i-1])
    U1[i,:], U2[i,:] = u1, u2
    β[i,:] = sampleβ(X, y, u1, u2, β[i-1,:], α, θ[i-1], σ[i-1], 100.)
    # β[i,:] = [2.1, 0.8]
    σ[i] = sampleσ(X, y, u1, u2, β[i, :], α, θ[i-1], 1, 1.)
    # σ[i] = 3.
    # θ[i] = sampleθ(θ[i-1], .1, X, y, u1, u2, β[i, :], α, σ[i])
    θ[i] = 1.
    if i % 5000 === 0
        interval = θinterval(X, y, u1, u2, β[i,:], α, σ[i])
        printfmt("iter: {1}, θ ∈ [{2:.2f}, {3:.2f}], σ = {4:.2f} \n", i, interval[1], interval[2], σ[i])
    end
end

plot(β[:, 2])
plot(σ)
plot(θ)
plot!(σ)


## Tests of output
id = 88
sampleσ(X, y, U1[id,:], U2[id,:], β[id, :], α, 1., 1, 1.)
sampleβ(X, y, U1[id,:], U2[id,:], β[760,:], α, θ[760], σ[760], 100.)
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
