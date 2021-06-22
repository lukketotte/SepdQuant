using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("../aepd.jl")
include("../../QuantileReg/QuantileReg.jl")
using .AEPD, .QuantileReg

using Plots, PlotThemes, Formatting, CSV, DataFrames, StatFiles, KernelDensity
theme(:juno)


function δ(α::Real, θ::Real)
    return 2*(α*(1-α))^θ / (α^θ + (1-α)^θ)
end

function sampleLatentBlock(X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T, θ::T, σ::T) where {T <: Real}
    u₁, u₂ = zeros(n), zeros(n)
    μ = X*β
    for i ∈ 1:n
        if y[i] <= μ[i]
            l = ((μ[i] - y[i]) / (σ^(1/θ) * α))^θ
            u₁[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        else
            l = ((y[i] - μ[i]) / (σ^(1/θ) * (1-α)))^θ
            u₂[i] = rand(truncated(Exponential(1/δ(α, θ)), l, Inf), 1)[1]
        end
    end
    u₁, u₂
end

function θBlockCond(θ::T, X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T) where {T <: Real}
    n = length(y)
    z  = y-X*β
    pos = findall(z .> 0)
    a = δ(α, θ)*(sum(abs.(z[Not(pos)]).^θ)/α^θ + sum(z[pos].^θ)/(1-α)^θ)
    n/θ * log(δ(α, θ))  - n*log(gamma(1+1/θ)) - n*log(a)/θ + loggamma(n/θ)
end

function sampleθBlock(θ::T, X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1},
    α::T, ε::T) where {T <: Real}
    prop = rand(Uniform(maximum([0., θ-ε]), θ + ε), 1)[1]
    θBlockCond(prop, X, y, β, α) - θBlockCond(θ, X, y, β, α) >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

function sampleσBlock(X::Array{T, 2}, y::Array{T, 1}, β::Array{T, 1}, α::T, θ::T) where {T, N <: Real}
    z = y - X*β
    pos = findall(z .> 0)
    b = (δ(α, θ) * sum(abs.(z[Not(pos)]).^θ) / α^θ) + (δ(α, θ) * sum(abs.(z[pos]).^θ) / (1-α)^θ)
    rand(InverseGamma(length(y)/θ, b), 1)[1]
end

function logβCond(β::Array{T, 1}, X::Array{T, 2}, y::Array{T, 1}, α::T, θ::T,
        σ::T, τ::T, λ::Array{T, 1}) where {T <: Real}
    z = y - X*β
    pos = findall(z .> 0)
    b = δ(α, θ)/σ * (sum(abs.(z[Not(pos)]).^θ) / α^θ + sum(abs.(z[pos]).^θ) / (1-α)^θ)
    -b -1/(2*τ) * β'*diagm(λ.^(-2))*β
end

"""
Computes the gradient of ∫ π(β,σ|⋅)dσ wrt. β
"""
function ∇ᵦ(β::Array{T, 1}, X::Array{T, 2}, y::Array{T, 1}, α::T, θ::T, σ::T,
        τ::T, λ::Array{T, 1}) where {T <: Real}
    z = y - X*β
    posId = findall(z.>0)
    p=length(β)
    ∇ = zeros(p)
    for k in 1:p
        ℓ₁ = θ/α^θ * sum(abs.(z[Not(posId)]).^(θ-1) .* X[Not(posId), k])
        ℓ₂ = θ/(1-α)^θ * sum(z[posId].^(θ-1) .* X[posId, k])
        ∇[k] = -δ(α,θ)/σ * (ℓ₁ - ℓ₂) - β[k]/(τ^2 * λ[k]^2)
    end
    ∇
end


function βMh(β::Array{T, 1}, ε::Array{T, 1},  X::Array{T, 2}, y::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    # prop = vec(rand(MvNormal(β, ε), 1))
    # try MALA sampling
    λ = abs.(rand(Cauchy(0,1), length(β)))
    ∇ = ∇ᵦ(β, X, y, α, θ, σ, 100., λ)
    μ = β + ε.^2 ./ 2 .* ∇
    # prop = β - ε^2/2 .* ∇
    prop = rand(MvNormal(μ, diagm(ε)), 1) |> vec
    α₁ = logβCond(prop, X, y, α, θ, σ, 100., λ) - logβCond(β, X, y, α, θ, σ, 100., λ)
    α₁ > log(rand(Uniform(0,1), 1)[1]) ? prop : β
end

function sampleβBlock(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, θ::T, σ::T, τ::T) where {T <: Real}
    n, p = size(X)
    βsim = zeros(p)
    for k in 1:p
        l, u = [-Inf], [Inf]
        for i in 1:n
            a = (y[i] - X[i, 1:end .!= k] ⋅  β[1:end .!= k]) / X[i, k]
            b₁ = α*σ^(1/θ)*(u₁[i]^(1/θ)) / X[i, k]
            b₂ = (1-α)*σ^(1/θ)*(u₂[i]^(1/θ)) / X[i, k]
            if (u₁[i] > 0) && (X[i, k] < 0)
                append!(l, a + b₁)
            elseif (u₂[i] > 0) && (X[i, k] > 0)
                append!(l, a - b₂)
            elseif (u₁[i] > 0) && (X[i, k] > 0)
                append!(u, a + b₁)
            elseif (u₂[i] > 0) && (X[i, k] < 0)
                append!(u, a - b₂)
            end
        end
        # Horse-shoe prior
        λ = 1.# abs(rand(Cauchy(0 , 1), 1)[1])
        βsim[k] =  maximum(l) < minimum(u) ? rand(truncated(Normal(0, λ*τ), maximum(l), minimum(u)), 1)[1] : β[k]
    end
    βsim
end

## test
n = 500;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(aepd(0., σ^(1/θ), θ, α), n);

par = MCMCparams(y, X, 10000, 10, 1000)
@time b, o, s = MCMC(par, 0.5, 100., 0.05, [0.05, 0.001], [2.1, 0.8], 2., 1., true)

nMCMC = 50000
β = zeros(nMCMC, 2)
β[1,:] = [2.1, 0.8]
σ, θ = zeros(nMCMC), zeros(nMCMC)
σ[1] = 2^(1/1.)
θ[1] = 1.
# seems like θ goes towards 1 with this sampling order
for i ∈ 2:nMCMC
    if i % 5000 === 0
        println("iter: ", i)
    end
    θ[i] = sampleθBlock(θ[i-1], X, y, β[i-1,:], α, 0.05)
    σ[i] = sampleσBlock(X, y, β[i-1,:], α, θ[i])
    β[i,:] = βMh(β[i-1,:], [0.05, 0.001], X, y, α, θ[i], σ[i], 100.)
end

plot(θ, label="θ")
plot(σ, label="σ")
plot(β[:, 1], label="β")

median(β[50000:nMCMC, 1])
median(σ[1000:nMCMC])
median(θ[1000:nMCMC])

1-((β[2:nMCMC, 1] .=== β[1:(nMCMC - 1), 1]) |> mean)
1-((b[2:length(o), 1] .=== b[1:(length(o) - 1), 1]) |> mean)

thin = ((par.burnIn:par.nMCMC) .% par.thin) .=== 0
(β[par.burnIn:par.nMCMC,:])[thin,:]
(β[par.burnIn:par.nMCMC,:])[thin,:]

length(thin[par.burnIn:par.nMCMC])

thin = ((1:nMCMC) .% 5) .=== 0

median(o)
median(s)
plot(o, label="o")

plot(β[:, 2], label="β")
plot(b[:, 2], label="b")

median(β[1000:nMCMC, 1])
median(b[:, 2])

@view b[1:10, 1]
view(b, 1:10, 1)
##
dat = load(string(pwd(), "/Tests/data/nsa_ff.dta")) |> DataFrame
dat = dat[:, Not(filter(c -> count(ismissing, dat[:,c])/size(dat,1) > 0.05, names(dat)))]
dropmissing!(dat)

y₁ = Float64.(dat."fatality_lag_ln")
colSub = [:intensity, :pop_dens_ln, :foreign_f, :ethnic, :rebstrength, :loot,
    :territorial,  :length, :govtbestfatal_ln]
X = Float64.(dat[:, colSub] |> Matrix)
y₁ = y₁[y₁.>0]
X = X[findall(y₁.>0),:]
X = hcat([1 for i in 1:260], X)
α, n = 0.5, length(y₁)


nMCMC = 1000000
β = zeros(nMCMC, 10)
β[1,:] = inv(X'*X)*X'*y₁
σ, θ = zeros(nMCMC), zeros(nMCMC)
σ[1] = 1.
θ[1] = 1.

for i ∈ 2:nMCMC
    if i % 10000 === 0
        println("iter: ", i)
    end
    global y = log.((exp.(y₁) + rand(Uniform(), length(y₁)) .- α))
    θ[i] = sampleθBlock(θ[i-1], X, y, β[i-1,:], α, 0.05)
    σ[i] = sampleσBlock(X, y, β[i-1,:], α, θ[i])
    global u1, u2 = sampleLatentBlock(X, y, β[i-1,:], α, θ[i], σ[i])
    β[i,:] = sampleβBlock(X, y, u1, u2, β[i-1,:], α, θ[i], σ[i], 100.)
end

thin = ((1:nMCMC) .% 20) .=== 0
plot(θ[thin], label="θ")
plot(σ[thin], label="σ")
plot(β[thin,10], label="β")

plot(cumsum(β[:,10]) ./ (1:length(σ)))
plot(cumsum(θ) ./ (1:length(σ)))
median(β[thin, 7])


p = 10
b1 = kde(β[thin, p])
x = range(median(β[thin, p])-1, median(β[thin, p])+1, length = 1000)
plot(x, pdf(b1, x))
