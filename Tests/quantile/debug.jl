include("QR.jl")
using .QR
using Distributions, LinearAlgebra, StatsBase, SpecialFunctions
using Plots, PlotThemes, CSV, DataFrames, StatFiles

##
n = 2000
β = [2.1, 0.8]
α, θ, σ = 0.5, 1., 1.
x₂ = rand(Uniform(-3, 3), n)
X = [repeat([1], n) x₂]
y = X * β .+ rand(Laplace(0, σ), n)

function θintervalNew(X::Array{T, 2}, y::Array{T, 1}, u₁::Array{T, 1}, u₂::Array{T, 1},
    β::Array{T, 1}, α::T, σ::T) where {T <: Real}
    n = length(y)
    id_pos = findall((X*β .- y) .> 0)
    id_neg = findall((y.-X*β) .> 0)
    ids1 = id_pos[findall(log.((X[id_pos,:]*β - y[id_pos])./(σ*α)) .< 0)]
    ids2 = id_pos[findall(log.((X[id_pos,:]*β - y[id_pos])./(σ*α)) .> 0)]
    ids3 = id_neg[findall(log.((y[id_neg]-X[id_neg,:]*β)./(σ*(1-α))) .< 0)]
    ids4 = id_neg[findall(log.((y[id_neg]-X[id_neg,:]*β)./(σ*(1-α))) .> 0)]

    l1 = maximum(log.(u1[ids1])./log.((X[ids1,:]*β - y[ids1])./(α*σ)))
    l2 = maximum(log.(u2[ids3])./log.(( y[ids3]-X[ids3,:]*β)./((1-α)*σ)))

    up1 = minimum(log.(u1[ids2]) ./ log.((X[ids2,:]*β - y[ids2])./(α*σ)))
    up2 = minimum(log.(u2[ids4]) ./log.((y[ids4]-X[ids4,:]*β)./((1-α)*σ)))

    [maximum([0 l1 l2]) minimum([up1 up2])]
end


u1, u2 = sampleLatent(X, y, β, α, θ, σ)

id_pos = findall((X*β .- y) .> 0)
id_neg = findall((y.-X*β) .> 0)

ids1 = id_pos[findall(log.((X[id_pos,:]*β - y[id_pos])./(σ*α)) .< 0)]
ids2 = id_pos[findall(log.((X[id_pos,:]*β - y[id_pos])./(σ*α)) .> 0)]

ids3 = id_neg[findall(log.((y[id_neg]-X[id_neg,:]*β)./(σ*(1-α))) .< 0)]
ids4 = id_neg[findall(log.((y[id_neg]-X[id_neg,:]*β)./(σ*(1-α))) .> 0)]

l1 = maximum(log.(u1[ids1])./log.((X[ids1,:]*β - y[ids1])./(α*σ)))
l2 = maximum(log.(u2[ids3])./log.(( y[ids3]-X[ids3,:]*β)./((1-α)*σ)))
maximum([0 l1 l2])

up1 = minimum(log.(u1[ids2]) ./ log.((X[ids2,:]*β - y[ids2])./(α*σ)))
up2 = minimum(log.(u2[ids4]) ./log.((y[ids4]-X[ids4,:]*β)./((1-α)*σ)))
minimum([up1 up2])

θinterval(X,y,u1,u2,β,α,σ)
θintervalNew(X,y,u1,u2,β,α,σ)
