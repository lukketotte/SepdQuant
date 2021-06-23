using StaticArrays, Distributions, LinearAlgebra, StatsBase, SpecialFunctions
include("../aepd.jl")
using .AEPD
using ProgressMeter

n = 500;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = ([repeat([1], n) rand(Uniform(10, 20), n)])
ϵ = rand(aepd(0., σ^(1/θ), θ, α), n);
y = X * β .+ ϵ
Xs = SMatrix{n,2}(X)
βs = SVector{2}(β)
ys = Xs*βs .+ SVector{n}(ϵ)

function mcmc(params::MCMCparams, α::Real, τ::Real, ε::Real = 0.05, εᵦ::Union{Real, Array{<:Real, 1}} = 0.01,
        β₁::Union{MixedVec, Nothing} = nothing, σ₁::Real = 1, θ₁::Real = 1, MALA::Bool = true)
    # TODO: validation
    n, p = size(params.X)
    β = zeros(params.nMCMC, p)
    σ = zeros(params.nMCMC)
    θ = similar(σ)
    β[1,:] = typeof(β₁) <: Nothing ? inv(X'*X)*X'*y : β₁
    σ[1], θ[1] = σ₁, θ₁

    p = Progress(params.nMCMC-1, dt=0.5,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen=50, color=:yellow)

    for i ∈ 2:params.nMCMC
        next!(p; showvalues=[(:iter,i) (:θ, round(θ[i-1], digits = 3)) (:σ, round(σ[i-1], digits = 3))])
        mcmcInner!(θ, σ, β, i, params, ε, εᵦ, α, τ, MALA)
    end
    return mcmcThin(θ, σ, β, params)
end

function mcmcInner!(θ::Array{<:Real, 1}, σ::Array{<:Real, 1}, β::MixedMat, i::Int, params::MCMCparams, ε::Real,
    εᵦ::Union{Real, Array{<:Real, 1}}, α::Real, τ::Real, MALA::Bool)
        θ[i] = sampleθ(θ[i-1], params.X, params.y, β[i-1,:], α, ε)
        σ[i] = sampleσ(params.X, params.y, β[i-1,:], α, θ[i])
        β[i,:] = sampleβ(β[i-1,:], εᵦ, params.X, params.y, α, θ[i], σ[i], τ, MALA)
        nothing
end

function mcmcThin(θ::Array{<:Real, 1}, σ::Array{<:Real, 1}, β::Array{<:Real, 2}, params::MCMCparams) where {M <: MVector, A <: MArray}
    thin = ((params.burnIn:params.nMCMC) .% params.thin) .=== 0

    β = (β[params.burnIn:params.nMCMC,:])[thin,:]
    θ = (θ[params.burnIn:params.nMCMC])[thin]
    σ = (σ[params.burnIn:params.nMCMC])[thin]
    return β, θ, σ
end

par = MCMCparams(y, X, 10000, 1, 1)
@time mcmc(par, α, 100.)

pars = MCMCparams(ys, Xs, 10000, 1, 1)
@time a,b,c = mcmc(pars, α, 100.)

SVector{size(a)[2]}(a[2-1,:])
@time mcmcInner!(b, c, a, 5, pars, 1., 1., α, 100, true)
