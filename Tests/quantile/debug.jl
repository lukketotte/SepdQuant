using Distributions, LinearAlgebra, StatsBase, SpecialFunctions, ForwardDiff

include("../../QuantileReg/QuantileReg.jl")
using.QuantileReg

using Traceur

n = 200;
β, α, σ = [2.1, 0.8], 0.5, 2.;
θ =  1.
X = [repeat([1], n) rand(Uniform(10, 20), n)]
y = X * β .+ rand(Normal(0., 1.), n);
s = Sampler(y, X, 0.5, 1000)

function πθ(θ::Real)
    θ^(-3/2) * √((1+1/θ) * trigamma(1+1/θ))
end

function θcond(s::Sampler, θ::Real, β::AbstractVector{<:Real})
    n = length(z)
    a = gamma(1+1/θ)^θ * kernel(s, β, θ)
    return -log(θ) + loggamma(n/θ) - (n/θ) * log(a) + log(πθ(θ))
end

function sampleθ(s::Sampler, θ::Real, β::AbstractVector{<:Real}, ε::Real)
    prop = rand(Truncated(Normal(θ, ε^2), 0.5, Inf), 1)[1]
    a = logpdf(Truncated(Normal(prop, ε^2), 0.5, Inf), θ) - logpdf(Truncated(Normal(θ, ε^2), 0.5, Inf), prop)
    return θcond(s, prop, β) - θcond(s, θ, β) + a >= log(rand(Uniform(0,1), 1)[1]) ? prop : θ
end

kernel(s::Sampler, β::AbstractVector{<:Real}, θ::Real) = s.y-s.X*β |> z -> (sum((.-z[z.<0]).^θ)/s.α^θ + sum(z[z.>0].^θ)/(1-s.α)^θ)



@trace πθ(2.)
@trace θcond(s, 2., β)


@trace kernel(s, β, 2)
